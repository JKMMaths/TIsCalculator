"""Reproducible network-pharmacology analysis using curated genes and public APIs."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import re
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

STRING_API = "https://string-db.org/api/tsv/network"
ENRICHR_API = "https://maayanlab.cloud/Enrichr"
DEFAULT_LIBRARIES = [
    "GO_Biological_Process_2025", "GO_Molecular_Function_2025",
    "GO_Cellular_Component_2025", "KEGG_2021_Human",
]


@dataclass
class NetworkPharmacologyResult:
    compound: str
    disease: str
    species: int
    drug_targets: pd.DataFrame
    disease_genes: pd.DataFrame
    intersections: pd.DataFrame
    interactions: pd.DataFrame
    hubs: pd.DataFrame
    enrichment: pd.DataFrame


def normalize_genes(values: Iterable[object] | str) -> list[str]:
    """Return unique, uppercase gene symbols while preserving first-seen order."""
    if isinstance(values, str):
        values = re.split(r"[\s,;|]+", values)
    seen, genes = set(), []
    for value in values:
        gene = str(value).strip().upper()
        if gene and gene not in {"NAN", "NONE"} and gene not in seen:
            seen.add(gene)
            genes.append(gene)
    return genes


def gene_table(genes: Iterable[object] | str, source: str, kind: str) -> pd.DataFrame:
    if not source.strip():
        raise ValueError(f"A source/database label is required for {kind} genes.")
    return pd.DataFrame({"Gene": normalize_genes(genes), "Source": source.strip(), "Evidence type": kind})


def intersect_targets(drug_targets: pd.DataFrame, disease_genes: pd.DataFrame) -> pd.DataFrame:
    drug = set(normalize_genes(drug_targets.get("Gene", [])))
    disease = set(normalize_genes(disease_genes.get("Gene", [])))
    return pd.DataFrame({"Gene": sorted(drug & disease)})


def fetch_string_interactions(
    genes: Iterable[str], species: int = 9606, required_score: int = 400,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    genes = normalize_genes(genes)
    columns = ["Protein A", "Protein B", "STRING ID A", "STRING ID B", "Combined score"]
    if len(genes) < 2:
        return pd.DataFrame(columns=columns)
    response = (session or requests.Session()).post(
        STRING_API,
        data={"identifiers": "\r".join(genes), "species": species,
              "required_score": required_score, "network_type": "physical",
              "caller_identity": "drug_topology_network_pharmacology"},
        timeout=(5, 45),
    )
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text), sep="\t")
    if raw.empty:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame({
        "Protein A": raw["preferredName_A"].str.upper(),
        "Protein B": raw["preferredName_B"].str.upper(),
        "STRING ID A": raw["stringId_A"], "STRING ID B": raw["stringId_B"],
        "Combined score": pd.to_numeric(raw["score"], errors="coerce"),
    })
    allowed = set(genes)
    return output[output["Protein A"].isin(allowed) & output["Protein B"].isin(allowed)].drop_duplicates().reset_index(drop=True)


def rank_hubs(genes: Iterable[str], interactions: pd.DataFrame) -> pd.DataFrame:
    graph = nx.Graph()
    graph.add_nodes_from(normalize_genes(genes))
    for row in interactions.itertuples(index=False):
        graph.add_edge(row[0], row[1], weight=float(row[4]))
    columns = ["Rank", "Gene", "Degree", "Degree centrality", "Betweenness centrality", "Closeness centrality", "Eigenvector centrality"]
    if not graph:
        return pd.DataFrame(columns=columns)
    degree = dict(graph.degree())
    dc = nx.degree_centrality(graph)
    bc = nx.betweenness_centrality(graph, weight=None)
    cc = nx.closeness_centrality(graph)
    try:
        ec = nx.eigenvector_centrality(graph, max_iter=1000, weight="weight")
    except nx.NetworkXException:
        ec = {node: 0.0 for node in graph}
    result = pd.DataFrame([{"Gene": gene, "Degree": degree[gene], "Degree centrality": dc[gene],
                            "Betweenness centrality": bc[gene], "Closeness centrality": cc[gene],
                            "Eigenvector centrality": ec[gene]} for gene in graph])
    result = result.sort_values(["Degree", "Betweenness centrality", "Gene"], ascending=[False, False, True]).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result[columns]


def fetch_enrichment(genes: Iterable[str], libraries: list[str] | None = None,
                     session: requests.Session | None = None) -> pd.DataFrame:
    genes = normalize_genes(genes)
    columns = ["Library", "Rank", "Term", "P-value", "Adjusted P-value", "Odds ratio", "Combined score", "Genes"]
    if not genes:
        return pd.DataFrame(columns=columns)
    client = session or requests.Session()
    added = client.post(f"{ENRICHR_API}/addList", files={"list": (None, "\n".join(genes)), "description": (None, "Network pharmacology overlap")}, timeout=(5, 45))
    added.raise_for_status()
    user_list_id = added.json()["userListId"]
    rows = []
    for library in libraries or DEFAULT_LIBRARIES:
        response = client.get(f"{ENRICHR_API}/enrich", params={"userListId": user_list_id, "backgroundType": library}, timeout=(5, 45))
        response.raise_for_status()
        for item in response.json().get(library, []):
            rows.append({"Library": library, "Rank": item[0], "Term": item[1], "P-value": item[2],
                         "Odds ratio": item[3], "Combined score": item[4], "Genes": ";".join(item[5]),
                         "Adjusted P-value": item[6]})
    return pd.DataFrame(rows, columns=columns).sort_values(["Adjusted P-value", "P-value"]).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)


def venn_figure(drug_count: int, disease_count: int, overlap_count: int) -> bytes:
    figure, axis = plt.subplots(figsize=(7, 5), dpi=160)
    from matplotlib.patches import Circle
    axis.add_patch(Circle((0.42, 0.5), 0.28, color="#3182bd", alpha=0.45))
    axis.add_patch(Circle((0.58, 0.5), 0.28, color="#e6550d", alpha=0.45))
    axis.text(0.27, 0.5, str(drug_count - overlap_count), ha="center", va="center", fontsize=16)
    axis.text(0.73, 0.5, str(disease_count - overlap_count), ha="center", va="center", fontsize=16)
    axis.text(0.50, 0.5, str(overlap_count), ha="center", va="center", fontsize=16, fontweight="bold")
    axis.text(0.27, 0.16, "Drug targets", ha="center")
    axis.text(0.73, 0.16, "Disease genes", ha="center")
    axis.set(xlim=(0, 1), ylim=(0, 1)); axis.axis("off"); figure.tight_layout()
    output = BytesIO(); figure.savefig(output, format="png", bbox_inches="tight"); plt.close(figure)
    return output.getvalue()


def network_figure(interactions: pd.DataFrame, hubs: pd.DataFrame) -> bytes:
    graph = nx.Graph()
    for row in interactions.itertuples(index=False):
        graph.add_edge(row[0], row[1], weight=float(row[4]))
    figure, axis = plt.subplots(figsize=(9, 7), dpi=160)
    if graph:
        positions = nx.spring_layout(graph, seed=42, weight="weight")
        degrees = dict(graph.degree())
        nx.draw_networkx(graph, positions, ax=axis, node_size=[250 + 130 * degrees[n] for n in graph],
                         node_color=[degrees[n] for n in graph], cmap="viridis", edge_color="#999999",
                         width=0.8, font_size=7)
    axis.set_title("Drug–disease intersection PPI network"); axis.axis("off"); figure.tight_layout()
    output = BytesIO(); figure.savefig(output, format="png", bbox_inches="tight"); plt.close(figure)
    return output.getvalue()


def create_report(result: NetworkPharmacologyResult) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame({"Setting": ["Compound", "Disease", "NCBI taxon"], "Value": [result.compound, result.disease, result.species]}).to_excel(writer, sheet_name="Summary", index=False)
        for name, frame in [("Drug Targets", result.drug_targets), ("Disease Genes", result.disease_genes),
                            ("Intersection", result.intersections), ("STRING PPI", result.interactions),
                            ("Hub Proteins", result.hubs), ("GO KEGG Enrichment", result.enrichment)]:
            frame.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()


def create_cytoscape_zip(result: NetworkPharmacologyResult) -> bytes:
    output = BytesIO()
    nodes = result.hubs.copy()
    nodes["Type"] = "intersection protein"
    edges = result.interactions.rename(columns={"Protein A": "source", "Protein B": "target", "Combined score": "weight"})
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("nodes.csv", nodes.to_csv(index=False))
        archive.writestr("edges.csv", edges.to_csv(index=False))
        archive.writestr("enrichment.csv", result.enrichment.to_csv(index=False))
        archive.writestr("README.txt", "Import nodes.csv as a node table and edges.csv as an undirected edge table in Cytoscape. Weight is the STRING combined score.\n")
    return output.getvalue()
