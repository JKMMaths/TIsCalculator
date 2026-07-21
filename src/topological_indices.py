"""Topological index calculations from the hydrogen-suppressed molecular graph."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem


# Catalogue order, names and formulas transcribed from
# Topological_Indices_Formula_Catalogue.docx.  ``None`` values are used when a
# formula needs a user-selected parameter/partition or an exact exponential
# algorithm; the UI labels these as "Requires specification".
INDEX_CATALOGUE = [
    ("M1", "First Zagreb Index", "Σᵥ dᵥ²"), ("M2", "Second Zagreb Index", "Σᵤᵥ dᵤdᵥ"),
    ("F", "Forgotten Index", "Σᵥ dᵥ³"), ("R", "Randić Index", "Σᵤᵥ 1/√(dᵤdᵥ)"),
    ("ABC", "Atom–Bond Connectivity Index", "Σᵤᵥ √[(dᵤ+dᵥ−2)/(dᵤdᵥ)]"),
    ("GA", "Geometric–Arithmetic Index", "Σᵤᵥ 2√(dᵤdᵥ)/(dᵤ+dᵥ)"),
    ("H", "Harmonic Index", "Σᵤᵥ 2/(dᵤ+dᵥ)"), ("HM", "Hyper-Zagreb Index", "Σᵤᵥ(dᵤ+dᵥ)²"),
    ("SCI", "Sum-Connectivity Index", "Σᵤᵥ 1/√(dᵤ+dᵥ)"),
    ("ISI", "Inverse Sum Indeg Index", "Σᵤᵥ dᵤdᵥ/(dᵤ+dᵥ)"),
    ("SO", "Sombor Index", "Σᵤᵥ √(dᵤ²+dᵥ²)"),
    ("SOred", "Reduced Sombor Index", "Σᵤᵥ √[(dᵤ−1)²+(dᵥ−1)²]"),
    ("mSO", "Modified Sombor Index", "Σᵤᵥ 1/√(dᵤ²+dᵥ²)"),
    ("NSO", "Normalized Sombor Index", "SO/m"),
    ("DPSO", "Degree-Product Sombor Index", "Σᵤᵥ dᵤdᵥ√(dᵤ²+dᵥ²)"),
    ("DSSO", "Degree-Sum Sombor Index", "Σᵤᵥ(dᵤ+dᵥ)√(dᵤ²+dᵥ²)"),
    ("Rα", "General Randić Index", "Σᵤᵥ(dᵤdᵥ)ᵅ"), ("RR", "Reciprocal Randić Index", "Σᵤᵥ√(dᵤdᵥ)"),
    ("χα", "General Sum-Connectivity Index", "Σᵤᵥ(dᵤ+dᵥ)ᵅ"), ("M2*", "Modified Second Zagreb Index", "Σᵤᵥ1/(dᵤdᵥ)"),
    ("Mα", "General Zagreb Index", "Σᵥdᵥᵅ"), ("AZI", "Augmented Zagreb Index", "Σᵤᵥ[dᵤdᵥ/(dᵤ+dᵥ−2)]³"),
    ("SDD", "Symmetric Division Degree Index", "Σᵤᵥ(dᵤ/dᵥ+dᵥ/dᵤ)"),
    ("Alb", "Albertson Irregularity Index", "Σᵤᵥ|dᵤ−dᵥ|"), ("M3", "Third Zagreb Index", "Σᵤᵥ|dᵤ−dᵥ|"),
    ("σ", "Sigma Index", "Σᵤᵥ(dᵤ−dᵥ)²"), ("ReZG1", "First Redefined Zagreb Index", "Σᵤᵥ(dᵤ+dᵥ)/(dᵤdᵥ)"),
    ("ReZG2", "Second Redefined Zagreb Index", "Σᵤᵥdᵤdᵥ/(dᵤ+dᵥ)"),
    ("ReZG3", "Third Redefined Zagreb Index", "Σᵤᵥdᵤdᵥ(dᵤ+dᵥ)"),
    ("AG", "Arithmetic–Geometric Index", "Σᵤᵥ(dᵤ+dᵥ)/(2√(dᵤdᵥ))"),
    ("HA", "Harmonic–Arithmetic Index", "Σᵤᵥ4dᵤdᵥ/(dᵤ+dᵥ)²"), ("N", "Nirmala Index", "Σᵤᵥ√(dᵤ+dᵥ)"),
    ("IN1", "Inverse Nirmala Index", "Σᵤᵥ√(1/dᵤ+1/dᵥ)"), ("0χ", "Zeroth-Order Connectivity Index", "Σᵥ1/√dᵥ"),
    ("0χα", "General Zeroth-Order Connectivity", "Σᵥdᵥᵅ"), ("Π1", "First Multiplicative Zagreb Index", "∏ᵥdᵥ²"),
    ("Π2", "Second Multiplicative Zagreb Index", "∏ᵥdᵥ^(dᵥ)"), ("NK", "Narumi–Katayama Index", "∏ᵥdᵥ"),
    ("ID", "Inverse Degree Index", "Σᵥ1/dᵥ"), ("SOα", "General Sombor Index", "Σᵤᵥ(dᵤ²+dᵥ²)ᵅ"),
    ("SOavr", "Average Sombor Index", "Σᵤᵥ√[(dᵤ−d̄)²+(dᵥ−d̄)²]"),
    ("ESO", "Elliptic Sombor Index", "Σᵤᵥ(dᵤ+dᵥ)√(dᵤ²+dᵥ²)"),
    ("ABC4", "Fourth Atom–Bond Connectivity Index", "Σᵤᵥ√[(Sᵤ+Sᵥ−2)/(SᵤSᵥ)]"),
    ("GA5", "Fifth Geometric–Arithmetic Index", "Σᵤᵥ2√(SᵤSᵥ)/(Sᵤ+Sᵥ)"),
    ("NM1", "Neighbourhood First Zagreb Index", "ΣᵥSᵥ²"), ("NM2", "Neighbourhood Second Zagreb Index", "ΣᵤᵥSᵤSᵥ"),
    ("NF", "Neighbourhood Forgotten Index", "ΣᵥSᵥ³"), ("NRα", "Neighbourhood Randić Index", "Σᵤᵥ(SᵤSᵥ)ᵅ"),
    ("NSCα", "Neighbourhood Sum-Connectivity Index", "Σᵤᵥ(Sᵤ+Sᵥ)ᵅ"),
    ("W", "Wiener Index", "Σᵤ<ᵥd(u,v)"), ("Wα", "Generalized Wiener Index", "Σᵤ<ᵥd(u,v)ᵅ"),
    ("WW", "Hyper-Wiener Index", "½Σᵤ<ᵥ[d(u,v)+d(u,v)²]"), ("Ha", "Harary Index", "Σᵤ<ᵥ1/d(u,v)"),
    ("Hα", "Generalized Harary Index", "Σᵤ<ᵥd(u,v)^(−α)"), ("Wp", "Wiener Polarity Index", "|{{u,v}:d(u,v)=3}|"),
    ("TW", "Terminal Wiener Index", "Σᵤ<ᵥ,dᵤ=dᵥ=1d(u,v)"),
    ("J", "Balaban J Index", "[m/(m−n+2)]Σᵤᵥ1/√(DᵤDᵥ)"), ("Sz", "Szeged Index", "Σₑnᵤ(e)nᵥ(e)"),
    ("Sz*", "Revised Szeged Index", "Σₑ[nᵤ+n₀/2][nᵥ+n₀/2]"), ("PIv", "Vertex Padmakar–Ivan Index", "Σₑ[nᵤ(e)+nᵥ(e)]"),
    ("PIe", "Edge Padmakar–Ivan Index", "Σₑ[mᵤ(e)+mᵥ(e)]"), ("Mo", "Mostar Index", "Σₑ|nᵤ(e)−nᵥ(e)|"),
    ("Moe", "Edge Mostar Index", "Σₑ|mᵤ(e)−mᵥ(e)|"), ("DD", "Degree Distance", "Σᵤ<ᵥ(dᵤ+dᵥ)d(u,v)"),
    ("Gut", "Gutman Index", "Σᵤ<ᵥdᵤdᵥd(u,v)"), ("RDD", "Reciprocal Degree Distance", "Σᵤ<ᵥ(dᵤ+dᵥ)/d(u,v)"),
    ("MTI", "Schultz Molecular Topological Index", "M₁+DD"), ("ξc", "Eccentric Connectivity Index", "Σᵥdᵥεᵥ"),
    ("Cξ", "Connective Eccentricity Index", "Σᵥdᵥ/εᵥ"), ("EDS", "Eccentric Distance Sum", "ΣᵥεᵥDᵥ"),
    ("ζ", "Total Eccentricity Index", "Σᵥεᵥ"), ("ω", "Detour Index", "Σᵤ<ᵥL(u,v)"),
    ("Kf", "Kirchhoff Index", "Σᵤ<ᵥrᵤᵥ"), ("RRi", "Reciprocal Resistance Index", "Σᵤ<ᵥ1/rᵤᵥ"),
    ("Z", "Hosoya Index", "Σₖ≥0pₖ(G)"), ("σG", "Merrifield–Simmons Index", "Σₖ≥0iₖ(G)"),
    ("M(G,x)", "Matching Polynomial", "Σₖ≥0(−1)ᵏpₖ(G)x^(n−2k)"), ("H(G,x)", "Hosoya Polynomial", "Σₖdₖ(G)xᵏ"),
    ("E", "Graph Energy", "Σᵢ|λᵢ|"), ("EE", "Estrada Index", "Σᵢexp(λᵢ)"),
    ("LE", "Laplacian Energy", "Σᵢ|μᵢ−2m/n|"), ("SLE", "Signless Laplacian Energy", "Σᵢ|qᵢ−2m/n|"),
    ("LEL", "Laplacian-Energy-Like Index", "Σᵢ√μᵢ"), ("IE", "Incidence Energy", "Σᵢ√qᵢ"),
    ("DE", "Distance Energy", "Σᵢ|θᵢ|"), ("LEE", "Laplacian Estrada Index", "Σᵢexp(μᵢ)"),
    ("SLEE", "Signless Laplacian Estrada Index", "Σᵢexp(qᵢ)"), ("ρ", "Adjacency Spectral Radius", "maxᵢ|λᵢ|"),
    ("a(G)", "Algebraic Connectivity", "μ₂"), ("IC(P)", "Information Content", "−Σᵢpᵢlog₂pᵢ"),
    ("TIC(P)", "Total Information Content", "n·IC(P)"), ("SIC(P)", "Structural Information Content", "IC(P)/log₂n"),
    ("CIC(P)", "Complementary Information Content", "log₂n−IC(P)"),
]
INDEX_ORDER = [row[0] for row in INDEX_CATALOGUE]
INDEX_METADATA = {symbol: {"name": name, "formula": formula} for symbol, name, formula in INDEX_CATALOGUE}


def _build_graph_data(mol: Chem.Mol) -> tuple[list[int], list[tuple[int, int]]]:
    """Build the hydrogen-suppressed graph data needed for all formula-based indices."""

    mol_without_h = Chem.RemoveHs(mol)
    if mol_without_h.GetNumAtoms() == 0:
        return [], []

    degrees = {
        atom.GetIdx(): atom.GetDegree()
        for atom in mol_without_h.GetAtoms()
    }
    vertex_degrees = list(degrees.values())
    edge_degree_pairs = [
        (
            degrees[bond.GetBeginAtomIdx()],
            degrees[bond.GetEndAtomIdx()],
        )
        for bond in mol_without_h.GetBonds()
    ]
    return vertex_degrees, edge_degree_pairs


def calculate_topological_indices(mol: Chem.Mol) -> dict[str, float | None]:
    """Compute the catalogue indices that have an unambiguous, practical definition.

    Parameterized, partition-dependent and exponential entries are returned as
    ``None`` so callers can distinguish them from a genuine numerical zero.
    """

    vertex_degrees, edge_degree_pairs = _build_graph_data(mol)
    if not vertex_degrees:
        return {name: (None if name in _REQUIRES_SPECIFICATION else 0.0) for name in INDEX_ORDER}

    degrees = vertex_degrees
    edge_pairs = edge_degree_pairs

    m1 = sum(d * d for d in degrees)
    m2 = sum(d_u * d_v for d_u, d_v in edge_pairs)
    f_index = sum(d * d * d for d in degrees)

    randic = sum(1.0 / math.sqrt(d_u * d_v) for d_u, d_v in edge_pairs)
    abc = sum(math.sqrt((d_u + d_v - 2) / (d_u * d_v)) for d_u, d_v in edge_pairs)
    ga = sum((2.0 * math.sqrt(d_u * d_v)) / (d_u + d_v) for d_u, d_v in edge_pairs)
    harmonic = sum(2.0 / (d_u + d_v) for d_u, d_v in edge_pairs)
    hyper_zagreb = sum((d_u + d_v) ** 2 for d_u, d_v in edge_pairs)
    sci = sum(1.0 / math.sqrt(d_u + d_v) for d_u, d_v in edge_pairs)
    isi = sum((d_u * d_v) / (d_u + d_v) for d_u, d_v in edge_pairs)
    so = sum(math.sqrt(d_u * d_u + d_v * d_v) for d_u, d_v in edge_pairs)
    rso = sum(math.sqrt((d_u - 1) ** 2 + (d_v - 1) ** 2) for d_u, d_v in edge_pairs)
    mso = sum(1.0 / math.sqrt(d_u * d_u + d_v * d_v) for d_u, d_v in edge_pairs)
    nso = so / len(edge_pairs) if edge_pairs else 0.0
    dpso = sum(d_u * d_v * math.sqrt(d_u * d_u + d_v * d_v) for d_u, d_v in edge_pairs)
    dsso = sum((d_u + d_v) * math.sqrt(d_u * d_u + d_v * d_v) for d_u, d_v in edge_pairs)

    values: dict[str, float | None] = {
        "M1": m1,
        "M2": m2,
        "F": f_index,
        "R": randic,
        "ABC": abc,
        "GA": ga,
        "H": harmonic,
        "HM": hyper_zagreb,
        "SCI": sci,
        "ISI": isi,
        "SO": so,
        "SOred": rso,
        "mSO": mso,
        "NSO": nso,
        "DPSO": dpso,
        "DSSO": dsso,
    }

    # Compatibility aliases retained for existing callers and saved reports.
    values["RSO"] = rso
    values["MSO"] = mso

    heavy = Chem.RemoveHs(mol)
    n, m = heavy.GetNumAtoms(), heavy.GetNumBonds()
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in heavy.GetBonds()]
    degree = np.asarray(degrees, dtype=float)
    adjacency = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1.0
    neighbour_sum = adjacency @ degree

    prod = lambda seq: float(math.prod(seq))
    values.update({
        "Rα": None, "RR": sum(math.sqrt(a*b) for a, b in edge_pairs), "χα": None,
        "M2*": sum(1/(a*b) for a, b in edge_pairs), "Mα": None,
        "AZI": sum((a*b/(a+b-2))**3 for a, b in edge_pairs if a+b > 2),
        "SDD": sum(a/b+b/a for a, b in edge_pairs),
        "Alb": sum(abs(a-b) for a, b in edge_pairs), "M3": sum(abs(a-b) for a, b in edge_pairs),
        "σ": sum((a-b)**2 for a, b in edge_pairs),
        "ReZG1": sum((a+b)/(a*b) for a, b in edge_pairs), "ReZG2": isi,
        "ReZG3": sum(a*b*(a+b) for a, b in edge_pairs),
        "AG": sum((a+b)/(2*math.sqrt(a*b)) for a, b in edge_pairs),
        "HA": sum(4*a*b/(a+b)**2 for a, b in edge_pairs),
        "N": sum(math.sqrt(a+b) for a, b in edge_pairs),
        "IN1": sum(math.sqrt(1/a+1/b) for a, b in edge_pairs),
        "0χ": sum(1/math.sqrt(d) for d in degrees if d), "0χα": None,
        "Π1": prod(d*d for d in degrees), "Π2": prod(d**d for d in degrees),
        "NK": prod(degrees), "ID": sum(1/d for d in degrees if d), "SOα": None,
        "SOavr": sum(math.hypot(a-2*m/n, b-2*m/n) for a, b in edge_pairs), "ESO": dsso,
    })

    s_pairs = [(neighbour_sum[u], neighbour_sum[v]) for u, v in edges]
    values.update({
        "ABC4": sum(math.sqrt((a+b-2)/(a*b)) for a, b in s_pairs if a*b and a+b >= 2),
        "GA5": sum(2*math.sqrt(a*b)/(a+b) for a, b in s_pairs if a+b),
        "NM1": float(np.sum(neighbour_sum**2)), "NM2": sum(a*b for a, b in s_pairs),
        "NF": float(np.sum(neighbour_sum**3)), "NRα": None, "NSCα": None,
    })

    # All-pairs shortest paths for the distance family.
    distance = np.full((n, n), np.inf)
    np.fill_diagonal(distance, 0)
    distance[adjacency > 0] = 1
    for k in range(n):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    upper = np.triu_indices(n, 1)
    pair_dist = distance[upper]
    transmission = distance.sum(axis=1)
    eccentricity = distance.max(axis=1)
    values.update({
        "W": float(pair_dist.sum()), "Wα": None,
        "WW": float(0.5*np.sum(pair_dist+pair_dist**2)), "Ha": float(np.sum(1/pair_dist)) if len(pair_dist) else 0.0,
        "Hα": None, "Wp": float(np.count_nonzero(pair_dist == 3)),
        "TW": float(sum(distance[u, v] for u in range(n) for v in range(u+1, n) if degree[u] == degree[v] == 1)),
        "J": (m/(m-n+2))*sum(1/math.sqrt(transmission[u]*transmission[v]) for u, v in edges)
             if m-n+2 != 0 and all(transmission) else None,
        "DD": float(sum((degree[u]+degree[v])*distance[u,v] for u in range(n) for v in range(u+1,n))),
        "Gut": float(sum(degree[u]*degree[v]*distance[u,v] for u in range(n) for v in range(u+1,n))),
        "RDD": float(sum((degree[u]+degree[v])/distance[u,v] for u in range(n) for v in range(u+1,n))),
        "MTI": float(m1 + sum((degree[u]+degree[v])*distance[u,v] for u in range(n) for v in range(u+1,n))),
        "ξc": float(np.dot(degree, eccentricity)),
        "Cξ": float(sum(degree[i]/eccentricity[i] for i in range(n) if eccentricity[i])),
        "EDS": float(np.dot(eccentricity, transmission)), "ζ": float(eccentricity.sum()), "ω": None,
    })

    sz = sz_star = piv = pie = mo = moe = 0.0
    edge_endpoints = np.asarray(edges, dtype=int)
    for u, v in edges:
        du, dv = distance[:, u], distance[:, v]
        nu, nv, n0 = np.sum(du < dv), np.sum(dv < du), np.sum(du == dv)
        sz += nu*nv; sz_star += (nu+n0/2)*(nv+n0/2); piv += nu+nv; mo += abs(nu-nv)
        if len(edges):
            eu = np.minimum(distance[edge_endpoints[:, 0], u], distance[edge_endpoints[:, 1], u])
            ev = np.minimum(distance[edge_endpoints[:, 0], v], distance[edge_endpoints[:, 1], v])
            mu, mv = np.sum(eu < ev), np.sum(ev < eu)
            pie += mu+mv; moe += abs(mu-mv)
    values.update({"Sz": float(sz), "Sz*": float(sz_star), "PIv": float(piv), "PIe": float(pie), "Mo": float(mo), "Moe": float(moe)})

    laplacian = np.diag(degree) - adjacency
    resistance = np.linalg.pinv(laplacian) if n > 1 else np.zeros((n, n))
    resistances = [resistance[u,u]+resistance[v,v]-2*resistance[u,v] for u in range(n) for v in range(u+1,n)]
    values["Kf"] = float(sum(resistances)); values["RRi"] = float(sum(1/r for r in resistances if r > 1e-12))

    adj_eig = np.linalg.eigvalsh(adjacency)
    lap_eig = np.clip(np.linalg.eigvalsh(laplacian), 0, None)
    signless_eig = np.clip(np.linalg.eigvalsh(np.diag(degree)+adjacency), 0, None)
    dist_eig = np.linalg.eigvalsh(distance)
    avg_degree = 2*m/n
    values.update({
        "E": float(np.abs(adj_eig).sum()), "EE": float(np.exp(adj_eig).sum()),
        "LE": float(np.abs(lap_eig-avg_degree).sum()), "SLE": float(np.abs(signless_eig-avg_degree).sum()),
        "LEL": float(np.sqrt(lap_eig).sum()), "IE": float(np.sqrt(signless_eig).sum()),
        "DE": float(np.abs(dist_eig).sum()), "LEE": float(np.exp(lap_eig).sum()),
        "SLEE": float(np.exp(signless_eig).sum()), "ρ": float(np.max(np.abs(adj_eig))),
        "a(G)": float(lap_eig[1]) if n > 1 else 0.0,
    })
    for symbol in _REQUIRES_SPECIFICATION:
        values.setdefault(symbol, None)
    return values


_REQUIRES_SPECIFICATION = {
    "Rα", "χα", "Mα", "0χα", "SOα", "NRα", "NSCα", "Wα", "Hα", "ω",
    "Z", "σG", "M(G,x)", "H(G,x)", "IC(P)", "TIC(P)", "SIC(P)", "CIC(P)",
}


def calculate_vertex_degree_sequence(mol: Chem.Mol) -> list[int]:
    """Return the vertex-degree sequence from the hydrogen-suppressed graph."""

    vertex_degrees, _ = _build_graph_data(mol)
    return vertex_degrees


def calculate_edge_degree_distribution(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Return a list of edge-degree distribution entries sorted by endpoint-degree pair."""

    _, edge_degree_pairs = _build_graph_data(mol)

    counts: Counter[tuple[int, int]] = Counter()
    for degree_u, degree_v in edge_degree_pairs:
        pair = tuple(sorted((degree_u, degree_v)))
        counts[pair] += 1

    rows = [
        {
            "edge_type": f"({first},{second})",
            "degree_1": first,
            "degree_2": second,
            "count": count,
        }
        for (first, second), count in sorted(counts.items())
    ]
    return rows
