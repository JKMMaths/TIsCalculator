"""Topological index calculations from the hydrogen-suppressed molecular graph."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from rdkit import Chem


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


def calculate_topological_indices(mol: Chem.Mol) -> dict[str, float]:
    """Compute all requested degree-based topological indices for a hydrogen-suppressed graph."""

    vertex_degrees, edge_degree_pairs = _build_graph_data(mol)
    if not vertex_degrees:
        return {name: 0.0 for name in INDEX_ORDER}

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

    return {
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
        "RSO": rso,
        "MSO": mso,
        "NSO": nso,
        "DPSO": dpso,
        "DSSO": dsso,
    }


INDEX_ORDER = [
    "M1",
    "M2",
    "F",
    "R",
    "ABC",
    "GA",
    "H",
    "HM",
    "SCI",
    "ISI",
    "SO",
    "RSO",
    "MSO",
    "NSO",
    "DPSO",
    "DSSO",
]


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
