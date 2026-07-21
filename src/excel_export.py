"""Excel workbook generation for the topology application."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from xlsxwriter import Workbook

from .molecule import MoleculeAnalysisResult


def create_excel_workbook(result: MoleculeAnalysisResult) -> bytes:
    """Generate an XLSX report in memory and return the workbook bytes."""

    output = BytesIO()
    workbook = Workbook(output, {"in_memory": True})
    bold = workbook.add_format({"bold": True, "color": "white", "bg_color": "#1F4E78"})
    header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7"})
    money = workbook.add_format({"num_format": "0.0000"})

    summary_sheet = workbook.add_worksheet("Molecule Summary")
    summary_sheet.freeze_panes(1, 0)
    summary_sheet.write_row(0, 0, ["Property", "Value"], header)

    summary_rows = [
        ("Drug name", result.drug_name or "-"),
        ("Entered SMILES", result.entered_smiles or "-"),
        ("Canonical SMILES", result.canonical_smiles or "-"),
        ("Molecular formula", result.formula or "-"),
        ("Molecular weight", result.molecular_weight),
        ("Exact molecular weight", result.exact_molecular_weight),
        ("Number of heavy atoms", result.heavy_atom_count),
        ("Number of vertices", result.vertex_count),
        ("Number of edges", result.edge_count),
        ("Number of rings", result.ring_count),
        ("Vertex degree sequence", ", ".join(str(value) for value in result.vertex_degree_sequence)),
        ("Hydrogen-bond donors", result.hb_donors),
        ("Hydrogen-bond acceptors", result.hb_acceptors),
        ("Topological polar surface area", result.tpsa),
        ("LogP", result.logp),
        ("Molar refractivity", result.molar_refractivity),
        ("3D embedding status", result.three_d_status),
        ("Force field", result.force_field or "-"),
        ("Random seed", result.random_seed),
        ("Calculation convention", result.calculation_convention),
    ]
    for row_idx, (prop, value) in enumerate(summary_rows, start=1):
        summary_sheet.write(row_idx, 0, prop, header)
        summary_sheet.write(row_idx, 1, value)

    indices_sheet = workbook.add_worksheet("Topological Indices")
    indices_sheet.freeze_panes(1, 0)
    indices_sheet.write_row(0, 0, ["No.", "Index", "Symbol", "Formula", "Calculated Value"], bold)
    for row_idx, (name, value) in enumerate(result.index_values.items(), start=1):
        indices_sheet.write_number(row_idx, 0, row_idx)
        indices_sheet.write(row_idx, 1, name)
        indices_sheet.write(row_idx, 2, name)
        indices_sheet.write(row_idx, 3, "")
        indices_sheet.write_number(row_idx, 4, value)

    edge_sheet = workbook.add_worksheet("Edge Distribution")
    edge_sheet.freeze_panes(1, 0)
    edge_sheet.write_row(0, 0, ["Edge type", "Endpoint degree 1", "Endpoint degree 2", "Number of edges"], bold)
    distribution_rows = list(result.edge_distribution)
    if distribution_rows:
        distribution_rows.append({"edge_type": "Total", "degree_1": "", "degree_2": "", "count": sum(row["count"] for row in distribution_rows)})
    for row_idx, row in enumerate(distribution_rows, start=1):
        edge_sheet.write(row_idx, 0, row["edge_type"])
        edge_sheet.write(row_idx, 1, row["degree_1"])
        edge_sheet.write(row_idx, 2, row["degree_2"])
        edge_sheet.write_number(row_idx, 3, row["count"])

    structure_sheet = workbook.add_worksheet("Structure")
    structure_sheet.write_row(0, 0, ["Drug name", "Formula", "Canonical SMILES"], bold)
    structure_sheet.write(1, 0, result.drug_name or "-")
    structure_sheet.write(1, 1, result.formula or "-")
    structure_sheet.write(1, 2, result.canonical_smiles or "-")
    if result.two_d_png:
        structure_sheet.insert_image(3, 0, "structure.png", {"image_data": BytesIO(result.two_d_png)})

    coords_sheet = workbook.add_worksheet("3D Coordinates")
    coords_sheet.write_row(0, 0, ["Atom index", "Element", "X", "Y", "Z"], bold)
    for row_idx, coord in enumerate(result.coordinates_3d, start=1):
        coords_sheet.write_number(row_idx, 0, coord["atom_index"])
        coords_sheet.write(row_idx, 1, coord["element"])
        coords_sheet.write_number(row_idx, 2, coord["x"])
        coords_sheet.write_number(row_idx, 3, coord["y"])
        coords_sheet.write_number(row_idx, 4, coord["z"])

    report = result.property_report or {}
    properties_sheet = workbook.add_worksheet("Physicochemical Properties")
    property_headers = ["Property", "Selected/display value", "Unit", "Value type", "Measurement conditions", "Source", "Reference", "Verification status"]
    properties_sheet.write_row(0, 0, property_headers, bold)
    records = report.get("records", [])
    seen = set()
    for row_idx, record in enumerate(records, start=1):
        # Each value is retained: conflicting records are intentionally not averaged.
        conditions = "; ".join(x for x in [record.get("temperature"), record.get("pressure")] if x) or "-"
        properties_sheet.write_row(row_idx, 0, [record.get("property"), record.get("normalized_value"), record.get("normalized_unit"), record.get("value_type"), conditions, record.get("source"), record.get("reference_number"), record.get("verification_status")])

    raw_sheet = workbook.add_worksheet("Raw Property Records")
    raw_headers = ["Property", "Original value", "Original unit", "Normalized value", "Normalized unit", "Temperature", "Pressure", "Description", "Value type", "Source", "Reference", "Source URL"]
    raw_sheet.write_row(0, 0, raw_headers, bold)
    for row_idx, record in enumerate(records, start=1):
        raw_sheet.write_row(row_idx, 0, [record.get("property"), record.get("original_value"), record.get("original_unit"), record.get("normalized_value"), record.get("normalized_unit"), record.get("temperature"), record.get("pressure"), record.get("description"), record.get("value_type"), record.get("source"), record.get("reference_number"), record.get("source_url")])

    sources_sheet = workbook.add_worksheet("Data Sources")
    sources_sheet.write_row(0, 0, ["PubChem CID", "InChIKey", "Retrieval timestamp", "Structure verified", "Source URL"], bold)
    sources_sheet.write_row(1, 0, [report.get("pubchem_cid"), report.get("inchikey") or result.inchikey, report.get("retrieved_at"), report.get("structure_verified"), f"https://pubchem.ncbi.nlm.nih.gov/compound/{report.get('pubchem_cid')}" if report.get("pubchem_cid") else "-"])

    for sheet in (properties_sheet, raw_sheet, sources_sheet):
        sheet.freeze_panes(1, 0)
        sheet.set_column(0, 12, 20)

    workbook.close()
    output.seek(0)
    return output.getvalue()
