import io

import pytest
from openpyxl import load_workbook

from src.excel_export import create_excel_workbook
from src.molecule import analyze_molecule, extract_3d_coordinates, generate_3d_sdf_bytes
from src.topological_indices import INDEX_CATALOGUE, INDEX_ORDER, calculate_edge_degree_distribution, calculate_topological_indices
from src.visualization import generate_3d_structure_png

ATENOL_OL_SMILES = "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O"


def test_atenolol_calculation():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)

    assert result is not None
    assert result.formula == "C14H22N2O3"
    assert result.canonical_smiles
    assert result.edge_count == 19
    assert result.vertex_count == 19
    assert result.heavy_atom_count == 19


def test_atenolol_edge_distribution():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    distribution = result.edge_distribution

    assert distribution == [
        {"edge_type": "(1,3)", "degree_1": 1, "degree_2": 3, "count": 5},
        {"edge_type": "(2,2)", "degree_1": 2, "degree_2": 2, "count": 4},
        {"edge_type": "(2,3)", "degree_1": 2, "degree_2": 3, "count": 10},
    ]


def test_vertex_degree_sequence_matches_graph():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    assert result.vertex_degree_sequence
    assert len(result.vertex_degree_sequence) == result.vertex_count
    assert sum(result.vertex_degree_sequence) == 2 * result.edge_count


def test_all_sixteen_indices_for_atenolol():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    values = result.index_values

    expected = {
        "M1": 86.0,
        "M2": 91.0,
        "F": 212.0,
        "R": 8.9692,
        "ABC": 13.9820,
        "GA": 18.1281,
        "H": 8.5000,
        "HM": 394.0,
        "SCI": 8.9721,
        "ISI": 19.7500,
        "SO": 63.1806,
        "RSO": 38.0175,
        "MSO": 5.7689,
        "NSO": 3.3253,
        "DPSO": 309.0221,
        "DSSO": 288.7780,
    }

    for name, value in expected.items():
        assert values[name] == pytest.approx(value, rel=1e-4, abs=1e-4)


def test_formula_catalogue_contains_all_93_entries():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)

    assert len(INDEX_CATALOGUE) == 93
    assert len(INDEX_ORDER) == 93
    assert all(symbol in result.index_values for symbol in INDEX_ORDER)
    assert result.index_values["W"] == pytest.approx(890.0)
    assert result.index_values["Rα"] is None


def test_invalid_smiles():
    result = analyze_molecule("Broken", "not-a-smiles")
    assert result.error_message is not None
    assert "invalid" in result.error_message.lower() or "could not be parsed" in result.error_message.lower()


def test_empty_smiles():
    result = analyze_molecule("", "")
    assert result.error_message is not None


def test_multi_fragment_smiles():
    result = analyze_molecule("Multi", "CCO.CCN")
    assert result.error_message is not None
    assert "single connected structure" in result.error_message.lower() or "multiple fragments" in result.error_message.lower()


def test_molecule_without_edges():
    result = analyze_molecule("Single atom", "[He]")
    assert result.error_message is None
    assert result.edge_count == 0
    assert result.index_values["M1"] == pytest.approx(0.0)


def test_canonical_smiles_generation():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    assert result.canonical_smiles.startswith("CC") or result.canonical_smiles.startswith("N")


def test_3d_coordinate_extraction():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    coords = extract_3d_coordinates(result.mol)
    assert len(coords) > 0


def test_excel_workbook_generation_and_sheets():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    workbook_bytes = create_excel_workbook(result)
    assert isinstance(workbook_bytes, bytes)
    assert len(workbook_bytes) > 0

    workbook = load_workbook(io.BytesIO(workbook_bytes), read_only=True)
    sheet = workbook["Topological Indices"]
    assert sheet["B1"].value == "Topological Index Name"
    assert sheet["B2"].value == "First Zagreb Index"
    assert sheet["C2"].value == "M1"


def test_3d_sdf_bytes_generation():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    sdf_bytes = generate_3d_sdf_bytes(result.mol)
    assert isinstance(sdf_bytes, bytes)
    assert len(sdf_bytes) > 0


def test_3d_structure_png_generation():
    result = analyze_molecule("Atenolol", ATENOL_OL_SMILES)
    png_bytes = generate_3d_structure_png(result.mol)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "c1ccccc1",
        "CCO",
        "CC(C)CCO",
        "c1ccncc1",
    ],
)
def test_valid_smiles_examples(smiles):
    result = analyze_molecule("Example", smiles)
    assert result.error_message is None
    assert result.index_values["M1"] >= 0
