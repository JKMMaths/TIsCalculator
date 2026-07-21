"""Offline contract tests for source-preserving property retrieval."""
from io import BytesIO
from unittest.mock import Mock

import openpyxl
import requests
from rdkit import Chem

from src.excel_export import create_excel_workbook
from src.molecule import analyze_molecule
from src.property_sources.property_parser import parse_pug_view
from src.property_sources.property_verification import build_property_report, verification_status
from src.property_sources.pubchem_client import PubChemClient
from src.property_sources.unit_normalizer import normalize_value


def test_successful_cid_resolution():
    client = PubChemClient()
    client._get = Mock(return_value={"IdentifierList": {"CID": [2244]}})
    assert client.resolve_cid("BSYNRYMUTXBXSQ-UHFFFAOYSA-N") == 2244


def test_missing_pubchem_compound():
    client = PubChemClient()
    client._get = Mock(return_value={"IdentifierList": {"CID": []}})
    assert client.resolve_cid("MISSING") is None


def test_nested_pug_view_parsing_and_experimental_classification():
    payload = {"Record": {"Section": [{"Information": [{"Value": {"StringWithMarkup": [{"String": "Melting point: 212 °F"}]}, "Description": "measured at 1 atm", "Reference": [{"ReferenceNumber": 7, "SourceName": "Independent source"}]}]}]}}
    rows = parse_pug_view(payload, "Melting Point", 1)
    assert len(rows) == 1
    assert rows[0]["normalized_value"] == 100
    assert rows[0]["normalized_unit"] == "degC"
    assert rows[0]["value_type"] == "Experimental"


def test_temperature_and_range_conversion():
    assert normalize_value(273.15, "K", "Melting Point") == (0.0, "degC")
    values, unit = normalize_value([32, 212], "°F", "Boiling Point")
    assert values == [0.0, 100.0] and unit == "degC"


def _record(value, source, ref, temp="25 C", kind="Experimental"):
    return {"property": "Melting Point", "numeric_value": value, "normalized_value": value, "normalized_unit": "degC", "temperature": temp, "source": source, "reference_number": ref, "value_type": kind}


def test_multiple_agreeing_independent_sources():
    assert verification_status([_record(100, "A", 1), _record(102, "B", 2)]) == "Cross-verified"


def test_conflicting_values_and_predicted_separation():
    assert verification_status([_record(100, "A", 1), _record(150, "B", 2)]) == "Conflicting values"
    assert verification_status([_record(100, "Model", 1, kind="Predicted")]) == "Predicted"


def test_api_timeout_is_nonfatal_and_missing_property_is_empty():
    client = PubChemClient()
    client._get = Mock(side_effect=requests.Timeout("timeout"))
    assert client.resolve_cid("TIMEOUT") is None
    assert client.computed_properties(1) == {}
    assert client.experimental_properties(1) == []


def test_report_keeps_rdkit_values_separate_from_experimental():
    client = Mock()
    client.resolve_cid.return_value = None
    report = build_property_report(Chem.MolFromSmiles("CCO"), client)
    assert {row["property"] for row in report["records"]} == {"MolLogP", "MolMR"}
    assert all(row["value_type"] == "RDKit calculated value" for row in report["records"])


def test_excel_source_sheet_generation():
    result = analyze_molecule("Ethanol", "CCO")
    result.property_report = {"pubchem_cid": 702, "inchikey": result.inchikey, "retrieved_at": "2026-01-01T00:00:00Z", "structure_verified": True, "records": [_record(100, "Source A", 1)]}
    workbook = openpyxl.load_workbook(BytesIO(create_excel_workbook(result)))
    assert {"Physicochemical Properties", "Raw Property Records", "Data Sources"}.issubset(workbook.sheetnames)
