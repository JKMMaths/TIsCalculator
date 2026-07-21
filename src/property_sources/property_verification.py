"""Identity checks and conservative experimental-property verification."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .pubchem_client import PubChemClient

def structure_identifiers(mol: Chem.Mol) -> tuple[str, str]:
    return Chem.MolToInchi(mol), Chem.MolToInchiKey(mol)

def _agree(a: dict, b: dict) -> bool:
    if (a.get("normalized_unit") != b.get("normalized_unit") or a.get("temperature") != b.get("temperature")
            or a.get("pressure") != b.get("pressure")):
        return False
    if a["property"] == "Vapor Pressure" and a.get("temperature") != b.get("temperature"):
        return False
    av, bv = a.get("normalized_value"), b.get("normalized_value")
    if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
        return False
    return abs(av - bv) <= max(0.01, 0.05 * max(abs(av), abs(bv), 1))

def verification_status(records: list[dict]) -> str:
    experimental = [r for r in records if r.get("value_type") == "Experimental" and r.get("numeric_value") is not None]
    if not records: return "Not available"
    if not experimental: return "Predicted"
    independent = {(r.get("source"), r.get("reference_number")) for r in experimental}
    if len(independent) == 1: return "Single experimental source"
    for i, first in enumerate(experimental):
        for second in experimental[i + 1:]:
            if (first.get("source"), first.get("reference_number")) != (second.get("source"), second.get("reference_number")) and _agree(first, second):
                return "Cross-verified"
    return "Conflicting values"

def build_property_report(mol: Chem.Mol, client: PubChemClient | None = None) -> dict[str, Any]:
    """Best-effort report; all network failures resolve to unavailable values."""
    inchi, inchikey = structure_identifiers(mol)
    client = client or PubChemClient()
    cid = client.resolve_cid(inchikey)
    computed, records, verified = {}, [], False
    if cid:
        computed = client.computed_properties(cid)
        verified = computed.get("InChIKey") == inchikey
        records = client.experimental_properties(cid)
        computed_labels = {
            "MolecularFormula": "Molecular Formula", "MolecularWeight": "Molecular Weight",
            "XLogP": "XLogP", "TPSA": "Topological Polar Surface Area",
            "HBondDonorCount": "Hydrogen Bond Donor Count", "HBondAcceptorCount": "Hydrogen Bond Acceptor Count",
        }
        for key, label in computed_labels.items():
            if computed.get(key) is not None:
                records.append({"property": label, "original_value": str(computed[key]), "numeric_value": computed[key],
                                "original_unit": None, "normalized_value": computed[key], "normalized_unit": None,
                                "temperature": None, "pressure": None, "description": "PubChem computed property",
                                "value_type": "Predicted", "source": "PubChem PUG REST", "reference_number": None,
                                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{key}/JSON"})
    # RDKit values are distinctly calculations, never experimental records.
    logp, mr = rdMolDescriptors.CalcCrippenDescriptors(mol)
    records += [
        {"property": "MolLogP", "original_value": str(logp), "numeric_value": logp, "original_unit": None, "normalized_value": logp, "normalized_unit": None, "temperature": None, "pressure": None, "description": "RDKit calculated value", "value_type": "RDKit calculated value", "source": "RDKit", "reference_number": None, "source_url": "https://www.rdkit.org/"},
        {"property": "MolMR", "original_value": str(mr), "numeric_value": mr, "original_unit": None, "normalized_value": mr, "normalized_unit": None, "temperature": None, "pressure": None, "description": "RDKit calculated value", "value_type": "RDKit calculated value", "source": "RDKit", "reference_number": None, "source_url": "https://www.rdkit.org/"},
    ]
    groups = defaultdict(list)
    for r in records: groups[r["property"]].append(r)
    for property_name, values in groups.items():
        status = verification_status(values)
        for value in values: value["verification_status"] = status
    return {"inchi": inchi, "inchikey": inchikey, "pubchem_cid": cid, "structure_verified": verified,
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "computed": computed, "records": records}
