"""Small, retrying PubChem PUG REST client (JSON APIs only)."""
from __future__ import annotations
import logging
from typing import Any
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from .property_parser import parse_pug_view

logger = logging.getLogger(__name__)
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
HEADINGS = ["Melting Point", "Boiling Point", "Flash Point", "Vapor Pressure", "Heat of Vaporization", "Surface Tension", "Molar Volume", "Polarizability", "LogP"]

class PubChemClient:
    def __init__(self, session: requests.Session | None = None, timeout: tuple[float, float] = (4, 20)):
        self.session, self.timeout = session or requests.Session(), timeout

    @retry(retry=retry_if_exception_type(requests.RequestException), wait=wait_exponential(min=1, max=4), stop=stop_after_attempt(3), reraise=True)
    def _get(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def resolve_cid(self, inchikey: str) -> int | None:
        try:
            data = self._get(f"{BASE}/pug/compound/inchikey/{inchikey}/cids/JSON")
            cids = data.get("IdentifierList", {}).get("CID", [])
            return int(cids[0]) if cids else None
        except requests.RequestException as exc:
            logger.warning("PubChem CID resolution unavailable: %s", exc)
            return None

    def computed_properties(self, cid: int) -> dict[str, Any]:
        props = "Title,IUPACName,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,CanonicalSMILES,InChI,InChIKey"
        try:
            return self._get(f"{BASE}/pug/compound/cid/{cid}/property/{props}/JSON").get("PropertyTable", {}).get("Properties", [{}])[0]
        except requests.RequestException as exc:
            logger.warning("PubChem properties unavailable: %s", exc)
            return {}

    def experimental_properties(self, cid: int) -> list[dict[str, Any]]:
        records = []
        for heading in HEADINGS:
            try:
                payload = self._get(f"{BASE}/pug_view/data/compound/{cid}/JSON?heading={requests.utils.quote(heading)}")
                records.extend(parse_pug_view(payload, heading, cid))
            except Exception as exc:  # Parsing variance must never stop the local application.
                logger.info("PubChem heading unavailable or unreadable (%s): %s", heading, exc)
        return records
