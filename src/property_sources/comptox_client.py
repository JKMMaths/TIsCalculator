"""Optional EPA CompTox client.  Failures are deliberately non-fatal."""
from __future__ import annotations
import logging
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
logger = logging.getLogger(__name__)

class CompToxClient:
    def __init__(self, session: requests.Session | None = None, timeout: tuple[float, float] = (4, 20)):
        self.session, self.timeout = session or requests.Session(), timeout
    @retry(retry=retry_if_exception_type(requests.RequestException), wait=wait_exponential(min=1, max=4), stop=stop_after_attempt(3), reraise=True)
    def query_inchikey(self, inchikey: str) -> list[dict]:
        """Return EPA API data when deployed; empty result keeps EPA strictly optional."""
        try:
            url = "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-search/search"
            response = self.session.get(url, params={"search": inchikey}, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else payload.get("results", [])
        except requests.RequestException as exc:
            logger.info("EPA CompTox unavailable: %s", exc)
            return []
