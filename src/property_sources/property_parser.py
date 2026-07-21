"""Recursive parser for PubChem PUG View information records."""
from __future__ import annotations

import re
from typing import Any, Iterator

from .unit_normalizer import normalize_value, numbers_from_text

def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)

def _value_text(info: dict[str, Any]) -> str:
    value = info.get("Value", {})
    if isinstance(value, dict):
        if "StringWithMarkup" in value:
            return " ".join(x.get("String", "") for x in value["StringWithMarkup"])
        for key in ("String", "Number"):
            if key in value:
                return str(value[key])
    return ""

def parse_pug_view(payload: dict[str, Any], property_name: str, cid: int | None = None) -> list[dict[str, Any]]:
    """Parse all nested Information entries, never selecting just the first report."""
    records: list[dict[str, Any]] = []
    for node in _walk(payload):
        for info in node.get("Information", []) if isinstance(node.get("Information"), list) else []:
            text = _value_text(info)
            if not text:
                continue
            unit_match = re.search(r"(°[CF]|\b[CFK]\b|kcal/mol|kJ/mol|mm\s*Hg|Pa|bar|mN/m|N/m)", text, re.I)
            unit = unit_match.group(1) if unit_match else None
            nums = numbers_from_text(text)
            numeric: float | list[float] | None = nums[0] if len(nums) == 1 else (nums[:2] if len(nums) >= 2 else None)
            normalized, normalized_unit = normalize_value(numeric, unit, property_name)
            refs = info.get("Reference", [])
            refs = refs if isinstance(refs, list) else [refs]
            if not refs:
                refs = [{}]
            for ref in refs:
                description = info.get("Description", "")
                lower = f"{text} {description}".lower()
                kind = "Predicted" if any(word in lower for word in ("predicted", "estimated", "calculated")) else "Experimental"
                records.append({
                    "property": property_name, "reference_number": ref.get("ReferenceNumber"),
                    "source": ref.get("SourceName") or "PubChem", "original_value": text,
                    "numeric_value": numeric, "original_unit": unit, "normalized_value": normalized,
                    "normalized_unit": normalized_unit, "temperature": _condition(text, "(?:at|@)\\s*([-+]?\\d+(?:\\.\\d+)?)\\s*°?([CFK])"),
                    "pressure": _condition(text, "(?:at|@)\\s*([-+]?\\d+(?:\\.\\d+)?)\\s*(mm\\s*Hg|Pa|bar)"),
                    "description": description, "value_type": kind,
                    "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else "https://pubchem.ncbi.nlm.nih.gov/",
                })
    return records

def _condition(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.I)
    return " ".join(match.groups()) if match else None
