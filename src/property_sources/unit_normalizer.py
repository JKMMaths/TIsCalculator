"""Unit handling for reported physicochemical measurements."""
from __future__ import annotations

import re
from typing import Any

from pint import UnitRegistry

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)

TARGET_UNITS = {
    "Melting Point": "degC", "Boiling Point": "degC", "Flash Point": "degC",
    "Heat of Vaporization": "kilojoule / mole",
}

_UNIT_ALIASES = {"°C": "degC", "C": "degC", "K": "kelvin", "°F": "degF", "F": "degF",
                 "kJ/mol": "kilojoule / mole", "kcal/mol": "kilocalorie / mole"}

def numbers_from_text(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text or "")]

def normalize_value(value: float | list[float] | None, unit: str | None, property_name: str) -> tuple[Any, str | None]:
    """Convert numeric values where a target unit is mandated; retain otherwise."""
    if value is None or not unit:
        return value, unit
    target = TARGET_UNITS.get(property_name)
    if not target:
        return value, unit
    source = _UNIT_ALIASES.get(unit.strip(), unit.strip())
    try:
        def convert(v: float) -> float:
            return round((v * ureg(source)).to(target).magnitude, 6)
        return ([convert(v) for v in value] if isinstance(value, list) else convert(value), target)
    except Exception:
        return value, unit
