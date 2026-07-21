"""Reliable, source-preserving physicochemical property retrieval."""

from .property_verification import build_property_report
from .pubchem_client import PubChemClient

__all__ = ["PubChemClient", "build_property_report"]
