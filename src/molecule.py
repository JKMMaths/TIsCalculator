"""Molecule parsing, validation, and 2D/3D generation helpers."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors

from .topological_indices import (
    calculate_edge_degree_distribution,
    calculate_topological_indices,
    calculate_vertex_degree_sequence,
)
from .visualization import generate_3d_structure_png

logger = logging.getLogger(__name__)


@dataclass
class MoleculeAnalysisResult:
    """Container for all molecule analysis outputs."""

    drug_name: str = ""
    entered_smiles: str = ""
    canonical_smiles: str = ""
    mol: Chem.Mol | None = None
    formula: str = ""
    molecular_weight: float = 0.0
    exact_molecular_weight: float = 0.0
    heavy_atom_count: int = 0
    vertex_count: int = 0
    edge_count: int = 0
    ring_count: int = 0
    hb_donors: int = 0
    hb_acceptors: int = 0
    tpsa: float = 0.0
    logp: float = 0.0
    molar_refractivity: float = 0.0
    edge_distribution: list[dict[str, Any]] = field(default_factory=list)
    index_values: dict[str, float] = field(default_factory=dict)
    vertex_degree_sequence: list[int] = field(default_factory=list)
    error_message: str | None = None
    two_d_png: bytes | None = None
    three_d_molblock: str | None = None
    three_d_sdf: bytes | None = None
    three_d_png: bytes | None = None
    three_d_status: str = "Not generated"
    force_field: str = ""
    random_seed: int = 42
    calculation_convention: str = "Hydrogen-suppressed molecular graph with bond order ignored"
    coordinates_3d: list[dict[str, Any]] = field(default_factory=list)
    canonical_smiles_generated: bool = False
    inchi: str = ""
    inchikey: str = ""
    property_report: dict[str, Any] = field(default_factory=dict)


def analyze_molecule(drug_name: str, smiles: str) -> MoleculeAnalysisResult:
    """Parse a SMILES string, validate it, and compute all requested descriptors."""

    result = MoleculeAnalysisResult(drug_name=drug_name or "", entered_smiles=smiles or "")

    if not smiles or not smiles.strip():
        result.error_message = "Please enter a SMILES string."
        return result

    cleaned_smiles = smiles.strip()

    try:
        mol = Chem.MolFromSmiles(cleaned_smiles)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("SMILES parsing failed")
        result.error_message = f"SMILES could not be parsed: {exc}"
        return result

    if mol is None:
        result.error_message = "The supplied SMILES is invalid. Please check the string and try again."
        return result

    if mol.GetNumAtoms() == 0:
        result.error_message = "The supplied SMILES did not produce a valid molecule."
        return result

    if mol.GetNumAtoms() != 0:
        fragment_count = Chem.GetMolFrags(mol, asMols=True)
        if len(fragment_count) > 1:
            result.error_message = (
                "This SMILES contains multiple disconnected components. "
                "Please enter the parent compound as a single connected structure."
            )
            return result

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Sanitization failed")
        result.error_message = f"Molecule sanitization failed: {exc}"
        return result

    if mol.GetNumAtoms() == 0:
        result.error_message = "The molecule could not be sanitized successfully."
        return result

    try:
        mol_no_h = Chem.RemoveHs(mol)
    except Exception:  # pragma: no cover - defensive logging
        mol_no_h = mol

    result.mol = mol
    result.canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    result.canonical_smiles_generated = True
    try:
        result.inchi = Chem.MolToInchi(mol)
        result.inchikey = Chem.MolToInchiKey(mol)
    except Exception:  # Identity generation must not affect topology.
        result.inchi = ""
        result.inchikey = ""

    try:
        result.formula = rdMolDescriptors.CalcMolFormula(mol_no_h)
    except Exception:
        result.formula = ""

    try:
        result.molecular_weight = Descriptors.MolWt(mol_no_h)
    except Exception:
        result.molecular_weight = 0.0

    try:
        result.exact_molecular_weight = Descriptors.ExactMolWt(mol_no_h)
    except Exception:
        result.exact_molecular_weight = 0.0

    result.heavy_atom_count = mol_no_h.GetNumHeavyAtoms()
    result.vertex_count = mol_no_h.GetNumHeavyAtoms()
    result.edge_count = mol_no_h.GetNumBonds()
    result.ring_count = rdMolDescriptors.CalcNumRings(mol_no_h)
    result.hb_donors = rdMolDescriptors.CalcNumHBD(mol_no_h)
    result.hb_acceptors = rdMolDescriptors.CalcNumHBA(mol_no_h)
    result.tpsa = rdMolDescriptors.CalcTPSA(mol_no_h)
    result.logp = rdMolDescriptors.CalcCrippenDescriptors(mol_no_h)[0]
    result.molar_refractivity = rdMolDescriptors.CalcCrippenDescriptors(mol_no_h)[1]

    try:
        result.edge_distribution = calculate_edge_degree_distribution(mol_no_h)
        result.vertex_degree_sequence = calculate_vertex_degree_sequence(mol_no_h)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Edge distribution calculation failed")
        result.error_message = f"Failed to calculate edge-degree distribution: {exc}"
        return result

    try:
        result.index_values = calculate_topological_indices(mol_no_h)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Topological index calculation failed")
        result.error_message = f"Failed to calculate topological indices: {exc}"
        return result

    try:
        result.two_d_png = generate_2d_structure_png(mol)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("2D structure generation failed")
        result.two_d_png = None

    try:
        result.three_d_molblock, result.three_d_sdf, result.three_d_status, result.force_field = generate_3d_structure(mol)
        if result.three_d_molblock:
            result.coordinates_3d = extract_3d_coordinates(mol)
            result.three_d_png = generate_3d_structure_png(mol)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("3D structure generation failed")
        result.three_d_status = "Failed"
        result.force_field = ""
        result.three_d_molblock = None
        result.three_d_sdf = None
        result.coordinates_3d = []

    return result


def generate_2d_structure_png(mol: Chem.Mol) -> bytes:
    """Render a 2D molecule image as a PNG byte stream."""

    molecule_for_drawing = Chem.Mol(mol)
    if molecule_for_drawing.GetNumAtoms() == 0:
        raise ValueError("Molecule has no atoms")

    AllChem.Compute2DCoords(molecule_for_drawing)

    drawer = Draw.MolDraw2DCairo(800, 550)
    opts = drawer.drawOptions()
    opts.setBackgroundColour((1, 1, 1))
    opts.addAtomIndices = False
    opts.bondLineWidth = 2
    opts.padding = 0.05
    Draw.PrepareAndDrawMolecule(drawer, molecule_for_drawing)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def generate_3d_structure(mol: Chem.Mol) -> tuple[str | None, bytes | None, str, str]:
    """Generate a 3D conformer and return a MolBlock, SDF bytes, status, and force field."""

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    try:
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    except Exception:
        try:
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("3D embedding failed")
            return None, None, "Failed", ""

    if mol_with_h.GetNumConformers() == 0:
        return None, None, "Failed", ""

    force_field = "MMFF"
    try:
        if AllChem.MMFFGetMoleculeProperties(mol_with_h) is not None:
            AllChem.MMFFOptimizeMolecule(mol_with_h, confId=0)
        else:
            force_field = "UFF"
            AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)
    except Exception:
        try:
            force_field = "UFF"
            AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Optimization failed")
            return None, None, "Failed", force_field

    molblock = Chem.MolToMolBlock(mol_with_h, includeStereo=True)
    sdf_bytes = _write_sdf_bytes(mol_with_h)
    return molblock, sdf_bytes, "Generated", force_field


def generate_3d_sdf_bytes(mol: Chem.Mol) -> bytes:
    """Export an optimized 3D conformer as SDF bytes."""

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    try:
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    except Exception:
        try:
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
        except Exception:
            raise

    if mol_with_h.GetNumConformers() == 0:
        raise ValueError("No 3D conformer available")

    try:
        if AllChem.MMFFGetMoleculeProperties(mol_with_h) is not None:
            AllChem.MMFFOptimizeMolecule(mol_with_h, confId=0)
        else:
            AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)

    return _write_sdf_bytes(mol_with_h)


def _write_sdf_bytes(mol: Chem.Mol) -> bytes:
    """Serialize a molecule with its conformer to SDF bytes."""

    stream = StringIO()
    writer = Chem.SDWriter(stream)
    writer.write(mol)
    writer.close()
    return stream.getvalue().encode("utf-8")


def extract_3d_coordinates(mol: Chem.Mol | None) -> list[dict[str, Any]]:
    """Extract coordinates from the first 3D conformer if available."""

    if mol is None:
        return []

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    try:
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    except Exception:
        try:
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
        except Exception:
            return []

    if mol_with_h.GetNumConformers() == 0:
        return []

    try:
        if AllChem.MMFFGetMoleculeProperties(mol_with_h) is not None:
            AllChem.MMFFOptimizeMolecule(mol_with_h, confId=0)
        else:
            AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol_with_h, confId=0)
        except Exception:
            return []

    conf = mol_with_h.GetConformer()
    coords: list[dict[str, Any]] = []
    for atom_idx in range(mol_with_h.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        coords.append(
            {
                "atom_index": atom_idx,
                "element": mol_with_h.GetAtomWithIdx(atom_idx).GetSymbol(),
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
            }
        )
    return coords
