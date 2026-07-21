"""Helpers for visualizing molecules in Streamlit."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def render_3d_viewer(molblock: str, show_hydrogens: bool = False) -> str:
    """Build an HTML snippet for py3Dmol to display an interactive 3D molecule."""

    viewer = py3Dmol.view(width=500, height=450)
    viewer.addModel(molblock, "mol")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()


def generate_2d_structure_png(mol: Chem.Mol) -> bytes:
    """Render a high-resolution 2D molecule image as a PNG byte stream."""

    molecule_for_drawing = Chem.Mol(mol)
    if molecule_for_drawing.GetNumAtoms() == 0:
        raise ValueError("Molecule has no atoms")

    AllChem.Compute2DCoords(molecule_for_drawing)
    drawer = Draw.MolDraw2DCairo(2400, 1800)
    opts = drawer.drawOptions()
    opts.setBackgroundColour((1, 1, 1))
    opts.addAtomIndices = False
    opts.bondLineWidth = 2
    opts.padding = 0.05
    Draw.PrepareAndDrawMolecule(drawer, molecule_for_drawing)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def generate_3d_structure_png(mol: Chem.Mol) -> bytes:
    """Generate a high-resolution 3D structure PNG using a headless 3D plot."""

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

    conf = mol_with_h.GetConformer()
    coords = [conf.GetAtomPosition(atom_idx) for atom_idx in range(mol_with_h.GetNumAtoms())]
    fig = plt.figure(figsize=(4, 4), dpi=600, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    for bond in mol_with_h.GetBonds():
        start = coords[bond.GetBeginAtomIdx()]
        end = coords[bond.GetEndAtomIdx()]
        ax.plot(
            [start.x, end.x],
            [start.y, end.y],
            [start.z, end.z],
            color="#4C4C4C",
            linewidth=1.6,
        )

    xs = [p.x for p in coords]
    ys = [p.y for p in coords]
    zs = [p.z for p in coords]
    ax.scatter(xs, ys, zs, s=40, color="#244B7A", depthshade=True)

    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return stream.getvalue()


def png_to_data_url(png_bytes: bytes) -> str:
    """Encode PNG bytes as a base64 data URL."""

    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
