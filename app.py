"""Streamlit web application for molecular structure and topological index analysis."""

from __future__ import annotations

import io
import logging
import os
import re
import string
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem

from src.excel_export import create_excel_workbook
from src.molecule import MoleculeAnalysisResult, analyze_molecule, extract_3d_coordinates, generate_3d_sdf_bytes
from src.qsar_analysis import create_qsar_report, run_qsar_analysis
from src.topological_indices import INDEX_METADATA, INDEX_ORDER
from src.visualization import png_to_data_url, render_3d_viewer
from src.property_sources.property_verification import build_property_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Drug Molecular Structure and Topological Index Calculator", page_icon="🧪", layout="wide")


EXAMPLE_SMILES = {
    "Atenolol": "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
    "Paracetamol": "CC(=O)NC1=CC=CC=C1O",
    "Caffeine": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
}

@st.cache_data(ttl=3600, show_spinner=False)
def retrieve_properties_cached(canonical_smiles: str) -> dict[str, Any]:
    """Cache best-effort REST retrieval without caching the RDKit Mol object."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    return build_property_report(mol) if mol is not None else {}

def property_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for record in report.get("records", []):
        conditions = "; ".join(v for v in [record.get("temperature"), record.get("pressure")] if v) or "-"
        rows.append({"Property": record.get("property"), "Selected/display value": record.get("normalized_value"), "Unit": record.get("normalized_unit"), "Value type": record.get("value_type"), "Measurement conditions": conditions, "Source": record.get("source"), "Reference": record.get("reference_number"), "Verification status": record.get("verification_status")})
    return pd.DataFrame(rows)


def _numeric_property_value(value: Any) -> float | None:
    """Extract one finite numeric value without inventing values for ranges/text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if pd.notna(value) else None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _numeric_property_value(value[0])
    return None


def remember_generated_properties(result: MoleculeAnalysisResult) -> None:
    """Add the current molecule to the session's modelling dataset."""
    row: dict[str, Any] = {
        "Drug name": result.drug_name or result.formula,
        "SMILES": result.canonical_smiles,
        "Molecular weight": result.molecular_weight,
        "Exact molecular weight": result.exact_molecular_weight,
        "TPSA": result.tpsa,
        "LogP (RDKit)": result.logp,
        "Molar refractivity": result.molar_refractivity,
    }
    # Retain the first usable value for each sourced property. Conflicting source
    # records are not averaged, because that would obscure experimental context.
    for record in (result.property_report or {}).get("records", []):
        value = _numeric_property_value(record.get("normalized_value"))
        name = str(record.get("property") or "").strip()
        unit = str(record.get("normalized_unit") or "").strip()
        if value is not None and name:
            row.setdefault(f"{name} ({unit})" if unit else name, value)
    history = st.session_state.setdefault("generated_property_rows", [])
    history[:] = [item for item in history if item.get("SMILES") != result.canonical_smiles]
    history.append(row)


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a drug name."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "molecular_topological_indices"


def display_summary_cards(result: MoleculeAnalysisResult) -> None:
    """Show summary cards for key molecular descriptors."""

    cols = st.columns(6)
    values = [
        ("Formula", result.formula or "-"),
        ("MW", f"{result.molecular_weight:.3f}"),
        ("Heavy atoms", result.heavy_atom_count),
        ("Vertices", result.vertex_count),
        ("Edges", result.edge_count),
        ("Rings", result.ring_count),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label=label, value=value)


def format_value(value: float | int) -> str:
    """Format numbers for display in the Streamlit table."""

    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.4f}"


def build_topology_dataframe(result: MoleculeAnalysisResult) -> pd.DataFrame:
    """Create the index dataframe shown in the app."""

    rows = []
    for idx, name in enumerate(INDEX_ORDER, start=1):
        value = result.index_values.get(name)
        metadata = INDEX_METADATA[name]
        rows.append(
            {
                "No.": idx,
                "Topological Index Name": metadata["name"],
                "Symbol": name,
                "Mathematical Formula": metadata["formula"],
                "Calculated Value": value if value is not None else "Requires specification",
            }
        )
    return pd.DataFrame(rows)


def render_qsar_workspace() -> None:
    """Render the dataset-level QSPR/QSAR analysis workspace."""

    st.title("QSPR/QSAR Analysis")
    st.caption("Compare classical machine learning, deep learning, graph-theory, and hybrid AI models on a labelled molecular dataset.")
    with st.expander("Dataset requirements", expanded=True):
        st.write("Upload a CSV containing one molecule per row, a SMILES column, and a measured or labelled endpoint. At least 12 valid labelled molecules are required. Larger, curated datasets produce more meaningful validation results.")
        template = pd.DataFrame({"SMILES": ["CCO", "CCCO", "c1ccccc1"], "Endpoint": [1.2, 1.8, 2.5]})
        st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"), "qspr_qsar_template.csv", "text/csv")

    source = st.radio("Dataset source", ["Automatically generated thermophysical data", "Upload CSV"], horizontal=True)
    if source == "Automatically generated thermophysical data":
        rows = st.session_state.get("generated_property_rows", [])
        data = pd.DataFrame(rows)
        if data.empty:
            st.info("No generated molecules are available yet. Use Molecular Analysis to generate properties; each successfully analysed molecule is added here automatically.")
            return
        st.info(f"Using {len(data)} unique molecule(s) generated in this session. At least 12 molecules with a common endpoint are required for modelling.")
    else:
        uploaded = st.file_uploader("Upload labelled CSV dataset", type=["csv"], key="qsar_csv")
        if uploaded is None:
            st.info("Upload a CSV dataset to configure and run the analysis.")
            return
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The CSV could not be read: {exc}")
            return
    if len(data.columns) < 2:
        st.error("The CSV must contain at least a SMILES column and an endpoint column.")
        return

    st.subheader("Dataset preview")
    st.dataframe(data.head(25), use_container_width=True, hide_index=True)
    left, middle, right = st.columns(3)
    with left:
        default_smiles = list(data.columns).index("SMILES") if "SMILES" in data.columns else 0
        smiles_column = st.selectbox("SMILES column", list(data.columns), index=default_smiles)
    with middle:
        target_options = [column for column in data.columns if column not in {smiles_column, "Drug name"}]
        if not target_options:
            st.error("No usable thermophysical endpoint column is available.")
            return
        target_column = st.selectbox("Thermophysical endpoint/target", target_options)
    with right:
        task = st.selectbox("Analysis task", ["Auto-detect", "Regression", "Classification"])
    test_percent = st.slider("Held-out test set", min_value=15, max_value=40, value=20, step=5)

    if st.button("Run QSPR/QSAR Analysis", type="primary"):
        with st.spinner("Building molecular features and training four models..."):
            try:
                analysis_result = run_qsar_analysis(data, smiles_column, target_column, task, test_percent / 100)
                analysis_result.configuration["Dataset source"] = source
                analysis_result.configuration["Endpoint"] = target_column
                st.session_state["qsar_result"] = analysis_result
            except Exception as exc:
                st.session_state.pop("qsar_result", None)
                st.error(str(exc))
                return

    result = st.session_state.get("qsar_result")
    if result is not None and (
        result.configuration.get("Dataset source") != source
        or result.configuration.get("Endpoint") != target_column
    ):
        result = None
    if result is None:
        return
    st.success(f"{result.task_type.title()} analysis completed.")
    metrics_tab, predictions_tab, report_tab = st.tabs(["Model Comparison", "Test Predictions", "Report and Download"])
    with metrics_tab:
        st.subheader("Held-out performance")
        st.dataframe(result.metrics, use_container_width=True, hide_index=True)
        metric = "R²" if result.task_type == "regression" else "Balanced Accuracy"
        chart = result.metrics.set_index("Model")[[metric]]
        st.bar_chart(chart)
        st.caption("Metrics are calculated only on the held-out test set. They are validation estimates, not experimental confirmation.")
    with predictions_tab:
        st.dataframe(result.predictions, use_container_width=True, hide_index=True)
        if not result.excluded_rows.empty:
            st.subheader("Excluded input rows")
            st.dataframe(result.excluded_rows, use_container_width=True, hide_index=True)
    with report_tab:
        st.subheader("Run configuration")
        st.dataframe(pd.DataFrame(result.configuration.items(), columns=["Setting", "Value"]), use_container_width=True, hide_index=True)
        report = create_qsar_report(result)
        st.download_button("Download QSPR/QSAR Excel Report", report, "qspr_qsar_analysis_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


INDEX_FORMULAS = {
    "M1": "Σ d_v²",
    "M2": "Σ d_u d_v",
    "F": "Σ d_v³",
    "R": "Σ 1/√(d_u d_v)",
    "ABC": "Σ √((d_u+d_v-2)/(d_u d_v))",
    "GA": "Σ 2√(d_u d_v)/(d_u+d_v)",
    "H": "Σ 2/(d_u+d_v)",
    "HM": "Σ (d_u+d_v)²",
    "SCI": "Σ 1/√(d_u+d_v)",
    "ISI": "Σ (d_u d_v)/(d_u+d_v)",
    "SO": "Σ √(d_u²+d_v²)",
    "RSO": "Σ √((d_u-1)²+(d_v-1)²)",
    "MSO": "Σ 1/√(d_u²+d_v²)",
    "NSO": "SO/|E|",
    "DPSO": "Σ d_u d_v √(d_u²+d_v²)",
    "DSSO": "Σ (d_u+d_v)√(d_u²+d_v²)",
}


def main() -> None:
    """Render the Streamlit UI and handle results."""

    workspace = st.sidebar.radio("Workspace", ["Molecular Analysis", "QSPR/QSAR Analysis"])
    if workspace == "QSPR/QSAR Analysis":
        render_qsar_workspace()
        return

    st.title("Drug Molecular Structure and Topological Index Calculator")
    st.caption(
        "Calculate degree-based topological indices from the hydrogen-suppressed molecular graph, then view 2D and 3D structure representations."
    )

    with st.sidebar:
        st.header("About")
        st.write("This app parses a SMILES string, validates it, and computes molecular descriptors and topological indices locally without external APIs.")
        st.header("Graph Convention")
        st.write("Indices are computed from the hydrogen-suppressed graph where heavy atoms are vertices and bonds are edges; bond order is ignored for graph calculations.")
        st.header("Software")
        st.write("Built with Python, Streamlit, RDKit, Pandas, NumPy, py3Dmol, Pillow, and XlsxWriter.")
        st.header("Disclaimer")
        st.write("Calculated descriptors are for research use and should not be treated as experimentally verified properties.")
        show_hydrogens = st.toggle("Show hydrogens in 3D viewer", value=False)

    with st.form("analysis_form"):
        drug_name = st.text_input("Drug name (optional)", value="")
        smiles_input = st.text_area("SMILES", height=120, placeholder="Enter a SMILES string")
        example_choice = st.selectbox("Example SMILES", list(EXAMPLE_SMILES.keys()))
        col1, col2 = st.columns([1, 1])
        with col1:
            submit = st.form_submit_button("Generate Structure and Calculate")
        with col2:
            reset = st.form_submit_button("Reset")

        if reset:
            st.session_state.pop("result", None)
            st.session_state.pop("error", None)
            st.rerun()

        if submit:
            if not smiles_input.strip():
                st.session_state["error"] = "Please enter a SMILES string."
                st.session_state.pop("result", None)
            else:
                with st.spinner("Generating molecule, structure, and indices..."):
                    st.session_state["result"] = analyze_molecule(drug_name, smiles_input)
                    st.session_state["error"] = st.session_state["result"].error_message

    if "result" in st.session_state and st.session_state["result"] is not None:
        result = st.session_state["result"]
        if result.error_message:
            st.error(result.error_message)
            return

        st.success("Analysis completed successfully.")
        display_summary_cards(result)
        if not result.property_report:
            with st.spinner("Retrieving available physicochemical properties..."):
                result.property_report = retrieve_properties_cached(result.canonical_smiles)
        remember_generated_properties(result)

        structures_tab, topology_tab, properties_tab, sources_tab, downloads_tab = st.tabs(["Molecular Structures", "Topological Indices", "Physicochemical Properties", "Sources and Verification", "Downloads"])
        with structures_tab:
            st.subheader("Structure Views")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("### 2D Structure")
                if result.two_d_png:
                    image = Image.open(io.BytesIO(result.two_d_png))
                    st.image(image, use_container_width=True)
                else:
                    st.warning("2D structure image was not generated.")
            with col_right:
                st.markdown("### 3D Structure")
                if result.three_d_molblock:
                    html = render_3d_viewer(result.three_d_molblock, show_hydrogens=show_hydrogens)
                    st.components.v1.html(html, height=500, scrolling=False)
                else:
                    st.warning("3D structure generation failed; the 2D image and topological indices are still available.")

            st.subheader("Molecular Summary")
            summary_df = pd.DataFrame(
                [
                    ["Drug name", result.drug_name or "-"], ["Entered SMILES", result.entered_smiles or "-"],
                    ["Canonical SMILES", result.canonical_smiles or "-"], ["InChIKey", result.inchikey or "-"],
                    ["Molecular formula", result.formula or "-"], ["Molecular weight", f"{result.molecular_weight:.3f}"],
                    ["Exact molecular weight", f"{result.exact_molecular_weight:.3f}"], ["Heavy atoms", result.heavy_atom_count],
                    ["Vertices", result.vertex_count], ["Edges", result.edge_count], ["Rings", result.ring_count],
                    ["Hydrogen-bond donors", result.hb_donors], ["Hydrogen-bond acceptors", result.hb_acceptors],
                    ["Vertex degree sequence", ", ".join(str(value) for value in result.vertex_degree_sequence)],
                    ["Topological polar surface area", f"{result.tpsa:.3f}"], ["LogP", f"{result.logp:.3f}"],
                    ["Molar refractivity", f"{result.molar_refractivity:.3f}"], ["3D embedding status", result.three_d_status], ["Force field", result.force_field or "-"],
                ], columns=["Property", "Value"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with topology_tab:
            st.subheader("Edge-Degree Distribution")
            edge_rows = list(result.edge_distribution)
            if edge_rows:
                edge_rows.append({"edge_type": "Total", "degree_1": "", "degree_2": "", "count": sum(item["count"] for item in edge_rows)})
            edge_df = pd.DataFrame(edge_rows)
            st.dataframe(edge_df, use_container_width=True, hide_index=True)

            st.subheader("Topological Indices")
            index_df = build_topology_dataframe(result)
            st.dataframe(index_df, use_container_width=True, hide_index=True)

        with properties_tab:
            st.caption("Experimental and predicted reports are retained separately. Unavailable database values are never inferred.")
            table = property_table(result.property_report)
            if table.empty:
                st.info("No external physicochemical properties are currently available.")
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)
        with sources_tab:
            report = result.property_report
            st.dataframe(pd.DataFrame([[report.get("pubchem_cid") or "Not available", report.get("inchikey") or result.inchikey, report.get("structure_verified"), report.get("retrieved_at")]], columns=["PubChem CID", "InChIKey", "PubChem structure match", "Retrieved (UTC)"]), use_container_width=True, hide_index=True)
            st.dataframe(pd.DataFrame(report.get("records", [])), use_container_width=True, hide_index=True)

        with downloads_tab:
            excel_bytes = create_excel_workbook(result)
            filename = f"{sanitize_filename(result.drug_name or 'molecular_topological_indices')}.xlsx"
            st.download_button("Download Excel Report", excel_bytes, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if result.two_d_png:
                st.download_button("Download 2D Structure PNG", result.two_d_png, file_name=f"{sanitize_filename(result.drug_name or 'molecule')}.png", mime="image/png")
            if result.three_d_sdf:
                st.download_button("Download 3D Structure SDF", result.three_d_sdf, file_name=f"{sanitize_filename(result.drug_name or 'molecule')}.sdf", mime="chemical/x-mdl-sdfile")
            if result.three_d_png:
                st.download_button("Download 3D Structure PNG", result.three_d_png, file_name=f"{sanitize_filename(result.drug_name or 'molecule')}_3d.png", mime="image/png")

    elif "error" in st.session_state and st.session_state["error"]:
        st.error(st.session_state["error"])


if __name__ == "__main__":
    main()
