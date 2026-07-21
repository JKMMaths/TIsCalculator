"""Dataset-level QSPR/QSAR modelling and report generation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .topological_indices import INDEX_METADATA, INDEX_ORDER, calculate_topological_indices


RANDOM_SEED = 42
MINIMUM_ROWS = 12
GRAPH_SYMBOLS = ["M1", "M2", "F", "R", "ABC", "GA", "H", "HM", "SCI", "ISI", "SO", "SOred", "mSO", "NSO"]


@dataclass
class QSARResult:
    task_type: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    excluded_rows: pd.DataFrame
    configuration: dict[str, Any]


def _molecular_descriptors(mol: Chem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol), rdMolDescriptors.CalcNumRings(mol),
        mol.GetNumHeavyAtoms(), Descriptors.FractionCSP3(mol), Descriptors.MolMR(mol),
    ]


def _graph_descriptors(mol: Chem.Mol) -> list[float]:
    values = calculate_topological_indices(mol)
    return [float(values[symbol] or 0.0) for symbol in GRAPH_SYMBOLS]


def _prepare_dataset(data: pd.DataFrame, smiles_column: str, target_column: str):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    descriptors, graph, fingerprints, targets, source_rows, excluded = [], [], [], [], [], []
    for row_number, (_, row) in enumerate(data.iterrows(), start=2):
        smiles, target = row.get(smiles_column), row.get(target_column)
        mol = Chem.MolFromSmiles(str(smiles).strip()) if pd.notna(smiles) else None
        if mol is None or pd.isna(target):
            excluded.append({"CSV row": row_number, "SMILES": smiles, "Reason": "Invalid SMILES" if mol is None else "Missing target"})
            continue
        descriptors.append(_molecular_descriptors(mol))
        graph.append(_graph_descriptors(mol))
        fingerprints.append(np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=float))
        targets.append(target)
        source_rows.append(row_number)
    return (
        np.asarray(descriptors), np.asarray(graph), np.asarray(fingerprints), np.asarray(targets),
        np.asarray(source_rows), pd.DataFrame(excluded, columns=["CSV row", "SMILES", "Reason"]),
    )


def add_topological_index_columns(data: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """Return a copy with every calculated scalar TI added as a numeric column."""
    augmented = data.copy()
    calculated: list[dict[str, float]] = []
    for value in augmented[smiles_column]:
        mol = Chem.MolFromSmiles(str(value).strip()) if pd.notna(value) else None
        row: dict[str, float] = {}
        if mol is not None:
            indices = calculate_topological_indices(mol)
            for symbol in INDEX_ORDER:
                index_value = indices.get(symbol)
                if index_value is not None and np.isfinite(index_value):
                    row[f"TI | {symbol} | {INDEX_METADATA[symbol]['name']}"] = float(index_value)
        calculated.append(row)
    ti_frame = pd.DataFrame(calculated, index=augmented.index)
    for column in ti_frame:
        augmented[column] = ti_frame[column]
    return augmented


def correlation_table(data: pd.DataFrame, property_column: str, ti_columns: list[str]) -> pd.DataFrame:
    """Calculate pairwise Pearson and Spearman correlations for one property."""
    rows = []
    prop = pd.to_numeric(data[property_column], errors="coerce")
    for column in ti_columns:
        ti = pd.to_numeric(data[column], errors="coerce")
        valid = prop.notna() & ti.notna() & np.isfinite(prop) & np.isfinite(ti)
        if valid.sum() < 3 or prop[valid].nunique() < 2 or ti[valid].nunique() < 2:
            continue
        rows.append({
            "Physicochemical Property": property_column,
            "Topological Index": column.removeprefix("TI | "),
            "TI Column": column,
            "Sample Count": int(valid.sum()),
            "Pearson r": float(prop[valid].corr(ti[valid], method="pearson")),
            "Spearman ρ": float(prop[valid].corr(ti[valid], method="spearman")),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Absolute Pearson r"] = result["Pearson r"].abs()
        result = result.sort_values("Absolute Pearson r", ascending=False).reset_index(drop=True)
    return result


def create_correlation_plot(data: pd.DataFrame, property_column: str, ti_column: str) -> bytes:
    """Render a publication-ready TI-versus-property scatter plot as PNG."""
    x = pd.to_numeric(data[ti_column], errors="coerce")
    y = pd.to_numeric(data[property_column], errors="coerce")
    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        raise ValueError("At least three paired numeric observations are required for a correlation graph.")
    x_values, y_values = x[valid].to_numpy(), y[valid].to_numpy()
    figure, axis = plt.subplots(figsize=(8, 5.5), dpi=160)
    axis.scatter(x_values, y_values, color="#1f77b4", alpha=0.8, edgecolors="white", linewidths=0.5)
    if np.unique(x_values).size > 1:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(x_values.min(), x_values.max(), 100)
        axis.plot(line_x, slope * line_x + intercept, color="#d62728", linewidth=2, label="Linear fit")
        axis.legend()
    pearson = pd.Series(x_values).corr(pd.Series(y_values), method="pearson")
    spearman = pd.Series(x_values).corr(pd.Series(y_values), method="spearman")
    axis.set_title(f"Topological Index vs {property_column}\nPearson r={pearson:.4f}; Spearman ρ={spearman:.4f}; n={len(x_values)}")
    axis.set_xlabel(ti_column.removeprefix("TI | "))
    axis.set_ylabel(property_column)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def create_all_property_graphs_zip(data: pd.DataFrame, property_columns: list[str], ti_columns: list[str]) -> bytes:
    """Create one graph per property using its strongest Pearson-correlated TI."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        summaries = []
        for number, prop in enumerate(property_columns, start=1):
            table = correlation_table(data, prop, ti_columns)
            if table.empty:
                continue
            best = table.iloc[0]
            png = create_correlation_plot(data, prop, best["TI Column"])
            safe_name = "".join(character if character.isalnum() else "_" for character in prop).strip("_")
            archive.writestr(f"{number:02d}_{safe_name}.png", png)
            summaries.append(table.drop(columns=["TI Column"]))
        if summaries:
            archive.writestr("correlation_summary.csv", pd.concat(summaries, ignore_index=True).to_csv(index=False))
    return output.getvalue()


def _resolve_task(target: np.ndarray, requested: str) -> str:
    if requested in {"Regression", "Classification"}:
        return requested.lower()
    numeric = pd.to_numeric(pd.Series(target), errors="coerce")
    if numeric.notna().all() and numeric.nunique() > max(10, int(len(target) * 0.1)):
        return "regression"
    return "classification"


def run_qsar_analysis(
    data: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    task: str = "Auto-detect",
    test_size: float = 0.2,
) -> QSARResult:
    """Train four reproducible QSPR/QSAR pipelines on a held-out split."""

    if smiles_column == target_column:
        raise ValueError("SMILES and endpoint columns must be different.")
    x_desc, x_graph, x_fp, raw_y, source_rows, excluded = _prepare_dataset(data, smiles_column, target_column)
    if len(raw_y) < MINIMUM_ROWS:
        raise ValueError(f"At least {MINIMUM_ROWS} valid labelled molecules are required; found {len(raw_y)}.")

    task_type = _resolve_task(raw_y, task)
    encoder = None
    if task_type == "regression":
        y = pd.to_numeric(pd.Series(raw_y), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Regression endpoints must all be numeric.")
        stratify = None
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_y.astype(str))
        counts = np.bincount(y)
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError("Classification requires at least two classes with two samples per class.")
        stratify = y

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify)
    hybrid = np.hstack([x_desc, x_graph, x_fp])
    if task_type == "regression":
        models = {
            "Classical ML (Random Forest)": (RandomForestRegressor(n_estimators=250, random_state=RANDOM_SEED, n_jobs=1), x_desc),
            "Deep Learning (MLP)": (make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_SEED)), x_fp),
            "Graph Theory (Random Forest)": (RandomForestRegressor(n_estimators=250, random_state=RANDOM_SEED, n_jobs=1), x_graph),
            "Hybrid AI (RF, all features)": (RandomForestRegressor(n_estimators=350, random_state=RANDOM_SEED, n_jobs=1), hybrid),
        }
    else:
        models = {
            "Classical ML (Random Forest)": (RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1), x_desc),
            "Deep Learning (MLP)": (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_SEED)), x_fp),
            "Graph Theory (Random Forest)": (RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1), x_graph),
            "Hybrid AI (RF, all features)": (RandomForestClassifier(n_estimators=350, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1), hybrid),
        }

    metric_rows, prediction_rows = [], []
    for model_name, (model, features) in models.items():
        model.fit(features[train_idx], y[train_idx])
        predicted = model.predict(features[test_idx])
        if task_type == "regression":
            metric_rows.append({"Model": model_name, "R²": r2_score(y[test_idx], predicted), "RMSE": mean_squared_error(y[test_idx], predicted) ** 0.5, "MAE": mean_absolute_error(y[test_idx], predicted)})
            actual_display, predicted_display = y[test_idx], predicted
        else:
            average = "binary" if len(np.unique(y)) == 2 else "weighted"
            row = {"Model": model_name, "Accuracy": accuracy_score(y[test_idx], predicted), "Balanced Accuracy": balanced_accuracy_score(y[test_idx], predicted), "F1": f1_score(y[test_idx], predicted, average=average)}
            if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
                row["ROC AUC"] = roc_auc_score(y[test_idx], model.predict_proba(features[test_idx])[:, 1])
            metric_rows.append(row)
            actual_display = encoder.inverse_transform(y[test_idx])
            predicted_display = encoder.inverse_transform(predicted)
        for idx, actual, prediction in zip(test_idx, actual_display, predicted_display):
            prediction_rows.append({"Model": model_name, "CSV row": int(source_rows[idx]), "SMILES": data.iloc[int(source_rows[idx]) - 2][smiles_column], "Actual": actual, "Predicted": prediction})

    return QSARResult(
        task_type=task_type,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.DataFrame(prediction_rows),
        excluded_rows=excluded,
        configuration={"Task": task_type.title(), "Valid molecules": len(y), "Excluded rows": len(excluded), "Training rows": len(train_idx), "Test rows": len(test_idx), "Test fraction": test_size, "Random seed": RANDOM_SEED, "Fingerprint": "Morgan radius 2, 1024 bits", "Graph features": ", ".join(GRAPH_SYMBOLS)},
    )


def create_qsar_report(result: QSARResult) -> bytes:
    """Create a downloadable Excel workbook containing the full modelling report."""

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(result.configuration.items(), columns=["Setting", "Value"]).to_excel(writer, sheet_name="Run Summary", index=False)
        result.metrics.to_excel(writer, sheet_name="Model Metrics", index=False)
        result.predictions.to_excel(writer, sheet_name="Test Predictions", index=False)
        result.excluded_rows.to_excel(writer, sheet_name="Excluded Rows", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
            sheet.set_column(0, 0, 32)
            sheet.set_column(1, 10, 20)
    return output.getvalue()
