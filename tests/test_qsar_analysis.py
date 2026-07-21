import io

import pandas as pd
from openpyxl import load_workbook

from src.qsar_analysis import create_qsar_report, run_qsar_analysis


def test_regression_analysis_and_report():
    smiles = ["C" * length for length in range(1, 13)] * 2
    data = pd.DataFrame({"SMILES": smiles, "Solubility": [float(len(value)) for value in smiles]})

    result = run_qsar_analysis(data, "SMILES", "Solubility", "Regression", test_size=0.25)

    assert result.task_type == "regression"
    assert len(result.metrics) == 4
    assert set(result.metrics["Model"]) == {
        "Classical ML (Random Forest)", "Deep Learning (MLP)",
        "Graph Theory (Random Forest)", "Hybrid AI (RF, all features)",
    }
    assert not result.predictions.empty

    workbook = load_workbook(io.BytesIO(create_qsar_report(result)), read_only=True)
    assert workbook.sheetnames == ["Run Summary", "Model Metrics", "Test Predictions", "Excluded Rows"]


def test_invalid_rows_are_reported():
    smiles = ["C" * length for length in range(1, 13)] + ["not-smiles"]
    data = pd.DataFrame({"SMILES": smiles, "Target": list(range(13))})

    result = run_qsar_analysis(data, "SMILES", "Target", "Regression")

    assert len(result.excluded_rows) == 1
    assert result.excluded_rows.iloc[0]["Reason"] == "Invalid SMILES"


def test_classification_analysis():
    smiles = ["C" * length for length in range(1, 13)] * 2
    labels = ["low" if len(value) <= 6 else "high" for value in smiles]
    data = pd.DataFrame({"SMILES": smiles, "Activity": labels})

    result = run_qsar_analysis(data, "SMILES", "Activity", "Classification", test_size=0.25)

    assert result.task_type == "classification"
    assert {"Accuracy", "Balanced Accuracy", "F1", "ROC AUC"}.issubset(result.metrics.columns)
    assert set(result.predictions["Actual"]).issubset({"low", "high"})
