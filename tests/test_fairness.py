from src.evaluation.fairness_audit import compute_slice_metrics
import numpy as np
import pandas as pd


def test_compute_slice_metrics():
    y_true = pd.Series([1, 0, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.4])

    metrics = compute_slice_metrics(y_true, y_pred, y_proba)

    assert metrics["Total_Municipios"] == 6
    assert metrics["Metas_Atingidas_Reais"] == 3
    assert metrics["Metas_Atingidas_Preditas"] == 3
    assert 0 <= metrics["Balanced_Accuracy"] <= 1
    assert 0 <= metrics["Recall_Risco_Classe0"] <= 1
    assert 0 <= metrics["ROC_AUC"] <= 1
