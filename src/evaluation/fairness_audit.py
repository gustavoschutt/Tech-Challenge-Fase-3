"""
Módulo de Auditoria de Equidade e Avaliação de Viés Algorítmico (Fairness Audit)
Avalia se o modelo preditivo apresenta disparidade de desempenho entre diferentes
regiões geográficas e faixas de desenvolvimento humano (IDHM), garantindo que
as recomendações de políticas públicas não introduzam vieses territoriais.
"""

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import load_and_split_data

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def compute_slice_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcula métricas de desempenho para uma fatia (slice) populacional."""
    if len(y_true) == 0:
        return {}
    
    unique_classes = len(np.unique(y_true))
    auc = roc_auc_score(y_true, y_proba) if unique_classes > 1 else np.nan

    return {
        "Total_Municipios": len(y_true),
        "Metas_Atingidas_Reais": int((y_true == 1).sum()),
        "Metas_Atingidas_Preditas": int((y_pred == 1).sum()),
        "Taxa_Atingimento_Real": float((y_true == 1).mean()),
        "Taxa_Atingimento_Predita": float((y_pred == 1).mean()),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Recall_Risco_Classe0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Precision_Risco_Classe0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Recall_Sucesso_Classe1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Precision_Sucesso_Classe1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_Macro": float(f1_score(y_true, y_pred, average="macro")),
        "ROC_AUC": float(auc),
        "Acuracia_Global": float(accuracy_score(y_true, y_pred)),
    }


def run_fairness_audit():
    print("=" * 75)
    print("⚖️ INICIANDO AUDITORIA DE EQUIDADE E VIÉS ALGORÍTMICO (FAIRNESS AUDIT)")
    print("=" * 75)

    model_path = MODELS_DIR / "best_model_pipeline.pkl"
    data_path = PROJECT_ROOT / "data" / "ml_features.parquet"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em '{model_path}'.")

    model = joblib.load(model_path)
    X_train, y_train, groups_train, X_test, y_test, df_raw = load_and_split_data(
        data_path, temporal_split=True, train_years=[2022, 2023], test_year=2024
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Reassociar metadados territoriais e socioeconômicos para fatiamento
    test_indices = X_test.index
    df_eval = df_raw.loc[test_indices].copy()
    df_eval["y_true"] = y_test.values
    df_eval["y_pred"] = y_pred
    df_eval["y_proba"] = y_proba

    # Categorização do IDHM em Tercis
    df_eval["Faixa_IDHM"] = pd.qcut(
        df_eval["IDHM"], q=3, labels=["Baixo IDHM", "Médio IDHM", "Alto IDHM"]
    )

    audit_records = []

    # 1. Auditoria por Grande Região
    print("\n🗺️ Calculando métricas por Grande Região...")
    for regiao, group in df_eval.groupby("regiao"):
        m = compute_slice_metrics(group["y_true"], group["y_pred"].values, group["y_proba"].values)
        m["Dimensao"] = "Grande Região"
        m["Grupo"] = regiao
        audit_records.append(m)

    # 2. Auditoria por Faixa de IDHM
    print("📈 Calculando métricas por Faixa de IDHM...")
    for faixa, group in df_eval.groupby("Faixa_IDHM", observed=False):
        m = compute_slice_metrics(group["y_true"], group["y_pred"].values, group["y_proba"].values)
        m["Dimensao"] = "Faixa de IDHM"
        m["Grupo"] = str(faixa)
        audit_records.append(m)

    # 3. Métrica Global de Referência
    m_global = compute_slice_metrics(df_eval["y_true"], df_eval["y_pred"].values, df_eval["y_proba"].values)
    m_global["Dimensao"] = "Nacional"
    m_global["Grupo"] = "Brasil (Global)"
    audit_records.append(m_global)

    df_fairness = pd.DataFrame(audit_records)
    # Reordenar colunas
    cols_order = ["Dimensao", "Grupo", "Total_Municipios", "Balanced_Accuracy", "ROC_AUC",
                  "Recall_Risco_Classe0", "Precision_Risco_Classe0", "Recall_Sucesso_Classe1",
                  "Taxa_Atingimento_Real", "Taxa_Atingimento_Predita", "Acuracia_Global"]
    df_fairness = df_fairness[cols_order]

    # Salvar Relatório
    report_file = REPORTS_DIR / "fairness_audit.csv"
    df_fairness.to_csv(report_file, index=False)
    print(f"\n💾 Relatório de auditoria salvo em: {report_file}")
    print("\n" + df_fairness.to_markdown(index=False))

    # 4. Geração de Gráfico de Paridade Regional e por IDHM
    print("\n🎨 Gerando visualização de equidade regional (images/11_fairness_regional_audit.png)...")
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Gráfico 1: Regiões
    df_reg = df_fairness[df_fairness["Dimensao"] == "Grande Região"]
    x = np.arange(len(df_reg))
    width = 0.35

    ax1.bar(x - width/2, df_reg["Recall_Risco_Classe0"] * 100, width, label="Recall Risco (Sensibilidade)", color="#e74c3c", edgecolor="black")
    ax1.bar(x + width/2, df_reg["Balanced_Accuracy"] * 100, width, label="Balanced Accuracy", color="#3498db", edgecolor="black")
    ax1.axhline(m_global["Recall_Risco_Classe0"] * 100, color="#c0392b", linestyle="--", alpha=0.7, label=f"Média Nacional Recall ({m_global['Recall_Risco_Classe0']:.1%})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_reg["Grupo"], rotation=15, fontweight="bold")
    ax1.set_ylabel("Percentual (%)", fontweight="bold")
    ax1.set_title("Equidade por Grande Região", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="lower right")
    ax1.set_ylim(60, 100)

    # Gráfico 2: IDHM
    df_idhm = df_fairness[df_fairness["Dimensao"] == "Faixa de IDHM"]
    x2 = np.arange(len(df_idhm))
    ax2.bar(x2 - width/2, df_idhm["Recall_Risco_Classe0"] * 100, width, label="Recall Risco", color="#e74c3c", edgecolor="black")
    ax2.bar(x2 + width/2, df_idhm["Balanced_Accuracy"] * 100, width, label="Balanced Accuracy", color="#2ecc71", edgecolor="black")
    ax2.axhline(m_global["Recall_Risco_Classe0"] * 100, color="#c0392b", linestyle="--", alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df_idhm["Grupo"], fontweight="bold")
    ax2.set_title("Equidade por Faixa de IDHM", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="lower right")

    plt.suptitle("Auditoria de Equidade (Fairness) — Detecção de Risco Educacional 2024", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = IMAGES_DIR / "11_fairness_regional_audit.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo em: {chart_path}")

    return df_fairness


if __name__ == "__main__":
    run_fairness_audit()
