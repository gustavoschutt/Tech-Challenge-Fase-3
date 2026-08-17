"""
Módulo de Visualização de Dados e Geração de Gráficos Executivos
Gera figuras de alta resolução (300 DPI) para relatórios, notebooks e README.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

IMAGES_DIR = Path("/home/gusvato/Documentos/FIAP_postech/Fase_3/Tech_Challenge/tech-challenge-fase3/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(y: pd.Series, output_name: str = "01_target_distribution.png"):
    """Gera gráfico de distribuição da variável alvo."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    labels = ["Meta Não Atingida (0)", "Meta Atingida (1)"]
    colors = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Quantidade de Municípios / Observações", fontweight="bold")
    ax.set_title("Distribuição da Variável Alvo (Atingimento da Meta de Alfabetização)", fontsize=13, fontweight="bold", pad=15)
    
    total = len(y)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_correlation_matrix(df: pd.DataFrame, output_name: str = "02_correlation_matrix.png"):
    """Gera matriz de correlação das variáveis socioeconômicas e históricas."""
    numeric_cols = [
        "indicador_lag1", "indicador_lag2", "tendencia_historica",
        "gap_historico_vs_meta_municipio", "IDHM", "PIB_per_capita",
        "quantidade_matriculas", "target_meta_atingida"
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()

    labels_map = {
        "indicador_lag1": "Ind. Lag 1 (t-1)",
        "indicador_lag2": "Ind. Lag 2 (t-2)",
        "tendencia_historica": "Tendência Hist.",
        "gap_historico_vs_meta_municipio": "Gap vs Meta",
        "IDHM": "IDHM",
        "PIB_per_capita": "PIB per capita",
        "quantidade_matriculas": "Matrículas",
        "target_meta_atingida": "Target (Meta Atingida)"
    }
    corr = corr.rename(index=labels_map, columns=labels_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Matriz de Correlação Linear (Features vs Target)", fontsize=13, fontweight="bold", pad=15)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_regional_performance(df: pd.DataFrame, output_name: str = "03_regional_performance.png"):
    """Gera comparativo regional de taxa de alfabetização e atingimento da meta."""
    if "regiao" not in df.columns:
        return
    
    reg_summary = df.groupby("regiao").agg(
        taxa_meta=("target_meta_atingida", "mean"),
        ind_medio=("indicador_lag1", "mean"),
        idhm_medio=("IDHM", "mean"),
        total=("id_municipio", "count")
    ).reset_index()
    reg_summary["taxa_meta_pct"] = reg_summary["taxa_meta"] * 100

    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    sns.barplot(data=reg_summary, x="regiao", y="taxa_meta_pct", palette="Blues_d", ax=ax1, edgecolor="black")
    ax1.set_ylabel("% Municípios com Meta Atingida", fontweight="bold", color="#1f77b4")
    ax1.set_xlabel("Grande Região do Brasil", fontweight="bold")
    ax1.set_title("Desempenho Educacional por Região Brasileira", fontsize=13, fontweight="bold", pad=15)
    ax1.set_ylim(0, 100)

    for i, row in reg_summary.iterrows():
        ax1.text(i, row["taxa_meta_pct"] + 2, f"{row['taxa_meta_pct']:.1f}%", ha="center", fontweight="bold")

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_roc_curves(models_dict: dict, X_test, y_test, output_name: str = "04_roc_curves_comparison.png"):
    """Plota comparação de curvas ROC para múltiplos modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=palette[idx % len(palette)], lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatório (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontweight="bold")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / Recall)", fontweight="bold")
    ax.set_title("Curvas ROC Comparativas dos Modelos de Machine Learning", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_confusion_matrix_heatmap(y_test, y_pred, model_name: str = "Random Forest", output_name: str = "05_confusion_matrix.png"):
    """Gera matriz de confusão anotada."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [["VN", "FP"], ["FN", "VP"]]
    annot = np.empty_like(cm).astype(str)
    
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Não Atingida (0)", "Atingida (1)"],
                yticklabels=["Não Atingida (0)", "Atingida (1)"], ax=ax)
    
    ax.set_xlabel("Predição do Modelo", fontweight="bold")
    ax.set_ylabel("Valor Real (Ground Truth)", fontweight="bold")
    ax.set_title(f"Matriz de Confusão - {model_name}", fontsize=13, fontweight="bold", pad=15)
    
    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")


def plot_risk_quadrant(df: pd.DataFrame, output_name: str = "06_risk_quadrant.png"):
    """Plota quadrante de vulnerabilidade: Indicador t-1 vs Tendência Histórica."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["indicador_lag1"],
        sample["tendencia_historica"],
        c=sample["IDHM"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="none",
        s=35
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("IDHM Municipal", fontweight="bold")
    
    # Linhas de quadrante
    ax.axvline(x=50.0, color="red", linestyle="--", alpha=0.7, label="Corte Crítico (50%)")
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.7, label="Estagnação (Tendência = 0)")
    
    ax.set_xlabel("Indicador de Alfabetização no Ano Anterior (%)", fontweight="bold")
    ax.set_ylabel("Tendência Histórica de Variação (p.p.)", fontweight="bold")
    ax.set_title("Quadrante de Risco Educacional Municipal (Vulnerabilidade vs Evolução)", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper left")
    
    # Anotações dos quadrantes
    ax.text(20, -8, "[ALTO RISCO]\nBaixo Ind. + Queda", color="darkred", fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    ax.text(75, 8, "[ALTO DESEMPENHO]\nAlto Ind. + Crescimento", color="darkgreen", fontweight="bold", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    plt.savefig(IMAGES_DIR / output_name, dpi=300)
    plt.close()
    print(f"📊 Gráfico salvo: {output_name}")
