"""
Módulo de Avaliação e Interpretabilidade com SHAP (Explainable AI - XAI)
Calcula valores SHAP para explicar as decisões do modelo nos âmbitos global e local,
identificando os fatores com maior impacto na alfabetização infantil.
"""

from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Garante inclusão do diretório raiz no sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_and_explain(sample_size: int = 2000):
    print("=" * 70)
    print("🧠 INICIANDO ANÁLISE DE INTERPRETABILIDADE COM SHAP (XAI)")
    print("=" * 70)

    pipeline_path = MODELS_DIR / "best_model_pipeline.pkl"
    data_path = MODELS_DIR / "train_test_data.pkl"

    if not pipeline_path.exists() or not data_path.exists():
        raise FileNotFoundError("Modelos não encontrados. Execute 'src/modeling/train.py' primeiro.")

    clf = joblib.load(pipeline_path)
    X_train, y_train, X_test, y_test = joblib.load(data_path)

    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["classifier"]

    # Amostragem para cálculo rápido e eficiente de SHAP
    if len(X_test) > sample_size:
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_test_sample = X_test

    print(f"📊 Transformando {len(X_test_sample)} amostras de teste...")
    X_test_transformed = preprocessor.transform(X_test_sample)

    # Obter nomes das features transformadas
    num_cols = NUMERIC_FEATURES
    cat_cols = preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = np.array(num_cols + cat_cols)

    # Dicionário de tradução para rótulos elegantes nos gráficos
    clean_labels = {
        "indicador_lag1": "Indicador Ano Anterior (t-1)",
        "indicador_lag2": "Indicador 2 Anos Anteriores (t-2)",
        "tendencia_historica": "Tendência Histórica (t-1 - t-2)",
        "gap_historico_vs_meta_municipio": "Gap vs Meta Municipal",
        "gap_historico_vs_meta_nacional": "Gap vs Meta Nacional",
        "meta_municipio": "Meta Municipal Pactuada",
        "meta_nacional": "Meta Nacional Brasil",
        "quantidade_matriculas": "Volume de Matrículas",
        "PIB_per_capita": "PIB per capita Municipal",
        "IDHM": "Índice de Desenv. Humano (IDHM)",
    }
    feature_names_clean = np.array([clean_labels.get(f, f.replace("sigla_uf_", "UF: ").replace("regiao_", "Região: ")) for f in feature_names])

    print("⚙️ Calculando SHAP Values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Formato do SHAP: para classificação binária pode ser lista [class0, class1] ou array 3D
    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]  # Classe 1: Meta Atingida
    elif len(np.shape(shap_values)) == 3:
        shap_values_target = shap_values[:, :, 1]
    else:
        shap_values_target = shap_values

    # 1. SHAP Summary Plot (Beeswarm)
    print("\n🎨 Gerando SHAP Summary Plot (Impacto e Direção das Features)...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values_target,
        X_test_transformed,
        feature_names=feature_names_clean,
        max_display=12,
        show=False,
    )
    plt.title("Impacto das Variáveis na Predição de Alfabetização (SHAP Beeswarm)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = IMAGES_DIR / "06_shap_summary_plot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {summary_path.name}")

    # 2. SHAP Bar Plot (Importância Média Global)
    print("🎨 Gerando SHAP Feature Importance (Bar Plot)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_target,
        X_test_transformed,
        feature_names=feature_names_clean,
        plot_type="bar",
        max_display=12,
        show=False,
    )
    plt.title("Ranking de Importância Global das Variáveis (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = IMAGES_DIR / "07_shap_bar_importance.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Salvo: {bar_path.name}")

    # 3. Tabela de Importância SHAP consolidada
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names_clean,
        "Nome_Original": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    importance_path = REPORTS_DIR / "shap_feature_importance.csv"
    df_importance.to_csv(importance_path, index=False)
    print(f"💾 Tabela de importância salva em: {importance_path}")

    print("\n" + "=" * 70)
    print("🏆 TOP 10 FATORES DETERMINANTES NA ALFABETIZAÇÃO (SHAP):")
    print("=" * 70)
    print(df_importance[["Feature", "Mean_Abs_SHAP"]].head(10).to_markdown(index=False))

    return df_importance


if __name__ == "__main__":
    evaluate_and_explain()
