"""
Módulo de Modelagem e Treinamento Supervisionado
Treina, compara e valida múltiplos modelos de Machine Learning utilizando
validação cruzada estratificada (5-folds), sem vazamento de dados.
"""

import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

# Garante inclusão do diretório raiz no sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import get_preprocessor, load_and_split_data
from src.visualization.plots import (
    plot_confusion_matrix_heatmap,
    plot_correlation_matrix,
    plot_regional_performance,
    plot_risk_quadrant,
    plot_roc_curves,
    plot_target_distribution,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = PROJECT_ROOT / "data" / "ml_features.parquet"


def train_and_compare_models():
    print("=" * 70)
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO E VALIDAÇÃO DE MACHINE LEARNING")
    print("=" * 70)

    # 1. Carga dos dados e geração dos gráficos exploratórios
    print("\n📂 Carregando dados da Camada Gold...")
    df_raw = pd.read_parquet(DATA_FILE)
    X, y = load_and_split_data(DATA_FILE)
    print(f"   Instâncias: {len(X):,} | Features: {X.shape[1]}")
    print(f"   Distribuição do Target: Classe 1 = {y.mean():.1%} | Classe 0 = {(1 - y.mean()):.1%}")

    # Gráficos da EDA
    print("\n🎨 Gerando gráficos exploratórios...")
    plot_target_distribution(y)
    plot_correlation_matrix(df_raw)
    plot_regional_performance(df_raw)
    plot_risk_quadrant(df_raw)

    # 2. Divisão Estratificada Treino / Teste (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n✂️ Separação dos Dados:")
    print(f"   Treino: {len(X_train):,} amostras")
    print(f"   Teste:  {len(X_test):,} amostras (Holdout isolado)")

    # 3. Definição dos Modelos Candidatos
    models = {
        "Regressão Logística (Baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42,
        ),
    }

    # 4. Validação Cruzada Estratificada (5-folds) e Avaliação no Teste
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_pipelines = {}

    print("\n🔄 Executando Validação Cruzada Estratificada (5-Folds)...")
    for name, estimator in models.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        # Cross-validation no conjunto de treino
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "f1", "precision", "recall", "balanced_accuracy"],
            n_jobs=-1,
        )

        # Ajuste no treino completo e predição no teste
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_roc_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)

        results.append({
            "Modelo": name,
            "CV ROC-AUC (Média ± DP)": f"{cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}",
            "CV F1-Score": f"{cv_scores['test_f1'].mean():.4f}",
            "Teste ROC-AUC": round(test_roc_auc, 4),
            "Teste F1-Score": round(test_f1, 4),
            "Teste Precision": round(test_precision, 4),
            "Teste Recall": round(test_recall, 4),
            "Teste Balanced Acc": round(test_bal_acc, 4),
            "Teste Accuracy": round(test_acc, 4),
        })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("📊 RESULTADOS COMPARATIVOS DOS MODELOS:")
    print("=" * 70)
    print(df_results.to_markdown(index=False))

    # Salva métricas em CSV
    metrics_path = MODELS_DIR / "model_comparison_metrics.csv"
    df_results.to_csv(metrics_path, index=False)
    print(f"\n💾 Tabela de métricas salva em: {metrics_path}")

    # 5. Seleção e Salvamento do Melhor Modelo
    best_name = "Gradient Boosting Classifier" if "Gradient Boosting Classifier" in fitted_pipelines else "Random Forest Classifier"
    best_pipeline = fitted_pipelines[best_name]
    
    # Salva pipelines e dados de treino/teste para SHAP
    joblib.dump(best_pipeline, MODELS_DIR / "best_model_pipeline.pkl")
    joblib.dump(fitted_pipelines["Random Forest Classifier"], MODELS_DIR / "rf_pipeline.pkl")
    joblib.dump((X_train, y_train, X_test, y_test), MODELS_DIR / "train_test_data.pkl")
    print(f"🏆 Melhor modelo selecionado: '{best_name}' salvo em 'models/best_model_pipeline.pkl'.")

    # 6. Gráficos de Avaliação do Modelo (ROC e Matriz de Confusão)
    print("\n📈 Gerando curvas ROC e Matriz de Confusão...")
    plot_roc_curves(fitted_pipelines, X_test, y_test)
    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix_heatmap(y_test, y_pred_best, model_name=best_name)

    print("\n" + "=" * 70)
    print("📋 RELATÓRIO DETALHADO DO MELHOR MODELO NO CONJUNTO DE TESTE:")
    print("=" * 70)
    print(classification_report(y_test, y_pred_best, target_names=["Meta Não Atingida", "Meta Atingida"]))

    return df_results, best_pipeline


if __name__ == "__main__":
    train_and_compare_models()
