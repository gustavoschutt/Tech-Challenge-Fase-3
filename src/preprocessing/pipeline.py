"""
Módulo de Pré-processamento e Construção de Pipelines Scikit-Learn
Implementa transformações robustas, tratamento de nulos, normalização e encoding
garantindo ZERO Data Leakage para o modelo de Machine Learning.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Definição estrita das features preditoras (sem vazamento da variável alvo)
NUMERIC_FEATURES = [
    "indicador_lag1",
    "indicador_lag2",
    "tendencia_historica",
    "gap_historico_vs_meta_municipio",
    "gap_historico_vs_meta_nacional",
    "meta_municipio",
    "meta_nacional",
    "quantidade_matriculas",
    "PIB_per_capita",
    "IDHM",
]

CATEGORICAL_FEATURES = [
    "sigla_uf",
    "regiao",
]

# Colunas identificadoras e targets a serem removidas da matriz X
EXCLUDE_COLUMNS = [
    "id_municipio",
    "nome",
    "id_uf",
    "ano",
    "indicador_alfabetizacao",
    "meta_atingida",
    "target_meta_atingida",
]


def get_preprocessor() -> ColumnTransformer:
    """
    Constrói o pipeline de pré-processamento Scikit-Learn.
    - Numéricas: Imputação pela mediana + Padronização (StandardScaler)
    - Categóricas: Imputação pelo mais frequente + OneHotEncoder
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def load_and_split_data(
    filepath: str | Path = "data/ml_features.parquet",
    target_col: str = "target_meta_atingida",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega os dados e separa a matriz de features X e o vetor de target y.
    Garante que o target e variáveis vazadas não estejam em X.
    """
    df = pd.read_parquet(filepath)
    
    # Validação do target
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset.")
    
    # Remoção de colunas não-preditoras
    cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col].astype(int)

    return X, y


if __name__ == "__main__":
    X, y = load_and_split_data()
    print(f"✅ Dados carregados com sucesso!")
    print(f"   X shape: {X.shape} | Features: {list(X.columns)}")
    print(f"   y shape: {y.shape} | Taxa de classe positiva: {y.mean():.2%}")
