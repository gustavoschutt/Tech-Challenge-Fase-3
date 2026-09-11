"""
API REST para Inferência em Tempo Real e em Lote (FastAPI)
Permite que secretarias municipais, estaduais e o MEC consultem o risco
educacional de municípios a partir do modelo campeão treinado.
"""

from pathlib import Path
import sys
from typing import List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import numpy as np
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model_pipeline.pkl"
RF_MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.pkl"

app = FastAPI(
    title="API de Predição de Risco Educacional e Alfabetização",
    description="Sistema Preditivo de Alerta Precoce com Explicabilidade XAI (SHAP) para o Compromisso Nacional Criança Alfabetizada",
    version="1.1.0",
)

# Carregamento do modelo e explainer
_pipeline = None
_explainer = None
_rf_pipeline = None
_feature_names = None

FEATURE_LABEL_MAP = {
    "indicador_lag1": "Taxa de Alfabetização Ano Anterior (t-1)",
    "indicador_lag2": "Taxa de Alfabetização há 2 Anos (t-2)",
    "tendencia_historica": "Tendência Histórica de Evolução",
    "gap_historico_vs_meta_municipio": "Distância em Relação à Meta Municipal",
    "gap_historico_vs_meta_nacional": "Distância em Relação à Meta Brasil",
    "meta_municipio": "Meta Municipal Pactuada",
    "meta_nacional": "Meta Nacional Brasil",
    "quantidade_matriculas": "Porte da Rede (Volume de Matrículas)",
    "PIB_per_capita": "PIB per capita Municipal",
    "IDHM": "Índice de Desenvolvimento Humano (IDHM)",
}


def get_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Modelo não encontrado em '{MODEL_PATH}'. Execute o treinamento primeiro.")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_explainer():
    global _explainer, _rf_pipeline, _feature_names
    if _explainer is None:
        if not RF_MODEL_PATH.exists():
            return None, None
        _rf_pipeline = joblib.load(RF_MODEL_PATH)
        preprocessor = _rf_pipeline.named_steps["preprocessor"]
        rf_model = _rf_pipeline.named_steps["classifier"]
        _explainer = shap.TreeExplainer(rf_model)

        cat_cols = preprocessor.transformers_[1][1].named_steps["encoder"].get_feature_names_out().tolist()
        num_cols = [
            "indicador_lag1", "indicador_lag2", "tendencia_historica",
            "gap_historico_vs_meta_municipio", "gap_historico_vs_meta_nacional",
            "meta_municipio", "meta_nacional", "quantidade_matriculas",
            "PIB_per_capita", "IDHM"
        ]
        _feature_names = num_cols + cat_cols
    return _explainer, _rf_pipeline


class MunicipioInput(BaseModel):
    indicador_lag1: float = Field(..., description="Taxa de alfabetização no ano anterior (t-1)", ge=0.0, le=100.0)
    indicador_lag2: Optional[float] = Field(None, description="Taxa de alfabetização há 2 anos (t-2)", ge=0.0, le=100.0)
    tendencia_historica: Optional[float] = Field(None, description="Variação histórica (lag1 - lag2)")
    gap_historico_vs_meta_municipio: Optional[float] = Field(None, description="Distância do indicador t-1 à meta municipal")
    gap_historico_vs_meta_nacional: Optional[float] = Field(None, description="Distância do indicador t-1 à meta nacional")
    meta_municipio: float = Field(..., description="Meta municipal pactuada para o ano t", ge=0.0, le=100.0)
    meta_nacional: float = Field(..., description="Meta nacional para o ano t", ge=0.0, le=100.0)
    quantidade_matriculas: int = Field(..., description="Número de matrículas da rede municipal", ge=0)
    PIB_per_capita: float = Field(..., description="PIB per capita municipal em R$", ge=0.0)
    IDHM: float = Field(..., description="Índice de Desenvolvimento Humano Municipal", ge=0.0, le=1.0)
    sigla_uf: str = Field(..., description="Sigla da Unidade Federativa (ex: 'MA', 'SP')", min_length=2, max_length=2)
    regiao: str = Field(..., description="Grande Região (ex: 'Nordeste', 'Sudeste')")


class FatorExplicativo(BaseModel):
    fator: str
    direcao: str
    impacto: str
    valor_shap: float


class PredictionResult(BaseModel):
    probabilidade_meta_atingida: float
    probabilidade_risco: float
    classificacao: str
    escore_risco_percentual: str
    alerta_prioridade: str
    recomendacao_politica_publica: str
    fatores_determinantes_locais: Optional[List[FatorExplicativo]] = None


def _prepare_dataframe(inputs: List[MunicipioInput]) -> pd.DataFrame:
    records = []
    for item in inputs:
        d = item.model_dump()

        # Preenchimento automático de derivações se não fornecidas explicitamente
        if d["tendencia_historica"] is None and d["indicador_lag2"] is not None:
            d["tendencia_historica"] = round(d["indicador_lag1"] - d["indicador_lag2"], 2)
        if d["gap_historico_vs_meta_municipio"] is None:
            d["gap_historico_vs_meta_municipio"] = round(d["indicador_lag1"] - d["meta_municipio"], 2)
        if d["gap_historico_vs_meta_nacional"] is None:
            d["gap_historico_vs_meta_nacional"] = round(d["indicador_lag1"] - d["meta_nacional"], 2)

        records.append(d)
    return pd.DataFrame(records)


def _compute_local_factors(df: pd.DataFrame, top_k: int = 3) -> List[FatorExplicativo]:
    """Calcula via SHAP (TreeExplainer) os fatores locais que mais influenciaram a decisão."""
    explainer, rf_pipeline = get_explainer()
    if explainer is None or rf_pipeline is None:
        return []
    try:
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(df)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif len(np.shape(shap_vals)) == 3:
            vals = shap_vals[0, :, 1]
        else:
            vals = shap_vals[0]

        top_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        factors = []
        for idx in top_indices:
            feat_name = _feature_names[idx] if idx < len(_feature_names) else f"feature_{idx}"
            label = FEATURE_LABEL_MAP.get(
                feat_name,
                feat_name.replace("sigla_uf_", "UF: ").replace("regiao_", "Região: ")
            )
            val = float(vals[idx])
            direcao = "PROTEGE_META" if val > 0 else "AUMENTA_RISCO"
            if val < 0:
                impacto = "Empurra o município em direção ao risco de não atingimento da meta pactuada."
            else:
                impacto = "Contribui positivamente para sustentar a probabilidade de cumprimento da meta."
            factors.append(
                FatorExplicativo(
                    fator=label,
                    direcao=direcao,
                    impacto=impacto,
                    valor_shap=round(val, 4),
                )
            )
        return factors
    except Exception:
        return []


def _format_prediction(proba_sucesso: float, local_factors: Optional[List[FatorExplicativo]] = None) -> PredictionResult:
    proba_risco = 1.0 - proba_sucesso

    if proba_risco >= 0.60:
        classificacao = "CRÍTICO: RISCO IMINENTE"
        alerta = "🔴 ALTA PRIORIDADE"
        rec = "Envio imediato de apoio pedagógico presencial do MEC e alocação suplementar de recursos FUNDEB."
    elif proba_risco >= 0.42:
        classificacao = "ALERTA: RISCO MODERADO"
        alerta = "🟡 MÉDIA PRIORIDADE"
        rec = "Monitoramento formativo bimestral e reforço de material estruturado para o corpo docente."
    else:
        classificacao = "REGULAR: META PROVÁVEL"
        alerta = "🟢 BAIXA PRIORIDADE (ACOMPANHAMENTO)"
        rec = "Manutenção das boas práticas pedagógicas e acompanhamento de rotina."

    return PredictionResult(
        probabilidade_meta_atingida=round(float(proba_sucesso), 4),
        probabilidade_risco=round(float(proba_risco), 4),
        classificacao=classificacao,
        escore_risco_percentual=f"{proba_risco:.1%}",
        alerta_prioridade=alerta,
        recomendacao_politica_publica=rec,
        fatores_determinantes_locais=local_factors,
    )


@app.get("/health", tags=["Status"])
def health_check():
    """Verifica a integridade do serviço e o status do modelo."""
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_file": str(MODEL_PATH.name),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inferência"])
def predict_municipio(input_data: MunicipioInput):
    """Gera predição de risco de alfabetização para um município individual com explicabilidade SHAP."""
    try:
        model = get_model()
        df = _prepare_dataframe([input_data])
        proba_sucesso = model.predict_proba(df)[0, 1]
        local_factors = _compute_local_factors(df, top_k=3)
        return _format_prediction(proba_sucesso, local_factors=local_factors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar predição: {str(e)}")


@app.post("/batch-predict", response_model=List[PredictionResult], tags=["Inferência"])
def predict_batch(inputs: List[MunicipioInput]):
    """Gera predição em lote para múltiplos municípios."""
    try:
        model = get_model()
        df = _prepare_dataframe(inputs)
        probas = model.predict_proba(df)[:, 1]
        return [_format_prediction(p) for p in probas]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar predições em lote: {str(e)}")

