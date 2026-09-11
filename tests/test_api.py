from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_success():
    payload = {
        "indicador_lag1": 65.5,
        "indicador_lag2": 60.0,
        "meta_municipio": 55.0,
        "meta_nacional": 60.0,
        "quantidade_matriculas": 1500,
        "PIB_per_capita": 25000.0,
        "IDHM": 0.720,
        "sigla_uf": "SP",
        "regiao": "Sudeste",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilidade_meta_atingida" in data
    assert "probabilidade_risco" in data
    assert "classificacao" in data
    assert "recomendacao_politica_publica" in data
    assert 0.0 <= data["probabilidade_meta_atingida"] <= 1.0
    assert "fatores_determinantes_locais" in data
    assert isinstance(data["fatores_determinantes_locais"], list)
    assert len(data["fatores_determinantes_locais"]) == 3
    first_factor = data["fatores_determinantes_locais"][0]
    assert "fator" in first_factor
    assert "direcao" in first_factor
    assert "impacto" in first_factor
    assert "valor_shap" in first_factor


def test_predict_endpoint_risk():
    # Município com queda histórica e grande defasagem
    payload = {
        "indicador_lag1": 40.0,
        "indicador_lag2": 45.0,
        "meta_municipio": 60.0,
        "meta_nacional": 65.0,
        "quantidade_matriculas": 800,
        "PIB_per_capita": 12000.0,
        "IDHM": 0.580,
        "sigla_uf": "MA",
        "regiao": "Nordeste",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probabilidade_risco"] > 0.50
    assert "RISCO" in data["classificacao"]
    assert "fatores_determinantes_locais" in data
    assert any(f["direcao"] == "AUMENTA_RISCO" for f in data["fatores_determinantes_locais"])


def test_batch_predict_endpoint():
    payload = [
        {
            "indicador_lag1": 65.5,
            "meta_municipio": 55.0,
            "meta_nacional": 60.0,
            "quantidade_matriculas": 1500,
            "PIB_per_capita": 25000.0,
            "IDHM": 0.720,
            "sigla_uf": "SP",
            "regiao": "Sudeste",
        },
        {
            "indicador_lag1": 38.0,
            "meta_municipio": 60.0,
            "meta_nacional": 65.0,
            "quantidade_matriculas": 800,
            "PIB_per_capita": 12000.0,
            "IDHM": 0.550,
            "sigla_uf": "BA",
            "regiao": "Nordeste",
        },
    ]
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
