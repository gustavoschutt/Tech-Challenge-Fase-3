# Documentação de Linhagem de Dados e Proveniência (Data Lineage)
### Tech Challenge – Fase 3 | PosTech FIAP

---

## 1. Visão Geral da Arquitetura e Proveniência

Este documento estabelece a linhagem técnica e formal da transição da **Camada Silver** (gerada na Fase 2 a partir do data lake no Google Cloud Storage / BigQuery) para a **Camada Gold** consumida na modelagem preditiva da Fase 3.

```
Fontes Oficiais (Base dos Dados / INEP / SAEB)
                     │
                     ▼
       ┌───────────────────────────┐
       │   Camada Silver (GCS)     │
       │  indicador_municipio.csv  │
       └─────────────┬─────────────┘
                     │  src/data_pipeline/build_gold.py
                     ▼
       ┌───────────────────────────┐
       │     Camada Gold (Local)   │
       │   ml_features.parquet     │
       │   evolucao_uf.parquet     │
       │   painel_nacional.parquet │
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   EDA & Clustering        Modelagem Supervisionada
  (notebooks/01_eda)     (HistGradientBoosting, RF, LogReg)
```

---

## 2. Auditoria e Resolução do Alvo de Negócio

### Contexto da Inconsistência Detectada
Em auditoria anterior de integridade, identificou-se que snapshots legados da base continham uma coluna `meta_atingida` que não refletia a regra formal de negócio estipulada pelo edital e pelas diretrizes do MEC. No snapshot legado, apontava-se artificialmente que mais de 93% dos municípios atingiram a meta em 2024, quando na realidade a meta municipal pactuada exigia patamar superior.

### Regra Formal de Derivação do Alvo
O alvo supervisionado é estrita e deterministicamente recalculado via código no módulo `src/data_pipeline/build_gold.py`:

$$\text{meta\_atingida} = (\text{indicador\_alfabetizacao} \ge \text{meta\_municipio})$$
$$\text{target\_meta\_atingida} = \begin{cases} 1, & \text{se meta atingida (Sucesso)} \\ 0, & \text{se meta não atingida (Risco Educacional)} \end{cases}$$

### Validação dos Dados Reconstruídos (Ano a Ano)

| Ano | Total Municípios | Metas Atingidas (Target = 1) | Municípios em Risco (Target = 0) | % Metas Atingidas |
| :---: | :---: | :---: | :---: | :---: |
| **2022** | 5.570 | 3.055 | 2.515 | **54,85%** |
| **2023** | 5.570 | 2.829 | 2.741 | **50,79%** |
| **2024** | 5.570 | **2.606** | **2.964** | **46,79%** |
| **Total** | **16.710** | **8.490** | **8.220** | **50,81%** |

> **Nota de Integridade:** Pela regra correta, em 2024 exatamente **2.606 municípios (46,79%)** atingiram suas respectivas metas pactuadas e **2.964 municípios (53,21%)** encontram-se em situação de risco educacional.

---

## 3. Contrato de Dados da Camada Silver (`validate_silver`)

Antes de qualquer derivação, a entrada Silver passa por validação estrita de contrato (`validate_silver`):
- **Obrigatoriedade de Atributos:** `id_municipio`, `nome`, `sigla_uf`, `id_uf`, `nome_uf`, `regiao`, `ano`, `indicador_alfabetizacao`, `meta_municipio`, `meta_nacional`, `quantidade_matriculas`, `PIB_per_capita`, `IDHM`.
- **Unicidade de Chave:** Nenhuma duplicata para o par `(id_municipio, ano)`.
- **Domínio Numérico:** 
  - `indicador_alfabetizacao`, `meta_municipio`, `meta_nacional` $\in [0, 100]$.
  - `IDHM` $\in [0, 1]$.
  - `quantidade_matriculas` $\ge 0$ e `PIB_per_capita` $\ge 0$.

---

## 4. Engenharia de Features e Prevenção de Data Leakage

Para alimentar os modelos preditivos sem vazamento temporal (*Zero Data Leakage*):
1. **Defasagens Temporais:**
   - `indicador_lag1`: Taxa de alfabetização municipal em $t-1$.
   - `indicador_lag2`: Taxa de alfabetização municipal em $t-2$.
   - `tendencia_historica`: Variação histórica real ($\text{lag}_1 - \text{lag}_2$).
2. **Gaps com Relação às Metas:**
   - `gap_historico_vs_meta_municipio`: Distância do indicador anterior ($t-1$) até a meta municipal do ano corrente.
   - `gap_historico_vs_meta_nacional`: Distância do indicador anterior ($t-1$) até a meta Brasil.
3. **Exclusão de Variáveis Contemporâneas:** O indicador do ano corrente (`indicador_alfabetizacao`) e colunas derivadas do ano corrente são estritamente removidos da matriz $X$.

---

## 5. Manifesto de Execução e Hashes Criptográficos

A execução mais recente do construtor reproduzível gerou o seguinte manifesto formal:

```json
{
  "schema_version": "1.1.0",
  "source_uri": "Base dos Dados - Camada Silver Consolidada (GCS/BigQuery)",
  "input_file": "data/indicador_municipio.csv",
  "input_sha256": "ef6d7d23fd30c97097c2a88ff2b2f821f50912eda61f81688e7ac75e6b18f187",
  "input_rows": 22280,
  "outputs": {
    "ml_features": 16710,
    "evolucao_uf": 108,
    "painel_nacional": 4
  },
  "target_rule": "indicador_alfabetizacao >= meta_municipio",
  "contract_validation": {
    "required_columns_count": 13,
    "ml_feature_columns_count": 19,
    "status": "VALIDATED"
  }
}
```

---

## 6. Instruções de Reprodução

Para reproduzir a compilação da Camada Gold a partir da Silver:

```bash
python -m src.data_pipeline.build_gold \
  --input data/indicador_municipio.csv \
  --output-dir data \
  --source-uri "Base dos Dados - Camada Silver Consolidada (GCS/BigQuery)"

# Executar suíte de testes de integridade
PYTHONPATH=. pytest -v tests/
```
