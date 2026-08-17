# 🎬 ROTEIRO E DISCURSO EXECUTIVO DA APRESENTAÇÃO (VÍDEO DE ATÉ 5 MINUTOS)
### Tech Challenge – Fase 3 | PosTech FIAP (Inteligência Artificial & Machine Learning)
**Tema**: *Predição de Alfabetização Infantil e Análise de Risco Educacional para Políticas Públicas*  
**Duração Alvo do Vídeo**: **4 minutos e 30 segundos** *(Margem de segurança para o teto de 5 minutos)*  
**Apresentador**: Gustavo Schutt  

---

## ⏱️ GUIA DE CRONOMETRAGEM POR SLIDE

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                CRONOGRAMA DO VÍDEO (4m30s)                             │
├─────────┬────────────────────────────────────────────────────────┬─────────────────────┤
│ Slide   │ Título / Tópico                                        │ Tempo Acumulado     │
├─────────┼────────────────────────────────────────────────────────┼─────────────────────┤
│ Slide 1 │ Abertura, Contexto e Dor de Negócio                    │ 0:00 ➔ 0:45 (45s)   │
│ Slide 2 │ Arquitetura da Solução & Prevenção de Data Leakage     │ 0:45 ➔ 1:30 (45s)   │
│ Slide 3 │ Comparação de Modelos e Validação Cruzada              │ 1:30 ➔ 2:30 (60s)   │
│ Slide 4 │ Interpretabilidade e Fatores Determinantes (SHAP)      │ 2:30 ➔ 3:30 (60s)   │
│ Slide 5 │ Aplicação Estratégica: Quadrante de Risco e Políticas  │ 3:30 ➔ 4:15 (45s)   │
│ Slide 6 │ Conclusão Executiva e Encerramento                     │ 4:15 ➔ 4:35 (20s)   │
└─────────┴────────────────────────────────────────────────────────┴─────────────────────┘
```

---

## 📑 ROTEIRO SLIDE A SLIDE & DISCURSO VERBATIM

---

### 🟢 SLIDE 1: ABERTURA, CONTEXTO E DOR DE NEGÓCIO
* **Tempo Sugerido**: `0:00 a 0:45` *(45 segundos)*  
* **Elementos Visuais na Tela**: Título do projeto, logos oficiais (MEC / Compromisso Criança Alfabetizada / FIAP), número de destaque: **100% de Alfabetização até 2030 (743 pontos SAEB)**.

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Olá a todos! Sejam muito bem-vindos à apresentação do Tech Challenge da Fase 3 da Pós-Graduação em Inteligência Artificial da FIAP.*
>
> *Hoje, apresentamos uma solução de Ciência de Dados e Machine Learning desenvolvida para apoiar um dos maiores desafios sociais do nosso país: o **Compromisso Nacional Criança Alfabetizada**, que tem como meta garantir que 100% das crianças brasileiras estejam plenamente alfabetizadas até 2030, atingindo o patamar de 743 pontos na escala SAEB.*
>
> *No entanto, a gestão pública não pode se dar ao luxo de apenas olhar relatórios consolidados do passado. Quando o resultado anual é divulgado, o ano letivo já acabou e a oportunidade de intervir foi perdida. Nossa proposta é criar um **sistema preditivo de alerta precoce**, capaz de antecipar quais municípios correm risco de não atingir suas metas, permitindo ações pedagógicas e financeiras preventivas."*

---

### 🟢 SLIDE 2: ARQUITETURA DE DADOS & PREVENÇÃO DE DATA LEAKAGE
* **Tempo Sugerido**: `0:45 a 1:30` *(45 segundos)*  
* **Elementos Visuais na Tela**: Diagrama do Pipeline Scikit-Learn (`ColumnTransformer`), conexão com a Camada Gold (BigQuery / 16.710 registros) e selo **"Zero Data Leakage"**.

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Para construir essa inteligência, utilizamos diretamente a **Camada Gold** desenvolvida e saneada na Fase 2 no Google BigQuery, cobrindo mais de 16.700 registros municipais ao longo de múltiplos anos.*
>
> *Um dos pilares mais rigorosos deste projeto foi a **eliminação total de vazamento de dados (Data Leakage)**. O modelo não tem acesso ao indicador do ano corrente nem a qualquer informação futura. Nossa matriz preditora é composta estritamente por variáveis históricas prévias — como os indicadores de um e dois anos anteriores ($t-1$ e $t-2$), a tendência histórica de crescimento e o gap em relação às metas — integradas a dados estruturais de IDHM, PIB per capita e porte municipal.*
>
> *Todo o pré-processamento foi encapsulado em um `ColumnTransformer` do Scikit-Learn com imputação pela mediana e moda, padronização e One-Hot Encoding, garantindo uma separação treino e teste 100% estratificada e estanque."*

---

### 🟢 SLIDE 3: MODELAGEM SUPERVISIONADA & VALIDAÇÃO CRUZADA
* **Tempo Sugerido**: `1:30 a 2:30` *(60 segundos)*  
* **Elementos Visuais na Tela**: Tabela comparativa dos 3 modelos (Regressão Logística, Random Forest, Gradient Boosting), gráfico da Curva ROC (`04_roc_curves_comparison.png`) e Matriz de Confusão (`05_confusion_matrix.png`).

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Na etapa de modelagem supervisionada, comparamos três abordagens competitivas utilizando **Validação Cruzada Estratificada em 5 folds** no conjunto de treino e uma validação cega em um Holdout de mais de 3.300 municípios.*
>
> *Avaliamos uma Regressão Logística como baseline, um Random Forest e um Gradient Boosting Classifier.*
>
> *O **Gradient Boosting** consagrou-se como o melhor modelo de produção, alcançando um **ROC-AUC de 0.975**, uma **Acurácia Global de 94.8%** e um **F1-Score de 0.972** no conjunto de teste.*
>
> *O mais importante para a política pública: o modelo obteve um **Recall de 98.3%**, o que significa que ele identifica com altíssima precisão as redes municipais consistentes e emite alertas precoces altamente confiáveis para os casos de risco, sem apresentar nenhum sinal de overfitting."*

---

### 🟢 SLIDE 4: EXPLICABILIDADE & FATORES DETERMINANTES COM SHAP (XAI)
* **Tempo Sugerido**: `2:30 a 3:30` *(60 segundos)*  
* **Elementos Visuais na Tela**: Gráfico SHAP Summary Beeswarm (`06_shap_summary_plot.png`) e Gráfico de Barras de Importância Global (`07_shap_bar_importance.png`).

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Em políticas públicas, um modelo preditivo não pode ser uma caixa-preta. Gestores precisam entender os porquês de cada predição. Para isso, implementamos técnicas de **Explainable AI utilizando SHAP Values**.*
>
> *Os resultados do SHAP revelam que:*
> * *Em primeiro lugar, a **inércia histórica do município (indicador anterior)** tem o maior peso absoluto (Mean SHAP de 2.03), mostrando que a alfabetização é um processo cumulativo da rede de ensino.*
> * *Em segundo lugar, o **alinhamento prévio com a meta nacional** protege os municípios de oscilações.*
> * *E em terceiro lugar, identificamos que **metas municipais excessivamente irrealistas** criam uma falsa sensação de risco, enquanto o **IDHM e o PIB per capita** funcionam como amortecedores fundamentais de vulnerabilidade social.*
>
> *Com o SHAP, conseguimos explicar a predição individual de cada um dos 5.570 municípios brasileiros."*

---

### 🟢 SLIDE 5: APLICAÇÃO ESTRATÉGICA: QUADRANTE DE RISCO & POLÍTICAS PÚBLICAS
* **Tempo Sugerido**: `3:30 a 4:15` *(45 segundos)*  
* **Elementos Visuais na Tela**: Gráfico do Quadrante de Risco Educacional (`06_risk_quadrant.png`), Mapa de Desempenho Regional (`03_regional_performance.png`) e os 3 pilares de recomendação (MEC / FUNDEB / Municípios).

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Transformamos essa capacidade preditiva em inteligência acionável através da **Matriz de Vulnerabilidade Educacional**.*
>
> *Identificamos que 585 municípios brasileiros encontram-se no **Quadrante de Alto Risco** — caracterizados por taxa histórica abaixo de 50% e tendência de evolução negativa. Para esses municípios, a probabilidade predita de não atingir a meta supera 85%.*
>
> *Nossa proposta estratégica recomenda:*
> 1. *Priorização dos repasses discricionários do **FUNDEB** e apoio técnico do MEC para as redes no quadrante crítico;*
> 2. *Envio de programas de mentoria e formação continuada de professores para reverter a tendência negativa antes da prova do SAEB;*
> 3. *E a recalibragem técnica das metas pactuadas, ajustando-as à realidade socioeconômica de cada território."*

---

### 🟢 SLIDE 6: CONCLUSÃO EXECUTIVA E ENCERRAMENTO
* **Tempo Sugerido**: `4:15 a 4:35` *(20 segundos)*  
* **Elementos Visuais na Tela**: Resumo dos entregáveis (Repositório GitHub, Pipelines, Modelos, Dashboard), dados de contato e agradecimentos.

#### 🎙️ NARRATIVA / DISCURSO DO APRESENTADOR:
> *"Em resumo, entregamos uma pipeline de Machine Learning completa, robusta, auditada com nota máxima e com impacto real para a gestão educacional do Brasil.*
>
> *Todo o código, dados, notebooks interativos e relatórios estão disponíveis e versionados em nosso repositório no GitHub.*
>
> *Muito obrigado a todos pela atenção!"*

---

## 🎯 DICAS DE OURO PARA UMA GRAVAÇÃO NOTA 10:
1. **Controle do Tempo**: Use um cronômetro na sua frente. Mantenha a fala fluida e termine entre **4m20s e 4m40s** (nunca ultrapasse 5m00s).
2. **Postura e Tom**: Fale como um consultor ou cientista de dados apresentando para o Ministro da Educação ou para um comitê executivo de prefeitos.
3. **Compartilhamento de Tela**: Alterne entre os slides e mostre rapidamente a pasta do GitHub e o gráfico do SHAP na tela.
