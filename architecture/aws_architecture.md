# Arquitetura AWS — Detecção de Fraude em Produção

> Diagrama completo (interativo): ver `aws_architecture.html` ou o artefato publicado.

Este documento descreve como o pipeline construído em `src/` (ingestão, pré-processamento,
treino, validação, explicabilidade) e em `deploy/`/`monitoring/` (deploy simulado,
benchmark de latência, monitoramento de drift) se traduziria em uma arquitetura de
produção real na AWS. Não é uma proposta genérica de "arquitetura de MLOps" — cada
decisão abaixo está amarrada a um número medido neste projeto.

## Números que sustentam as decisões de arquitetura

| Medição | Valor | Onde foi medido |
|---|---|---|
| Latência de inferência (LightGBM), p50 | 1,0 ms | `deploy/benchmark_latency.py` |
| Latência de inferência, p95 / p99 | 1,7 ms / 2,1 ms | `deploy/benchmark_latency.py` |
| Throughput em lote | 511.247 transações/s | `deploy/benchmark_latency.py` |
| Tuning completo (40 trials Optuna + treino final) | ~39 s | `src/train.py`, ~200k linhas |
| Treino em volume simulado de ~4M linhas | ~43 s | `src/evaluate.py` (`scale_benchmark`) |
| Cálculo de SHAP (5.000 amostras) | 0,45 s | `src/explainability.py` |
| PSI real (treino vs. teste), pior caso | 7,5 (`hour_cos`) | `monitoring/drift_monitor.py` |
| PSI simulado (choque de +60% em `Amount`) | 0,17 (moderado) | `monitoring/drift_monitor.py` |

## 1. Ingestão em tempo real & Data Lake

- **Amazon Kinesis Data Streams** recebe cada evento de transação do gateway de
  pagamentos. Um stream permite múltiplos consumidores independentes do mesmo evento —
  necessário aqui porque a mesma transação alimenta tanto o scoring em tempo real quanto
  o data lake, sem competirem pela mensagem (diferente de uma fila tradicional como SQS).
- **Kinesis Data Firehose** persiste o evento bruto, sem transformação, em **S3 (raw
  zone)** — nunca se perde o dado original.
- **AWS Glue ETL / SageMaker Processing** executa a mesma lógica de
  `src/preprocessing.py` (limpeza, features cíclicas de hora do dia, split temporal),
  containerizada, gravando em **S3 curated** (Parquet particionado por data).
- **SageMaker Feature Store** materializa as features em duas camadas: *offline*
  (consumida pelo treino) e *online* (latência de leitura < 10 ms, consumida pelo
  endpoint de inferência) — garantindo que treino e serving usem exatamente o mesmo
  código de feature engineering (evita o problema clássico de *training-serving skew*).

## 2. Treinamento & Registro (offline / agendado)

- **SageMaker Training Job**, em instância **Spot** (~70% mais barata), executa o mesmo
  `src/train.py` — baseline + LightGBM com tuning via Optuna — containerizado. Como o
  treino completo leva ~39s neste projeto, o risco de interrupção do Spot tem custo
  desprezível frente à economia.
- Disparado por **EventBridge** (agendamento semanal) ou pelo alarme de drift do Model
  Monitor (seção 4), fechando o ciclo de retreino automático.
- Antes de virar produção, o modelo passa por um **gate de aprovação automático**: só é
  promovido no **SageMaker Model Registry** se suas métricas de validação igualarem ou
  superarem o modelo em produção — a mesma checagem que fizemos manualmente no
  Checkpoint 1 deste projeto (comparação baseline vs. modelo tunado), aqui automatizada.

## 3. Serving em tempo real — o caminho medido

Esta é a única camada do diagrama com números **medidos**, não estimados. O modelo
responde uma predição em **p50 = 1,0 ms / p99 = 2,1 ms**, com throughput de **511 mil
predições/s** em lote. Isso justifica hospedar o modelo em um **SageMaker Real-Time
Endpoint** multi-AZ com auto-scaling: a latência do modelo é uma fração pequena do
orçamento de uma autorização de cartão (tipicamente algumas centenas de ms), sobrando
margem para rede, validação (**API Gateway + Lambda** na frente do endpoint) e retries.

**Alternativas consideradas e descartadas:**

- *KNN*, apesar de competitivo em métricas de validação: o benchmark de complexidade
  (Fase 3) mediu sua latência subindo de 2,2 ms para 7,5 ms conforme o volume de treino
  cresce (inferência O(n)) — inviável para um endpoint cujo dataset de treino só cresce
  ao longo dos anos.
- *SageMaker Serverless Inference* em vez de endpoint dedicado: reservado para ambientes
  de menor tráfego (ex.: staging) — para volume de autorização de cartão, um endpoint
  sempre ativo evita cold start.

## 4. Monitoramento & Retreino (feedback loop)

- Toda inferência é capturada (input + output) e comparada, de forma agendada, pelo
  **SageMaker Model Monitor** contra a distribuição de referência do treino — a mesma
  técnica de PSI/KS implementada em `monitoring/drift_monitor.py`, que já demonstrou
  neste projeto: PSI real de até 7,5 nas features de horário entre treino e teste
  (esperado, dado o split temporal em uma janela curta de ~48h), e detecção correta de
  um choque simulado de +60% em `Amount` (PSI 0,17, classificado como moderado).
- Quando o drift ultrapassa o limiar, um **CloudWatch Alarm** dispara em paralelo: um
  alerta via **SNS** para o time de risco, e um retrain automático de volta à seção 2 —
  sem esperar o agendamento semanal.

**Indicadores de acompanhamento propostos:**

| Indicador | Método | Frequência |
|---|---|---|
| Data drift | PSI/KS por feature vs. baseline offline | Diária |
| Concept drift | PR-AUC recalculado quando rótulos de chargeback chegam (atraso de dias/semanas) | A cada lote de rótulos confirmados |
| Performance operacional | Latência p99 e taxa de erro do endpoint | Contínua (CloudWatch) |

Concept drift é o indicador mais crítico e o mais difícil de operacionalizar: como o
rótulo de fraude confirmado (chargeback) chega com atraso, a degradação real do modelo
só é observável depois que o dano já ocorreu — por isso o monitoramento de data drift
(que não depende de rótulo) funciona como sinal antecipado.

## 5. Segurança & Governança (nota breve)

- IAM com políticas de privilégio mínimo por serviço (Training Job, Endpoint, Feature
  Store cada um com sua própria role).
- Criptografia em repouso (S3 SSE-KMS) e em trânsito (TLS) em todos os saltos.
- VPC endpoints para SageMaker/S3, evitando tráfego pela internet pública.
- SageMaker Clarify para detecção de viés no modelo — complementa a explicabilidade via
  SHAP já feita neste projeto (`src/explainability.py`), agora como checagem contínua em
  produção, não só pontual no desenvolvimento.
