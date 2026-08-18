# Detecção de Fraude em Cartão de Crédito — Trilha A

Projeto de referência de Engenharia de Aprendizado de Máquina: pipeline completo de
detecção de fraude, do dado bruto ao monitoramento em produção, com ênfase em
**performance mensurada** (não estimada) e em uma **arquitetura de produção real**.

> Relatório completo com todos os números, gráficos e discussão crítica dos resultados:
> [`reports/relatorio.md`](reports/relatorio.md). Este README cobre organização e como
> executar.

## Por que este repositório está organizado assim

Este projeto é construído como um repositório de engenharia de ML de verdade, não como
uma coleção de notebooks soltos — a organização abaixo *é* parte do que se espera de
uma solução profissional, tanto quanto o modelo em si.

**Scripts (`src/`), não notebook, são a fonte da verdade do pipeline.** Um notebook é
ótimo para explorar dados e narrar resultados, mas péssimo para rodar em produção: não
é testável isoladamente, não versiona bem em diffs, e mistura estado de execução com
código. Por isso, `src/` contém o pipeline real — modular, reprodutível, com um único
entry point (`run_pipeline.py`) que qualquer sistema de orquestração (cron, Step
Functions, Airflow) poderia chamar. O notebook (`notebooks/`) existe separadamente,
para demonstração e para rodar no Google Colab — ele narra a mesma lógica, mas não é
de onde o pipeline "vive".

**Configuração central, nada hardcoded.** Todo caminho, hiperparâmetro, threshold e
seed vive em [`config/config.yaml`](config/config.yaml), lido por
[`src/utils.py`](src/utils.py). Mudar o dataset, o modelo ou o espaço de busca de
hiperparâmetros não deveria exigir editar código Python em vários arquivos.

**Cada etapa do pipeline é um script independente, executável isoladamente**
(`python -m src.ingestion`, `python -m src.evaluate`, etc.) — útil tanto para debug
quanto porque, em produção, nem toda etapa roda com a mesma frequência (retreino é
semanal; deploy e monitoramento são contínuos).

**Todo número reportado vem de uma execução real, documentada.** Não há métrica,
gráfico ou tempo de execução neste projeto que não tenha sido gerado rodando o código
deste repositório — incluindo os números "desconfortáveis" (overfitting inicial
detectado e corrigido, gap de generalização temporal, baseline competitivo com o
modelo mais sofisticado). Ver seção "Achados honestos" do relatório.

## Estrutura

```
config/config.yaml           # configuração central — paths, hiperparâmetros, thresholds, seeds
data/                        # cache local do dataset (gitignored)
src/
  ingestion.py                # baixa e valida o dataset bruto (OpenML, sem autenticação)
  eda.py                      # análise exploratória (exploratório, fora do pipeline de retreino)
  preprocessing.py            # split temporal, engenharia de atributos, escalonamento
  train.py                    # baseline + LightGBM com tuning via Optuna (early stopping)
  evaluate.py                 # CV, métricas, curva de aprendizado, benchmarks de escala/complexidade
  explainability.py           # SHAP (TreeExplainer)
  experiment_tracking.py      # registro de experimentos no MLflow
  utils.py                    # config loader, logging, timing
run_pipeline.py               # entry point único: ingestão → ... → treino
deploy/
  api.py                       # FastAPI /predict — deploy simulado
  benchmark_latency.py         # mede latência p50/p95/p99 e throughput
  demo_faker.py                # gera transações sintéticas (Faker) e roda inferência real
monitoring/
  drift_monitor.py             # PSI/KS — drift real (treino vs. teste) e simulado
notebooks/
  STM_Fraude_TrilhaA.ipynb     # notebook didático, para rodar no Google Colab
architecture/
  aws_architecture.md          # arquitetura de produção AWS — decisões e justificativas
  aws_architecture.drawio      # diagrama com ícones oficiais AWS (abrir em app.diagrams.net)
  aws_architecture.html        # versão navegável do diagrama
reports/
  relatorio.md                 # relatório completo do projeto
  figures/                     # todos os gráficos gerados
  *.json                       # métricas, benchmarks e resumos, em formato estruturado
models/                        # artefatos do modelo treinado (gerado ao rodar o pipeline)
smoke_test.py                  # demonstração funcional mínima
main.py                        # ⭐ ponto de entrada único: modelo "em operação" — demo ou API
.github/workflows/ci.yml       # esteira de validação (GitHub Actions)
```

## Como executar

Requer Python 3.10+.

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Isso baixa o dataset (cache local, sem necessidade de credenciais), roda
pré-processamento e treino, e salva os artefatos em `models/`. Treina em ~1 minuto
(seção "Resultados principais" abaixo).

## O modelo em operação — `main.py`

Depois de treinado (`run_pipeline.py`), `main.py` é o ponto de entrada único para
mostrar o modelo funcionando de verdade, das duas formas que um modelo roda em
produção:

```bash
python main.py                 # demonstracao no terminal: transacoes sinteticas (Faker) + inferencia real
python main.py --n 10          # gera 10 transacoes em vez de 6

python main.py --serve         # sobe a API de inferencia real (FastAPI/uvicorn), Ctrl+C para parar
python main.py --serve --port 8080
```

Com `--serve`, a API fica disponível em `http://127.0.0.1:8000/docs` (Swagger
interativo) — o mesmo tipo de processo que rodaria atrás de um load balancer em
produção. Sem `--serve`, roda a demonstração de terminal direto, sem precisar de
navegador nem Postman — melhor para uma demo rápida em sala.

## Esteira de validação (CI — GitHub Actions) e os três ambientes

`.github/workflows/ci.yml` roda automaticamente a cada `push`/PR nas branches `dev`,
`hom` e `prod` (e manualmente pela aba **Actions**, botão "Run workflow"). Diferente da
esteira do `esteira-kit` da turma (que só confere formato de arquivos), esta roda o
pipeline de verdade, do zero, em uma máquina limpa da GitHub — com um princípio central:
**`prod` nunca re-treina o modelo**, ele promove o exato artefato já validado em `hom`.

Em `dev`/`hom`:
1. Instala as dependências (`requirements.txt`)
2. Roda `run_pipeline.py` (ingestão → pré-processamento → treino) — a mesma prova de
   reprodutibilidade feita localmente, mas agora em uma máquina que nunca viu este
   código
3. Roda validação (`src.evaluate`), explicabilidade (`src.explainability`) e
   monitoramento de drift (`monitoring.drift_monitor`)
4. Roda `smoke_test.py` — se a esteira ficar verde, uma predição real foi executada
5. Publica os gráficos e métricas gerados como artefato do workflow
6. **Só em `hom`**: publica o modelo treinado como artefato (`modelo-treinado`) — é
   esse binário exato, e nenhum outro, que `prod` vai usar

Em `prod`: roda `prepare_data.py` (só ingestão/pré-processamento, sem treino), baixa o
`modelo-treinado` do último build bem-sucedido de `hom`, roda `smoke_test.py` sobre esse
artefato e faz o deploy simulado (sobe a API real, confere `/health`, derruba).

O repositório simula três ambientes (`dev` → `hom` → `prod`) para mostrar como a
promoção de um modelo passa por estágios de rigor crescente sem nunca re-treinar fora
do estágio de validação — ver **[`BRANCHING.md`](BRANCHING.md)** para o fluxo completo
(o porquê de "build once, promote the artifact", como promover uma mudança via Pull
Request, como configurar aprovação manual antes de produção).

**Pré-requisito para isso rodar**: é preciso `git init`, criar o repositório no GitHub
e dar `git push` — nada nisso acontece sozinho.

Para gerar os relatórios de validação, explicabilidade e monitoramento manualmente
(com os gráficos), sem o CI:

```bash
python -m src.evaluate
python -m src.explainability
python -m monitoring.drift_monitor
python -m src.experiment_tracking
```

Outras demonstrações pontuais:

```bash
python smoke_test.py                 # predição real sobre uma transação do conjunto de teste
python -m deploy.demo_faker           # transações sintéticas via Faker + inferência real
python -m deploy.benchmark_latency    # latência p50/p95/p99 medida
```

## Resultados principais (resumo — números completos no relatório)

| | |
|---|---|
| PR-AUC (teste, temporal) | 0,7265 |
| Latência de inferência (p50 / p99) | 1,0 ms / 2,1 ms |
| Tuning completo (40 trials Optuna) | ~39 s |
| Treino em ~4M linhas simuladas | ~43 s |

## Reprodutibilidade

- Dataset público, sem autenticação (OpenML `data_id=1597`).
- Nenhum caminho local hardcoded — tudo relativo à raiz do projeto via
  `src/utils.py::resolve_path`.
- Seed fixa (`config.yaml::project.random_seed`) em todos os pontos de aleatoriedade.
- `notebooks/STM_Fraude_TrilhaA.ipynb` roda do início ao fim sem alterações, no Google
  Colab ou localmente.
