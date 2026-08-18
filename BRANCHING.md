# Estratégia de Branches — dev / hom / prod

Este repositório simula três ambientes, o suficiente para demonstrar em aula como a
promoção de um modelo de ML passa por estágios com rigor crescente — sem precisar de
infraestrutura real por trás (o "deploy" em `prod` é simulado, mas roda de verdade:
sobe a API, chama `/health`, derruba).

**Princípio central: build once, promote the artifact — nunca retreina a cada
estágio.** `prod` não roda `run_pipeline.py`. Ele baixa o exato binário do modelo que
já foi treinado e validado em `hom` e só roda a demonstração funcional + deploy em cima
dele. Se cada ambiente retreinasse do zero, não haveria garantia de que o modelo
testado em `hom` é o mesmo que chega em produção — e essa garantia é o ponto central de
uma esteira de promoção de verdade (o mesmo problema que motiva Model Registry no
SageMaker, ver `architecture/aws_architecture.md`).

```
dev  ──treina──▶  push
                    │
                   PR
                    ▼
hom  ──treina + esteira completa──▶  publica artefato "modelo-treinado"
                    │
                   PR
                    ▼
prod ──baixa o artefato de hom (NÃO retreina)──▶  smoke test + deploy simulado
```

## O que roda em cada branch

| | `dev` | `hom` (homologação) | `prod` (produção) |
|---|---|---|---|
| **Propósito** | Iteração rápida, experimentação | Validação final — o gate real | Promove o que já foi validado, sem retreinar |
| **Treina o modelo?** | Sim (`run_pipeline.py`) | Sim (`run_pipeline.py`) | **Não** — baixa o artefato publicado por `hom` |
| **Tuning (Optuna)** | 5 trials (`HPO_N_TRIALS=5`) | Completo — 40 trials | — (não treina) |
| **Benchmark de escala** | Reduzido (`1,2`) | Reduzido (`1,5,10`) — o estudo completo até 20x já está no relatório | — (não treina) |
| **Esteira (CI)** | Pipeline completo + validação + explicabilidade + drift + smoke test | Igual a `dev`, e **publica o modelo treinado** como artefato do workflow | Baixa o artefato de `hom` + `prepare_data.py` (só ingestão/pré-processamento, sem treino) + smoke test + deploy simulado |
| **GitHub Environment** | `desenvolvimento` (sem gate) | `homologacao` (sem gate, por padrão) | `producao` (pode exigir aprovação manual) |
| **Merge permitido de** | — | `dev` (via Pull Request) | `hom` (via Pull Request) |

## Por que `prod` roda `prepare_data.py` mas não `run_pipeline.py`

`prepare_data.py` só faz ingestão + pré-processamento — determinístico e idempotente
(mesma seed, mesma fonte de dados, sempre produz o mesmo `data/processed/*.parquet`).
Rodar isso de novo em `prod` é aceitável e até esperado (cada ambiente pode ter acesso
a dados diferente na prática). O que **nunca** pode ser refeito é o treino: o modelo
que roda em produção precisa ser byte-a-byte o mesmo binário que passou pela esteira
completa em `hom`, não uma cópia "quase igual" retreinada — mesmo com seed fixa, isso
introduziria uma diferença entre "o que foi testado" e "o que está rodando", que é
exatamente o tipo de risco que uma esteira de promoção existe para eliminar.

## Como promover uma mudança, na prática

1. Trabalhe em `dev` (ou em uma branch de feature a partir de `dev`). A esteira roda
   rápido a cada push, dando feedback imediato.
2. Quando estiver pronto, abra um **Pull Request de `dev` para `hom`**. A esteira
   completa roda automaticamente — só é possível fazer o merge se ela passar (ative
   "Require status checks to pass" em Settings → Branches → Branch protection rules
   para `hom`). Ao final, o modelo treinado fica publicado como artefato do workflow
   (aba Actions → execução → Artifacts → `modelo-treinado`).
3. Em `hom`, revise os artefatos publicados pela esteira (métricas, gráficos, relatório
   de drift) — a mesma checagem manual que um time de ML faria antes de aprovar
   produção.
4. Abra um **Pull Request de `hom` para `prod`**. A esteira de `prod` **não treina
   nada** — ela procura o último run bem-sucedido de `hom`, baixa o `modelo-treinado`
   de lá, e roda smoke test + deploy simulado sobre esse artefato exato.
5. Se configurar `prod` como um GitHub Environment com "Required reviewers" (Settings →
   Environments → producao), o job **pausa esperando aprovação manual** antes de rodar
   — o mesmo tipo de gate humano que existiria antes de um deploy real em produção.

## Configurando os Environments no GitHub (uma vez, pela interface)

1. Settings → Environments → New environment → nomes: `desenvolvimento`,
   `homologacao`, `producao`.
2. Em `producao`, marque **Required reviewers** e adicione você mesmo (ou outro
   colaborador) — isso é o que faz o job de `prod` parar e pedir aprovação antes de
   baixar o artefato e rodar o deploy simulado.
3. (Opcional) Em Settings → Branches, adicione uma regra de proteção para `hom` e
   `prod` exigindo que a esteira passe antes de permitir o merge.

## Limitação conhecida deste setup didático

Os artefatos do GitHub Actions expiram (`retention-days: 90` para o modelo, 30 para os
relatórios) — em um ambiente real de produção, o "Model Registry" que guarda essas
versões de forma permanente e rastreável seria o SageMaker Model Registry (ou MLflow
Model Registry hospedado), não um artefato de CI com prazo de validade. Ver
`architecture/aws_architecture.md`, seção 2, para o desenho completo com esse
componente.
