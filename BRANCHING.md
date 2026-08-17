# Estratégia de Branches — dev / hom / prod

Este repositório simula três ambientes, o suficiente para demonstrar em aula como a
promoção de um modelo de ML passa por estágios com rigor crescente — sem precisar de
infraestrutura real por trás (o "deploy" em `prod` é simulado, mas roda de verdade:
sobe a API, chama `/health`, derruba).

```
dev  ──PR──▶  hom  ──PR──▶  prod
(rápido)      (esteira      (esteira completa
               completa)     + deploy simulado,
                              atrás de aprovação)
```

## O que muda em cada branch

| | `dev` | `hom` (homologação) | `prod` (produção) |
|---|---|---|---|
| **Propósito** | Iteração rápida, experimentação | Validação final antes de ir para produção | Espelha exatamente o que seria produção |
| **Tuning (Optuna)** | 5 trials (`HPO_N_TRIALS=5`, ~segundos) | Completo — 40 trials (o mesmo do `config.yaml`) | Completo — 40 trials |
| **Esteira (CI)** | Roda o pipeline completo, mas mais rápido | Esteira completa: pipeline + validação + explicabilidade + drift + smoke test | Igual a `hom`, **mais** um passo de deploy simulado |
| **GitHub Environment** | `desenvolvimento` (sem gate) | `homologacao` (sem gate, por padrão) | `producao` (pode exigir aprovação manual) |
| **Merge permitido de** | — | `dev` (via Pull Request) | `hom` (via Pull Request) |

## Por que essa diferença de tuning entre `dev` e `hom`/`prod`

Em desenvolvimento, o que importa é feedback rápido — 5 trials do Optuna rodam em
segundos e são suficientes para saber se o código está funcionando, sem esperar o
tuning completo a cada push. `hom` e `prod` sempre rodam o tuning **completo** (mesmo
espaço de busca, mesmo número de trials): é isso que garante que o modelo validado em
`hom` é exatamente o modelo que vai para `prod` — nenhuma das duas usa atalho.

## Como promover uma mudança, na prática

1. Trabalhe em `dev` (ou em uma branch de feature a partir de `dev`). A esteira roda
   rápido a cada push, dando feedback imediato.
2. Quando estiver pronto, abra um **Pull Request de `dev` para `hom`**. A esteira
   completa roda automaticamente — só é possível fazer o merge se ela passar (ative
   "Require status checks to pass" em Settings → Branches → Branch protection rules
   para `hom`).
3. Em `hom`, revise os artefatos publicados pela esteira (aba Actions → execução →
   Artifacts: métricas, gráficos, relatório de drift) — a mesma checagem manual que um
   time de ML faria antes de aprovar produção.
4. Abra um **Pull Request de `hom` para `prod`**. A esteira roda de novo (mesmo
   resultado esperado, já que `hom` e `prod` usam a mesma configuração), e o passo
   extra de deploy simulado confirma que a API sobe e responde.
5. Se configurar `prod` como um GitHub Environment com "Required reviewers" (Settings →
   Environments → producao), o job da esteira **pausa esperando aprovação manual**
   antes de rodar — o mesmo tipo de gate humano que existiria antes de um deploy real
   em produção.

## Configurando os Environments no GitHub (uma vez, pela interface)

1. Settings → Environments → New environment → nomes: `desenvolvimento`,
   `homologacao`, `producao`.
2. Em `producao`, marque **Required reviewers** e adicione você mesmo (ou outro
   colaborador) — isso é o que faz o job da esteira parar e pedir aprovação antes de
   rodar o passo de deploy simulado.
3. (Opcional) Em Settings → Branches, adicione uma regra de proteção para `hom` e
   `prod` exigindo que a esteira (`validar-pipeline`) passe antes de permitir o merge.
