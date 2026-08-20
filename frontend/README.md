# Visualizador do ciclo de vida do modelo

App React (Vite + React Flow + D3 + Framer Motion) que visualiza o ciclo de vida
completo do modelo de detecção de fraude — não só uma predição isolada — consumindo o
backend real (`deploy/api.py`) via REST e Server-Sent Events (SSE). Ver a seção
"Visualizador do ciclo de vida do modelo" no `README.md` da raiz do projeto para o
contexto completo.

## Rodando localmente

```bash
# 1. backend, a partir da raiz do projeto (Sistematizacao/)
python main.py --serve --port 8001

# 2. captura uma execução real do pipeline, uma vez, para o modo "replay"
python -m deploy.capture_training_events

# 3. este app
cd frontend
npm install
npm run dev        # http://localhost:5173
```

`VITE_API_BASE` (em `.env`) aponta para o backend — ajuste se rodar em outra porta.

## Estrutura

- `src/flow/` — o grafo React Flow (`LifecycleFlow.jsx`) e o nó custom (`StageNode.jsx`)
- `src/charts/` — `DriftChart.jsx` (D3, PSI real por atributo) e `MetricsPanel.jsx`
- `src/hooks/` — `useSSE`, `useReplay` (`/events/train-replay`), `useLive` (`/events/live`)
- `src/theme.css` — os mesmos tokens visuais usados no material de aula (IBM Plex Mono/Sans, âmbar)
