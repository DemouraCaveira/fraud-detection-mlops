import { useSSE } from "./useSSE";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

/** Consome /events/train-replay — reprodução de uma execução real e já capturada. */
export function useReplay(active, speed) {
  const url = `${API_BASE}/events/train-replay?speed=${speed}`;
  const { events, connected, clear } = useSSE(url, active);
  const finished = events.some((e) => e.type === "replay_finished");
  return { events: events.filter((e) => e.stage), connected, clear, finished };
}
