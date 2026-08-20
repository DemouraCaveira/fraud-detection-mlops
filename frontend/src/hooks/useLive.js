import { useSSE } from "./useSSE";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

/** Consome /events/live — cada POST /predict real, enquanto o stream estiver aberto. */
export function useLive(active) {
  const url = `${API_BASE}/events/live`;
  const { events, connected, clear } = useSSE(url, active);
  return { events: events.filter((e) => e.type === "prediction"), connected, clear };
}
