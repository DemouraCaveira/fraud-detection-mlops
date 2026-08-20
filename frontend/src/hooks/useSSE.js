import { useEffect, useRef, useState, useCallback } from "react";

/**
 * Wrapper reutilizavel de EventSource. Fica desconectado ate `active` virar true (para
 * o usuario controlar o play/pause em vez do stream abrir sozinho ao montar o
 * componente), e reconecta do zero sempre que `url`/`active` mudam.
 */
export function useSSE(url, active) {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef(null);

  const clear = useCallback(() => setEvents([]), []);

  useEffect(() => {
    if (!active || !url) {
      setConnected(false);
      return undefined;
    }

    const source = new EventSource(url);
    sourceRef.current = source;

    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        setEvents((prev) => [...prev, parsed]);
      } catch {
        // heartbeats/comentarios SSE (linhas ": ...") nao chegam em onmessage
      }
    };

    return () => {
      source.close();
      setConnected(false);
    };
  }, [url, active]);

  return { events, connected, clear };
}
