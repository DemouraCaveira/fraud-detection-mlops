import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

function fmtBRL(n) {
  return Number(n).toLocaleString("pt-BR", { style: "currency", currency: "BRL" });
}

/**
 * Simulação didática de uma maquininha de cartão: gera uma transação sintética real
 * (GET /generate-sample) e roda a inferência real (POST /predict) — a mesma chamada
 * que o app "ao vivo" escuta, só que aqui apresentada como um cliente passando o cartão,
 * para tornar o fluxo "dado chega -> passa no modelo -> decisão" tangível em sala.
 */
export default function CardMachine() {
  const [phase, setPhase] = useState("idle"); // idle | reading | processing | result | error
  const [customer, setCustomer] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const autoTimer = useRef(null);
  const autoPlayRef = useRef(false);

  // laço "automático": um cliente por vez, com uma pausa curta entre um resultado e o
  // próximo — pra explicar volume/regime permanente em aula sem clicar a cada transação
  useEffect(() => {
    autoPlayRef.current = autoPlay;
    if (autoPlay) scheduleNext(400);
    return () => clearTimeout(autoTimer.current);
  }, [autoPlay]);

  const scheduleNext = (delay) => {
    clearTimeout(autoTimer.current);
    autoTimer.current = setTimeout(() => {
      if (autoPlayRef.current) runCustomer();
    }, delay);
  };

  const runCustomer = async () => {
    setPhase("reading");
    setResult(null);
    setError(null);
    try {
      const sampleRes = await fetch(`${API_BASE}/generate-sample`);
      if (!sampleRes.ok) throw new Error("falha ao gerar transação sintética");
      const sample = await sampleRes.json();
      setCustomer(sample);

      await new Promise((r) => setTimeout(r, 900)); // tempo de "leitura do cartão", só cosmético
      setPhase("processing");

      const predictRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sample.transaction),
      });
      if (!predictRes.ok) throw new Error("falha em /predict");
      const prediction = await predictRes.json();
      setResult(prediction);
      setPhase("result");
      if (autoPlayRef.current) scheduleNext(1800);
    } catch (err) {
      setError(err.message);
      setPhase("error");
      if (autoPlayRef.current) scheduleNext(3000);
    }
  };

  const reset = () => {
    setPhase("idle");
    setCustomer(null);
    setResult(null);
    setError(null);
  };

  const approved = result?.decision === "APROVAR";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20, padding: "32px 16px" }}>
      <div
        style={{
          width: 300, borderRadius: 26, padding: "20px 18px 26px",
          background: "linear-gradient(160deg, var(--surface-alt), var(--surface))",
          border: "1px solid var(--border-strong)", boxShadow: "var(--shadow)",
          position: "relative",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: "var(--text-faint)", letterSpacing: "0.08em" }}>
            POS · fraud-detection-mlops
          </span>
          <span
            className="mono"
            style={{
              fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 5,
              background: phase === "idle" ? "var(--surface-alt)" : "var(--accent-soft)",
              color: phase === "idle" ? "var(--text-faint)" : "var(--accent-text)",
            }}
          >
            {phase === "idle" ? "pronta" : phase === "reading" ? "lendo cartão" : phase === "processing" ? "processando" : phase === "error" ? "erro" : "concluído"}
          </span>
        </div>

        {/* tarja/slot do cartão */}
        <div
          style={{
            height: 8, borderRadius: 4, background: "var(--canvas-bg)",
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.35)", marginBottom: 14, position: "relative", overflow: "visible",
          }}
        >
          <AnimatePresence>
            {(phase === "reading" || phase === "processing") && (
              <motion.div
                initial={{ y: -46, opacity: 0 }}
                animate={{ y: -8, opacity: 1 }}
                exit={{ y: -46, opacity: 0 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
                style={{
                  position: "absolute", left: "50%", marginLeft: -46, top: 0, width: 92, height: 56,
                  borderRadius: 7, background: "linear-gradient(135deg, var(--accent), var(--accent-text))",
                  boxShadow: "0 6px 14px rgba(0,0,0,0.35)",
                }}
              >
                <div style={{ width: 20, height: 14, background: "rgba(255,255,255,0.55)", borderRadius: 3, margin: "10px 0 0 10px" }} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* "tela" da maquininha */}
        <div
          style={{
            minHeight: 128, borderRadius: 12, background: "var(--canvas-bg)", color: "#cfe0d8",
            padding: 14, fontFamily: '"IBM Plex Mono"', fontSize: 12.5, display: "flex",
            flexDirection: "column", justifyContent: "center", gap: 6,
          }}
        >
          {phase === "idle" && <div style={{ color: "#8fa39c", textAlign: "center" }}>toque para simular um cliente</div>}

          {phase === "reading" && customer && (
            <>
              <div style={{ color: "#8fa39c", fontSize: 10 }}>lendo cartão…</div>
              <div style={{ fontWeight: 600 }}>{customer.metadata.titular}</div>
              <div style={{ color: "#8fa39c" }}>{customer.metadata.estabelecimento} · {customer.metadata.cidade}</div>
              <div style={{ fontSize: 18, fontWeight: 700, marginTop: 2 }}>{fmtBRL(customer.transaction.Amount)}</div>
            </>
          )}

          {phase === "processing" && (
            <div style={{ textAlign: "center" }}>
              <motion.div
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1, repeat: Infinity }}
                style={{ color: "var(--accent-text)" }}
              >
                processando na API real…
              </motion.div>
              <div style={{ fontSize: 10, color: "#8fa39c", marginTop: 4 }}>POST /predict</div>
            </div>
          )}

          {phase === "result" && result && customer && (
            <>
              <div style={{ fontWeight: 600 }}>{customer.metadata.titular}</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{fmtBRL(customer.transaction.Amount)}</div>
              <motion.div
                initial={{ scale: 0.85, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                style={{
                  marginTop: 4, fontWeight: 800, fontSize: 15, letterSpacing: "0.04em",
                  color: approved ? "#7fc9a6" : "#e2867f",
                }}
              >
                {approved ? "✓ APROVADA" : "✕ NEGADA"}
              </motion.div>
              <div style={{ color: "#8fa39c", fontSize: 10.5 }}>
                p(fraude)={result.fraud_probability.toFixed(3)} · threshold={result.threshold_used.toFixed(3)} · {result.latency_ms.toFixed(2)}ms
              </div>
            </>
          )}

          {phase === "error" && <div style={{ color: "#e2867f" }}>erro: {error} — a API está no ar em {API_BASE}?</div>}
        </div>
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <button
          className="btn primary"
          onClick={phase === "idle" || phase === "result" || phase === "error" ? runCustomer : undefined}
          disabled={phase === "reading" || phase === "processing" || autoPlay}
        >
          {phase === "result" || phase === "error" ? "novo cliente" : phase === "idle" ? "inserir cartão" : "aguarde…"}
        </button>
        <button
          className="btn"
          onClick={() => setAutoPlay((v) => !v)}
          style={autoPlay ? { background: "var(--danger-soft)", borderColor: "var(--danger)", color: "var(--danger)" } : undefined}
        >
          {autoPlay ? "⏸ parar automático" : "▶ automático"}
        </button>
      </div>
      {phase === "result" && !autoPlay && (
        <button className="btn" onClick={reset} style={{ marginTop: -10 }}>
          limpar tela
        </button>
      )}

      {autoPlay && (
        <div className="mono" style={{ fontSize: 11, color: "var(--accent-text)" }}>
          gerando clientes automaticamente — bom pra deixar rodando enquanto você explica
        </div>
      )}

      <div style={{ maxWidth: 320, textAlign: "center", fontSize: 11.5, color: "var(--text-faint)" }}>
        Cada cliente é uma transação sintética real (Faker + jitter estatístico) — a decisão vem de uma chamada real
        e ao vivo a <code className="mono">POST /predict</code>, não de um valor pré-calculado.
      </div>
    </div>
  );
}
