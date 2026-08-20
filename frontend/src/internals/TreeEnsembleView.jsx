import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import TreeDiagram from "./TreeDiagram";
import SigmoidChart from "../charts/SigmoidChart";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8001";

function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

/**
 * Inspirado em bbycroft.net/llm: em vez de resumir a predição, mostra o cômputo real
 * acontecendo — as 202 árvores reais do modelo (models/lightgbm_fraud.joblib), com o
 * caminho real que UMA transação percorre em cada uma, aceso ao mesmo tempo em todas.
 * Clique em qualquer árvore para ver os splits e o valor de folha reais em detalhe.
 */
export default function TreeEnsembleView() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [customer, setCustomer] = useState(null);
  const [zoomed, setZoomed] = useState(null);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const sampleRes = await fetch(`${API_BASE}/generate-sample`);
      const sample = await sampleRes.json();
      const res = await fetch(`${API_BASE}/model-internals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sample.transaction),
      });
      if (!res.ok) throw new Error("falha em /model-internals");
      const internals = await res.json();
      setCustomer(sample);
      setData(internals);
    } catch {
      setError("não foi possível carregar as árvores — a API está no ar?");
    } finally {
      setLoading(false);
    }
  };

  const approved = data && data.fraud_probability < (data.threshold ?? 0.724);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexWrap: "wrap", gap: 10 }}>
        <div>
          <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: "var(--accent)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            as 202 árvores reais — inspirado em bbycroft.net/llm
          </div>
          <div style={{ fontSize: 13, color: "var(--text-faint)", marginTop: 2 }}>
            {customer ? `caminho real de ${customer.metadata.titular} · R$ ${customer.transaction.Amount.toFixed(2)}` : "cada árvore é real — models/lightgbm_fraud.joblib"}
          </div>
        </div>
        <button className="btn primary" onClick={load} disabled={loading}>
          {loading ? "calculando…" : data ? "nova transação" : "gerar transação"}
        </button>
      </div>

      {error && <div style={{ color: "var(--danger)", fontSize: 13 }}>{error}</div>}

      {!data && !loading && !error && (
        <div className="panel" style={{ textAlign: "center", color: "var(--text-faint)", padding: 40, marginTop: 12 }}>
          clique em "gerar transação" para ver o caminho real dela nas 202 árvores
        </div>
      )}

      {data && (
        <>
          <div className="panel" style={{ display: "flex", gap: 22, alignItems: "center", flexWrap: "wrap", margin: "14px 0" }}>
            <Stat label="soma das 202 folhas" value={fmt(data.leaf_sum, 4)} />
            <span className="mono" style={{ color: "var(--text-faint)" }}>+</span>
            <Stat label="score inicial (base_score)" value={fmt(data.init_score, 4)} />
            <span className="mono" style={{ color: "var(--text-faint)" }}>=</span>
            <Stat label="score bruto (logit)" value={fmt(data.raw_score, 4)} />
            <span className="mono" style={{ color: "var(--text-faint)" }}>→ sigmoid →</span>
            <Stat
              label="probabilidade de fraude"
              value={fmt(data.fraud_probability, 4)}
              highlight
              color={data.fraud_probability >= 0.724 ? "var(--danger)" : "var(--ok)"}
            />
          </div>

          <div className="panel" style={{ margin: "0 0 14px" }}>
            <div className="panel-title">De "soma das árvores" a probabilidade — a curva sigmoide real</div>
            <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <SigmoidChart rawScore={data.raw_score} probability={data.fraud_probability} />
              <p style={{ fontSize: 12, color: "var(--text-muted)", maxWidth: 260, margin: 0 }}>
                As 202 árvores nunca devolvem uma probabilidade — devolvem um número qualquer (o score bruto <code className="mono">z</code>).
                A sigmoide <code className="mono">1/(1+e⁻ᶻ)</code> é o que espreme esse número para o intervalo 0–1. O ponto acima é onde
                <b> esta</b> transação caiu.
              </p>
            </div>
          </div>

          <div
            className="panel"
            style={{
              display: "flex", flexWrap: "wrap", gap: 10, maxHeight: "48vh", overflowY: "auto",
              padding: "16px 18px",
            }}
          >
            {data.trees.map((t, i) => (
              <motion.button
                key={t.index}
                onClick={() => setZoomed(t)}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: Math.min(i, 60) * 0.006 }}
                title={`árvore ${t.index} — clique para ampliar (folha real: ${fmt(t.leaf_value, 3)})`}
                style={{
                  border: "1px solid var(--border)", borderRadius: 6, padding: 3, cursor: "zoom-in",
                  background: "var(--surface)",
                }}
              >
                <TreeDiagram structure={t.structure} />
                <div className="mono" style={{ fontSize: 8.5, color: "var(--text-faint)", textAlign: "center", marginTop: 1 }}>
                  #{t.index}
                </div>
              </motion.button>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 8 }}>
            {data.n_trees} árvores reais · linha âmbar = o caminho que esta transação percorreu · ponto vermelho/verde = folha
            real onde ela chegou (empurra para fraude/legítima)
          </div>
        </>
      )}

      <AnimatePresence>
        {zoomed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setZoomed(null)}
            style={{
              position: "fixed", inset: 0, background: "rgba(10,16,13,0.72)",
              display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50, cursor: "zoom-out",
            }}
          >
            <motion.div
              initial={{ scale: 0.94, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.96, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="panel"
              style={{ maxWidth: "min(720px, 92vw)" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ fontWeight: 600 }}>Árvore #{zoomed.index} — models/lightgbm_fraud.joblib</div>
                <button className="btn" onClick={() => setZoomed(null)}>fechar ✕</button>
              </div>
              <div style={{ overflowX: "auto" }}>
                <TreeDiagram structure={zoomed.structure} width={640} height={340} detailed />
              </div>
              <div className="mono" style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
                contribuição desta árvore para o score: <b style={{ color: zoomed.leaf_value >= 0 ? "var(--danger)" : "var(--ok)" }}>{fmt(zoomed.leaf_value, 4)}</b>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Stat({ label, value, highlight, color }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      <div className="mono" style={{ fontSize: highlight ? 17 : 14, fontWeight: 700, color: color || "var(--text)" }}>{value}</div>
    </div>
  );
}
