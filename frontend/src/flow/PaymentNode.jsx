import { Handle, Position } from "reactflow";
import { motion } from "framer-motion";

/**
 * Nó especial: a pessoa passando o cartão, antes de qualquer coisa chegar na API. Existe
 * porque "Deploy/API" sozinho não deixa claro que tem um aplicativo de pagamento (com um
 * usuário de verdade) do outro lado da chamada — este nó é esse elo.
 */
export default function PaymentNode({ data }) {
  const pulsing = !!data.pulsing;
  return (
    <div
      style={{
        background: "var(--surface)", border: `1.6px solid ${pulsing ? "var(--accent)" : "var(--border-strong)"}`,
        borderRadius: 12, padding: "12px 16px", minWidth: 150, textAlign: "center",
        boxShadow: pulsing ? "var(--shadow), 0 0 0 4px var(--accent-soft)" : "var(--shadow)",
        fontFamily: '"IBM Plex Mono", monospace',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--border-strong)" }} />
      <Handle type="source" position={Position.Right} style={{ background: "var(--border-strong)" }} />

      <div style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        aplicativo de pagamento
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2 }}>Cliente passa o cartão</div>

      {/* pessoa + cartão indo para a maquininha, simples e vetorial */}
      <svg width="86" height="44" style={{ margin: "8px auto 0", display: "block" }}>
        {/* pessoa: cabeça + corpo */}
        <circle cx="16" cy="12" r="6" fill="var(--text-faint)" />
        <path d="M6 34 Q16 18 26 34 Z" fill="var(--text-faint)" />
        {/* maquininha */}
        <rect x="52" y="14" width="26" height="24" rx="3" fill="var(--surface-alt)" stroke="var(--border-strong)" />
        <rect x="56" y="19" width="18" height="4" rx="1" fill="var(--border-strong)" />
        {/* cartão animando entre os dois */}
        <motion.rect
          x="30" y="18" width="16" height="10" rx="2"
          fill="var(--accent)"
          animate={pulsing ? { x: [30, 54, 30], opacity: [1, 1, 0] } : { x: 30, opacity: 1 }}
          transition={pulsing ? { duration: 0.9, ease: "easeInOut" } : { duration: 0.2 }}
        />
      </svg>

      <div style={{ fontSize: 9.5, color: "var(--text-faint)", marginTop: 4 }}>
        {pulsing ? "enviando POST /predict…" : "aguardando transação"}
      </div>
    </div>
  );
}
