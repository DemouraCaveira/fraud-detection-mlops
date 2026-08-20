import { Handle, Position } from "reactflow";
import { motion } from "framer-motion";
import LiveScoreSparkline from "../charts/LiveScoreSparkline";

const STATUS_STYLE = {
  idle: { border: "var(--border-strong)", glow: "none" },
  running: { border: "var(--accent)", glow: "0 0 0 4px var(--accent-soft)" },
  done: { border: "var(--ok)", glow: "0 0 0 3px var(--ok-soft)" },
  armed: { border: "var(--danger)", glow: "0 0 0 4px var(--danger-soft)" },
};

const STATUS_LABEL = {
  idle: "aguardando",
  running: "rodando…",
  done: "concluído",
  armed: "gatilho ativo",
};

/**
 * Nó custom do React Flow para uma etapa do ciclo de vida. `data.status` decide cor e
 * badge; a transição para "running"/"done" dispara uma animação de pulso via Framer
 * Motion — a mudança de estado em si vem de um evento real (SSE de replay/live ou do
 * /drift-report), o componente só a anima.
 */
export default function StageNode({ data }) {
  const status = data.status || "idle";
  const style = STATUS_STYLE[status] || STATUS_STYLE.idle;
  const isCentral = data.central;

  return (
    <motion.div
      animate={
        status === "running"
          ? { scale: [1, 1.035, 1] }
          : { scale: 1 }
      }
      transition={status === "running" ? { duration: 1.1, repeat: Infinity, ease: "easeInOut" } : { duration: 0.3 }}
      style={{
        background: "var(--surface)",
        border: `${isCentral ? 2.4 : 1.6}px solid ${style.border}`,
        borderRadius: 12,
        padding: isCentral ? "14px 18px" : "11px 14px",
        width: data.io ? 210 : isCentral ? 190 : 150,
        boxShadow: `var(--shadow), ${style.glow}`,
        fontFamily: '"IBM Plex Mono", monospace',
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--border-strong)" }} />
      <Handle type="source" position={Position.Right} style={{ background: "var(--border-strong)" }} />

      <div style={{ fontSize: 10, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {data.kicker}
      </div>
      <div style={{ fontSize: isCentral ? 14.5 : 13, fontWeight: 600, color: "var(--text)", marginTop: 2 }}>
        {data.label}
      </div>
      {data.metric && (
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.5 }}>{data.metric}</div>
      )}
      {data.io && (
        <div
          style={{
            marginTop: 8, fontSize: 9.5, lineHeight: 1.5, background: "var(--canvas-bg)",
            borderRadius: 6, padding: "6px 8px", color: "#cfe0d8",
          }}
        >
          <div style={{ color: "#7fc9a6" }}>
            ↓ recebe: <span style={{ color: "#cfe0d8" }}>{data.io.in}</span>
          </div>
          <div style={{ color: "#e0a93d", marginTop: 3 }}>
            ↑ envia: <span style={{ color: "#cfe0d8" }}>{data.io.out}</span>
          </div>
        </div>
      )}
      {data.liveScores && (
        <div style={{ marginTop: 8 }}>
          <LiveScoreSparkline events={data.liveScores} />
        </div>
      )}
      {data.figures?.length > 0 && (
        <div style={{ display: "flex", gap: 5, marginTop: 8 }}>
          {data.figures.map((fig) => (
            <img
              key={fig.src}
              src={fig.src}
              alt={fig.caption}
              title={`${fig.caption} — clique para ampliar`}
              onClick={(e) => {
                e.stopPropagation();
                data.onOpenFigure?.(fig);
              }}
              style={{
                width: 40, height: 30, objectFit: "cover", borderRadius: 4,
                border: "1px solid var(--border)", cursor: "zoom-in", background: "#fff",
              }}
            />
          ))}
        </div>
      )}
      <div
        style={{
          marginTop: 8,
          fontSize: 9.5,
          fontWeight: 700,
          color: style.border,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {STATUS_LABEL[status]}
      </div>
    </motion.div>
  );
}
