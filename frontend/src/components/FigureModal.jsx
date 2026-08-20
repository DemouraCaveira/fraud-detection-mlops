import { motion, AnimatePresence } from "framer-motion";

/** Lightbox simples para ver em tamanho grande um gráfico real gerado pelo pipeline. */
export default function FigureModal({ figure, onClose }) {
  return (
    <AnimatePresence>
      {figure && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          style={{
            position: "fixed", inset: 0, background: "rgba(10,16,13,0.72)",
            display: "flex", alignItems: "center", justifyContent: "center",
            zIndex: 50, padding: 32, cursor: "zoom-out",
          }}
        >
          <motion.div
            initial={{ scale: 0.94, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={(e) => e.stopPropagation()}
            style={{
              background: "var(--surface)", borderRadius: 14, padding: 18,
              maxWidth: "min(920px, 92vw)", maxHeight: "88vh", overflow: "auto",
              boxShadow: "var(--shadow)", border: "1px solid var(--border)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
              <div>
                <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: "var(--text-faint)" }}>{figure.source}</div>
                <div style={{ fontSize: 15, fontWeight: 600 }}>{figure.caption}</div>
              </div>
              <button className="btn" onClick={onClose}>fechar ✕</button>
            </div>
            <img src={figure.src} alt={figure.caption} style={{ width: "100%", display: "block", borderRadius: 8 }} />
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
