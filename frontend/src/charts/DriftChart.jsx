import { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * D3 puro (o componente so monta o <svg>; D3 possui o conteudo dele) — barras horizontais
 * de PSI real por feature, vindas de /drift-report (monitoring/drift_monitor.py), com as
 * mesmas bandas de decisao (0.10 / 0.25) usadas no resto do material de aula.
 */
export default function DriftChart({ rows, thresholds }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    if (!rows || rows.length === 0) return;

    const width = ref.current.clientWidth || 320;
    const barHeight = 16;
    const gap = 6;
    const margin = { top: 4, right: 44, bottom: 4, left: 78 };
    const sorted = [...rows].sort((a, b) => b.psi - a.psi).slice(0, 10);
    const height = sorted.length * (barHeight + gap) + margin.top + margin.bottom;
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);

    const maxPsi = Math.max(thresholds?.significativo ?? 0.25, d3.max(sorted, (d) => d.psi) ?? 0.1) * 1.15;
    const x = d3.scaleLinear().domain([0, maxPsi]).range([0, width - margin.left - margin.right]);

    const color = (psi) => {
      if (psi >= (thresholds?.significativo ?? 0.25)) return "var(--danger)";
      if (psi >= (thresholds?.moderado ?? 0.1)) return "var(--accent)";
      return "var(--ok)";
    };

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // linhas guia nos thresholds
    [thresholds?.moderado, thresholds?.significativo].forEach((t) => {
      if (t == null) return;
      g.append("line")
        .attr("x1", x(t)).attr("x2", x(t))
        .attr("y1", 0).attr("y2", height - margin.top - margin.bottom)
        .attr("stroke", "var(--border-strong)")
        .attr("stroke-dasharray", "3 3");
    });

    const rowsG = g
      .selectAll("g.row")
      .data(sorted)
      .join("g")
      .attr("class", "row")
      .attr("transform", (_, i) => `translate(0, ${i * (barHeight + gap)})`);

    rowsG
      .append("text")
      .attr("x", -8)
      .attr("y", barHeight / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("font-family", '"IBM Plex Mono", monospace')
      .attr("font-size", 10.5)
      .attr("fill", "var(--text-muted)")
      .text((d) => d.feature);

    rowsG
      .append("rect")
      .attr("height", barHeight)
      .attr("rx", 3)
      .attr("width", 0)
      .attr("fill", (d) => color(d.psi))
      .transition()
      .duration(500)
      .attr("width", (d) => Math.max(2, x(d.psi)));

    rowsG
      .append("text")
      .attr("x", (d) => x(d.psi) + 6)
      .attr("y", barHeight / 2)
      .attr("dy", "0.35em")
      .attr("font-family", '"IBM Plex Mono", monospace')
      .attr("font-size", 10.5)
      .attr("fill", "var(--text-faint)")
      .text((d) => d.psi.toFixed(3));
  }, [rows, thresholds]);

  return <svg ref={ref} style={{ display: "block", width: "100%" }} />;
}
