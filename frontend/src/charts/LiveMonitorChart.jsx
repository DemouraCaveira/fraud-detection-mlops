import { useEffect, useRef } from "react";
import * as d3 from "d3";

/**
 * Gráfico ao vivo, D3, que se move de verdade: a cada predição real recebida via
 * /events/live, um ponto novo entra pela direita e a janela desliza — como um monitor
 * de sinais vitais. Sem valor simulado: cada ponto é a fraud_probability de uma
 * predição real que aconteceu.
 */
export default function LiveMonitorChart({ events, threshold = 0.724, windowSize = 30 }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    const width = ref.current.clientWidth || 300;
    const height = 120;
    const margin = { top: 10, right: 10, bottom: 4, left: 30 };
    svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);

    const data = events.slice(-windowSize);
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const x = d3.scaleLinear().domain([0, Math.max(windowSize - 1, 1)]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

    let g = svg.select("g.plot");
    if (g.empty()) {
      g = svg.append("g").attr("class", "plot").attr("transform", `translate(${margin.left},${margin.top})`);
      g.append("g").attr("class", "y-axis");
      g.append("line").attr("class", "threshold-line");
      g.append("path").attr("class", "score-line");
      g.append("g").attr("class", "dots");
    }

    g.select(".y-axis")
      .call(d3.axisLeft(y).ticks(3).tickSize(-innerW).tickFormat(d3.format(".1f")))
      .call((sel) => sel.select(".domain").remove())
      .call((sel) => sel.selectAll(".tick line").attr("stroke", "var(--border)").attr("stroke-dasharray", "2 3"))
      .call((sel) => sel.selectAll(".tick text").attr("fill", "var(--text-faint)").attr("font-size", 9).attr("font-family", '"IBM Plex Mono", monospace'));

    g.select(".threshold-line")
      .attr("x1", 0).attr("x2", innerW)
      .attr("y1", y(threshold)).attr("y2", y(threshold))
      .attr("stroke", "var(--accent)")
      .attr("stroke-dasharray", "3 3")
      .attr("stroke-width", 1);

    if (data.length > 1) {
      const line = d3.line()
        .x((_, i) => x(i))
        .y((d) => y(d.fraud_probability))
        .curve(d3.curveMonotoneX);
      g.select(".score-line")
        .datum(data)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "var(--text-faint)")
        .attr("stroke-width", 1.2);
    } else {
      g.select(".score-line").attr("d", null);
    }

    const dots = g.select(".dots").selectAll("circle").data(data, (d) => d.ts);
    dots.exit().remove();
    dots
      .enter()
      .append("circle")
      .attr("r", 0)
      .merge(dots)
      .attr("cx", (_, i) => x(i))
      .attr("cy", (d) => y(d.fraud_probability))
      .attr("fill", (d) => (d.decision === "BLOQUEAR" ? "var(--danger)" : "var(--ok)"))
      .transition()
      .duration(200)
      .attr("r", (_, i) => (i === data.length - 1 ? 4 : 2.6));
  }, [events, threshold, windowSize]);

  if (!events || events.length === 0) {
    return <div style={{ fontSize: 11.5, color: "var(--text-faint)", padding: "12px 0" }}>aguardando predições reais…</div>;
  }

  return <svg ref={ref} style={{ display: "block", width: "100%" }} />;
}
