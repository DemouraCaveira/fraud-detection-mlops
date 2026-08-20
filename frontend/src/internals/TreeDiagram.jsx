/**
 * Desenha uma árvore real do modelo (nó = split real com feature/threshold reais, ou
 * folha = valor real de contribuição). `on_path` (calculado no backend a partir da
 * transação atual) decide o que fica em destaque — o caminho real que a transação
 * percorreu dentro desta árvore específica.
 */
function layoutTree(root) {
  let leafCounter = 0;
  const nodes = [];
  const edges = [];
  let maxDepth = 0;

  function visit(node, depth) {
    maxDepth = Math.max(maxDepth, depth);
    if (node.leaf) {
      const x = leafCounter++;
      nodes.push({ ...node, x, y: depth });
      return x;
    }
    const leftX = visit(node.left, depth + 1);
    const rightX = visit(node.right, depth + 1);
    const x = (leftX + rightX) / 2;
    nodes.push({ ...node, x, y: depth });
    edges.push({ x1: x, y1: depth, x2: leftX, y2: depth + 1, onPath: node.left.on_path });
    edges.push({ x1: x, y1: depth, x2: rightX, y2: depth + 1, onPath: node.right.on_path });
    return x;
  }
  visit(root, 0);
  return { nodes, edges, maxDepth: Math.max(maxDepth, 1), leafCount: Math.max(leafCounter, 1) };
}

export default function TreeDiagram({ structure, width = 74, height = 56, detailed = false }) {
  const { nodes, edges, maxDepth, leafCount } = layoutTree(structure);
  const pad = detailed ? 46 : 5;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;
  const sx = (x) => pad + (leafCount <= 1 ? innerW / 2 : (x / (leafCount - 1)) * innerW);
  const sy = (y) => pad + (y / maxDepth) * innerH;

  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      {edges.map((e, i) => (
        <line
          key={i}
          x1={sx(e.x1)} y1={sy(e.y1)} x2={sx(e.x2)} y2={sy(e.y2)}
          stroke={e.onPath ? "var(--accent)" : "var(--border)"}
          strokeWidth={e.onPath ? (detailed ? 2.4 : 1.6) : 1}
        />
      ))}
      {nodes.map((n, i) => {
        if (n.leaf) {
          const color = n.on_path ? (n.value >= 0 ? "var(--danger)" : "var(--ok)") : "var(--border-strong)";
          return (
            <g key={i}>
              <circle cx={sx(n.x)} cy={sy(n.y)} r={n.on_path ? (detailed ? 6 : 3) : detailed ? 3.5 : 1.8} fill={color} />
              {detailed && n.on_path && (
                <text x={sx(n.x)} y={sy(n.y) + 20} textAnchor="middle" className="mono" fontSize="10.5" fill={color} fontWeight="700">
                  {n.value.toFixed(3)}
                </text>
              )}
            </g>
          );
        }
        return (
          <g key={i}>
            <circle
              cx={sx(n.x)} cy={sy(n.y)}
              r={n.on_path ? (detailed ? 4.5 : 2.4) : detailed ? 3 : 1.4}
              fill={n.on_path ? "var(--accent)" : "var(--surface-alt)"}
              stroke={n.on_path ? "var(--accent)" : "var(--border-strong)"}
            />
            {detailed && n.on_path && (
              <text x={sx(n.x)} y={sy(n.y) - 9} textAnchor="middle" className="mono" fontSize="10" fill="var(--accent-text)">
                {n.feature} ≤ {n.threshold.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
