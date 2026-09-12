// Weights for the faint "anatomy" colour of the brain views. Every cell adds the same faint grey, so only
// the packed optic lobes used to show up; cells in sparse regions (central brain, nerve cord) get up to
// `maxBoost` times more so the whole nervous system stays visible. 0 = no soma position.

export function sparseBoost(pos: Float32Array, n: number, maxBoost = 5, cellUm = 20): Float32Array {
  const keys = new Float64Array(n);
  const counts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(pos[3 * i])) {
      keys[i] = -1;
      continue;
    }
    const k = (Math.floor(pos[3 * i] / cellUm) * 4096 + Math.floor(pos[3 * i + 1] / cellUm)) * 4096 + Math.floor(pos[3 * i + 2] / cellUm);
    keys[i] = k;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  // Reference density: 80 % of cells live in grid cells at least this sparse; denser keeps weight 1.
  const perCell: number[] = [];
  for (let i = 0; i < n; i++) if (keys[i] >= 0) perCell.push(counts.get(keys[i])!);
  perCell.sort((a, b) => a - b);
  const ref = perCell[Math.floor(0.8 * (perCell.length - 1))] ?? 1;
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (keys[i] >= 0) w[i] = Math.min(maxBoost, Math.max(1, ref / counts.get(keys[i])!));
  }
  return w;
}
