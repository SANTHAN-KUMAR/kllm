// Decode/prefill speed estimates from the relay's own request log.
// Shared by the Fleet slide-over (Overview), the Fleet map tiles and the Wall page.
// These are per-answer timings averaged over recent answers, not GPU profiling:
// decode = output tokens / (total - ttft). prefill ≈ uncached prompt tokens / ttft.
import { get } from "./api.js";

const TTL = 15000;
let cache = { ts: 0, rows: null };

export async function speedRows(force) {
  if (!force && cache.ts && Date.now() - cache.ts < TTL) return cache.rows || [];
  cache.ts = Date.now();                       // reserve the slot even on failure
  try {
    const r = await get("/admin/api/requests?n=300");
    cache = { ts: Date.now(), rows: r.ok ? (r.data.requests || []) : [] };
  } catch (e) {
    cache = { ts: Date.now(), rows: [] };
  }
  return cache.rows || [];
}

export const tpsOf = q => (q && q.ttft_s != null && q.ttft_s >= 0 && q.total_s != null && q.total_s > q.ttft_s && q.out != null)
  ? q.out / (q.total_s - q.ttft_s) : null;

export const prefillOf = q => (q && q.ttft_s > 0 && q.new_in != null) ? q.new_in / q.ttft_s : null;

export const avgOf = (rows, kind, n = 5) => {
  const f = kind === "decode" ? tpsOf : prefillOf;
  const d = (rows || []).slice(0, n).map(f).filter(v => v != null && Number.isFinite(v));
  return d.length ? d.reduce((a, b) => a + b, 0) / d.length : null;
};

export const byBackend = rows => {
  const out = {};
  for (const q of rows || []) {
    if (q && q.backend) (out[q.backend] = out[q.backend] || []).push(q);
  }
  return out;
};
