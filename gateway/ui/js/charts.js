// Inline SVG charts. Each takes real series from an API response and draws nothing it was not given.
import { html, raw, esc, compact } from "./ui.js";

const fmt = v => compact(v);

function scale(values, h, pad = 4) {
  const max = Math.max(1e-9, ...values);
  return { max, y: v => h - pad - (v / max) * (h - 2 * pad) };
}

// Area chart: line in text colour, area in a state colour fading out, the latest point marked.
export function area(values, { w = 420, h = 150, state = "live", label = "", xlabels = [] } = {}) {
  if (!values.length) return html`<p class="k-note">No data in this range.</p>`;
  const { max, y } = scale(values, h - 18);
  const step = values.length > 1 ? w / (values.length - 1) : w;
  const pts = values.map((v, i) => [i * step, y(v)]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
  const fill = `${line}L${w},${h - 18}L0,${h - 18}Z`;
  const last = pts[pts.length - 1];
  const id = "g" + Math.abs(hash(label + values.length));
  const xl = axisX(xlabels, w, h);
  return html`<svg class="k-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgb(var(--k-${state}))" stop-opacity=".32"/><stop offset="1" stop-color="rgb(var(--k-${state}))" stop-opacity="0"/></linearGradient></defs>
    <line class="k-grid" x1="0" x2="${w}" y1="${y(max)}" y2="${y(max)}"/><line class="k-grid" x1="0" x2="${w}" y1="${y(max / 2)}" y2="${y(max / 2)}"/>
    <text class="k-axis" x="0" y="${y(max) - 4}">${fmt(max)}</text>
    <path d="${fill}" fill="url(#${id})"/><path class="k-line" d="${line}"/>
    <circle class="k-halo" cx="${last[0]}" cy="${last[1]}" r="7"/><circle class="k-last" cx="${last[0]}" cy="${last[1]}" r="2.6"/>${xl}</svg>`;
}

// Stacked bars: rows of {a, b}; a in text at 80%, b in lead at 60% (new versus cached tokens).
export function stacked(rows, { w = 420, h = 150, label = "", xlabels = [] } = {}) {
  if (!rows.length) return html`<p class="k-note">No data in this range.</p>`;
  const totals = rows.map(r => (r.a || 0) + (r.b || 0));
  const { max, y } = scale(totals, h - 18);
  const bw = w / rows.length;
  const bars = rows.map((r, i) => {
    const bwi = Math.min(bw * 0.64, 22), x = i * bw + (bw - bwi) / 2, ya = y(r.a || 0), yt = y(totals[i]);
    const base = h - 22;
    return `<rect x="${x.toFixed(1)}" y="${ya.toFixed(1)}" width="${bwi.toFixed(1)}" height="${Math.max(0, base - ya).toFixed(1)}" rx="2" fill="var(--k-text)" fill-opacity=".8"/>`
      + `<rect x="${x.toFixed(1)}" y="${yt.toFixed(1)}" width="${bwi.toFixed(1)}" height="${Math.max(0, ya - yt).toFixed(1)}" rx="2" fill="rgb(var(--k-lead))" fill-opacity=".6"/>`;
  }).join("");
  const xl = axisX(xlabels, w, h);
  return html`<svg class="k-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">
    <line class="k-grid" x1="0" x2="${w}" y1="${y(max)}" y2="${y(max)}"/><text class="k-axis" x="0" y="${y(max) - 4}">${fmt(max)}</text>
    ${raw(bars)}${xl}</svg>`;
}

// Plain bars with a value label (speed by prompt size).
export function bars(rows, { w = 420, h = 150, label = "", unit = "" } = {}) {
  if (!rows.length) return html`<p class="k-note">No data in this range.</p>`;
  const { y } = scale(rows.map(r => r.v || 0), h - 34);
  const bw = w / Math.max(rows.length, 6);              // a lone bucket stays a bar, not a block
  return html`<svg class="k-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}">${raw(rows.map((r, i) => {
    const bwi = Math.min(bw * 0.56, 34), x = i * bw + (bw - bwi) / 2, yy = y(r.v || 0) + 14, cx = x + bwi / 2;
    return `<rect x="${x.toFixed(1)}" y="${yy.toFixed(1)}" width="${bwi.toFixed(1)}" height="${Math.max(0, h - 20 - yy).toFixed(1)}" rx="3" fill="var(--k-text)" fill-opacity=".55"/>`
      + `<text class="k-axis" x="${cx.toFixed(1)}" y="${(yy - 5).toFixed(1)}" text-anchor="middle">${esc((r.v || 0).toFixed(1))}${esc(unit)}</text>`
      + `<text class="k-axis" x="${cx.toFixed(1)}" y="${h - 4}" text-anchor="middle">${esc(r.label)}</text>`;
  }).join(""))}</svg>`;
}

// Sparkline for live series collected from successive real responses.
export function spark(values, { w = 280, h = 44, state = "live", label = "" } = {}) {
  if (values.length < 2) return html`<p class="k-note">Collecting samples.</p>`;
  return area(values, { w, h: h + 18, state, label });
}

// Session orb: one ring per serving kernel, sized by the share of its session left.
export function orb(rings, { size = 236, center = "", sub = "" } = {}) {
  const c = size / 2, gapR = 15;
  const paths = rings.slice(0, 5).map((r, i) => {
    const rad = c - 12 - i * gapR, circ = 2 * Math.PI * rad, frac = Math.max(0.02, Math.min(1, r.frac));
    const col = r.state === "wait" ? "rgb(var(--k-wait))" : "rgb(var(--k-live))";
    return `<circle cx="${c}" cy="${c}" r="${rad}" fill="none" stroke="var(--k-fill-2)" stroke-width="7"/>`
      + `<circle cx="${c}" cy="${c}" r="${rad}" fill="none" stroke="${col}" stroke-width="7" stroke-linecap="round" stroke-opacity="${(0.95 - i * 0.12).toFixed(2)}"`
      + ` stroke-dasharray="${(circ * frac).toFixed(1)} ${circ.toFixed(1)}" transform="rotate(-90 ${c} ${c})"><title>${esc(r.title)}</title></circle>`;
  }).join("");
  return html`<svg class="k-orb" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img" aria-label="${rings.map(r => r.title).join(", ")}">
    ${raw(paths)}<text x="${c}" y="${c + 4}" text-anchor="middle" class="k-orb-num">${center}</text>
    <text x="${c}" y="${c + 24}" text-anchor="middle" class="k-orb-sub">${sub}</text></svg>`;
}

// x-axis labels: [first, last] or [first, middle, last]
function axisX(xl, w, h) {
  if (!xl || !xl.length) return "";
  const last = xl[xl.length - 1], mid = xl.length > 2 ? xl[1] : null;
  return html`<text class="k-axis" x="0" y="${h - 2}">${xl[0]}</text>${mid ? html`<text class="k-axis" x="${w / 2}" y="${h - 2}" text-anchor="middle">${mid}</text>` : ""}<text class="k-axis" x="${w}" y="${h - 2}" text-anchor="end">${last}</text>`;
}

function hash(s) { let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return h; }
