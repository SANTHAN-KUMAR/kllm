// Shared UI helpers: escaped templates, formatting, sheets, toasts, freshness, empty/gap/error states.
import { marked } from "../vendor/marked.esm.js";
import DOMPurify from "../vendor/purify.es.js";
import { RELAY_DOWN } from "./api.js";

// ---- templates: html`...` escapes every interpolation unless wrapped in raw() --------------------------------
class Raw { constructor(s) { this.s = s; } toString() { return this.s; } }
export const raw = s => new Raw(s == null ? "" : String(s));
export function esc(v) {
  return String(v == null ? "" : v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function part(v) {
  if (v == null || v === false) return "";
  if (v instanceof Raw) return v.s;
  if (Array.isArray(v)) return v.map(part).join("");
  return esc(v);
}
export function html(strings, ...vals) {
  let out = strings[0];
  vals.forEach((v, i) => { out += part(v) + strings[i + 1]; });
  return new Raw(out);
}
export function render(el, content) { el.innerHTML = part(content); return el; }
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
export const icon = (name, cls = "k-ic") => raw(`<svg class="${cls}" aria-hidden="true"><use href="#k-i-${name}"/></svg>`);

// ---- formatting -------------------------------------------------------------------------------------------------
const WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];
export const word = n => (n >= 0 && n < 10 ? WORDS[n] : String(n));
export const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + "s")}`;
export function num(n, d = 0) {
  if (n == null || isNaN(n)) return "";
  return Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
}
export function compact(n) {
  if (n == null || isNaN(n)) return "";
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  return String(Math.round(n));
}
export function dur(s) {
  if (s == null || isNaN(s)) return "";
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`;
  if (h) return `${h}h ${m % 60}m`;
  return `${m} min`;
}
export function bytes(n, d = 1) {
  if (n == null || isNaN(n)) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (Math.abs(n) >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i < 2 ? 0 : d)} ${u[i]}`;
}
export const gb = (n, d = 1) => (n / 2 ** 30).toFixed(d);
export function clock(ts, secs = false) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", ...(secs ? { second: "2-digit" } : {}) });
}
export function when(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000), now = new Date();
  if (d.toDateString() === now.toDateString()) return clock(ts);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) + " " + clock(ts);
}
export function ago(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  return s < 5 ? "just now" : `${dur(s)} ago`;
}
export const pct = (a, b) => (b ? Math.round((100 * a) / b) : 0);
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---- states ---------------------------------------------------------------------------------------------------------
export const loading = () => html`<div class="k-loading" aria-label="Loading"><div class="k-shimmer"></div><div class="k-shimmer"></div><div class="k-shimmer"></div></div>`;
export const gap = (what, need) => html`<div class="k-gap">${icon("info")}<span>Not available yet: ${what}${need ? html`, needs <code>${need}</code>` : ""}.</span></div>`;
export const empty = (title, text, action) => html`<div class="k-empty"><b>${title}</b>${text ? html`<p>${text}</p>` : ""}${action || ""}</div>`;
export const errorBox = msg => html`<div class="k-error-box" role="alert">${msg || RELAY_DOWN}</div>`;
// the fill is its own element: an <i> inherited list styles (step circles are <i>) and shrank to a sliver
export const bar = (frac, cls = "") => html`<span class="k-bar ${cls}" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(clamp(frac || 0, 0, 1) * 100)}"><span class="k-bar-fill" style="transform:scaleX(${clamp(frac || 0, 0, 1).toFixed(3)})"></span></span>`;

// ---- toast ----------------------------------------------------------------------------------------------------------------
let toastEl = null, toastTimer = null;
export function toast(msg, bad = false) {
  if (!msg) return;
  if (!toastEl) { toastEl = document.createElement("div"); toastEl.setAttribute("role", "status"); document.body.append(toastEl); }
  toastEl.className = "k-toast k-float" + (bad ? " k-bad" : "");
  toastEl.textContent = msg;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.hidden = true; }, bad ? 9000 : 4500);
}

// Run an API action from a button: disabled while waiting, success only after the answer (contract rule 7).
export async function act(btn, fn, { ok = null, quiet = false } = {}) {
  const label = btn ? btn.innerHTML : "";
  if (btn) { btn.disabled = true; btn.setAttribute("aria-busy", "true"); }
  let r;
  try { r = await fn(); } finally {
    if (btn) { btn.disabled = false; btn.removeAttribute("aria-busy"); btn.innerHTML = label; }
  }
  if (!r) return r;
  if (r.ok) { if (!quiet) toast(ok || (r.data && r.data.msg) || "Done."); }
  else toast(r.error, true);
  return r;
}

export async function copy(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    if (btn) { const t = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = t; }, 1400); }
    return true;
  } catch { toast("The browser blocked the clipboard: select the text and copy it.", true); return false; }
}

// ---- sheets ------------------------------------------------------------------------------------------------------------
let openSheet = null;
export function sheet({ title, sub, body, foot, wide = false, onClose, scrim = true }) {
  closeSheet();
  const wrap = document.createElement("div");
  render(wrap, html`${scrim ? html`<div class="k-scrim k-clear" data-close></div>` : ""}
    <aside class="k-sheet k-float ${wide ? "k-wide" : ""}" role="dialog" aria-modal="true" aria-label="${title}">
      <div class="k-sheet-head"><h2>${title}</h2>${sub ? html`<p>${sub}</p>` : ""}
        <button class="k-icon-btn k-close" data-close aria-label="Close" title="Close (Esc)">${icon("x")}</button></div>
      <div class="k-sheet-body"></div>
      ${foot !== false ? html`<div class="k-sheet-foot"></div>` : ""}
    </aside>`);
  document.body.append(wrap);
  const el = $(".k-sheet", wrap);
  const prev = document.activeElement;
  const s = {
    el, body: $(".k-sheet-body", el), foot: $(".k-sheet-foot", el), head: $(".k-sheet-head", el),
    close() {
      if (!wrap.isConnected) return;
      wrap.remove(); document.removeEventListener("keydown", key);
      if (openSheet === s) openSheet = null;
      onClose && onClose();
      if (prev && prev.focus) prev.focus();
    },
  };
  const key = e => { if (e.key === "Escape" && !document.querySelector(".k-palette")) s.close(); };
  document.addEventListener("keydown", key);
  wrap.addEventListener("click", e => { if (e.target.closest("[data-close]")) s.close(); });
  if (body) render(s.body, body);
  if (foot && s.foot) render(s.foot, foot);
  openSheet = s;
  setTimeout(() => { const f = el.querySelector("input, select, textarea, button:not(.k-close)"); (f || el.querySelector(".k-close")).focus(); }, 30);
  return s;
}
export function closeSheet() { if (openSheet) openSheet.close(); }

// ---- freshness: "updated N s ago", stale after 2 intervals (contract rule 4) ------------------------------------------
export class Fresh {
  constructor(root, intervalMs, noteEl) {
    this.root = root; this.interval = intervalMs; this.note = noteEl; this.at = null;
    this.timer = setInterval(() => this.paint(), 1000);
  }
  mark() { this.at = Date.now(); this.paint(); }
  paint() {
    if (!this.root.isConnected) return this.stop();
    if (!this.at) { if (this.note) this.note.textContent = "waiting for data"; return; }
    const age = (Date.now() - this.at) / 1000;
    const stale = age * 1000 > 2 * this.interval;
    this.root.classList.toggle("k-stale", stale);
    if (this.note) this.note.textContent = stale ? `Stale, last update ${new Date(this.at).toLocaleTimeString("en-GB")}`
      : age < 3 ? "updated just now" : `updated ${Math.round(age)} s ago`;
  }
  stop() { clearInterval(this.timer); }
}

// ---- minimal safe markdown (escape first, then add a few tags) ---------------------------------------------------
// Markdown: marked (GFM: tables, task lists, nested lists, numbering) then DOMPurify. Code blocks keep their
// language in data-lang for the copy header; links open in a new tab; file: and relative links keep the label only.
marked.use({ gfm: true, breaks: true, renderer: {
  code({ text, lang }) { return `<pre data-lang="${esc((lang || "").split(/\s/)[0])}"><code>${esc(text)}</code></pre>`; },
} });
DOMPurify.addHook("afterSanitizeAttributes", node => {
  if (node.tagName !== "A") return;
  const href = node.getAttribute("href") || "";
  if (/^https?:\/\//i.test(href)) { node.setAttribute("target", "_blank"); node.setAttribute("rel", "noopener noreferrer"); }
  else node.removeAttribute("href");
});
export function md(src) {
  try { return DOMPurify.sanitize(marked.parse(String(src || "")), { ADD_ATTR: ["target", "data-lang"] }); }
  catch { return `<p>${esc(src)}</p>`; }
}

export function diffHtml(text) {
  return String(text || "").split("\n").map(l => {
    const c = l.startsWith("+++") || l.startsWith("---") || l.startsWith("diff ") ? "k-file" : l.startsWith("@@") ? "k-hunk"
      : l.startsWith("+") ? "k-add" : l.startsWith("-") ? "k-del" : "";
    return c ? `<span class="${c}">${esc(l)}</span>` : esc(l);
  }).join("\n");
}

// Debounce for search fields.
export function debounce(fn, ms = 200) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// Read a stored preference (localStorage may be unavailable).
export function pref(key, fallback) { try { const v = localStorage.getItem("kllm." + key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; } }
export function setPref(key, v) { try { localStorage.setItem("kllm." + key, JSON.stringify(v)); } catch { /* private window */ } }
