// Fetch wrapper: never throws into page code, always returns {ok, data, error, status}.
// Errors are sentences a person can act on. State-changing calls carry X-Kllm-Panel (admin_api.py guard).

const listeners = new Set();
export const relay = { down: false, since: null };

function setDown(down) {
  if (relay.down === down) return;
  relay.down = down;
  relay.since = down ? Date.now() : null;
  listeners.forEach(fn => fn(down));
}
export function onRelay(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export const RELAY_DOWN = "The relay is not running. Start it with `kllm serve`, then this page reconnects.";

function message(status, data, text) {
  if (data && typeof data === "object") {
    const m = data.msg || data.detail || (data.error && data.error.message);
    if (m) return typeof m === "string" ? m : JSON.stringify(m);
  }
  if (status === 404) return "Not found. The relay may be older than this page: restart it with `kllm serve`.";
  if (status === 403) return "Refused by the relay's local guard. Open the panel from this machine.";
  return (text && text.slice(0, 200)) || `The relay answered HTTP ${status}.`;
}

export async function api(path, { method = "GET", body, signal, raw = false } = {}) {
  const opts = { method, signal, headers: {} };
  if (method !== "GET") opts.headers["X-Kllm-Panel"] = "1";
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    if (e && e.name === "AbortError") return { ok: false, error: "Cancelled.", status: 0, aborted: true };
    setDown(true);
    return { ok: false, error: RELAY_DOWN, status: 0 };
  }
  setDown(false);
  if (raw) return { ok: res.ok, res, status: res.status, error: res.ok ? null : `The relay answered HTTP ${res.status}.` };
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!res.ok || (data && data.ok === false)) {
    return { ok: false, data, status: res.status, error: message(res.status, data, text) };
  }
  return { ok: true, data, status: res.status, error: null };
}

export const get = (p, o) => api(p, o);
export const post = (p, b = {}) => api(p, { method: "POST", body: b });
export const put = (p, b = {}) => api(p, { method: "PUT", body: b });
export const del = p => api(p, { method: "DELETE" });

// Poll fn every ms while the page is visible. fn may be async; the next run waits for it. Returns stop().
export function poll(fn, ms) {
  let timer = null, stopped = false, running = false;
  const run = async () => {
    if (stopped || running) return;
    if (document.hidden) { timer = null; return; }
    running = true;
    try { await fn(); } finally { running = false; }
    if (!stopped) timer = setTimeout(run, typeof ms === "function" ? ms() : ms);
  };
  const vis = () => { if (!document.hidden && !timer && !running) run(); };
  document.addEventListener("visibilitychange", vis);
  run();
  return () => { stopped = true; clearTimeout(timer); document.removeEventListener("visibilitychange", vis); };
}
