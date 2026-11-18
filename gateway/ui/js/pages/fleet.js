// Fleet: where traffic goes; the kernel sheet (#fleet/kg3), the launch sheet (#launch?model=) and Starting (#starting/kg7).
// Sources: GET /admin/api/backends (shared, 3 s here), /admin/api/autoscale/config (min_quota_hours), /admin/api/sessions,
// /admin/api/requests, /admin/api/backends/{a}/telemetry and the telemetry stream, POST /admin/api/backends/*.
import { get, post, poll } from "../api.js";
import { html, render, $, $$, icon, esc, dur, num, bytes, gb, bar, pct, clock, empty, errorBox, gap, loading, sheet, act, toast, Fresh, pref, setPref } from "../ui.js";
import { spark } from "../charts.js";
import { stream } from "../live.js";
import { activeStep, stepFrac } from "./home.js";
import { speedRows, avgOf, byBackend } from "../speed.js";

let ctxRef = null, el = null, minQuota = 0, sheetState = null, view = pref("fleetView", "map");
let speedByBackend = {};

export async function mount(root, ctx) {
  ctxRef = ctx; el = root;
  if (ctx.route.parts[0] === "starting") return mountStarting(root, ctx);
  render(el, html`<div class="k-page k-fleet">
    <div class="k-fleet-top"><div><span class="k-kicker">Fleet · live</span><h1 class="k-head k-s" id="f-head">${""}</h1></div>
      <div class="k-acts" style="margin-top:0"><div class="k-seg" role="group" aria-label="View">
        <button data-view="map" aria-pressed="${view === "map"}">Map</button><button data-view="list" aria-pressed="${view === "list"}">List</button></div>
        <a class="k-btn k-primary" href="#launch">${icon("plus")}Launch</a></div></div>
    <div id="f-body" class="k-staleable">${loading()}</div>
    <p class="k-fresh" id="f-fresh"></p></div>`);
  const fresh = new Fresh($(".k-fleet", el), 6000, $("#f-fresh", el));
  el.addEventListener("click", e => {
    const v = e.target.closest("[data-view]");
    if (v) { view = v.dataset.view; setPref("fleetView", view); $$("[data-view]", el).forEach(b => b.setAttribute("aria-pressed", b.dataset.view === view)); paint(ctx.shared); }
  });
  get("/admin/api/autoscale/config").then(r => { if (r.ok) { minQuota = r.data.settings.min_quota_hours || 0; paint(ctx.shared); } });
  const paint = s => {
    if (!el.isConnected) return;
    if (!s.backends) { if (s.error) render($("#f-body", el), errorBox(s.error)); return; }
    fresh.mark();
    speedRows().then(rows => { speedByBackend = byBackend(rows); });
    const b = s.backends;
    const rpm = (b.traffic && b.traffic.rpm != null ? b.traffic.rpm : b.backends.reduce((a, x) => a + (x.rpm || 0), 0));
    const free = freeAccounts(b);
    render($("#f-head", el), html`${b.backends.length} serving<br><span>${b.pending.length} loading · ${free.length} free</span>`);
    render($("#f-body", el), view === "list" ? listView(b, free) : mapView(b, free, rpm));
    if (view === "map") requestAnimationFrame(() => { wires(); spotlight($("#f-canvas", el)); });
    if (sheetState && sheetState.refresh) sheetState.refresh(b);
  };
  const off = ctx.shared.on(paint);
  paint(ctx.shared);
  const onResize = () => view === "map" && wires();
  window.addEventListener("resize", onResize);
  update(ctx.route);
  return () => { off(); fresh.stop(); window.removeEventListener("resize", onResize); if (sheetState) sheetState.close(); };
}

export const screen = r => (r.parts[0] === "starting" ? "starting:" + (r.parts[1] || "") : "fleet");
export function update(r) {
  if (r.parts[0] === "starting") return false;                      // a different screen: remount
  if (sheetState) { const s = sheetState; sheetState = null; s.close(true); }
  if (r.parts[0] === "launch") launchSheet(r.query);
  else if (r.parts[1]) kernelSheet(r.parts[1]);
}

export function freeAccounts(b) {
  // formula (FRONTEND_GUIDE): not running, not launching, not disabled, allowed, left_h above autoscale.min_quota_hours
  return b.accounts.filter(a => !a.busy && !a.disabled && a.allowed !== false && a.left_h > minQuota).sort((x, y) => y.left_h - x.left_h);
}

function mapView(b, free, rpm) {
  const sessions = b.backends.reduce((a, x) => a + (x.sessions || 0), 0);
  const spd = a => { const r = speedByBackend["kaggle:" + a]; const v = r ? avgOf(r, "decode") : null; return v == null ? "" : ` · ${v.toFixed(1)} t/s`; };
  const kernels = [
    ...b.backends.map(x => html`<a class="k-tile k-raised ${x.healthy ? "" : "k-tile-bad"}" href="#fleet/${x.account}" data-node="${x.account}" data-up="${!!x.healthy}" data-traffic="${(x.rpm || 0) + (x.inflight || 0) > 0}">
      <b>${x.account}<i class="k-dot ${x.healthy ? "k-ok" : "k-risk"}"></i></b><small>${x.model_key}</small>${x.spec && (x.spec.quant || x.spec.vram_gb) ? html`<small class="k-tile-spec">${[x.spec.quant, x.spec.vram_gb ? "~" + x.spec.vram_gb + " GB" : ""].filter(Boolean).join(" · ")}</small>` : ""}<small>${dur(x.left_s)} left${x.pinned ? " · pinned" : ""}${spd(x.account)}</small>
      ${x.res && x.res.gpu_total_mib ? html`<span class="k-tile-meter" title="GPU memory">${bar(x.res.gpu_used_mib / x.res.gpu_total_mib, "k-live")}</span>` : ""}</a>`),
    ...b.pending.map(p => html`<a class="k-tile k-raised k-tile-load ${p.failed ? "k-tile-bad" : ""}" href="#starting/${p.account}" data-node="${p.account}" data-pending="1">
      <b>${p.account}<i class="k-dot ${p.failed ? "k-risk" : "k-wait"}"></i></b><small>${activeStep(p).label.toLowerCase()}${p.failed ? " failed" : ""}</small>
      <small>${p.bytes_done ? `${gb(p.bytes_done)}${p.bytes_total ? ` of ${gb(p.bytes_total)}` : ""} GB` : `${dur(p.age_s)} so far`}</small></a>`),
  ];
  // kernels and launch slots share one grid that grows sideways with the screen; a few free accounts are shown as
  // slots, the rest folded into "+N more"
  const slot = a => html`<a class="k-tile k-tile-free" href="#launch?account=${a.name}" data-node="free-${a.name}" data-free="1">${icon("plus")}<small>${a.name} free</small></a>`;
  const shownFree = free.slice(0, 2);
  kernels.push(...shownFree.map(slot));
  if (free.length > shownFree.length) kernels.push(html`<a class="k-tile k-tile-free" href="#accounts">${icon("plus")}<small>+${free.length - shownFree.length} more free</small></a>`);
  const traffic = b.traffic || { series: [], rpm, ttft_p50: null };
  const trafficHasData = (traffic.series || []).some(v => v > 0) || rpm > 0;
  const clientTiles = (b.clients && b.clients.length) ? b.clients.map(c => html`<div class="k-tile k-raised k-tile-client" data-node="client-${c.kind}"><b>${icon("term")}${c.kind}</b><small>${c.sessions} active ${c.sessions === 1 ? "session" : "sessions"}${c.rpm ? ` · ${num(c.rpm, 1)} req/min` : ""}</small></div>`) : html`<div class="k-tile k-raised" data-node="clients"><b>${icon("term")}Clients</b><small>${sessions} active ${sessions === 1 ? "session" : "sessions"}</small></div>`;
  return html`<div class="k-canvas k-aurora" id="f-canvas">
    <div class="k-spot" aria-hidden="true"></div>
    <svg class="k-wires" id="f-wires" aria-hidden="true"></svg>
    <div class="k-col k-col-clients">${clientTiles}</div>
    <div class="k-col k-col-relay"><div class="k-tile k-raised k-tile-relay" data-node="relay" data-traffic="${rpm > 0}"><b>relay</b><small>${sessions} sessions</small><small>${num(rpm, 1)} req/min</small></div></div>
    <div class="k-col k-col-kernels">${kernels.length ? kernels : html`<a class="k-tile k-tile-free" href="#accounts"><small>No free account</small></a>`}</div>
    <div class="k-traffic k-raised"><div class="k-traffic-head"><span>Traffic, last hour</span><span class="k-mono">${num(rpm, 1)} req/min${traffic.ttft_p50 != null ? ` · p50 ${traffic.ttft_p50.toFixed(1)} s` : ""}</span></div>
      <p><b class="k-num">${num(rpm, 1)}</b> requests a minute</p>${trafficHasData ? spark(traffic.series, { label: "Requests per minute, last 60 minutes" }) : html`<p class="k-note" style="margin:2px 0 0">Collecting samples. The line appears once requests flow.</p>`}</div>
  </div>`;
}

function listView(b, free) {
  return html`<div class="k-scroll-x"><table class="k-table"><thead><tr><th>Account</th><th>Model</th><th>State</th><th class="k-r">Sessions</th><th class="k-r">Req/min</th><th class="k-r">GPU memory</th><th class="k-r">Time left</th></tr></thead><tbody>
    ${b.backends.map(x => html`<tr class="k-click" data-href="#fleet/${x.account}" tabindex="0"><td class="k-mono">${x.account}</td><td>${x.model_key}${x.spec && (x.spec.quant || x.spec.vram_gb) ? html`<br><small class="k-dim">${[x.spec.quant, x.spec.vram_gb ? "~" + x.spec.vram_gb + " GB" : ""].filter(Boolean).join(" · ")}</small>` : ""}</td><td>${x.healthy ? "serving" : "not answering"}${x.draining ? ", handing off" : ""}</td>
      <td class="k-r">${x.sessions || 0}</td><td class="k-r">${x.rpm || 0}</td><td class="k-r">${x.res && x.res.gpu_total_mib ? `${(x.res.gpu_used_mib / 1024).toFixed(1)} of ${(x.res.gpu_total_mib / 1024).toFixed(1)} GB` : "no sample"}</td><td class="k-r">${dur(x.left_s)}</td></tr>`)}
    ${b.pending.map(p => html`<tr class="k-click" data-href="#starting/${p.account}" tabindex="0"><td class="k-mono">${p.account}</td><td>${p.model_key}</td><td>${activeStep(p).label}${p.failed ? " failed" : ""}</td><td class="k-r"></td><td class="k-r"></td><td class="k-r"></td><td class="k-r">${dur(p.age_s)} so far</td></tr>`)}
    ${!b.backends.length && !b.pending.length ? html`<tr><td colspan="7" class="k-dim">Nothing is running. ${free.length} accounts are free.</td></tr>` : ""}
  </tbody></table></div>`;
}

// Wires: SVG paths between tile edges. A serving link carries a slow, faint pulse (the link is up); while the backend
// reports traffic (rpm or inflight) the pulses run faster and brighter. A kernel still loading gets a moving dash.
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)");
function wires() {
  const c = $("#f-canvas", el), svg = $("#f-wires", el);
  if (!c || !svg) return;
  const box = c.getBoundingClientRect();
  const pos = n => { const t = $(`[data-node="${n}"]`, c); if (!t) return null; const r = t.getBoundingClientRect(); return { l: r.left - box.left, r: r.right - box.left, y: r.top - box.top + r.height / 2, t }; };
  const relayP = pos("relay"), cli = pos("clients");
  if (!relayP) return;
  const path = (a, bb) => { const mx = (a.x + bb.x) / 2; return `M${a.x},${a.y} C${mx},${a.y} ${mx},${bb.y} ${bb.x},${bb.y}`; };
  let out = `<defs><linearGradient id="f-pulse" x1="0" x2="1"><stop offset="0" style="stop-color:rgb(var(--k-live));stop-opacity:0"/><stop offset="1" style="stop-color:rgb(var(--k-live));stop-opacity:.95"/></linearGradient></defs>`;
  let n = 0;
  const pulse = (id, busy) => REDUCE.matches ? "" : [0, 1].slice(0, busy ? 2 : 1).map(k => {
    const dur = busy ? 1.4 : 4.2, begin = (k * dur / 2 + (n % 5) * 0.37).toFixed(2);
    return `<rect class="k-pulse ${busy ? "k-pulse-busy" : ""}" width="14" height="3" rx="1.5" fill="url(#f-pulse)"><animateMotion dur="${dur}s" begin="-${begin}s" repeatCount="indefinite" rotate="auto"><mpath href="#${id}"/></animateMotion></rect>`;
  }).join("");
  const wire = (a, bb, cls, up, busy) => { const id = `f-w${n++}`; return `<path id="${id}" class="${cls}" d="${path(a, bb)}"/>` + (up ? pulse(id, busy) : ""); };
  const relayBusy = relayP.t.dataset.traffic === "true";
  const clientNodes = $$('[data-node^="client-"]', c);
  const anyUp = $$('[data-up="true"]', c).length > 0;
  if (clientNodes.length) {
    clientNodes.forEach(t => { const p = pos(t.dataset.node); if (p) out += wire({ x: p.r, y: p.y }, { x: relayP.l, y: relayP.y }, `k-wire ${anyUp ? "k-wire-on" : ""}`, anyUp, relayBusy); });
  } else if (cli) out += wire({ x: cli.r, y: cli.y }, { x: relayP.l, y: relayP.y }, `k-wire ${anyUp ? "k-wire-on" : ""}`, anyUp, relayBusy);
  $$("[data-node]", c).forEach(t => {
    const id = t.dataset.node;
    if (id === "relay" || id === "clients") return;
    if (t.dataset.free) return;                                           // free slots are not connected yet
    const p = pos(id), up = t.dataset.up === "true", busy = t.dataset.traffic === "true";
    const cls = t.dataset.pending ? "k-wire k-wire-dash k-wire-load" : t.dataset.free ? "k-wire k-wire-dash" : up ? "k-wire k-wire-on" : "k-wire";
    out += wire({ x: relayP.r, y: relayP.y }, { x: p.l, y: p.y }, cls, up, busy);
  });
  svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  svg.innerHTML = out;
}
// the light follows the cursor across the map, like the prototype
// across the whole page, not only the map's box
function spotlight(c) {
  const page = c && c.closest(".k-page"), spot = c && $(".k-spot", c);
  if (!page || page.dataset.spot) return;
  page.dataset.spot = "1";
  if (spot) page.prepend(spot);
  page.addEventListener("pointermove", e => { const r = page.getBoundingClientRect(); page.style.setProperty("--mx", `${e.clientX - r.left}px`); page.style.setProperty("--my", `${e.clientY - r.top}px`); });
}

// ---- kernel sheet --------------------------------------------------------------------------------------------------
// Decode/prefill rates from the relay's request log (see ../speed.js): decode = out / (total - ttft),
// prefill ≈ uncached prompt tokens / ttft. Approximate, not GPU profiling.

function kernelSheet(account) {
  const b = ctxRef.shared.backends;
  const s = sheet({ title: account, sub: "Reading the kernel", body: loading(), foot: html``, onClose: () => { if (sheetState === st) { sheetState = null; if (location.hash.startsWith("#fleet/")) history.replaceState(null, "", "#fleet"); } } });
  let tab = "overview", stopT = null, recent = null, sessions = null;
  const st = { close: (silent) => { if (stopT) stopT(); s.close(); }, refresh: bb => paintOverview(bb) };
  sheetState = st;
  const tabs = html`<div class="k-seg" role="tablist"><button role="tab" data-tab="overview" aria-pressed="true">Overview</button><button role="tab" data-tab="resources" aria-pressed="false">Resources</button></div>`;
  s.head.insertAdjacentHTML("beforeend", `<div class="k-sheet-tabs">${tabs}</div>`);
  s.el.addEventListener("click", async e => {
    const t = e.target.closest("[data-tab]");
    if (t) { tab = t.dataset.tab; $$("[data-tab]", s.el).forEach(x => x.setAttribute("aria-pressed", x === t)); if (stopT) { stopT(); stopT = null; } tab === "overview" ? paintOverview(ctxRef.shared.backends) : resources(); }
    const a = e.target.closest("[data-act]");
    if (!a) return;
    const k = a.dataset.act;
    if (k === "rotate") { const r = await act(a, () => post("/admin/api/backends/rotate", { account })); if (r.ok) ctxRef.shared.refresh(); }
    if (k === "pin" || k === "unpin") { const r = await act(a, () => post("/admin/api/backends/pin", { account, pinned: k === "pin" })); if (r.ok) ctxRef.shared.refresh(); }
    if (k === "stop") confirmStop(s, account);
    if (k === "forget") { const r = await act(a, () => post("/admin/api/sessions/forget", { id: a.dataset.id })); if (r.ok) loadSessions(); }
    if (k === "move") moveSession(a.dataset.id, a.dataset.model);
  });
  const loadSessions = async () => {
    // one shared cache (../speed.js) feeds the numbers AND the Recent list, so the sheet, the
    // map tiles and the Wall always agree: avg over the same last-5 answers, same 15 s snapshot
    const [ss, rows] = await Promise.all([get("/admin/api/sessions"), speedRows()]);
    sessions = ss.ok ? ss.data.sessions.filter(x => x.backend === "kaggle:" + account) : null;
    recent = rows.filter(x => x.backend === "kaggle:" + account).slice(0, 5);
    if (tab === "overview") paintOverview(ctxRef.shared.backends);
  };
  const paintOverview = bb => {
    if (tab !== "overview" || !s.el.isConnected) return;
    const x = bb && bb.backends.find(y => y.account === account);
    const p = bb && bb.pending.find(y => y.account === account);
    if (!x) {
      render(s.body, p ? html`<p class="k-sec">${account} is still starting: ${activeStep(p).label.toLowerCase()}.</p><p style="margin-top:14px"><a class="k-btn k-sm" href="#starting/${account}">Open Starting</a></p>`
        : empty(`${account} is not running.`, "It may have stopped or rotated. Pick another kernel on the map."));
      render(s.foot, html``);
      return;
    }
    $(".k-sheet-head p", s.el).textContent = `${x.user ? "@" + x.user + " · " : ""}serving ${x.model_key} for ${dur(x.up_s)}`;
    const total = x.left_s + x.up_s;
    render(s.body, html`<div class="k-sheet-big"><b>${dur(x.left_s)}</b><span>left of ${dur(total)}</span></div>${bar(x.left_s / Math.max(1, total), "k-live k-full")}
      <div class="k-kv" style="margin-top:18px">
        <div><span>Model</span><span>${x.model_key}</span></div>
        <div><span>Sessions</span><span>${x.sessions || 0}</span></div>
        <div><span>Requests in flight</span><span>${x.inflight || 0}</span></div>
        <div><span>Requests a minute</span><span>${x.rpm || 0}</span></div>
        <div><span>Errors, last 5 min</span><span class="${x.errors_5m ? "k-risk-t" : ""}">${x.errors_5m || 0}</span></div>
        <div><span>Last good answer</span><span>${x.ok_age_s == null ? "none yet" : `${dur(x.ok_age_s)} ago`}</span></div>
        <div><span>Week quota</span><span>${x.quota_used_h} of ${x.quota_limit_h != null ? x.quota_limit_h : "?"} h</span></div>
        <div><span>Max context</span><span>${x.ctx ? num(x.ctx) + " tokens" : "unknown"}</span></div>
        <div><span>Decode rate</span><span>${recent && recent.length && avgOf(recent, "decode") != null ? avgOf(recent, "decode").toFixed(1) + " t/s" : (recent ? "-" : "estimating after one answer")}</span></div>
        <div><span>Prefill rate</span><span>${recent && recent.length && avgOf(recent, "prefill") != null ? avgOf(recent, "prefill").toFixed(1) + " t/s" : (recent ? "-" : "estimating after one answer")}</span></div>
        <div><span>Tunnel</span><span class="k-mono">${x.host || ""}</span></div></div>
      ${x.cooldown_s ? html`<p class="k-note k-wait-t" style="margin-top:8px">Skipped by the router for ${dur(x.cooldown_s)} after a failure.</p>` : ""}
      <div class="k-sect k-sect-s"><h2>Sessions</h2><span>${sessions ? sessions.length : ""}</span></div>
      ${sessions == null ? loading() : sessions.length ? sessions.slice(0, 8).map(q => html`<div class="k-row k-hov"><span class="k-t"><b class="k-mono">${q.id}</b><small>${q.requests} requests · idle ${dur(q.idle_s)}${q.ctx_est ? ` · about ${num(q.ctx_est)} tokens` : ""}</small></span>
        <span class="k-row-acts"><button class="k-btn k-xs" data-act="move" data-id="${q.id}" data-model="${q.model}">Move</button><button class="k-btn k-xs" data-act="forget" data-id="${q.id}">Forget</button></span></div>`)
        : html`<p class="k-note">No client is attached right now.</p>`}
      <div class="k-sect k-sect-s"><h2>Recent</h2></div>
      ${recent == null ? loading() : recent.length ? html`<div class="k-kv">${recent.map(q => html`<div><span class="k-mono">${clock(q.ts)}</span><span>${q.new_in != null ? `${num((q.new_in || 0) + (q.cached_in || 0))} in, ${num(q.out || 0)} out, ` : (q.kind || "")}${q.total_s != null ? q.total_s + " s" : ""}${q.status && q.status >= 400 ? html`<span class="k-risk-t">HTTP ${q.status}</span>` : ""}</span></div>`)}</div>`
        : html`<p class="k-note">No requests logged for ${account} yet.</p>`}`);
    render(s.foot, html`<a class="k-btn k-sm" href="#logs?backend=kaggle:${account}">Open log</a><button class="k-btn k-sm" data-act="rotate">Rotate now</button>
      <button class="k-btn k-sm" data-act="${x.pinned ? "unpin" : "pin"}">${x.pinned ? "Unpin" : "Keep warm"}</button><button class="k-btn k-sm k-push" data-act="stop">Stop</button>`);
  };
  const resources = () => {
    render(s.body, loading());
    let samples = [], status = "waiting", age = null;
    const fresh = new Fresh(s.body, 10000, null);
    const draw = () => {
      if (tab !== "resources" || !s.el.isConnected) return;
      const last = samples[samples.length - 1];
      if (status === "unsupported") { render(s.body, gap(`${account} runs a kernel without resource telemetry`, "a relaunch on the current kernel template")); return; }
      if (!last) { render(s.body, html`<p class="k-note">${status.startsWith("error") ? `Cannot reach the kernel: ${status.slice(7)}.` : "Waiting for the first sample. The kernel reports every 5 s."}</p>`); return; }
      const gpus = last.gpu || [];
      const memPct = last.cg_limit ? last.cg_used / last.cg_limit : null;
      const oom = samples.length && last.cg_ev_oom_kill != null ? last.cg_ev_oom_kill - (samples[0].cg_ev_oom_kill || 0) : null;
      const disk = (last.disk || {})["/kaggle/working"];
      const slots = last.slots || null;
      const ctxUsed = slots ? slots.reduce((a, q) => a + (q.used || 0), 0) : null;
      const ctxMax = slots && slots.length ? slots[0].n_ctx : null;
      render(s.body, html`<div class="k-res-head"><span class="k-note">From the kernel every 5 s</span><span class="k-fresh" id="r-fresh"></span></div>
        <div class="k-gpus k-staleable">${gpus.map(g => html`<div class="k-gpu k-raised"><div class="k-gpu-top"><b>T4 · ${g.i}</b><span class="${g.temp > 85 ? "k-wait-t" : "k-dim"}">${g.temp} °C</span></div>
          <p><b class="k-num">${(g.used_mib / 1024).toFixed(1)}</b> <span class="k-dim">of ${(g.total_mib / 1024).toFixed(1)} GB</span></p>${bar(g.used_mib / g.total_mib, "k-live k-full")}
          <div class="k-gpu-meta"><span>${g.util}% busy</span><span>${g.power_w != null ? Math.round(g.power_w) + " W" : ""}</span></div>
          ${spark(samples.slice(-120).map(q => ((q.gpu || []).find(z => z.i === g.i) || {}).util || 0), { w: 200, h: 26, label: `GPU ${g.i} utilisation` })}</div>`)}</div>
        <div class="k-kv k-staleable" style="margin-top:12px">
          <div><span>Memory</span><span class="${memPct > 0.9 ? "k-wait-t" : ""}">${last.cg_used ? html`<b>${gb(last.cg_used)}</b> of ${last.cg_limit ? gb(last.cg_limit) : "?"} GB` : "not reported"}</span></div></div>
        ${memPct != null ? bar(memPct, `k-full ${memPct > 0.9 ? "k-wait" : ""}`) : ""}
        <p class="k-note" style="margin-top:6px">${last.srv_VmRSS ? `llama-server uses ${gb(last.srv_VmRSS)} GB` : "llama-server memory not reported"}${oom != null ? ` · out-of-memory kills ${oom}` : ""}</p>
        <div class="k-kv k-staleable" style="margin-top:12px">
          <div><span>Disk, /kaggle/working</span><span class="${disk && disk.used / disk.total > 0.9 ? "k-wait-t" : ""}">${disk ? html`<b>${gb(disk.used)}</b> of ${gb(disk.total)} GB` : "not reported"}</span></div></div>
        ${disk ? bar(disk.used / disk.total, "k-full") : ""}
        <div class="k-kv k-staleable" style="margin-top:12px">
          <div><span>CPU</span><span>${last.cpu_pct != null ? `${last.cpu_pct}%` : "not reported"}${last.load1 != null ? ` · load ${last.load1.toFixed(1)}` : ""}</span></div>
          <div><span>Network</span><span>${last.net_rx_bps != null ? `in ${bytes(last.net_rx_bps)}/s · out ${bytes(last.net_tx_bps)}/s` : "not reported"}</span></div>
          <div><span>Context in use</span><span>${slots ? `${num(ctxUsed)} of ${num(ctxMax)}, ${slots.length} slots` : "not reported"}</span></div></div>
        ${last.decode_tps != null || last.prefill_tps != null ? html`<div class="k-sect k-sect-s"><h2>Speed, live</h2><span>llama.cpp /metrics</span></div>
          <div class="k-kv k-staleable" style="margin-top:4px">
            <div><span>Decode</span><span><b>${last.decode_tps != null ? last.decode_tps.toFixed(1) : "-"}</b> t/s</span></div>
            <div><span>Prefill</span><span><b>${last.prefill_tps != null ? last.prefill_tps.toFixed(1) : "-"}</b> t/s</span></div>
            <div><span>Speed curve, last 120 samples</span><span>${spark(samples.slice(-120).map(q => q.decode_tps || 0), { w: 220, h: 30, label: "decode t/s" })}</span></div></div>` : html`<p class="k-note" style="margin-top:12px">Live speeds appear here once a kernel is pushed by the current template (relaunch/rotate picks them up). Last-answer rates are in Overview.</p>`}
        ${last.cpu_pct == null || !last.disk ? html`<div style="margin-top:14px">${gap("CPU, disk, network and context for this kernel", "the current kernel template (relaunch to get it)")}</div>` : ""}
        <p class="k-note" style="margin-top:16px">kllm warns you when memory passes 90% of the limit, a GPU passes 85 °C, or the disk passes 90%.</p>`);
      fresh.note = $("#r-fresh", s.body);
      fresh.paint();
    };
    get(`/admin/api/backends/${account}/telemetry`).then(r => {
      if (!r.ok) { render(s.body, errorBox(r.error)); return; }
      samples = r.data.samples; status = r.data.status;
      if (samples.length) fresh.mark();
      draw();
    });
    const stopS = stream("/admin/api/telemetry/stream", {
      named: ["sample"],
      onEvent: d => {
        if (d.account !== account) return;
        status = d.status;
        if (d.sample && (!samples.length || d.sample.ts > samples[samples.length - 1].ts)) { samples.push(d.sample); if (samples.length > 720) samples.shift(); fresh.mark(); }
        draw();
      },
    });
    stopT = () => { stopS(); fresh.stop(); };
  };
  paintOverview(b);
  loadSessions();
}

function confirmStop(s, account) {
  render(s.foot, html`<span class="k-sec" style="align-self:center">Stop ${account} and free its GPU?</span><button class="k-btn k-sm k-risk k-push" data-yes>Stop ${account}</button><button class="k-btn k-sm" data-no>Keep it</button>`);
  $("[data-no]", s.foot).onclick = () => ctxRef.shared.refresh();
  $("[data-yes]", s.foot).onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/backends/stop", { account })); if (r.ok) { s.close(); ctxRef.shared.refresh(); } };
}

async function moveSession(id, model) {
  const b = ctxRef.shared.backends;
  const targets = b.backends.filter(x => x.model_key === model || x.model_key === undefined);
  const s = sheet({ title: "Move session", sub: `${id} uses ${model}. kllm replays its last request on the target so the cache is warm.`,
    body: targets.length > 1 ? html`<div class="k-form"><label><span>Move to</span><select class="k-select" id="mv-to">${targets.map(t => html`<option value="${t.account}">${t.account} · ${t.sessions || 0} sessions</option>`)}</select></label>
      <label class="k-check"><input type="checkbox" id="mv-cold"> Move cold if warming fails (the next turn re-reads the whole prompt)</label></div>`
      : html`<p class="k-sec">No other kernel serves ${model}. Launch one first.</p>`,
    foot: targets.length > 1 ? html`<button class="k-btn k-primary" data-go>Move session</button>` : html`<a class="k-btn" href="#launch?model=${encodeURIComponent(model)}">Launch ${model}</a>` });
  const btn = $("[data-go]", s.el);
  if (btn) btn.onclick = async () => { const r = await act(btn, () => post("/admin/api/sessions/move", { id, backend: $("#mv-to", s.el).value, cold: $("#mv-cold", s.el).checked })); if (r.ok) s.close(); };
}

// ---- launch sheet -----------------------------------------------------------------------------------------------------
async function launchSheet(q) {
  const s = sheet({ title: "Launch a model", sub: "kllm pushes a kernel to Kaggle and adds it to the pool when it is ready.", body: loading(), foot: html``,
    onClose: () => { if (sheetState === st) { sheetState = null; if (location.hash.startsWith("#launch")) history.replaceState(null, "", "#fleet"); } } });
  const st = { close: () => s.close() };
  sheetState = st;
  const [m, c] = await Promise.all([get("/admin/api/models"), get("/admin/api/config")]);
  if (!m.ok) { render(s.body, errorBox(m.error)); return; }
  const models = m.data.models.filter(x => !(x.problems || []).length);
  const b = ctxRef.shared.backends || { accounts: [] };
  const free = freeAccounts(b);
  const want = q.model || m.data.default_model;
  const sessionH = c.ok ? Math.round(c.data.session_seconds / 3600 * 10) / 10 : null;
  render(s.body, html`<div class="k-form">
    <label><span>Model</span><select class="k-select" id="l-model">${models.map(x => html`<option value="${x.key}" ${x.key === want ? "selected" : ""}>${x.key}${x.running && x.running.length ? " (running)" : ""}</option>`)}</select></label>
    <p class="k-spec" id="l-spec"></p>
    <label><span>Account</span><select class="k-select" id="l-acct"><option value="">Pick automatically${free[0] ? ` (${free[0].name}, ${free[0].left_h} h left)` : ""}</option>
      ${free.map(a => html`<option value="${a.name}" ${a.name === q.account ? "selected" : ""}>${a.name} · ${a.left_h} h left</option>`)}</select></label>
    <p class="k-note">${free.length} free accounts can launch (enabled, allowed by account_allow_prefixes, above ${minQuota} h of quota).</p>
    <label class="k-check"><button type="button" class="k-toggle" role="switch" aria-checked="false" id="l-warm" aria-label="Keep warm"></button> Keep warm: autoscale never stops it for being idle</label>
    <p class="k-note">Sessions last ${sessionH ? sessionH + " h" : "up to 12 h"}; kllm starts a successor before the cap. First launch on an account builds llama.cpp, about 20 min; later ones take 6 to 10 min.</p>
    <div id="l-err"></div></div>`);
  const spec = () => { const x = models.find(y => y.key === $("#l-model", s.body).value); $("#l-spec", s.body).textContent = x ? [x.kind, x.ctx ? Math.round(x.ctx / 1024) + "K context" : "", x.quant, x.vram_gb ? x.vram_gb + " GB" : "", (x.tags || []).slice(0, 2).join(" ")].filter(Boolean).join(" · ") : ""; };
  const label = () => { const a = $("#l-acct", s.body).value; render(s.foot, html`<button class="k-btn k-primary" data-launch>${icon("plus")}Launch on ${a || "the next free account"}</button>`); $("[data-launch]", s.foot).onclick = e => go(e.currentTarget); };
  $("#l-model", s.body).onchange = spec; $("#l-acct", s.body).onchange = label;
  $("#l-warm", s.body).onclick = e => e.currentTarget.setAttribute("aria-checked", e.currentTarget.getAttribute("aria-checked") !== "true");
  spec(); label();
  const go = async (btn, force = false) => {
    const model = $("#l-model", s.body).value, account = $("#l-acct", s.body).value;
    const r = await act(btn, () => post("/admin/api/backends/launch", { model, account: account || undefined, force }), { quiet: true });
    if (!r.ok) {
      render($("#l-err", s.body), html`<div class="k-error-box">${r.error}</div>${r.data && r.data.can_force ? html`<button class="k-btn k-sm k-risk" style="margin-top:10px" data-force>Launch anyway</button>` : ""}`);
      const f = $("[data-force]", s.body); if (f) f.onclick = () => go(f, true);
      return;
    }
    toast(r.data.msg);
    if (account && $("#l-warm", s.body).getAttribute("aria-checked") === "true") {
      setTimeout(() => post("/admin/api/backends/pin", { account, pinned: true }).then(p => !p.ok && toast(p.error, true)), 4000);
    }
    s.close();
    ctxRef.shared.refresh();
  };
}

// ---- Starting ------------------------------------------------------------------------------------------------------
function mountStarting(root, ctx) {
  const account = ctx.route.parts[1];
  render(root, html`<div class="k-page k-starting"><span class="k-kicker">Starting · ${account}</span><h1 class="k-head" id="s-head"></h1>
    <div id="s-body" class="k-staleable">${loading()}</div><p class="k-fresh" id="s-fresh"></p></div>`);
  const fresh = new Fresh($(".k-starting", root), 6000, $("#s-fresh", root));
  const paint = s => {
    if (!s.backends) { if (s.error) render($("#s-body", root), errorBox(s.error)); return; }
    fresh.mark();
    const p = s.backends.pending.find(x => x.account === account);
    const live = s.backends.backends.find(x => x.account === account);
    const head = $("#s-head", root);
    if (live) {
      render(head, html`${account} is ready.<br><span>${live.model_key} is serving.</span>`);
      render($("#s-body", root), html`<div class="k-acts"><a class="k-btn k-primary" href="#fleet/${account}">Open ${account}</a><a class="k-btn" href="#chat">Chat with it</a></div>`);
      return;
    }
    if (!p) {
      render(head, html`${account} is not starting.<br><span>Nothing is launching there.</span>`);
      render($("#s-body", root), html`<div class="k-acts"><a class="k-btn k-primary" href="#launch?account=${account}">Launch on ${account}</a><a class="k-btn" href="#fleet">Open fleet</a></div>`);
      return;
    }
    const step = activeStep(p);
    render(head, p.failed ? html`The launch failed.<br><span>${p.model_key} on ${account}.</span>` : html`${p.model_key}<br><span>${step.label.toLowerCase()} on ${account}.</span>`);
    render($("#s-body", root), html`<ol class="k-steps">${p.steps.map(x => html`<li class="k-step-${x.state}"><i>${x.state === "done" ? icon("check") : ""}</i>
        <span><b>${x.label}</b>${x.key === "download" && x.bytes_done != null ? html`<small>${gb(x.bytes_done)}${x.bytes_total ? ` of ${gb(x.bytes_total)}` : ""} GB${x.bytes_total ? ` · ${Math.min(100, pct(x.bytes_done, x.bytes_total))}%${x.bytes_done > x.bytes_total ? "+" : ""}` : ""}</small>${x.bytes_total ? bar(x.bytes_done / x.bytes_total, "k-live") : ""}`
          : x.key === "download" && x.state === "active" ? html`<small>download size not reported by this kernel yet</small>` : ""}</span>
        <em class="k-mono">${x.secs != null ? dur(x.secs) : x.state === "active" ? "now" : ""}</em></li>`)}</ol>
      <p class="k-note" style="margin-top:14px">Started ${dur(p.age_s)} ago${p.queue_s != null ? ` · waited ${dur(p.queue_s)} in Kaggle's queue` : ""}${p.origin ? ` · ${p.origin}` : ""}.</p>
      ${p.failed && p.note ? html`<div class="k-error-box" style="margin-top:14px">${p.note}</div>` : ""}
      <div class="k-acts"><button class="k-btn" data-stop>Stop this launch</button><a class="k-btn" href="#logs">Open logs</a></div>`);
    $("[data-stop]", root).onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/backends/stop", { account })); if (r.ok) ctx.shared.refresh(); };
  };
  const off = ctx.shared.on(paint);
  paint(ctx.shared);
  ctx.shared.refresh();
  return () => { off(); fresh.stop(); };
}
