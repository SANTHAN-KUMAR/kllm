// Shell: hash router, Console and Agents sidebars, sidebar collapse, command palette, alert banners, shared polls.
import { get, post, poll, onRelay, relay, RELAY_DOWN } from "./api.js";
import { html, render, $, $$, icon, esc, toast, sheet, closeSheet, pref, setPref, act } from "./ui.js";

// ---- store: shared, persisted preferences (FRONTEND_GUIDE: one store object) -----------------------------------
export const store = {
  side: pref("side", "shown"), theme: pref("theme", "system"), thinking: pref("thinking", true),
  last: pref("last", { console: "#home", agents: "#agents" }),
  set(k, v) { this[k] = v; setPref(k, v); },
};
if (location.pathname === "/chat") store.side = "hidden";
// a narrow screen opens with the sidebar closed (it slides over the page there); the saved choice is for wide screens
const NARROW = matchMedia("(max-width: 900px)");
if (NARROW.matches) store.side = "hidden";

// ---- shared data: polled once for the sidebar counts and reused by pages -------------------------------------------
export const shared = {
  backends: null, agents: null, approvals: 0, error: null, at: 0, subs: new Set(),
  on(fn) { this.subs.add(fn); return () => this.subs.delete(fn); },
  emit() { this.subs.forEach(fn => { try { fn(this); } catch (e) { console.error(e); } }); },
  async refresh() {
    const r = await get("/admin/api/backends");
    if (r.ok) { this.backends = r.data; this.error = null; this.at = Date.now(); } else this.error = r.error;
    this.emit();
    paintSide();
    return r;
  },
  async refreshAgents() {
    const [s, a] = await Promise.all([get("/admin/api/agents/sessions"), get("/admin/api/agents/approvals")]);
    if (s.ok) this.agents = s.data.sessions;
    if (a.ok) {
      const n = a.data.approvals.length;
      // desktop notification when a new approval arrives (permission asked on the Inbox page)
      if (n > this.approvals && this.agents !== null && document.hidden && "Notification" in window && Notification.permission === "granted") {
        const top = a.data.approvals[0];
        try { new Notification("kllm Agents needs you", { body: top ? `${top.worker_id || "Lead"}: ${top.summary}` : `${n} requests wait` }); } catch { /* not allowed here */ }
      }
      this.approvals = n;
    }
    paintSide();
  },
};

const CONSOLE = [
  ["Run", [["home", "Home"], ["fleet", "Fleet", () => shared.backends && shared.backends.backends.length]]],
  ["Studio", [["chat", "Chat"], ["image", "Image"], ["video", "Video"],
              ["agents", "Agents", () => (shared.agents || []).filter(s => ["running", "needs_you", "queued", "waiting"].includes(s.status)).length || ""]]],
  ["Manage", [["models", "Models", () => shared.backends && shared.backends.models.length],
              ["accounts", "Accounts", () => shared.backends && shared.backends.accounts.filter(a => a.allowed !== false).length],
              ["autoscale", "Autoscale"], ["usage", "Usage"]]],
  ["System", [["logs", "Logs"], ["wall", "Wall"], ["connect", "Connect"], ["settings", "Settings"]]],
];
const AGENTS = [
  ["Work", [["agents", "Sessions", () => { const n = (shared.agents || []).filter(s => ["running", "needs_you", "queued", "waiting"].includes(s.status)).length; return n ? `${n} live` : ""; }],
            ["agents/board", "Board"], ["agents/inbox", "Inbox", () => shared.approvals || "", true],
            ["agents/playbooks", "Playbooks"], ["agents/schedules", "Schedules"]]],
  ["Setup", [["agents/access", "Access"], ["agents/memory", "Memory"], ["agents/harnesses", "Harnesses"],
             ["agents/workers", "Workers"], ["agents/guardrails", "Guardrails"]]],
];

// ---- routing ------------------------------------------------------------------------------------------------------
const ROUTES = {
  home: "home", fleet: "fleet", launch: "fleet", starting: "fleet", chat: "chat", image: "studio", video: "studio",
  models: "models", accounts: "accounts", autoscale: "autoscale", usage: "usage", logs: "logs", connect: "connect",
  settings: "settings", welcome: "welcome", wall: "wall",
};
const AGENT_ROUTES = {
  "": "agents/board", s: "agents/session", new: "agents/new", board: "agents/board", inbox: "agents/inbox",
  playbooks: "agents/library", schedules: "agents/library", memory: "agents/library", access: "agents/setup",
  harnesses: "agents/setup", workers: "agents/setup", guardrails: "agents/setup", compare: "agents/compare",
};

export function parse(hash = location.hash) {
  const h = (hash || "").replace(/^#/, "");
  const [path, qs] = h.split("?");
  const parts = path.split("/").filter(Boolean);
  const query = Object.fromEntries(new URLSearchParams(qs || ""));
  const ws = parts[0] === "agents" ? "agents" : "console";
  let mod;
  if (ws === "agents") mod = AGENT_ROUTES[parts[1] || ""];
  else mod = ROUTES[parts[0] || "home"];
  return { hash: "#" + h, path, parts, query, ws, mod: mod || null };
}

export const go = hash => { if (location.hash === hash) route(); else location.hash = hash; };

let current = { cleanup: null, route: null, token: 0 };

async function route() {
  if (!location.hash) {
    const hasAccounts = shared.backends ? shared.backends.accounts.length > 0 : true;
    history.replaceState(null, "", location.pathname === "/chat" ? "#chat" : hasAccounts ? "#home" : "#welcome");
  }
  const r = parse();
  if (NARROW.matches && document.body.dataset.side !== "hidden") setSide("hidden");   // a pick closes the drawer
  const view = $("#k-view");
  if (!r.mod) {
    // invalid route: hide any keep-alive page, clear its cleanup-tracking, paint the sidebar for this route
    // (otherwise a kept Chat page stays on screen and the sidebar keeps pointing at the previous page)
    for (const [k, box] of kept) box.el.hidden = true;
    view.hidden = false;
    document.body.dataset.ws = r.ws;
    current.route = r; current.page = null;
    paintSide(r);
    paintAlerts();
    render(view, html`<div class="k-page"><h1 class="k-head">Nothing lives here.<br><span>Pick a page in the sidebar.</span></h1></div>`);
    return;
  }
  // one module can serve several screens (Fleet: the map and a launch's progress page): a page's screen() names the
  // screen a route shows, and moving to another screen of the same module remounts instead of calling update()
  const scr = (pg, rt) => (pg && pg.screen && rt ? pg.screen(rt) : "");
  const same = current.route && current.route.mod === r.mod && scr(current.page, current.route) === scr(current.page, r);
  store.last[r.ws] = r.hash; setPref("last", store.last);
  document.body.dataset.ws = r.ws;
  paintSide(r);
  paintAlerts();
  // the same page module handles sub-routes (sheets, tabs) without a remount when it exports update()
  if (same && current.page && current.page.update && current.page.update(r) !== false) { current.route = r; return; }
  if (current.cleanup) { try { current.cleanup(); } catch (e) { console.error(e); } }
  closeSheet();
  current.cleanup = null;
  const token = ++current.token;
  let page;
  try { page = await import(`./pages/${r.mod}.js`); }
  catch (e) { console.error(e); render(view, html`<div class="k-page">${raw500(e)}</div>`); return; }
  if (token !== current.token) return;
  current.page = page; current.route = r;
  // keep-alive pages (Chat, Image, Video) mount once and are only hidden when you leave, so a reply or a render
  // that is still running keeps going and is there when you come back
  const keepKey = page.keepAlive ? (page.keepKey ? page.keepKey(r) : r.mod) : null;
  for (const [k, box] of kept) box.el.hidden = k !== keepKey;
  view.hidden = !!keepKey;
  if (keepKey) {
    let box = kept.get(keepKey);
    if (!box) {
      const el = document.createElement("div"); el.className = "k-kept";
      view.after(el);
      box = { el, page };
      kept.set(keepKey, box);
      try { await page.mount(el, { route: r, go, shared, store }); }
      catch (e) { console.error(e); render(el, html`<div class="k-page">${raw500(e)}</div>`); }
    } else {
      box.el.hidden = false;
      if (page.update) page.update(r);
    }
    current.cleanup = () => { if (page.hide) page.hide(); };
    return;
  }
  view.innerHTML = "";
  const el = document.createElement("div");
  view.append(el);
  try { current.cleanup = (await page.mount(el, { route: r, go, shared, store })) || null; }
  catch (e) { console.error(e); render(el, html`<div class="k-page">${raw500(e)}</div>`); }
  const main = $("#k-main"); if (main) main.scrollTop = 0;
}
const kept = new Map();
const raw500 = e => html`<div class="k-error-box">This page failed to load: ${String(e && e.message || e)}. Reload, and check the relay log if it repeats.</div>`;

// ---- sidebar ---------------------------------------------------------------------------------------------------------
// ---- update notice: one check per page load (the relay caches PyPI for a day) ---------------------------------------------
let UPD = null;
get("/admin/api/update").then(r => { if (r.ok && r.data && r.data.update_available) { UPD = r.data; paintSide(); } }).catch(() => {});
document.addEventListener("click", async e => {
  if (!e.target.closest("#k-upd") || !UPD) return;
  try { await navigator.clipboard.writeText(UPD.command); } catch (_) { /* clipboard blocked: the toast still shows it */ }
  toast(`Update to ${UPD.latest}: run "${UPD.command}" (copied), then restart kllm serve.`);
});

function paintSide(r = current.route || parse()) {
  const side = $("#k-side");
  if (!side) return;
  const ws = r.ws;
  const curr = ws === "agents" ? (r.parts[1] === "s" || r.parts[1] === "compare" ? "agents" : ["agents", r.parts[1]].filter(Boolean).join("/"))
    : ({ launch: "fleet", starting: "fleet" }[r.parts[0]] || r.parts[0] || "home");
  const groups = ws === "agents" ? AGENTS : CONSOLE;
  const sup = shared.backends ? shared.backends.supervisor : null;
  const live = (shared.agents || []).filter(s => ["running", "needs_you", "queued", "waiting"].includes(s.status));
  render(side, html`
    <div class="k-wm">${icon("mark", "k-mark")}<span>kllm${ws === "agents" ? html` <span class="k-wm-sub">Agents</span>` : ""}</span>
      <button class="k-icon-btn" id="k-collapse" aria-label="Hide sidebar" title="Hide sidebar (Ctrl \\)">${icon("panel")}</button></div>
    <div class="k-ws" role="tablist" aria-label="Workspace">
      <a href="${ws === "console" ? r.hash : store.last.console}" role="tab" aria-current="${ws === "console"}">Console</a>
      <a href="${ws === "agents" ? r.hash : store.last.agents}" role="tab" aria-current="${ws === "agents"}">Agents</a></div>
    ${ws === "agents" ? html`<a class="k-btn k-primary" href="#agents/new" style="justify-content:center">${icon("plus")}New session</a>` : ""}
    <button class="k-trig" id="k-trig">${icon("search")}Search or run<kbd>Ctrl K</kbd></button>
    ${groups.map(([label, items]) => html`<div class="k-nav"><span>${label}</span>${items.map(([key, name, count, wait]) => {
      const c = count ? count() : "";
      return html`<a href="#${key}" aria-current="${key === curr ? "page" : "false"}">${name}${c ? html`<em class="${wait ? "k-wait-t" : ""}">${c}</em>` : ""}</a>`;
    })}</div>`)}
    ${ws === "agents" && live.length ? html`<div class="k-nav"><span>Running</span><div class="k-side-sessions">${live.slice(0, 6).map(s => html`
      <a href="#agents/s/${s.id}" aria-current="${r.parts[2] === s.id ? "page" : "false"}"><b><i class="k-dot ${s.status === "needs_you" ? "k-wait" : "k-ok"}"></i>${s.title}</b>
      <small>${s.workers} ${s.workers === 1 ? "worker" : "workers"}${s.approvals_pending ? ` · ${s.approvals_pending} waiting` : ""}</small></a>`)}</div></div>` : ""}
    <div class="k-side-foot">
      ${sup == null ? html`<span><i class="k-dot"></i>checking supervisor</span>` : html`<span><i class="k-dot ${sup ? "k-ok" : "k-risk"}"></i>supervisor ${sup ? "running" : "stopped"}</span>`}
      <span class="k-mono">relay ${esc(location.host)}</span>
      ${UPD && UPD.update_available ? html`<button class="k-upd" id="k-upd" type="button" title="Copy the update command">${icon("arrowdown")}kllm ${esc(UPD.latest)} is available</button>` : ""}</div>`);
}

function setSide(v) {
  if (NARROW.matches) store.side = v; else store.set("side", v);
  document.body.dataset.side = v;
}

// ---- command palette --------------------------------------------------------------------------------------------------
const PAGES = [["home", "Home"], ["fleet", "Fleet"], ["chat", "Chat"], ["image", "Image"], ["video", "Video"], ["models", "Models"],
  ["accounts", "Accounts"], ["autoscale", "Autoscale"], ["usage", "Usage"], ["logs", "Logs"], ["wall", "Wall"], ["connect", "Connect"], ["settings", "Settings"],
  ["agents", "Agents sessions"], ["agents/new", "New agents session"], ["agents/inbox", "Agents inbox"], ["agents/board", "Agents board"],
  ["agents/harnesses", "Harnesses"], ["agents/guardrails", "Guardrails"]];

function score(q, text) {
  q = q.toLowerCase(); text = text.toLowerCase();
  if (!q) return 1;
  let i = 0, s = 0, run = 0;
  for (const ch of text) {
    if (ch === q[i]) { i++; run++; s += run; } else run = 0;
    if (i === q.length) return s + (text.startsWith(q) ? 20 : 0);
  }
  return 0;
}

async function openPalette(initial = "") {
  if ($(".k-palette")) return;
  const wrap = document.createElement("div");
  render(wrap, html`<div class="k-scrim" data-close></div><div class="k-palette k-float" role="dialog" aria-label="Command palette">
    <div class="k-palette-in">${icon("search")}<input aria-label="Search or run" placeholder="launch qwen, rotate kg3, logs kg3, go models, stop all" autocomplete="off"></div>
    <div class="k-palette-list" role="listbox"></div>
    <div class="k-palette-foot"><span><kbd>Enter</kbd> run</span><span><kbd>Up</kbd> <kbd>Down</kbd> move</span><span><kbd>Esc</kbd> close</span></div></div>`);
  document.body.append(wrap);
  const input = $("input", wrap), list = $(".k-palette-list", wrap);
  input.value = initial;
  const prev = document.activeElement;
  const close = () => { wrap.remove(); document.removeEventListener("keydown", key, true); if (prev && prev.focus) prev.focus(); };
  wrap.addEventListener("click", e => { if (e.target.closest("[data-close]")) close(); });
  const [m, b] = await Promise.all([get("/admin/api/models"), shared.backends ? { ok: true, data: shared.backends } : get("/admin/api/backends")]);
  const models = m.ok ? m.data.models.filter(x => !x.problems || !x.problems.length) : [];
  const running = b.ok ? b.data.backends.map(x => x.account) : [];
  const accounts = b.ok ? b.data.accounts : [];
  const items = [
    ...models.map(x => ({ g: "Launch", t: `launch ${x.key}`, k: `${x.kind} · ${x.ctx ? Math.round(x.ctx / 1024) + "K" : ""}`, run: () => go(`#launch?model=${encodeURIComponent(x.key)}`) })),
    ...running.map(a => ({ g: "Actions", t: `rotate ${a}`, k: "opens the kernel", run: () => go(`#fleet/${a}`) })),
    ...running.map(a => ({ g: "Actions", t: `logs ${a}`, k: "requests", run: () => go(`#logs?backend=kaggle:${a}`) })),
    ...accounts.map(a => ({ g: "Actions", t: `account ${a.name}`, k: `${a.left_h} h left`, run: () => go(`#accounts?q=${a.name}`) })),
    { g: "Actions", t: "stop all", k: "asks first", run: () => stopAll() },
    { g: "Actions", t: "new agents session", k: "agents", run: () => go("#agents/new") },
    ...PAGES.map(([h, n]) => ({ g: "Pages", t: `go ${n.toLowerCase()}`, k: "page", run: () => go("#" + h) })),
  ];
  let sel = 0, shown = [];
  const paint = () => {
    const q = input.value.trim();
    shown = items.map(it => ({ it, s: score(q, it.t) })).filter(x => x.s > 0).sort((a, z) => z.s - a.s).slice(0, 40).map(x => x.it);
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    let g = null;
    render(list, shown.length ? shown.map((it, i) => html`${it.g !== g ? (g = it.g, html`<span>${it.g}</span>`) : ""}
      <button role="option" data-i="${i}" aria-selected="${i === sel}">${icon(it.g === "Launch" ? "chip" : it.g === "Pages" ? "arrow" : "bolt")}${it.t}<small>${it.k}</small></button>`)
      : html`<p class="k-note" style="padding:14px">Nothing matches. Try a model name, an account such as kg3, or a page.</p>`);
    const s = $(`[aria-selected="true"]`, list); if (s) s.scrollIntoView({ block: "nearest" });
  };
  const run = i => { const it = shown[i]; if (it) { close(); it.run(); } };
  const key = e => {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(shown.length - 1, sel + 1); paint(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel - 1); paint(); }
    else if (e.key === "Enter") { e.preventDefault(); run(sel); }
  };
  document.addEventListener("keydown", key, true);
  input.addEventListener("input", () => { sel = 0; paint(); });
  list.addEventListener("click", e => { const btn = e.target.closest("button[data-i]"); if (btn) run(+btn.dataset.i); });
  paint();
  input.focus();
}
export { openPalette };

function stopAll() {
  const n = shared.backends ? shared.backends.backends.length + shared.backends.pending.length : 0;
  const s = sheet({
    title: "Stop everything?", sub: n ? `${n} kernels stop and free their GPU quota. Clients wait or fail until something runs again.` : "Nothing is running. kllm still sends a stop to every account.",
    body: html`<p class="k-sec">Autoscale may start kernels again when requests arrive. Turn it off on the Autoscale page to keep everything down.</p>`,
    foot: html`<button class="k-btn k-risk" data-stop>Stop everything</button><button class="k-btn k-push" data-close>Keep running</button>`,
  });
  $("[data-stop]", s.el).onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/backends/stop", {})); if (r && r.ok) { s.close(); shared.refresh(); } };
}

// ---- alert banners (GET /admin/api/alerts every 10 s) ---------------------------------------------------------------
const dismissed = new Set(pref("dismissed", []));
let alerts = [];
// banners show for a while and then fade: warnings 10 s, critical 20 s. One comes back only when its text changes
// (memory 96% -> 99% is news; the same sentence every poll is not). Dismissing by hand hides it for good.
const ALERT_TTL = { warn: 10000, crit: 20000 };
const alertSeen = new Map();          // id|text -> first shown (ms)
let alertTimer = null;
function paintAlerts() {
  let box = $("#k-alerts");
  if (!box) { box = document.createElement("div"); box.id = "k-alerts"; box.className = "k-alerts"; $("#k-main").insertBefore(box, $("#k-view")); }
  const now = Date.now();
  const fresh = a => { const k = a.id + "|" + a.text; if (!alertSeen.has(k)) alertSeen.set(k, now); return now - alertSeen.get(k) < (ALERT_TTL[a.level] || ALERT_TTL.warn); };
  const show = alerts.filter(a => a.level !== "info" && !dismissed.has(a.id) && fresh(a)).slice(0, 4);
  box.hidden = !show.length || ["chat", "studio", "agents/session"].includes(parse().mod);
  clearTimeout(alertTimer);
  if (show.length) {
    const next = Math.min(...show.map(a => alertSeen.get(a.id + "|" + a.text) + (ALERT_TTL[a.level] || ALERT_TTL.warn) - now));
    alertTimer = setTimeout(paintAlerts, Math.max(250, next));
  }
  render(box, html`<div class="k-banners">${show.map(a => { const left = alertSeen.get(a.id + "|" + a.text) + (ALERT_TTL[a.level] || ALERT_TTL.warn) - now;
    return html`<div class="k-banner ${a.level === "crit" ? "k-crit" : "k-warn"}" style="--k-fade-delay:${Math.max(0, left - 600)}ms"><i class="k-dot"></i><span>${a.text}</span>
    <button class="k-icon-btn" data-dismiss="${a.id}" aria-label="Dismiss" title="Dismiss">${icon("x")}</button></div>`; })}</div>`);
}

// ---- boot -------------------------------------------------------------------------------------------------------------
function boot() {
  document.body.dataset.side = store.side;
  document.addEventListener("click", e => {
    if (e.target.closest("#k-collapse")) setSide("hidden");
    else if (e.target.closest("#k-float-open")) setSide("shown");
    else if (e.target.closest("#k-trig")) openPalette();
    const row = e.target.closest("[data-href]");
    if (row && !e.target.closest("a, button")) go(row.dataset.href);
    const d = e.target.closest("[data-dismiss]");
    if (d) { dismissed.add(d.dataset.dismiss); setPref("dismissed", [...dismissed].slice(-50)); paintAlerts(); }
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target.matches && e.target.matches("[data-href]")) go(e.target.dataset.href);
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); }
    if (mod && e.key === "\\") { e.preventDefault(); setSide(store.side === "hidden" ? "shown" : "hidden"); }
  });
  onRelay(down => {
    const b = $("#k-relay-down");
    b.hidden = !down;
    b.textContent = down ? RELAY_DOWN : "";
  });
  window.addEventListener("hashchange", route);
  let routed = false;
  shared.on(() => { if (!routed) { routed = true; route(); } });
  poll(() => shared.refresh(), () => (["fleet"].includes(parse().mod) ? 3000 : 10000));
  poll(async () => { const r = await get("/admin/api/alerts"); if (r.ok) { alerts = r.data.alerts; paintAlerts(); } }, 10000);
  poll(() => shared.refreshAgents(), () => (parse().ws === "agents" ? 5000 : 30000));
}
boot();
