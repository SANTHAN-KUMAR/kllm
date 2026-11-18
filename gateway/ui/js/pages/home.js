// Home: what is running, and a way to start something.
// Sources: GET /admin/api/backends (shared poll), GET /admin/api/metrics?range=24h (30 s).
import { get, poll } from "../api.js";
import { html, render, $, icon, word, dur, compact, num, bar, empty, errorBox, loading, Fresh } from "../ui.js";
import { area, stacked, orb } from "../charts.js";
import { openPalette } from "../app.js";


export function quotaStrip(accounts, runningSet) {
  const rows = accounts.map(a => ({ ...a, used: Math.max(0, a.limit_h - a.left_h) })).sort((x, y) => y.used - x.used);
  return html`<div class="k-quota" role="img" aria-label="${rows.length} accounts, sorted by GPU hours used this week">${rows.map(a =>
    html`<i class="${runningSet.has(a.name) ? "k-run" : ""}" title="${a.name}: ${a.left_h} h left of ${a.limit_h} h"><b style="transform:scaleY(${a.limit_h ? (a.used / a.limit_h).toFixed(3) : "0"})"></b></i>`)}</div>
    <div class="k-quota-cap"><span>most used</span><span>${accounts[0] ? num(accounts[0].limit_h, 0) : "?"} h per account · blue = running now</span></div>`;
}

export function poolHours(b) {
  // headline formula: sum of left_h over enabled accounts kllm may launch on (account_allow_prefixes)
  return b.accounts.filter(a => !a.disabled && a.allowed !== false).reduce((s, a) => s + Math.max(0, a.left_h), 0);
}

export function stepFrac(p) { const done = p.steps.filter(s => s.state === "done").length; return done / p.steps.length; }
export function activeStep(p) { return p.steps.find(s => s.state === "active" || s.state === "failed") || p.steps[p.steps.length - 1]; }

export async function mount(el, { shared, go }) {
  render(el, html`<div class="k-page k-home">
    <div class="k-home-top"><div>
      <span class="k-kicker" id="h-kick"></span>
      <h1 class="k-head" id="h-head">${""}</h1>
      <form class="k-cmd k-glow" id="h-cmd" role="search"><span class="k-dim">${icon("search")}</span>
        <input id="h-in" aria-label="Search or run" placeholder="launch a model, rotate kg3, go models" autocomplete="off">
        <button class="k-btn k-primary k-sm" type="submit">Run</button></form>
      <div class="k-cmd-hints"><span><b>launch</b> model</span><span><b>rotate</b> account</span><span><b>logs</b> account</span><span><b>stop</b> all</span></div>
    </div><div class="k-home-orb" id="h-orb"></div></div>
    <div class="k-sect"><h2>Running now</h2><span id="h-run-note"></span><a class="k-link" href="#fleet">Open fleet</a></div>
    <div id="h-run" class="k-staleable">${loading()}</div><p class="k-fresh" id="h-fresh"></p>
    <div class="k-sect"><h2>Last 24 hours</h2><span id="h-m-note"></span><a class="k-link" href="#usage">Usage</a></div>
    <div id="h-metrics">${loading()}</div>
    <div class="k-sect"><h2>GPU pool</h2><span id="h-pool-note"></span><a class="k-link" href="#accounts">Accounts</a></div>
    <div id="h-pool"></div>
  </div>`);
  $("#h-cmd", el).onsubmit = e => { e.preventDefault(); openPalette($("#h-in", el).value); };
  const fresh = new Fresh($(".k-home", el), 20000, $("#h-fresh", el));
  const tick = () => {
    const d = new Date();
    $("#h-kick", el).textContent = `Home · ${d.toLocaleDateString("en-GB", { weekday: "long" })} ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
  };
  tick();
  const clockT = setInterval(tick, 30000);

  const paint = s => {
    if (!s.backends) { if (s.error) render($("#h-run", el), errorBox(s.error)); return; }
    fresh.mark();
    const b = s.backends, healthy = b.backends.filter(x => x.healthy);
    const models = new Set(healthy.map(x => x.model_key));
    const hours = Math.floor(poolHours(b));
    const head = $("#h-head", el);
    if (!b.backends.length && !b.pending.length) {
      render(head, html`Nothing is running.<br><span>${num(hours)} GPU hours left.</span>`);
    } else if (!b.backends.length) {
      render(head, html`${word(b.pending.length)} ${b.pending.length === 1 ? "kernel is" : "kernels are"} starting.<br><span>${num(hours)} GPU hours left.</span>`);
    } else {
      render(head, html`${word(models.size)} ${models.size === 1 ? "model is" : "models are"} serving.<br><span>${num(hours)} GPU hours left.</span>`);
    }
    // orb: one ring per kernel, sized by the share of its session left (left_s / (left_s + up_s))
    const rings = b.backends.map(x => ({ frac: x.left_s / Math.max(1, x.left_s + x.up_s), title: `${x.account}: ${dur(x.left_s)} left`, state: "live" }))
      .concat(b.pending.map(p => ({ frac: stepFrac(p), title: `${p.account}: ${activeStep(p).label}`, state: "wait" })));
    const longest = Math.max(0, ...b.backends.map(x => x.left_s));
    render($("#h-orb", el), rings.length ? html`${orb(rings, { center: b.backends.length ? dur(longest) : "starting", sub: b.backends.length ? "longest left" : "" })}
      <p class="k-orb-cap">${[...b.backends.map(x => x.account), ...b.pending.map(p => p.account)].slice(0, 5).join(" · ")}<br>session time left, outer to inner</p>` : "");
    const rows = [
      ...b.backends.map(x => html`<a class="k-row" href="#fleet/${x.account}"><span class="k-glyph">${icon("chip")}</span>
        <span class="k-t"><b>${x.model_key}</b><small>${x.account} · ${x.sessions || 0} ${x.sessions === 1 ? "session" : "sessions"} · ${x.rpm || 0} req/min${x.healthy ? "" : " · not answering"}</small></span>
        <span class="k-v"><span>${x.draining ? "handing off, " : ""}${dur(x.left_s)} left</span>${bar(x.left_s / Math.max(1, x.left_s + x.up_s))}</span></a>`),
      ...b.pending.map(p => html`<a class="k-row" href="#starting/${p.account}"><span class="k-glyph">${icon("chip")}</span>
        <span class="k-t"><b>${p.model_key}</b><small>${p.account} · ${activeStep(p).label.toLowerCase()}${p.failed ? " failed" : ""}</small></span>
        <span class="k-v"><span>${p.failed ? "failed" : `started ${dur(p.age_s)} ago`}</span>${bar(stepFrac(p), p.failed ? "k-risk" : "k-live")}</span></a>`),
    ];
    render($("#h-run", el), rows.length ? rows : empty("Nothing is running.", "Launch a model to start. Autoscale can also start one when a request arrives.",
      html`<a class="k-btn k-primary" href="#launch">${icon("plus")}Launch a model</a>`));
    const hand = b.backends.find(x => x.replaced_by);
    $("#h-run-note", el).textContent = hand ? `${hand.account} hands off to ${hand.replaced_by}` : b.autoscale ? "autoscale on" : "autoscale off";
    const run = new Set([...b.backends.map(x => x.account), ...b.pending.map(p => p.account)]);
    const allowed = b.accounts.filter(a => a.allowed !== false);
    render($("#h-pool", el), allowed.length ? quotaStrip(allowed, run) : empty("No accounts yet.", "Add a Kaggle account to run models.", html`<a class="k-btn k-primary" href="#welcome">Add account</a>`));
    $("#h-pool-note", el).textContent = `${allowed.length} accounts`;
  };
  const off = shared.on(paint);
  paint(shared);

  const stopM = poll(async () => {
    const r = await get(`/admin/api/metrics?range=24h&tz=${-new Date().getTimezoneOffset()}`);
    const box = $("#h-metrics", el);
    if (!r.ok) { render(box, errorBox(r.error)); return; }
    const m = r.data, t = m.totals;
    $("#h-m-note", el).textContent = t.n ? `${num(t.n)} requests, ${Math.round((t.cache_hit || 0) * 100)}% from prompt cache` : "no requests yet";
    const hrs = m.hourly.map(h => new Date(h.t * 1000).getHours().toString().padStart(2, "0") + ":00");
    render(box, html`<div class="k-two">
      <div><p class="k-chart-title">Requests per hour</p>${area(m.hourly.map(h => h.n), { label: "Requests per hour, last 24 hours", xlabels: [hrs[0], hrs[hrs.length - 1]] })}</div>
      <div><p class="k-chart-title">Prompt tokens per hour</p>${stacked(m.hourly.map(h => ({ a: h.new_in, b: h.cached_in })), { label: "New and cached prompt tokens per hour", xlabels: [hrs[0], hrs[hrs.length - 1]] })}
        <div class="k-legend"><span><i style="background:var(--k-text)"></i>new ${compact(t.new_in)}</span><span><i style="background:rgb(var(--k-lead))"></i>cached ${compact(t.cached_in)}</span></div></div></div>`);
  }, 30000);
  return () => { off(); stopM(); clearInterval(clockT); fresh.stop(); };
}
