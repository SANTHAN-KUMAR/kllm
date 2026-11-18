// Autoscale: rules as sentences, live demand, planner decisions.
// GET/POST /admin/api/autoscale/config (10 s), GET /__relay/demand (5 s).
import { get, post, poll } from "../api.js";
import { html, render, $, $$, icon, num, when, dur, errorBox, loading, empty, toast, Fresh } from "../ui.js";
import { spark } from "../charts.js";

// Each rule is a sentence with one or more settings in it ({key}).
const RULES = [
  "Keep at least {min_backends} and at most {max_backends} kernels running.",
  "Keep {spare_backends} spare kernel ready while traffic grows.",
  "Put up to {sessions_per_backend} active sessions on one kernel.",
  "Stop a kernel after {idle_stop_minutes} min with no requests.",
  "Skip accounts with less than {min_quota_hours} h of weekly quota left.",
  "Let a request wait up to {max_wait_seconds} s for a kernel.",
];
const ADVANCED = [
  "Count a session as active for {session_active_seconds} s after its last request.",
  "Give up on a launch stuck in Kaggle's queue after {queue_timeout_seconds} s.",
  "Give up on a kernel that has not loaded after {load_timeout_minutes} min.",
  "Rest an account for {fail_cooldown_minutes} min after a failed launch, {queue_cooldown_minutes} min after a queue timeout.",
  "Queue at most {max_queued_per_model} launches per model, with {slots_per_backend} request slots per kernel.",
];

export async function mount(el) {
  let cfg = null, demandSeries = [];
  render(el, html`<div class="k-page k-auto"><span class="k-kicker" id="as-kick">Autoscale</span><h1 class="k-head" id="as-head">${""}</h1>
    <div class="k-acts" id="as-acts"></div>
    <div class="k-sect"><h2>Rules</h2><span>saved as you change them</span></div><div id="as-rules">${loading()}</div>
    <details class="k-adv" style="margin-top:16px"><summary>More rules</summary><div id="as-adv"></div></details>
    <div class="k-sect"><h2>Demand</h2><span id="as-d-note"></span></div><div id="as-demand" class="k-staleable">${loading()}</div><p class="k-fresh" id="as-fresh"></p>
    <div class="k-sect"><h2>Planner decisions</h2><span>newest first</span></div><div id="as-dec">${loading()}</div></div>`);
  const fresh = new Fresh($(".k-auto", el), 5000, $("#as-fresh", el));
  const sentence = (tpl) => html`<p class="k-rule">${tpl.split(/(\{\w+\})/).map(p => {
    const m = p.match(/^\{(\w+)\}$/);
    if (!m) return p;
    const k = m[1], sp = cfg.spec[k], v = cfg.settings[k];
    return html`<span class="k-step"><button type="button" data-dec="${k}" aria-label="Less">${icon("minus")}</button><input data-k="${k}" type="number" value="${v}" min="${sp.min}" max="${sp.max}" step="${sp.type === "float" ? 0.5 : 1}" aria-label="${k.replace(/_/g, " ")}"><button type="button" data-inc="${k}" aria-label="More">${icon("plus")}</button></span>`;
  })}<span class="k-note k-saved" data-note></span></p>`;
  const paint = () => {
    const on = cfg.settings.enabled;
    $("#as-kick", el).textContent = `Autoscale · supervisor ${cfg.supervisor ? "running" : "stopped"}`;
    render($("#as-head", el), on ? html`Autoscale is on.<br><span>Models start when used.</span>` : html`Autoscale is off.<br><span>You start models yourself.</span>`);
    render($("#as-acts", el), html`<button class="k-btn ${on ? "" : "k-primary"}" id="as-toggle">${on ? "Turn autoscale off" : "Turn autoscale on"}</button>
      ${!cfg.supervisor ? html`<span class="k-note k-risk-t">The supervisor is not running, so nothing starts or stops. Run <code>kllm serve</code>.</span>` : ""}`);
    render($("#as-rules", el), RULES.map(sentence));
    render($("#as-adv", el), ADVANCED.map(sentence));
    const dec = cfg.decisions || [];
    render($("#as-dec", el), dec.length ? html`<ol class="k-timeline">${dec.slice(0, 40).map(d => html`<li><span class="k-mono k-dim">${when(d.ts)}</span>
      <b>${d.action}${d.target ? " " + d.target : ""}${d.model ? html` <span class="k-dim">${d.model}</span>` : ""}</b><small>${d.why}</small></li>`)}</ol>`
      : empty("No decisions yet.", "The planner writes one line here each time it starts, stops or skips a kernel.", ""));
  };
  const load = async () => {
    const r = await get("/admin/api/autoscale/config");
    if (!r.ok) { if (!cfg) render($("#as-rules", el), errorBox(r.error)); return; }
    const editing = document.activeElement && document.activeElement.matches("[data-k]") && el.contains(document.activeElement);
    cfg = r.data;
    if (!editing) paint();
  };
  const save = async (k, input) => {
    const v = input.value;
    const note = input.closest(".k-rule").querySelector("[data-note]");
    note.textContent = "Saving";
    const r = await post("/admin/api/autoscale/config", { settings: { [k]: v } });
    if (r.ok) { cfg.settings[k] = r.data.settings[k]; note.textContent = "Saved"; setTimeout(() => { note.textContent = ""; }, 1800); }
    else { note.textContent = ""; toast(r.error, true); input.value = cfg.settings[k]; }
  };
  el.addEventListener("change", e => { const i = e.target.closest("[data-k]"); if (i) save(i.dataset.k, i); });
  el.addEventListener("click", async e => {
    const b = e.target.closest("[data-inc],[data-dec]");
    if (b) {
      const k = b.dataset.inc || b.dataset.dec, i = $(`[data-k="${k}"]`, el);
      const step = +i.step || 1, v = Math.min(+i.max, Math.max(+i.min, +i.value + (b.dataset.inc ? step : -step)));
      if (v !== +i.value) { i.value = v; save(k, i); }
    }
    if (e.target.closest("#as-toggle")) {
      const r = await post("/admin/api/autoscale/config", { settings: { enabled: !cfg.settings.enabled } });
      if (r.ok) { toast(r.data.msg); load(); } else toast(r.error, true);
    }
  });
  const stopC = poll(load, 10000);
  const stopD = poll(async () => {
    const r = await get("/__relay/demand");
    if (!r.ok) { render($("#as-demand", el), errorBox(r.error)); return; }
    fresh.mark();
    const models = Object.entries(r.data.models || {});
    const total = models.reduce((s, [, d]) => s + d.sessions + d.waiting, 0);
    demandSeries.push(total); if (demandSeries.length > 180) demandSeries.shift();
    $("#as-d-note", el).textContent = `${r.data.sessions ? r.data.sessions.length : 0} active sessions`;
    render($("#as-demand", el), html`${models.length ? html`<table class="k-table"><thead><tr><th>Model</th><th class="k-r">Active sessions</th><th class="k-r">Busy</th><th class="k-r">Waiting for a kernel</th><th class="k-r">In flight</th></tr></thead><tbody>
      ${models.map(([m, d]) => html`<tr><td>${m}</td><td class="k-r">${d.sessions}</td><td class="k-r">${d.busy || 0}</td><td class="k-r ${d.waiting ? "k-wait-t" : ""}">${d.waiting}</td><td class="k-r">${d.inflight}</td></tr>`)}</tbody></table>`
      : html`<p class="k-note">No client is using a model right now.</p>`}
      <div style="margin-top:16px;max-width:420px"><p class="k-chart-title">Sessions plus waiting requests, since you opened this page</p>${spark(demandSeries, { label: "Demand over time" })}</div>`);
  }, 5000);
  return () => { stopC(); stopD(); fresh.stop(); };
}
