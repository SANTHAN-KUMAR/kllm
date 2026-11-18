// Logs: requests, the service log and the doctor. GET /admin/api/logs/requests (+ .csv), /admin/api/logs/service,
// GET /admin/api/doctor, POST /admin/api/doctor/run.
import { get, post, poll } from "../api.js";
import { html, render, $, $$, num, when, clock, errorBox, loading, empty, act, Fresh } from "../ui.js";

export async function mount(el, ctx) {
  let tab = ctx.route.query.tab || "requests";
  const f = { status: "", backend: ctx.route.query.backend || "", q: "", kind: "", slow: false };
  let doctor = null;
  render(el, html`<div class="k-page k-logs"><span class="k-kicker">Logs</span><h1 class="k-head" id="l-head">${""}</h1>
    <div class="k-acts"><button class="k-btn k-primary" id="l-doc">Run doctor</button><a class="k-btn" id="l-csv">Export CSV</a></div>
    <div class="k-tabs" role="tablist" style="margin-top:34px">${[["requests", "Requests"], ["service", "Service log"], ["doctor", "Doctor"]].map(([k, n]) => html`<button role="tab" data-tab="${k}" aria-selected="${k === tab}">${n}</button>`)}</div>
    <div id="l-filters"></div><div id="l-body" class="k-staleable">${loading()}</div><p class="k-fresh" id="l-fresh"></p></div>`);
  const fresh = new Fresh($(".k-logs", el), 20000, $("#l-fresh", el));
  const qs = () => {
    const p = new URLSearchParams();
    if (f.status) p.set("status", f.status);
    if (f.backend) p.set("backend", f.backend);
    if (f.q) p.set("q", f.q);
    if (f.kind) p.set("kind", f.kind);
    if (f.slow) p.set("min_total", "10");
    return p.toString();
  };
  const headline = () => {
    const h = $("#l-head", el);
    if (!doctor) { render(h, html`Logs<br><span>Checking the last doctor run.</span>`); return; }
    if (doctor.running) render(h, html`The doctor is running.<br><span>It sends a few real requests, about a minute.</span>`);
    else if (!doctor.finished) render(h, html`No doctor run yet.<br><span>Run it to check relay, pool, context and tool calls.</span>`);
    else if (doctor.failed) render(h, html`${doctor.failed} doctor ${doctor.failed === 1 ? "check" : "checks"} failed.<br><span>Last run ${when(doctor.finished)}.</span>`);
    else render(h, html`Every doctor check passed.<br><span>Last run ${when(doctor.finished)}.</span>`);
  };
  const filters = () => render($("#l-filters", el), tab === "requests" ? html`<div class="k-log-filters">
    <div class="k-chips" role="group" aria-label="Kind and speed">
      <button class="k-chip" data-kind="" aria-pressed="${!f.kind && !f.slow}">All</button>
      <button class="k-chip" data-kind="chat" aria-pressed="${f.kind === "chat"}">Chat</button>
      <button class="k-chip" data-kind="video" aria-pressed="${f.kind === "video"}">Video</button>
      <button class="k-chip" data-slow="1" aria-pressed="${f.slow}">Slow over 10 s</button>
    </div>
    <div class="k-chips" role="group" aria-label="Status">${[["", "All"], ["ok", "OK"], ["error", "Errors"]].map(([k, n]) => html`<button class="k-chip" data-st="${k}" aria-pressed="${f.status === k}">${n}</button>`)}</div>
    <input class="k-field" id="l-backend" placeholder="backend, e.g. kaggle:kg3" value="${f.backend}" aria-label="Backend">
    <input class="k-field" id="l-q" placeholder="Search text" value="${f.q}" aria-label="Search"></div>` : "");
  const loadDoctor = async () => { const r = await get("/admin/api/doctor"); if (r.ok) { doctor = r.data; headline(); if (tab === "doctor") paint(); } };
  const paint = async () => {
    const box = $("#l-body", el);
    $("#l-csv", el).href = `/admin/api/logs/requests.csv?${qs()}`;
    if (tab === "doctor") {
      if (!doctor) { render(box, loading()); return; }
      fresh.mark();
      render(box, doctor.checks.length ? html`${doctor.summary ? html`<p class="k-sec">${doctor.summary}</p>` : ""}<div style="margin-top:12px">${doctor.checks.map(c => html`
        <div class="k-row"><i class="k-dot ${c.ok ? "k-ok" : "k-risk"}"></i><span class="k-t"><b>${c.name}</b><small>${c.detail}</small>${c.fix ? html`<small class="k-wait-t">${c.fix}</small>` : ""}</span></div>`)}</div>`
        : empty(doctor.running ? "Running checks." : "No results yet.", "Run doctor to check the relay, the pool, context length, tool calls and speed.", ""));
      return;
    }
    if (tab === "service") {
      const r = await get("/admin/api/logs/service?n=400");
      if (!r.ok) { render(box, errorBox(r.error)); return; }
      fresh.mark();
      render(box, html`<p class="k-note">From ${r.data.source}</p><pre class="k-code k-wrap k-log-tail">${r.data.lines.join("\n")}</pre>`);
      const pre = $(".k-log-tail", box); pre.scrollTop = pre.scrollHeight;
      return;
    }
    const r = await get(`/admin/api/logs/requests?limit=200&${qs()}`);
    if (!r.ok) { render(box, errorBox(r.error)); return; }
    fresh.mark();
    const rows = r.data.records;
    render(box, rows.length ? html`<p class="k-note">${rows.length} newest of ${num(r.data.scanned)} scanned</p><div class="k-scroll-x"><table class="k-table"><thead><tr><th>Time</th><th>Backend</th><th>Model</th><th class="k-r">In</th><th class="k-r">Cached</th><th class="k-r">Out</th><th class="k-r">First token</th><th class="k-r">Total</th><th class="k-r">Status</th></tr></thead><tbody>
      ${rows.map(x => html`<tr><td class="k-mono">${when(x.ts)}</td><td class="k-mono">${(x.backend || "").replace("kaggle:", "")}</td><td>${x.model || x.kind || ""}</td>
        <td class="k-r">${x.new_in != null ? num(x.new_in) : ""}</td><td class="k-r">${x.cached_in != null ? num(x.cached_in) : ""}</td><td class="k-r">${x.out != null ? num(x.out) : ""}</td>
        <td class="k-r">${x.ttft_s != null ? x.ttft_s + " s" : ""}</td><td class="k-r">${x.total_s != null ? x.total_s + " s" : ""}</td>
        <td class="k-r ${x.status >= 400 || x.error ? "k-risk-t" : ""}">${x.status || ""}${x.error ? " " + String(x.error).slice(0, 40) : ""}</td></tr>`)}</tbody></table></div>`
      : empty("No requests match.", f.status || f.backend || f.q || f.kind || f.slow ? "Clear the filters to see every request." : "Requests appear here as clients use the relay.", ""));
  };
  el.addEventListener("click", async e => {
    const t = e.target.closest("[data-tab]");
    if (t) { tab = t.dataset.tab; $$("[data-tab]", el).forEach(x => x.setAttribute("aria-selected", x === t)); filters(); render($("#l-body", el), loading()); paint(); }
    const s = e.target.closest("[data-st]"); if (s) { f.status = s.dataset.st; filters(); paint(); }
    const kk = e.target.closest("[data-kind]"); if (kk) { f.kind = kk.dataset.kind; if (kk.dataset.kind) f.slow = false; filters(); paint(); }
    const sl = e.target.closest("[data-slow]"); if (sl) { f.slow = !f.slow; if (f.slow) f.kind = ""; filters(); paint(); }
    if (e.target.closest("#l-doc")) { const r = await act(e.target.closest("#l-doc"), () => post("/admin/api/doctor/run", {})); if (r.ok) { tab = "doctor"; $$("[data-tab]", el).forEach(x => x.setAttribute("aria-selected", x.dataset.tab === "doctor")); filters(); loadDoctor(); } }
  });
  el.addEventListener("change", e => {
    if (e.target.id === "l-backend") { f.backend = e.target.value.trim(); paint(); }
    if (e.target.id === "l-q") { f.q = e.target.value.trim(); paint(); }
  });
  filters(); headline();
  const stopD = poll(loadDoctor, () => (doctor && doctor.running ? 3000 : 20000));
  const stopP = poll(() => (tab === "doctor" ? null : paint()), 20000);
  return () => { stopD(); stopP(); fresh.stop(); };
}
