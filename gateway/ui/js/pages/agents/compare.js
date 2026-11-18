// Compare attempts (best of N). GET /admin/api/agents/sessions/{id}/attempts, POST /attempts/{id}/choose.
import { get, post, poll } from "../../api.js";
import { html, render, $, icon, act, empty, errorBox, loading } from "../../ui.js";
import { statusBadge } from "./board.js";

export async function mount(el, ctx) {
  const sid = ctx.route.parts[2];
  if (!sid) {
    render(el, html`<div class="k-page"><span class="k-kicker">Agents · Compare</span><h1 class="k-head">Compare attempts.<br><span>Open a session that runs as attempts.</span></h1><div id="cp-body" style="margin-top:34px">${empty("Nothing to compare yet.", "Sessions started with more than one attempt can be compared side by side. Open such a session and choose “Compare attempts”.", html`<a class="k-btn k-primary" style="margin-top:18px" href="#agents">Go to sessions</a>`)}</div></div>`);
    return () => {};
  }
  render(el, html`<div class="k-page"><span class="k-kicker">Agents · Compare</span><h1 class="k-head" id="cp-head">${""}</h1><div id="cp-body" style="margin-top:34px">${loading()}</div></div>`);
  const load = async () => {
    const [s, a] = await Promise.all([get(`/admin/api/agents/sessions/${sid}`), get(`/admin/api/agents/sessions/${sid}/attempts`)]);
    if (!a.ok) { render($("#cp-body", el), errorBox(a.error)); return; }
    const rows = a.data.attempts, done = rows.filter(r => !["queued", "running", "needs_you", "waiting"].includes(r.status)).length, chosen = rows.find(r => r.chosen);
    render($("#cp-head", el), html`${s.ok ? s.data.title : "Attempts"}<br><span>${chosen ? `Attempt ${chosen.n} was chosen.` : `${done} of ${rows.length} attempts finished.`}</span>`);
    render($("#cp-body", el), html`<div class="k-attempts">${rows.map(r => html`<div class="k-attempt k-raised ${r.chosen ? "k-glow" : ""}">
      <div class="k-attempt-top"><b>Attempt ${r.n}</b>${statusBadge(r.status)}</div>
      <p class="k-note">${r.worker.harness || ""}${r.worker.model ? " · " + r.worker.model : ""} · ${r.worker.turns || 0} turns</p>
      <div class="k-stats" style="gap:24px;margin-top:14px"><div><b class="k-ok-t">+${r.added || 0}</b><small>lines added</small></div><div><b class="k-risk-t">-${r.removed || 0}</b><small>removed</small></div><div><b>${(r.files || []).length}</b><small>files</small></div></div>
      ${r.verify ? html`<p class="${r.verify.code === 0 ? "k-ok-t" : "k-risk-t"} k-note" style="margin-top:12px">${r.verify.code === 0 ? "Verification passed" : `Verification failed (${r.verify.code})`}: <code>${r.verify.command}</code></p>
        <pre class="k-code k-wrap" style="max-height:120px;margin-top:6px">${r.verify.output.split("\n").slice(-8).join("\n")}</pre>` : html`<p class="k-note" style="margin-top:12px">No verification command set.</p>`}
      <p class="k-spec" style="margin-top:10px">${(r.files || []).slice(0, 6).map(f => f.file).join(", ")}</p>
      <div class="k-acts" style="margin-top:14px"><a class="k-btn k-sm" href="#agents/s/${r.id}">Open</a>${!chosen && !["queued", "running", "needs_you", "waiting"].includes(r.status) ? html`<button class="k-btn k-sm k-primary" data-choose="${r.id}">Keep attempt ${r.n}</button>` : ""}
        ${r.chosen ? html`<a class="k-btn k-sm k-primary" href="#agents/s/${r.id}">Apply it</a>` : ""}</div></div>`)}</div>
      ${!chosen ? html`<p class="k-note" style="margin-top:18px">Keeping one attempt stops the others and removes their worktrees.</p>` : ""}`);
    el.onclick = async e => { const c = e.target.closest("[data-choose]"); if (c) { const r = await act(c, () => post(`/admin/api/agents/attempts/${c.dataset.choose}/choose`, {})); if (r.ok) load(); } };
  };
  const stop = poll(load, 8000);
  return () => stop();
}
