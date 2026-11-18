// Inbox: approvals waiting for you across sessions. GET /admin/api/agents/approvals, POST /approvals/{id}.
import { get, post, poll } from "../../api.js";
import { html, render, raw, $, icon, when, act, empty, errorBox, loading, diffHtml, debounce, Fresh } from "../../ui.js";
import { stream } from "../../live.js";

export async function mount(el, ctx) {
  let show = "pending";
  render(el, html`<div class="k-page k-inbox"><span class="k-kicker">Agents · Inbox</span><h1 class="k-head" id="i-head">${""}</h1>
    <div class="k-chips" style="margin-top:26px" role="group" aria-label="Show"><button class="k-chip" data-show="pending" aria-pressed="true">Waiting</button><button class="k-chip" data-show="all" aria-pressed="false">All</button></div>
    <div id="i-list" class="k-staleable" style="margin-top:18px">${loading()}</div><p class="k-fresh" id="i-fresh"></p></div>`);
  const fresh = new Fresh($(".k-inbox", el), 20000, $("#i-fresh", el));
  const load = async () => {
    const r = await get(`/admin/api/agents/approvals?status=${show}`);
    if (!r.ok) { render($("#i-list", el), errorBox(r.error)); return; }
    fresh.mark();
    const rows = r.data.approvals, waiting = rows.filter(a => a.status === "pending").length;
    render($("#i-head", el), waiting ? html`${waiting} ${waiting === 1 ? "request waits" : "requests wait"} for you.<br><span>Agents pause until you decide.</span>` : html`Nothing waits for you.<br><span>Ask-mode sessions send requests here.</span>`);
    render($("#i-list", el), rows.length ? rows.map(a => html`<div class="k-approval ${a.status === "pending" ? "" : "k-approval-done"}">
      <div class="k-approval-top"><span class="k-tag">${a.kind}</span><b>${a.worker_id || "Lead"} in <a href="#agents/s/${a.session_id}">${a.session_title || a.session_id}</a></b><em class="k-dim">${when(a.created_at)}</em></div>
      <p>${a.summary}</p>
      ${a.options && a.options.command ? html`<pre class="k-code" style="margin-top:10px">${a.options.command}</pre>` : ""}
      ${a.diff ? html`<pre class="k-code k-diff" style="margin-top:10px">${raw(diffHtml(a.diff))}</pre>` : ""}
      ${a.status === "pending" ? html`<div class="k-acts" style="margin-top:12px"><button class="k-btn k-sm k-primary" data-decide="${a.id}" data-d="accept">Accept</button><button class="k-btn k-sm" data-decide="${a.id}" data-d="reject">Reject</button>
        <button class="k-btn k-sm" data-decide="${a.id}" data-d="lead">Let the lead decide</button>${a.kind === "tool" && !a.waiting ? html`<span class="k-note">The worker is no longer waiting on this one.</span>` : ""}</div>`
        : html`<p class="k-note">${a.status}${a.decided_by ? " by " + a.decided_by : ""}${a.decided_at ? " · " + when(a.decided_at) : ""}</p>`}</div>`)
      : empty(show === "pending" ? "All clear." : "No requests yet.", "Ask-mode sessions send permission requests, scope checks and lead questions here.", ""));
  };
  el.addEventListener("click", async e => {
    const sh = e.target.closest("[data-show]"); if (sh) { show = sh.dataset.show; el.querySelectorAll("[data-show]").forEach(b => b.setAttribute("aria-pressed", b === sh)); load(); }
    const d = e.target.closest("[data-decide]");
    if (d) { const r = await act(d, () => post(`/admin/api/agents/approvals/${d.dataset.decide}`, { decision: d.dataset.d })); if (r.ok) { load(); ctx.shared.refreshAgents(); } }
  });
  const reload = debounce(load, 300);
  const stopS = stream("/admin/api/agents/events", { onEvent: ev => { if (ev.type.startsWith("approval")) reload(); } });
  const stopP = poll(load, 20000);
  if ("Notification" in window && Notification.permission === "default") {
    render($("#i-fresh", el).parentElement.appendChild(document.createElement("div")), html`<button class="k-btn k-sm" style="margin-top:18px" data-notify>${icon("info")}Notify me on this computer</button>`);
    el.addEventListener("click", e => { if (e.target.closest("[data-notify]")) Notification.requestPermission().then(() => e.target.closest("[data-notify]").remove()); });
  }
  return () => { stopS(); stopP(); fresh.stop(); };
}
