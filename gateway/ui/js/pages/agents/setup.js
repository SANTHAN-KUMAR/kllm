// Access, Harnesses, Workers and Guardrails. GET/POST /admin/api/agents/{access,guardrails}, GET /harnesses,
// POST /harnesses/{name}/test, GET /workers?running=, POST /admin/api/subagents/{id}/cancel.
import { get, post, poll } from "../../api.js";
import { html, render, $, $$, icon, when, dur, act, toast, empty, errorBox, loading, Fresh } from "../../ui.js";

export async function mount(el, ctx) {
  const v = ctx.route.parts[1];
  return ({ access, harnesses, workers, guardrails })[v](el, ctx);
}
export function update() { return false; }

async function access(el) {
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">Agents · Access</span><h1 class="k-head" id="ac-head">${""}</h1><div id="ac-body">${loading()}</div></div>`);
  const r = await get("/admin/api/agents/access");
  if (!r.ok) { render($("#ac-body", el), errorBox(r.error)); return; }
  const a = r.data.access;
  const LV = { off: "Off", read: "Read only", full: "Full" };
  const paint = () => {
    render($("#ac-head", el), html`${a.cli === "full" ? "Agents have full access." : a.cli === "read" ? "Agents can read kllm." : "Agents cannot use kllm."}<br><span>Every kllm action is logged.</span>`);
    render($("#ac-body", el), html`
      <div class="k-setting"><p>kllm commands in a worker's shell</p><div class="k-seg" role="group">${Object.entries(LV).map(([k, n]) => html`<button data-k="cli" data-v="${k}" aria-pressed="${a.cli === k}">${n}</button>`)}</div>
        <p class="k-note">Read only allows <code>kllm status</code>, <code>models</code>, <code>sessions</code>, <code>doctor</code>; Full also allows <code>kllm up</code>, <code>down</code>, <code>rotate</code>. Workers get <code>KLLM_ACCESS</code> in their environment.</p></div>
      <div class="k-setting"><p>Admin API calls marked as coming from a session</p><div class="k-seg" role="group">${Object.entries(LV).map(([k, n]) => html`<button data-k="api" data-v="${k}" aria-pressed="${a.api === k}">${n}</button>`)}</div>
        <p class="k-note">Requests with the <code>X-Kllm-Agent-Session</code> header are logged and held to this level. The admin API is open to any process on this machine, so the shell setting and the harness's own permission prompts are the hard boundary.</p></div>
      <p class="k-note">Changes apply to workers started from now on.</p>`);
  };
  el.addEventListener("click", async e => {
    const b = e.target.closest("[data-k]"); if (!b) return;
    const r2 = await act(b, () => post("/admin/api/agents/access", { [b.dataset.k]: b.dataset.v }));
    if (r2.ok) { Object.assign(a, r2.data.access); paint(); }
  });
  paint();
}

async function harnesses(el) {
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">Agents · Harnesses</span><h1 class="k-head" id="h-head">${""}</h1>
    <div class="k-acts"><button class="k-btn" id="h-refresh">Detect again</button></div><div id="h-list" style="margin-top:30px">${loading()}</div></div>`);
  const tests = {};
  const load = async (refresh) => {
    const r = await get(`/admin/api/agents/harnesses${refresh ? "?refresh=1" : ""}`);
    if (!r.ok) { render($("#h-list", el), errorBox(r.error)); return; }
    const hs = r.data.harnesses, ok = hs.filter(h => h.usable);
    render($("#h-head", el), html`${ok.length} of ${hs.length} harnesses ready.<br><span>${hs.filter(h => h.acp).length} ${hs.filter(h => h.acp).length === 1 ? "streams" : "stream"} live thinking.</span>`);
    render($("#h-list", el), hs.map(h => html`<div class="k-row k-hov k-harness"><span class="k-glyph">${icon("term")}</span>
      <span class="k-t"><b>${h.name} ${h.usable ? html`<span class="k-badge ${h.acp ? "k-ok" : ""}">${h.acp ? "ACP" : "stream-json"}</span>` : html`<span class="k-badge">not usable</span>`}</b>
        <small>${h.version || "not installed"}${h.path ? " · " + h.path : ""}</small><small>${h.usable ? `${h.ask ? "Ask and Skip" : "Skip only"} · ${h.note || ""}` : h.why_not}</small>
        ${tests[h.id] ? html`<small class="${tests[h.id].ok ? "k-ok-t" : "k-risk-t"}">${tests[h.id].text}</small>` : ""}</span>
      ${h.usable ? html`<button class="k-btn k-sm" data-test="${h.id}">Test</button>` : ""}</div>`));
  };
  el.addEventListener("click", async e => {
    if (e.target.closest("#h-refresh")) load(true);
    const t = e.target.closest("[data-test]");
    if (t) {
      const r = await act(t, () => post(`/admin/api/agents/harnesses/${t.dataset.test}/test`, {}), { quiet: true });
      tests[t.dataset.test] = r.ok ? { ok: true, text: r.data.agent ? `Handshake OK in ${r.data.secs} s: ${r.data.agent.name} ${r.data.agent.version}, ${r.data.models.length} models${r.data.load_session ? ", resumes sessions" : ""}.` : `OK: ${r.data.version}. ${r.data.note}` } : { ok: false, text: r.error };
      load();
    }
  });
  await load();
}

async function workers(el) {
  render(el, html`<div class="k-page k-workers"><span class="k-kicker">Agents · Workers</span><h1 class="k-head" id="wk-head">${""}</h1>
    <div class="k-chips" style="margin-top:26px" role="group"><button class="k-chip" data-run="1" aria-pressed="true">Running</button><button class="k-chip" data-run="0" aria-pressed="false">All</button></div>
    <div id="wk-list" class="k-staleable" style="margin-top:16px">${loading()}</div><p class="k-fresh" id="wk-fresh"></p></div>`);
  let running = 1;
  const fresh = new Fresh($(".k-workers", el), 10000, $("#wk-fresh", el));
  const load = async () => {
    const r = await get(`/admin/api/agents/workers?running=${running}`);
    if (!r.ok) { render($("#wk-list", el), errorBox(r.error)); return; }
    fresh.mark();
    const ws = r.data.workers, subs = r.data.subagents;
    render($("#wk-head", el), html`${ws.filter(w => ["starting", "working", "needs_you"].includes(w.status)).length} workers are busy.<br><span>The limit is ${r.data.limit} across sessions.</span>`);
    render($("#wk-list", el), html`${ws.length ? html`<div class="k-scroll-x"><table class="k-table"><thead><tr><th>Worker</th><th>Session</th><th>Harness</th><th>Status</th><th class="k-r">Turns</th><th class="k-r">Tools</th><th class="k-r">Context</th><th>Started</th></tr></thead><tbody>
      ${ws.map(w => html`<tr class="k-click" data-href="#agents/s/${w.session_id}" tabindex="0"><td class="k-mono">${w.id}</td><td>${w.session_title || w.session_id}</td><td>${w.harness}${w.model ? html` <span class="k-dim">${w.model}</span>` : ""}</td>
        <td class="${w.status === "needs_you" ? "k-wait-t" : w.status === "failed" ? "k-risk-t" : ""}">${w.status.replace("_", " ")}</td><td class="k-r">${w.turns}</td><td class="k-r">${w.tool_calls}</td>
        <td class="k-r">${w.context_max ? Math.round(100 * (w.context_used || 0) / w.context_max) + "%" : ""}</td><td class="k-mono">${when(w.created_at)}</td></tr>`)}</tbody></table></div>`
      : empty(running ? "No worker is running." : "No workers yet.", "Workers appear when a session starts them.", "")}
      <div class="k-sect"><h2>Subagents</h2><span>started by a lead over MCP or <code>kllm spawn</code></span></div>
      ${subs.length ? html`<div class="k-scroll-x"><table class="k-table"><thead><tr><th>Id</th><th>Harness</th><th>Status</th><th class="k-r">Time</th><th class="k-r">Files</th><th></th></tr></thead><tbody>
        ${subs.slice(0, 60).map(x => html`<tr><td class="k-mono">${x.subagent_id}</td><td>${x.harness}</td><td>${x.status}</td><td class="k-r">${dur(x.execution_time_seconds)}</td><td class="k-r">${(x.files_modified || []).length}</td>
          <td>${["PENDING", "RUNNING"].includes(x.status) ? html`<button class="k-btn k-xs" data-cancel="${x.subagent_id}">Cancel</button>` : ""}</td></tr>`)}</tbody></table></div>`
        : html`<p class="k-note">None ${running ? "running" : "recorded"}.</p>`}`);
  };
  el.addEventListener("click", async e => {
    const c = e.target.closest("[data-run]"); if (c) { running = +c.dataset.run; $$("[data-run]", el).forEach(b => b.setAttribute("aria-pressed", b === c)); load(); }
    const x = e.target.closest("[data-cancel]"); if (x) { const r = await act(x, () => post(`/admin/api/subagents/${x.dataset.cancel}/cancel`, {})); if (r.ok) load(); }
  });
  const stop = poll(load, 5000);
  return () => { stop(); fresh.stop(); };
}

async function guardrails(el) {
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">Agents · Guardrails</span><h1 class="k-head">Defaults for new sessions.<br><span>kllm enforces them anywhere.</span></h1><div id="g-body" style="margin-top:30px">${loading()}</div></div>`);
  const r = await get("/admin/api/agents/guardrails");
  if (!r.ok) { render($("#g-body", el), errorBox(r.error)); return; }
  const g = r.data.guardrails;
  const n = (k, min, max) => html`<input class="k-field k-inline k-inline-num" type="number" data-k="${k}" value="${g[k]}" min="${min}" max="${max}" aria-label="${k.replace(/_/g, " ")}">`;
  const tog = k => html`<button class="k-toggle" role="switch" data-t="${k}" aria-checked="${!!g[k]}" aria-label="${k.replace(/_/g, " ")}"></button>`;
  const line = (c, note) => html`<div class="k-setting"><p>${c}</p><span class="k-note k-saved" data-note></span>${note ? html`<p class="k-note">${note}</p>` : ""}</div>`;
  render($("#g-body", el), html`
    ${line(html`Never let agents edit these paths:`, "One pattern per line. kllm refuses permission requests that touch them and reverts any change to them after each turn.")}
    <textarea class="k-textarea k-mono" id="g-never" rows="6">${g.never_touch.join("\n")}</textarea><div class="k-acts" style="margin-top:10px"><button class="k-btn k-sm" id="g-never-save">Save paths</button></div>
    ${line(html`${tog("delete_breaker")} Refuse <code>rm -rf</code>, <code>git reset --hard</code> and force pushes in Skip mode, and undo a turn that deletes more than ${n("delete_limit", 0, 1000)} files.`)}
    ${line(html`${tog("worktree")} Give every session its own git worktree.`, "Skip mode always needs one.")}
    ${line(html`Run at most ${n("max_workers", 1, 20)} workers per session and ${n("worker_limit", 1, 50)} across all sessions.`)}
    ${line(html`Stop a worker after ${n("max_turns", 1, 1000)} turns.`)}
    ${line(html`Keep up to ${n("thought_cap_mb", 0, 100)} MB of thinking per session for replay.`)}
    ${line(html`After each best-of-N attempt, run <input class="k-field k-inline k-mono" data-k="verify_command" value="${g.verify_command || ""}" placeholder="pytest -q" aria-label="Verification command" style="width:220px">.`, "Leave empty to skip verification.")}`);
  const save = async (body, node) => {
    const note = node.closest(".k-setting") && node.closest(".k-setting").querySelector("[data-note]");
    const r2 = await post("/admin/api/agents/guardrails", body);
    if (r2.ok) { Object.assign(g, r2.data.guardrails); if (note) { note.textContent = "Saved"; setTimeout(() => { note.textContent = ""; }, 1800); } else toast("Saved."); }
    else toast(r2.error, true);
    return r2;
  };
  el.addEventListener("change", e => { const i = e.target.closest("[data-k]"); if (i) save({ [i.dataset.k]: i.dataset.k === "verify_command" ? (i.value.trim() || null) : +i.value }, i); });
  el.addEventListener("click", async e => {
    const t = e.target.closest("[data-t]"); if (t) { const on = t.getAttribute("aria-checked") !== "true"; const r2 = await save({ [t.dataset.t]: on }, t); if (r2.ok) t.setAttribute("aria-checked", on); }
    if (e.target.closest("#g-never-save")) save({ never_touch: $("#g-never", el).value.split("\n").map(x => x.trim()).filter(Boolean) }, $("#g-never", el));
  });
}
