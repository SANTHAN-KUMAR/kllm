// One agents session: header, workers, the live timeline, the composer and the inspector.
// GET /admin/api/agents/sessions/{id}, the event stream /sessions/{id}/events (resumable), POST message/pause/resume/stop/
// permission/apply, GET changes/checkpoints, POST checkpoints/{id}/restore|fork, workers/{ref}/move|compact|attach,
// POST approvals/{id}.
import { get, post } from "../../api.js";
import { html, render, raw, $, $$, icon, esc, md, dur, clock, when, num, sheet, act, toast, copy, errorBox, loading, empty, diffHtml, pref, setPref } from "../../ui.js";
import { stream } from "../../live.js";
import { statusBadge } from "./board.js";

const LINE_TYPES = new Set(["peer", "kllm_action", "auto_applied", "guardrail", "checkpoint", "verify", "approval_resolved"]);

export async function mount(el, ctx) {
  const sid = ctx.route.parts[2];
  let s = null, events = [], live = { text: {}, open: false }, insp = null, paintQ = false, lastId = 0;
  let thinking = pref("thinking", true);
  render(el, html`<div class="k-page k-tool k-session"><div id="ss-head" class="k-ss-head"></div><div id="ss-chips" class="k-ss-chips"></div>
    <div class="k-cv-scroll k-ss-log" id="ss-log">${loading()}</div>
    <button class="k-cv-tobottom k-float" id="ss-down" aria-label="Scroll to the latest event" hidden>${icon("arrowdown")}</button>
    <div class="k-cv-dock"><form class="k-cv-comp k-raised" id="ss-form">
      <div class="k-slash k-float" id="ss-slash" role="listbox" hidden></div>
      <textarea id="ss-in" rows="1" aria-label="Message" placeholder="Message an agent. Type / for commands"></textarea>
      <div class="k-cv-tools"><select id="ss-to" aria-label="Recipient" hidden></select><span class="k-to-wrap"><button type="button" class="k-cv-to" id="ss-to-btn" aria-haspopup="listbox" aria-expanded="false"></button><div class="k-menu k-float k-to-menu" id="ss-to-menu" role="listbox" hidden></div></span>
        <button type="button" class="k-cv-tool" data-interrupt aria-pressed="false" title="Off: delivered when the agent's current turn ends. On: stops the turn and delivers now">Interrupt</button>
        <button class="k-send" type="submit" aria-label="Send" title="Send (Enter)">${icon("send")}</button></div></form></div></div>`);
  const log = $("#ss-log", el);
  const openThink = new Map();           // details key -> open, so repaints keep what you opened or closed
  log.addEventListener("toggle", e => { const d = e.target; if (d.dataset && d.dataset.key) openThink.set(d.dataset.key, d.open); }, true);
  log.addEventListener("scroll", () => { $("#ss-down", el).hidden = log.scrollHeight - log.scrollTop - log.clientHeight < 140; });
  $("#ss-down", el).onclick = () => log.scrollTo({ top: log.scrollHeight, behavior: "smooth" });
  const autosize = () => { const t = $("#ss-in", el); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 200) + "px"; t.style.overflowY = t.scrollHeight > 200 ? "auto" : "hidden"; };
  const syncTo = () => { const s0 = $("#ss-to", el), o = s0.options[s0.selectedIndex]; render($("#ss-to-btn", el), html`<span>${o ? o.textContent : "Pick an agent"}</span>${icon("down")}`); };
  const toMenu = open => {
    const m = $("#ss-to-menu", el), b = $("#ss-to-btn", el), s0 = $("#ss-to", el);
    open = open == null ? m.hidden : open;
    m.hidden = !open; b.setAttribute("aria-expanded", open);
    if (open) render(m, [...s0.options].map(o => html`<button type="button" role="option" data-pickto="${o.value}" aria-selected="${o.value === s0.value}"><b>${o.textContent}</b></button>`));
  };
  el.addEventListener("click", e => {
    if (e.target.closest("#ss-to-btn")) { toMenu(); return; }
    const pk = e.target.closest("[data-pickto]");
    if (pk) { const s0 = $("#ss-to", el); s0.value = pk.dataset.pickto; s0.dispatchEvent(new Event("change")); syncTo(); toMenu(false); $("#ss-in", el).focus(); return; }
    if (!e.target.closest("#ss-to-menu")) toMenu(false);
  });
  $("#ss-in", el).addEventListener("input", autosize);

  const loadSession = async () => {
    const r = await get(`/admin/api/agents/sessions/${sid}`);
    if (!r.ok) { render(log, r.status === 404 ? empty("This session does not exist.", "It may have been deleted.", html`<a class="k-btn k-primary" href="#agents">All sessions</a>`) : errorBox(r.error)); return false; }
    s = r.data; paintHead(); paintChips(); return true;
  };

  // ---- header ----
  const paintHead = () => {
    const running = ["queued", "running", "needs_you", "waiting"].includes(s.status);
    const paused = s.workers.some(w => w.status === "paused");
    const lead = s.workers.find(w => w.role === "lead");
    const age = (s.finished_at || Date.now() / 1000) - s.created_at;
    render($("#ss-head", el), html`<div class="k-ss-title"><h1>${s.title}</h1>
      <p><i class="k-dot ${["needs_you", "waiting"].includes(s.status) ? "k-wait" : running ? "k-ok" : s.status === "failed" ? "k-risk" : ""}"></i>${statusBadge(s.status)} <span class="k-dim">${running ? "for" : "took"} ${dur(age)}</span>
        ${lead ? html`<span class="k-dim">Lead: ${lead.harness}${lead.model ? " " + lead.model : ""}</span>` : ""}
        <button class="k-badge ${s.permission_mode === "skip" ? "k-risk" : "k-ok"}" data-perm title="Change permissions">${s.permission_mode === "skip" ? "Skips permissions" : "Asks before risky changes"}</button>
        <span id="ss-flow" class="k-ss-flow"></span></p></div>
      <div class="k-ss-actions"><button class="k-btn k-sm" data-insp="${s.counts.checkpoints > 1 ? "changes" : s.mode === "flow" ? "map" : "worker"}">Inspect${s.counts.checkpoints > 1 ? html` <span class="k-dim">${s.counts.checkpoints - 1} changes</span>` : ""}</button>
        ${running ? html`${paused ? html`<button class="k-btn k-sm" data-ctl="resume">Resume</button>` : html`<button class="k-btn k-sm" data-ctl="pause">Pause</button>`}
          <button class="k-btn k-sm k-risk" data-ctl="stop">Stop</button>` : ""}
        <button class="k-icon-btn" data-more aria-label="More" title="More">${icon("dots")}</button></div>`);
    if (s.permission_mode === "skip" && running) {
      const k = "kllm.skipNote." + s.id;
      try { if (!sessionStorage.getItem(k)) { sessionStorage.setItem(k, "1"); toast("Permissions skipped: agents act without asking. Never-touch paths and the delete breaker still apply, and every change has a checkpoint."); } } catch { /* private window */ }
    }
    const to = $("#ss-to", el), cur = to.value;
    const reachable = s.workers.filter(w => w.role !== "lead" && !["moved"].includes(w.status));
    render(to, html`${lead ? html`<option value="lead">To the lead (${lead.harness})</option>` : ""}${reachable.map(w => html`<option value="${w.id}">To ${w.id} (${w.harness}${["done", "stopped", "failed"].includes(w.status) ? ", " + w.status : ""})</option>`)}${s.workers.length > 1 ? html`<option value="all">To everyone</option>` : ""}`);
    if (cur && [...to.options].some(o => o.value === cur)) to.value = cur;
    syncTo();
    paintFlow();
  };
  // a flow session: its steps in order, with what each waits for and who runs it
  const paintFlow = () => {
    const f = s.flow && s.flow.steps;
    if (!f) { render($("#ss-flow", el), ""); return; }
    const by = Object.fromEntries(f.map(x => [x.id, x]));
    const wk = id => s.workers.find(w => w.id === id);
    render($("#ss-flow", el), html`<ol class="k-flowbar">${f.map(x => { const w = x.worker && wk(x.worker), live = w && !["done", "stopped", "failed", "moved"].includes(w.status);
      return html`<li class="k-flow-${x.status}" title="${x.objective}">
      <button class="k-flowbar-main" ${x.worker ? html`data-worker="${x.worker}"` : ""} title="${[x.objective, x.model, x.after.length ? "after " + x.after.map(a => (by[a] || {}).title || a).join(", ") : ""].filter(Boolean).join("\n")}"><i class="k-dot ${x.status === "running" ? "k-live" : x.status === "done" ? "k-ok" : x.status === "failed" ? "k-risk" : ""}"></i>
      <b>${x.title}</b><small>${x.harness}${x.worker ? " · " + x.worker : ""}${w && w.status === "working" ? ` · turn ${w.turns + 1}` : ""}</small></button>
      ${live ? html`<span class="k-flowbar-acts"><button class="k-icon-btn" data-wto="${x.worker}" title="Write to ${x.worker}" aria-label="Write to ${x.worker}">${icon("edit")}</button><button class="k-icon-btn" data-wstop="${x.worker}" title="Stop ${x.worker}" aria-label="Stop ${x.worker}">${icon("stop")}</button></span>` : ""}</li>`; })}</ol>`);
  };
  const ENDED = ["done", "stopped", "failed", "moved"];
  const paintChips = () => render($("#ss-chips", el), s.flow ? "" : s.workers.map(w => html`<div class="k-wchip ${w.status === "needs_you" ? "k-wchip-wait" : ""} ${ENDED.includes(w.status) ? "k-wchip-ended" : ""}">
    <button class="k-wchip-main" data-worker="${w.id}" title="Open ${w.id} in the inspector"><span class="k-wav ${w.role === "lead" ? "k-lead-av" : ""}">${w.id}</span>${w.harness}<span class="k-dim">${(w.contract && w.contract.objective ? w.contract.objective : "").slice(0, 28)}</span>
    <span class="${w.status === "needs_you" ? "k-wait-t" : w.status === "failed" ? "k-risk-t" : "k-dim"}">${w.status === "working" ? `turn ${w.turns + 1}` : w.status.replace("_", " ")}</span></button>
    ${!ENDED.includes(w.status) ? html`<button class="k-icon-btn" data-wto="${w.role === "lead" ? "lead" : w.id}" title="Write to ${w.id}" aria-label="Write to ${w.id}">${icon("edit")}</button>
      <button class="k-icon-btn" data-wstop="${w.id}" title="Stop ${w.id}" aria-label="Stop ${w.id}">${icon("stop")}</button>` : ""}</div>`));

  // ---- timeline model ----
  const build = () => {
    const items = [], turn = {}, startAt = {};
    const open = w => turn[w] || (turn[w] = { kind: "turn", w, tools: [], thought: "", tStart: null, tEnd: null, text: "" });
    const close = (w, at) => { const t = turn[w]; if (!t) return; t.done = true; t.at = t.at || at; items.push(t); delete turn[w]; delete live.text[w]; };
    for (const ev of events) {
      const d = ev.data, w = ev.worker;
      // a turn ends with its message, or with an error or the worker going quiet (a crash leaves no message)
      if (w && turn[w] && ((ev.type === "message" && d.from === "kllm") || (ev.type === "status" && ["idle", "paused", "stopped", "failed", "done", "moved"].includes(d.status)))) close(w, ev.at);
      switch (ev.type) {
        case "thought_chunk": { const t = open(w); if (!t.thought) t.tStart = ev.at; t.thought += d.text; t.tEnd = ev.at; break; }
        case "tool_call": open(w).tools.push({ id: d.id, title: d.title || d.kind, kind: d.kind, status: d.status, paths: d.paths, out: "" }); break;
        case "tool_update": { const t = turn[w]; const tl = t && t.tools.find(x => x.id === d.id); if (tl) tl.status = d.status; break; }
        case "tool_output": { const t = turn[w]; const tl = t && t.tools.find(x => x.id === d.id); if (tl) tl.out = d.text; break; }
        case "message":
          if (d.from === "you") items.push({ kind: "you", text: d.text, to: d.to, mode: d.mode, at: ev.at });
          else if (d.from === "kllm" || !w) items.push({ kind: "line", icon: d.error ? "x" : "info", cls: d.error ? "k-risk-t" : "", text: d.text, at: ev.at });
          else { const t = open(w); t.text = d.text; t.at = ev.at; t.done = true; items.push(t); delete turn[w]; delete live.text[w]; }
          break;
        case "plan": items.push({ kind: "plan", w, steps: d.steps, at: ev.at }); break;
        case "approval_requested": items.push({ kind: "approval", id: d.id, w, akind: d.kind, summary: d.summary, at: ev.at }); break;
        case "status":
          if (w && d.status === "starting") items.push({ kind: "line", icon: "plus", text: `${w} started in ${d.harness}${d.model ? " on " + d.model : ""}${d.objective ? ": " + d.objective.slice(0, 140) : ""}`, at: ev.at });
          else if (w && ["stopped", "failed", "moved"].includes(d.status)) items.push({ kind: "line", icon: "stop", text: `${w} ${d.status}${d.to ? " to " + d.to + " in " + d.harness : ""}`, at: ev.at });
          else if (!w && d.status === "done") items.push({ kind: "line", icon: "check", cls: "k-ok-t", text: "Session finished.", at: ev.at });
          else if (!w && d.applied) items.push({ kind: "line", icon: "branch", cls: "k-ok-t", text: d.applied.branch ? `Applied to branch ${d.applied.branch}.` : `Committed ${String(d.applied.commit).slice(0, 10)}.`, at: ev.at });
          else if (!w && d.permission_mode) items.push({ kind: "line", icon: "shield", text: d.permission_mode === "skip" ? "Permissions switched to Skip." : "Permissions switched to Ask.", at: ev.at });
          else if (!w && d.why) items.push({ kind: "line", icon: "info", text: `Session ${d.status}: ${d.why}`, at: ev.at });
          break;
        default:
          if (LINE_TYPES.has(ev.type)) items.push({ kind: "line", type: ev.type, w, d, at: ev.at });
      }
    }
    Object.values(turn).forEach(t => items.push(t));                         // turns still in progress
    Object.entries(live.text).forEach(([w, t]) => { if (t) { const x = items.find(i => i.kind === "turn" && i.w === w && !i.done) || (items.push(open(w)), open(w)); x.streaming = t; } });
    return items;
  };
  const avatar = w => html`<span class="k-wav ${w === "L" ? "k-lead-av" : ""}">${w}</span>`;
  const worker = w => s.workers.find(x => x.id === w) || {};
  const lineHtml = it => {
    if (it.text) return html`<div class="k-evline k-sys ${it.cls || ""}">${icon(it.icon || "info")}<span>${it.text}</span><em>${clock(it.at)}</em></div>`;
    const d = it.d;
    const t = {
      peer: () => d.text,
      kllm_action: () => html`${it.w === "L" ? "Lead" : it.w || "A worker"} ${d.allowed ? "ran" : "was refused"} <code>${d.action}</code>${d.via === "api" ? " on the admin API" : ""}`,
      auto_applied: () => html`${it.w} edited outside its part: ${d.files.join(", ")} ${d.undo_to ? html`<button class="k-link" data-restore="${d.undo_to}">Undo to checkpoint</button>` : ""}`,
      guardrail: () => html`<span class="k-risk-t">Guardrail: ${d.refused ? `refused ${d.refused}` : `reverted ${(d.reverted || []).join(", ")}`} (${d.why})</span>`,
      checkpoint: () => d.restored ? `Restored ${d.label}.` : `Checkpoint ${d.n}: ${(d.files || []).length} ${d.files && d.files.length === 1 ? "file" : "files"} (${(d.files || []).slice(0, 4).join(", ")}${(d.files || []).length > 4 ? ", ..." : ""})`,
      verify: () => html`Verification <code>${d.command}</code> ${d.code === 0 ? html`<span class="k-ok-t">passed</span>` : html`<span class="k-risk-t">failed (${d.code})</span>`}`,
      approval_resolved: () => `${d.by === "you" ? "You" : d.by} ${d.decision === "accept" ? "accepted" : d.decision === "reject" ? "rejected" : "handed to the lead"} a request from ${it.w || "the lead"}.`,
    }[it.type];
    return html`<div class="k-evline">${icon(it.type === "guardrail" ? "shield" : it.type === "checkpoint" ? "branch" : it.type === "kllm_action" ? "term" : "info")}<span>${t ? t() : it.type}</span><em>${clock(it.at)}</em></div>`;
  };
  const turnHtml = it => {
    const w = worker(it.w), thoughtS = it.tStart && it.tEnd ? Math.max(1, Math.round(it.tEnd - it.tStart)) : null;
    const liveTurn = !it.done;
    return html`<div class="k-amsg"><div class="k-amsg-who">${avatar(it.w)}<b>${it.w === "L" ? "Lead" : it.w}</b><span class="k-dim">${w.harness || ""}${w.model ? " · " + w.model : ""}${it.at ? " · " + clock(it.at) : ""}</span></div>
      ${it.thought ? html`<details class="k-cv-think" data-key="${it.w}-${it.tStart || 0}" ${(openThink.has(`${it.w}-${it.tStart || 0}`) ? openThink.get(`${it.w}-${it.tStart || 0}`) : liveTurn && !it.streaming) ? "open" : ""} ${thinking ? "" : "hidden"}><summary><i class="k-dot k-lead ${liveTurn ? "" : "k-still"}"></i>${liveTurn && !it.text && !it.streaming ? "Thinking" : `Thought for ${thoughtS || 1} s`}</summary><div class="k-cv-think-body">${it.thought.slice(-6000)}</div></details>` : ""}
      ${it.tools.length ? (() => { const t = it.tools[it.tools.length - 1];
        return html`<button class="k-tools1" data-worker="${it.w}" title="Open ${it.w} in the inspector"><i class="k-dot ${t.status === "completed" ? "k-ok" : t.status === "failed" ? "k-risk" : liveTurn ? "k-live" : ""}"></i><span>${t.title}</span>${it.tools.length > 1 ? html`<em>${it.tools.length} tool calls</em>` : ""}</button>`; })() : ""}
      ${it.text ? html`<div class="k-md">${raw(md(it.text))}</div>` : it.streaming ? html`<div class="k-md">${raw(md(it.streaming))}</div>` : liveTurn && !it.thought && !it.tools.length ? html`<p class="k-dim">Working</p>` : ""}</div>`;
  };
  const approvalHtml = (it, pending) => html`<div class="k-approval ${pending ? "" : "k-approval-done"}"><div class="k-approval-top"><span class="k-tag">${it.akind}</span><b>${it.w || "Lead"} asks</b><em class="k-dim">${clock(it.at)}</em></div>
    <p>${it.summary}</p>${pending ? html`<div class="k-acts" style="margin-top:12px"><button class="k-btn k-sm k-primary" data-decide="${it.id}" data-d="accept">Accept</button><button class="k-btn k-sm" data-decide="${it.id}" data-d="reject">Reject</button>
      ${s.workers.some(w => w.role === "lead") && it.w !== "L" ? html`<button class="k-btn k-sm" data-decide="${it.id}" data-d="lead">Let the lead decide</button>` : ""}${it.akind === "scope" || it.akind === "tool" ? html`<a class="k-link" href="#agents/inbox">Open in Inbox</a>` : ""}</div>` : html`<p class="k-note">Decided.</p>`}</div>`;
  const planHtml = it => html`<div class="k-plan k-raised"><ol>${it.steps.map((p, i) => html`<li class="k-plan-${p.status || "todo"}"><i>${p.status === "done" ? icon("check") : i + 1}</i>
    <span><b>${p.text}</b>${p.files && p.files.length ? html`<small class="k-mono">${p.files.join(", ")}</small>` : ""}</span>${p.owner ? html`<em>${avatar(p.owner)}${(p.status || "todo").replace("_", " ")}</em>` : ""}</li>`)}</ol></div>`;
  const paintLog = () => {
    paintQ = false;
    if (!s) return;
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 120;
    const items = build();
    const resolved = new Set(events.filter(e => e.type === "approval_resolved").map(e => e.data.id));
    const lastPlan = items.map(i => i.kind).lastIndexOf("plan");
    render(log, html`<div class="k-cv-thread k-ss-inner"><div class="k-cv-u"><div class="k-cv-bub"><div class="k-cv-text">${s.goal}</div></div></div>
      ${s.attempt_ids ? html`<div class="k-evline">${icon("layers")}<span>This task runs as ${s.attempt_ids.length} attempts in separate worktrees. <a href="#agents/compare/${s.id}">Compare attempts</a></span></div>` : ""}
      ${items.map((it, i) => it.kind === "you" ? html`<div class="k-cv-u"><div class="k-cv-bub"><div class="k-cv-text">${it.text}</div></div><small class="k-cv-to-note">${it.to && it.to !== "lead" ? "to " + it.to + " · " : ""}${it.mode === "interrupt" ? "interrupt" : "next turn"}</small></div>`
        : it.kind === "turn" ? turnHtml(it) : it.kind === "plan" ? (i === lastPlan ? planHtml(it) : html`<div class="k-evline">${icon("page")}<span>Plan updated by ${it.w}.</span><em>${clock(it.at)}</em></div>`)
        : it.kind === "approval" ? (s.permission_mode === "skip" && !resolved.has(it.id) ? html`<div class="k-evline">${icon("info")}<span>${it.w} asked: ${it.summary}</span></div>` : approvalHtml(it, !resolved.has(it.id) && ["running", "needs_you", "waiting"].includes(s.status)))
        : lineHtml(it))}
      ${s.status === "done" || s.status === "stopped" || s.status === "failed" ? finishedCard() : s.status === "waiting" && s.workers.some(w => w.role === "lead") ? html`<div class="k-evline k-sys">${icon("info")}<span>The lead is waiting for you.</span></div>` : ""}</div>`);
    if (atEnd) log.scrollTop = log.scrollHeight;
  };
  const finishedCard = () => {
    const changed = (s.counts.checkpoints || 0) > 1;          // checkpoint 0 is the start
    return html`<div class="k-finished k-raised"><b>${s.status === "done" ? "Finished" : s.status === "failed" ? "Failed" : "Stopped"}${s.finished_at ? " at " + clock(s.finished_at) : ""}.</b>
    ${s.summary ? html`<div class="k-md">${raw(md(s.summary))}</div>` : ""}
    ${!s.worktree_path ? html`<p class="k-note">The worktree is gone: the changes were applied, or another attempt was chosen.</p>`
      : changed ? html`<p class="k-sec">${s.counts.checkpoints - 1} ${s.counts.checkpoints === 2 ? "checkpoint" : "checkpoints"} with changes wait in the session's worktree.</p><div class="k-acts" style="margin-top:12px"><button class="k-btn k-primary k-sm" data-insp="changes">Review and apply</button></div>`
      : html`<p class="k-note">No files changed.</p>`}</div>`;
  };
  const schedule = () => { if (!paintQ) { paintQ = true; requestAnimationFrame(paintLog); } };

  // ---- live events ----
  let reloadT = null;
  const onEvent = ev => {
    if (ev.type === "message_chunk") { live.text[ev.worker] = (live.text[ev.worker] || "") + ev.data.text; schedule(); return; }
    if (ev.id != null) { if (ev.id <= lastId) return; lastId = ev.id; }
    events.push(ev);
    if (ev.type === "status" || ev.type === "checkpoint" || ev.type.startsWith("approval") || ev.type === "plan") {
      clearTimeout(reloadT); reloadT = setTimeout(async () => { await loadSession(); schedule(); if (insp && insp.refresh) insp.refresh(); ctx.shared.refreshAgents(); }, 250);
    }
    schedule();
  };
  if (!(await loadSession())) return;
  const stopS = stream(`/admin/api/agents/sessions/${sid}/events?after=0`, { onEvent, onState: up => { el.classList.toggle("k-stream-down", !up); } });

  // ---- composer ----
  let mode = "next_turn";
  // / commands: kllm's own (act on the session) plus the ones the recipient's harness advertised over ACP. Harness
  // commands go to that one agent as its message; with several harnesses, pick the agent first to see its commands.
  const KLLM_CMDS = [["stop", "Stop every worker", () => confirmStop()], ["pause", "Pause after the current turns", () => ctl("pause")],
    ["resume", "Resume paused workers", () => ctl("resume")], ["changes", "Review the changes", () => inspector("changes")],
    ["map", "Show who works on what", () => inspector("map")], ["trace", "Show the timeline of turns and tools", () => inspector("trace")]];
  const recipient = () => { const v = $("#ss-to", el).value; return v === "lead" ? s.workers.find(w => w.role === "lead") : s.workers.find(w => w.id === v); };
  const slashItems = q => {
    const w = recipient(), hc = w ? (w.command_info && w.command_info.length ? w.command_info : (w.commands || []).map(n => ({ name: n, description: "" }))) : [];
    const k = KLLM_CMDS.filter(c => c[0].startsWith(q)).map(c => ({ name: c[0], description: c[1], kllm: true }));
    const h = hc.filter(c => c.name.startsWith(q)).map(c => ({ name: c.name, description: c.description || `${w.harness} command`, who: w.id }));
    return [...k, ...h];
  };
  let slashSel = 0;
  const paintSlash = () => {
    const v = $("#ss-in", el).value, box = $("#ss-slash", el);
    const m = v.match(/^\/(\S*)$/);
    const items = m ? slashItems(m[1]) : [];
    box.hidden = !items.length;
    if (!items.length) return;
    slashSel = Math.min(slashSel, items.length - 1);
    const w = recipient();
    render(box, html`${items.some(i => i.kllm) ? html`<span>kllm</span>` : ""}${items.filter(i => i.kllm).map(i => html`<button type="button" role="option" data-slash="${i.name}" aria-selected="${items.indexOf(i) === slashSel}"><b>/${i.name}</b><small>${i.description}</small></button>`)}
      ${items.some(i => !i.kllm) ? html`<span>${w ? `${w.id} in ${w.harness}` : "Agent"}</span>` : ""}${items.filter(i => !i.kllm).map(i => html`<button type="button" role="option" data-slash="${i.name}" data-harness="1" aria-selected="${items.indexOf(i) === slashSel}"><b>/${i.name}</b><small>${i.description}</small></button>`)}
      ${!items.some(i => !i.kllm) && w ? html`<p class="k-note">${w.harness} has not listed commands${w.driver === "acp" ? " yet" : " (it runs without ACP)"}.</p>` : ""}`);
  };
  const runSlash = (name, harness) => {
    const c = !harness && KLLM_CMDS.find(x => x[0] === name);
    $("#ss-slash", el).hidden = true;
    if (c) { $("#ss-in", el).value = ""; autosize(); c[2](); return; }
    $("#ss-in", el).value = "/" + name + " "; autosize(); $("#ss-in", el).focus();
  };
  const ctl = async what => { const r = await post(`/admin/api/agents/sessions/${sid}/${what}`, {}); if (r.ok) loadSession(); else toast(r.error, true); };
  $("#ss-in", el).addEventListener("input", () => { slashSel = 0; paintSlash(); });
  $("#ss-to", el).addEventListener("change", paintSlash);
  const send = async () => {
    const text = $("#ss-in", el).value.trim();
    if (!text) return;
    const kc = text.match(/^\/(\w+)$/) && KLLM_CMDS.find(x => x[0] === text.slice(1));
    if (kc) { runSlash(kc[0], false); return; }
    const r = await act($("[type=submit]", $("#ss-form", el)), () => post(`/admin/api/agents/sessions/${sid}/message`, { to: $("#ss-to", el).value, text, mode }), { quiet: true });
    if (r.ok) { $("#ss-in", el).value = ""; autosize(); }
  };
  $("#ss-form", el).onsubmit = e => { e.preventDefault(); send(); };
  $("#ss-in", el).addEventListener("keydown", e => {
    const box = $("#ss-slash", el);
    if (!box.hidden) {
      const opts = $$("[data-slash]", box);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); slashSel = (slashSel + (e.key === "ArrowDown" ? 1 : -1) + opts.length) % opts.length; paintSlash(); return; }
      if ((e.key === "Enter" || e.key === "Tab") && opts[slashSel]) { e.preventDefault(); runSlash(opts[slashSel].dataset.slash, !!opts[slashSel].dataset.harness); return; }
      if (e.key === "Escape") { box.hidden = true; return; }
    }
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  // ---- clicks ----
  el.addEventListener("click", async e => {
    const it = e.target.closest("[data-interrupt]"); if (it) { mode = mode === "interrupt" ? "next_turn" : "interrupt"; it.setAttribute("aria-pressed", mode === "interrupt"); }
    const th = e.target.closest("[data-think]"); if (th) { thinking = !thinking; setPref("thinking", thinking); ctx.store.thinking = thinking; th.setAttribute("aria-checked", thinking); paintLog(); }
    const c = e.target.closest("[data-ctl]");
    if (c) {
      if (c.dataset.ctl === "stop") return confirmStop();
      const r = await act(c, () => post(`/admin/api/agents/sessions/${sid}/${c.dataset.ctl}`, {})); if (r.ok) loadSession();
    }
    const d = e.target.closest("[data-decide]");
    if (d) { const r = await act(d, () => post(`/admin/api/agents/approvals/${d.dataset.decide}`, { decision: d.dataset.d })); if (r.ok) loadSession(); }
    const rs = e.target.closest("[data-restore]"); if (rs) restore(rs.dataset.restore);
    if (e.target.closest("[data-perm]")) permSheet();
    const i = e.target.closest("[data-insp]"); if (i) inspector(i.dataset.insp);
    const w = e.target.closest("[data-worker]"); if (w) inspector("worker", w.dataset.worker);
    if (e.target.closest("[data-more]")) moreSheet();
    const sl = e.target.closest("[data-slash]"); if (sl) runSlash(sl.dataset.slash, !!sl.dataset.harness);
    const wt = e.target.closest("[data-wto]"); if (wt) { $("#ss-to", el).value = wt.dataset.wto; syncTo(); paintSlash(); $("#ss-in", el).focus(); }
    const ws = e.target.closest("[data-wstop]");
    if (ws) { const r = await act(ws, () => post(`/admin/api/agents/workers/${sid}:${ws.dataset.wstop}/stop`, {}), { quiet: true }); if (r.ok) { toast(`Stopped ${ws.dataset.wstop}.`); loadSession(); } }
  });

  const confirmStop = () => {
    const sh = sheet({ title: "Stop this session?", sub: "Every worker stops now. Changes stay in the worktree with their checkpoints.", body: html`<p class="k-sec">Waiting approvals are rejected. You can still apply what is there.</p>`,
      foot: html`<button class="k-btn k-risk" data-yes>Stop every worker</button><button class="k-btn k-push" data-close>Keep working</button>` });
    $("[data-yes]", sh.el).onclick = async e => { const r = await act(e.currentTarget, () => post(`/admin/api/agents/sessions/${sid}/stop`, {})); if (r.ok) { sh.close(); loadSession(); } };
  };
  const permSheet = () => {
    const to = s.permission_mode === "skip" ? "ask" : "skip";
    const sh = sheet({ title: to === "skip" ? "Skip permissions?" : "Ask before risky changes?",
      sub: to === "skip" ? "Agents act without asking. Waiting tool requests are approved now. kllm still refuses never-touch paths and the delete breaker." : "Permission requests from the harness wait in your Inbox from the next request on.",
      body: html`<p class="k-sec">${to === "skip" ? "Every change keeps a checkpoint you can undo." : "Workers that run without ACP cannot ask; stop them first."}</p>`,
      foot: html`<button class="k-btn ${to === "skip" ? "k-risk" : "k-primary"}" data-yes>${to === "skip" ? "Skip permissions" : "Ask first"}</button><button class="k-btn k-push" data-close>Cancel</button>` });
    $("[data-yes]", sh.el).onclick = async e => { const r = await act(e.currentTarget, () => post(`/admin/api/agents/sessions/${sid}/permission`, { mode: to })); if (r.ok) { sh.close(); loadSession(); } };
  };
  const moreSheet = () => {
    const done = !["queued", "running", "needs_you", "waiting"].includes(s.status);
    const sh = sheet({ title: "Session", sub: s.id, body: html`<div class="k-kv"><div><span>Folder</span><span class="k-mono">${s.cwd}</span></div>
      <div><span>Worktree</span><span class="k-mono">${s.worktree_path || "none"}</span></div><div><span>Base commit</span><span class="k-mono">${(s.base_commit || "").slice(0, 12)}</span></div>
      <div><span>Started</span><span>${when(s.created_at)}</span></div><div><span>Events</span><span>${num(s.counts.events)}</span></div></div>
      <div class="k-cv-prow" style="margin-top:18px"><span><b>Show thinking</b><small>The agents' reasoning in the timeline</small></span><button type="button" class="k-toggle" data-think role="switch" aria-checked="${thinking}" aria-label="Show thinking"></button></div>
      <div id="mo-confirm"></div>`,
      foot: html`<a class="k-btn k-sm" href="/admin/api/agents/sessions/${sid}/export" download>Export transcript</a>${done ? html`<button class="k-btn k-sm k-push" data-del>Delete session</button>` : ""}` });
    const del = $("[data-del]", sh.el);
    if (del) del.onclick = () => {
      render($("#mo-confirm", sh.el), html`<div class="k-confirm k-raised" style="margin-top:16px"><p class="k-sec">This removes the session, its worktree and its checkpoints. Applied branches stay.</p>
        <div class="k-acts" style="margin-top:10px"><button class="k-btn k-sm k-risk" data-yes>Delete session</button></div></div>`);
      $("[data-yes]", sh.el).onclick = async e => { const r = await act(e.currentTarget, () => post(`/admin/api/agents/sessions/${sid}/delete`, {})); if (r.ok) { sh.close(); ctx.shared.refreshAgents(); ctx.go("#agents"); } };
    };
  };
  const restore = async cid => {
    const r = await post(`/admin/api/agents/checkpoints/${cid}/restore`, {});
    if (r.ok) toast("Restored. The worktree matches that checkpoint."); else toast(r.error, true);
    if (insp && insp.refresh) insp.refresh();
  };

  // ---- inspector ----
  const inspector = (tab, wid) => {
    if (insp) insp.close();
    const sh = sheet({ title: "Inspector", body: loading(), foot: false, scrim: false,
      onClose: () => { if (insp === me) { insp = null; el.firstElementChild.classList.remove("k-insp-open"); } } });
    sh.el.classList.add("k-insp");
    const room = () => el.firstElementChild.classList.add("k-insp-open");
    room();
    const me = { close: () => sh.close(), refresh: () => paintTab() };
    insp = me;
    const TABS = [["worker", "Worker"], ["map", s.mode === "flow" ? "Flow" : "Map"], ["changes", "Changes"], ["trace", "Trace"], ["checkpoints", "Saves"]];
    sh.head.insertAdjacentHTML("beforeend", `<div class="k-insp-tabs" role="tablist">${TABS.map(([t, n]) => `<button role="tab" data-tab="${t}" aria-selected="${t === tab}">${n}</button>`).join("")}</div>`);
    let selW = wid || (s.workers.find(w => w.status === "working") || s.workers[0] || {}).id, file = null;
    sh.el.addEventListener("click", async e => {
      const t = e.target.closest("[data-tab]"); if (t) { tab = t.dataset.tab; $$("[data-tab]", sh.el).forEach(x => x.setAttribute(x.closest(".k-insp-tabs") ? "aria-selected" : "aria-pressed", x === t)); paintTab(); }
      const pw = e.target.closest("[data-pickw]"); if (pw) { selW = pw.dataset.pickw; paintTab(); }
      const f = e.target.closest("[data-file]"); if (f) { file = f.dataset.file === file ? null : f.dataset.file; paintTab(); }
      const a = e.target.closest("[data-apply]");
      if (a) {
        const body = a.dataset.apply === "commit" ? { target: "commit", message: $("#ap-msg", sh.el).value.trim() } : { target: "new_branch", name: $("#ap-branch", sh.el).value.trim(), message: $("#ap-msg", sh.el).value.trim() };
        const r = await act(a, () => post(`/admin/api/agents/sessions/${sid}/apply`, body));
        if (r.ok) {
          await loadSession();
          const d = r.data;
          render(sh.body, html`<div class="k-empty"><b>${d.branch ? `Applied to ${d.branch}.` : "Committed on the current branch."}</b>
            <p>Commit <code>${String(d.commit).slice(0, 12)}</code>. The worktree is removed; checkpoints stay until you delete the session.</p>
            ${d.branch ? html`<div class="k-copy" style="margin-top:12px"><code>git switch ${d.branch}</code><button class="k-btn k-xs" data-copy="git switch ${d.branch}">Copy</button></div>` : ""}</div>`);
        }
      }
      const rc = e.target.closest("[data-rc]"); if (rc) { await restore(rc.dataset.rc); }
      const fk = e.target.closest("[data-fork]"); if (fk) { const r = await act(fk, () => post(`/admin/api/agents/checkpoints/${fk.dataset.fork}/fork`, {})); if (r.ok) { ctx.shared.refreshAgents(); ctx.go(`#agents/s/${r.data.id}`); } }
      const wa = e.target.closest("[data-wact]");
      if (wa) {
        const ref = `${sid}:${selW}`, k = wa.dataset.wact;
        if (k === "attach") { const r = await act(wa, () => post(`/admin/api/agents/workers/${ref}/attach`, {}), { quiet: true }); if (r.ok) render($("#w-attach", sh.el), html`<div class="k-copy" style="margin-top:10px"><code>${r.data.command}</code><button class="k-btn k-xs" data-copy="${r.data.command}">Copy</button></div><p class="k-note">${r.data.note}</p>`); }
        if (k === "compact") await act(wa, () => post(`/admin/api/agents/workers/${ref}/compact`, {}));
        if (k === "move") { const r = await act(wa, () => post(`/admin/api/agents/workers/${ref}/move`, { harness: $("#w-move", sh.el).value })); if (r.ok) { await loadSession(); selW = r.data.to; paintTab(); } }
      }
      const cp = e.target.closest("[data-copy]"); if (cp) copy(cp.dataset.copy, cp);
    });
    const paintTab = async () => {
      if (!sh.el.isConnected) return;
      const box = sh.body;
      if (tab === "worker") {
        const w = s.workers.find(x => x.id === selW);
        if (!w) { render(box, empty("No workers yet.", "Workers appear when the session starts them.", "")); return; }
        const h = await get("/admin/api/agents/harnesses");
        const others = h.ok ? h.data.harnesses.filter(x => x.usable && x.id !== w.harness) : [];
        const c = w.contract || {};
        render(box, html`<div class="k-chips" style="margin-bottom:16px">${s.workers.map(x => html`<button class="k-chip" data-pickw="${x.id}" aria-pressed="${x.id === selW}">${x.id}</button>`)}</div>
          <h3 class="k-sub-title">${w.id === "L" ? "Lead" : w.id} · ${w.harness}${w.model ? " · " + w.model : ""}</h3><p class="k-note">${w.driver === "acp" ? "Agent Client Protocol: live thinking, tools and permission requests" : "stream-json: one process per turn, permissions skipped"}</p>
          <div class="k-kv" style="margin-top:12px"><div><span>Status</span><span>${w.status.replace("_", " ")}</span></div><div><span>Turns</span><span>${w.turns}</span></div><div><span>Tool calls</span><span>${w.tool_calls}</span></div>
            ${w.context_max ? html`<div><span>Context</span><span>${num(w.context_used || 0)} of ${num(w.context_max)}</span></div>` : ""}${w.tokens_out ? html`<div><span>Tokens</span><span>${num(w.tokens_in)} in, ${num(w.tokens_out)} out</span></div>` : ""}
            ${w.queued ? html`<div><span>Messages queued</span><span>${w.queued}</span></div>` : ""}</div>
          ${c.objective ? html`<p class="k-chart-title" style="margin-top:16px">Objective</p><p class="k-sec">${c.objective}</p>` : ""}
          ${(c.may_edit || []).length ? html`<p class="k-note" style="margin-top:8px">May edit: ${c.may_edit.join(", ")}</p>` : ""}${(c.must_not_touch || []).length ? html`<p class="k-note">Must not touch: ${c.must_not_touch.join(", ")}</p>` : ""}
          ${w.last_text ? html`<p class="k-chart-title" style="margin-top:16px">Last message</p><div class="k-md k-sec">${raw(md(w.last_text))}</div>` : ""}
          ${w.error ? html`<div class="k-error-box" style="margin-top:12px">${w.error}</div>` : ""}
          <div class="k-acts"><button class="k-btn k-sm" data-wact="attach">Attach in a terminal</button>${(w.commands || []).includes("compact") ? html`<button class="k-btn k-sm" data-wact="compact">Compact context</button>` : ""}</div><div id="w-attach"></div>
          ${others.length && w.status !== "moved" ? html`<div class="k-acts" style="margin-top:14px"><select class="k-select" id="w-move" aria-label="Harness">${others.map(o => html`<option value="${o.id}">${o.name}</option>`)}</select><button class="k-btn k-sm" data-wact="move">Move to this harness</button></div>
            <p class="k-note">Moving stops ${w.id} and starts a new worker with its last message and the changes so far.</p>` : ""}`);
      } else if (tab === "map" && s.flow) {
        const f = s.flow.steps, by = Object.fromEntries(f.map(x => [x.id, x]));
        render(box, html`<ol class="k-flowlist">${f.map((x, i) => html`<li class="k-flow-${x.status}">
          <div class="k-flowlist-top"><span class="k-glyph">${i + 1}</span><b>${x.title}</b><span class="k-badge ${x.status === "done" ? "k-ok" : x.status === "failed" ? "k-risk" : x.status === "running" ? "k-live" : ""}">${x.status}</span></div>
          <p class="k-note">${x.harness}${x.model ? " · " + x.model : ""}${x.worker ? " · " + x.worker : ""}${x.after.length ? " · after " + x.after.map(a => (by[a] || {}).title || a).join(", ") : " · starts first"}</p>
          ${x.summary ? html`<div class="k-md k-sec">${raw(md(String(x.summary).slice(0, 1200)))}</div>` : ""}
          ${x.worker ? html`<button class="k-link" data-pickw="${x.worker}" data-tab="worker">Open ${x.worker}</button>` : ""}</li>`)}</ol>`);
      } else if (tab === "map") {
        const lead = s.workers.find(w => w.role === "lead"), ws = s.workers.filter(w => w.role !== "lead");
        render(box, html`<div class="k-amap">${lead ? html`<div class="k-amap-lead k-raised"><b>${avatar("L")} Lead</b><small>${lead.harness} · ${lead.status.replace("_", " ")}</small></div>` : html`<div class="k-amap-lead k-raised"><b>You</b><small>${s.mode === "one" ? "one agent" : "playbook"}</small></div>`}
          <div class="k-amap-workers">${ws.map(w => html`<button class="k-amap-w k-raised ${w.status === "working" ? "k-amap-live" : ""} ${w.status === "needs_you" ? "k-amap-wait" : ""}" data-pickw="${w.id}" data-tab="worker">
            <b>${avatar(w.id)} ${w.harness}</b><small>${(w.contract && w.contract.objective || "").slice(0, 60)}</small><small class="k-dim">${w.status.replace("_", " ")} · ${w.turns} turns</small></button>`)}
          ${!ws.length ? html`<p class="k-note">No workers besides the lead yet.</p>` : ""}</div></div>
          <p class="k-note" style="margin-top:14px">${events.filter(e => e.type === "peer").length} messages between agents so far.</p>`);
      } else if (tab === "changes") {
        render(box, loading());
        const r = await get(`/admin/api/agents/sessions/${sid}/changes${file ? "?file=" + encodeURIComponent(file) : ""}`);
        if (!r.ok) { render(box, r.status === 409 || r.status === 410 ? empty("No worktree.", r.error, "") : errorBox(r.error)); return; }
        if (file) { render(box, html`<button class="k-link" data-file="${file}">All files</button><pre class="k-code k-diff" style="margin-top:12px">${raw(diffHtml(r.data.diff))}</pre>`); return; }
        const fs = r.data.files, running = ["running", "queued"].includes(s.status);
        render(box, fs.length ? html`${r.data.conflicts.length ? html`<div class="k-banner k-warn"><i class="k-dot"></i><span>More than one worker edited ${r.data.conflicts.join(", ")}.</span></div>` : ""}
          <table class="k-table" style="margin-top:8px"><thead><tr><th>File</th><th>By</th><th class="k-r">Added</th><th class="k-r">Removed</th></tr></thead><tbody>${fs.map(f => html`<tr class="k-click" data-file="${f.file}"><td class="k-mono">${f.file}</td><td>${f.owners.join(", ")}</td><td class="k-r k-ok-t">${f.added == null ? "binary" : "+" + f.added}</td><td class="k-r k-risk-t">${f.removed == null ? "" : "-" + f.removed}</td></tr>`)}</tbody></table>
          <div class="k-apply k-raised"><b>Apply ${fs.length} ${fs.length === 1 ? "file" : "files"}</b>${running ? html`<p class="k-note k-wait-t">A worker is still running: pause or stop the session first.</p>` : ""}
            <div class="k-form" style="margin-top:10px"><label><span>Branch</span><input class="k-field" id="ap-branch" value="kllm/${s.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 32)}"></label>
            <label><span>Commit message</span><input class="k-field" id="ap-msg" value="${s.title}"></label></div>
            <div class="k-acts" style="margin-top:12px"><button class="k-btn k-primary k-sm" data-apply="new_branch">Apply to the new branch</button><button class="k-btn k-sm" data-apply="commit">Commit on the current branch</button></div>
            <p class="k-note">Applying removes the worktree. The checkpoints stay until you delete the session.</p></div>`
          : empty("No changes yet.", "Files the workers change show up here.", ""));
      } else if (tab === "trace") {
        render(box, traceSvg());
      } else {
        const r = await get(`/admin/api/agents/sessions/${sid}/checkpoints`);
        if (!r.ok) { render(box, errorBox(r.error)); return; }
        const cps = r.data.checkpoints.slice().reverse();
        render(box, cps.length ? html`<p class="k-note">Before each change kllm saves the worktree. Restore puts it back in place; Fork starts a new session from it.</p>
          ${cps.map(c => html`<div class="k-row k-hov"><span class="k-glyph">${c.n}</span><span class="k-t"><b>${c.label}</b><small>${clock(c.at)} · ${(c.files || []).length} files · <span class="k-mono">${(c.commit_sha || "").slice(0, 10)}</span></small></span>
            <span class="k-row-acts" style="opacity:1">${s.worktree_path ? html`<button class="k-btn k-xs" data-rc="${c.id}">Restore</button>` : ""}<button class="k-btn k-xs" data-fork="${c.id}">Fork</button></span></div>`)}`
          : empty("No checkpoints.", "This session runs without a worktree.", ""));
      }
    };
    paintTab();
  };

  // Trace waterfall: one row per agent; turns (working to idle), tools (call to completion), waiting for you.
  const traceSvg = () => {
    if (!events.length) return empty("Nothing to trace yet.", "", "");
    const t0 = s.created_at, t1 = Math.max(events[events.length - 1].at, s.finished_at || 0, Date.now() / 1000 * (["running", "needs_you", "waiting"].includes(s.status) ? 1 : 0)) || t0 + 1;
    const rows = s.workers.map(w => w.id), W = 500, rowH = 26, left = 40, span = Math.max(1, t1 - t0);
    const x = t => left + ((t - t0) / span) * (W - left);
    const spans = [];
    const open = {}, tools = {};
    events.forEach(ev => {
      const w = ev.worker; if (!w) return;
      if (ev.type === "status") {
        if (ev.data.status === "working") open[w] = { t: ev.at, k: "turn" };
        else if (open[w]) { spans.push({ w, a: open[w].t, b: ev.at, k: open[w].k }); open[w] = ev.data.status === "needs_you" ? { t: ev.at, k: "wait" } : null; }
      }
      if (ev.type === "tool_call") tools[ev.data.id] = { w, a: ev.at };
      if (ev.type === "tool_update" && tools[ev.data.id] && ["completed", "failed"].includes(ev.data.status)) { spans.push({ ...tools[ev.data.id], b: ev.at, k: "tool" }); delete tools[ev.data.id]; }
    });
    Object.entries(open).forEach(([w, o]) => o && spans.push({ w, a: o.t, b: t1, k: o.k }));
    const msgs = events.filter(e => e.type === "peer" || (e.type === "message" && e.data.from === "you"));
    const H = rows.length * rowH + 30;
    return html`<svg class="k-chart k-trace" viewBox="0 0 ${W} ${H}" role="img" aria-label="Trace: when each agent worked, ran tools and waited">
      ${raw(rows.map((r, i) => `<text class="k-axis" x="0" y="${i * rowH + 17}">${esc(r)}</text><line class="k-grid" x1="${left}" x2="${W}" y1="${i * rowH + rowH}" y2="${i * rowH + rowH}"/>`).join(""))}
      ${raw(spans.map(sp => { const i = rows.indexOf(sp.w); if (i < 0) return ""; const y = i * rowH + (sp.k === "tool" ? 13 : 6), h = sp.k === "tool" ? 6 : 14;
        const fill = sp.k === "tool" ? "rgb(var(--k-lead))" : sp.k === "wait" ? "rgb(var(--k-wait))" : "var(--k-text)"; const op = sp.k === "wait" ? 0.3 : sp.k === "tool" ? 0.9 : 0.25;
        return `<rect x="${x(sp.a).toFixed(1)}" y="${y}" width="${Math.max(1.5, x(sp.b) - x(sp.a)).toFixed(1)}" height="${h}" rx="2" fill="${fill}" fill-opacity="${op}"/>`; }).join(""))}
      ${raw(msgs.map(m => `<line x1="${x(m.at).toFixed(1)}" x2="${x(m.at).toFixed(1)}" y1="0" y2="${rows.length * rowH}" stroke="var(--k-text-3)" stroke-dasharray="2 3"/>`).join(""))}
      <text class="k-axis" x="${left}" y="${H - 4}">${clock(t0, true)}</text><text class="k-axis" x="${W}" y="${H - 4}" text-anchor="end">${clock(t1, true)}</text></svg>
      <div class="k-legend"><span><i style="background:var(--k-text);opacity:.4"></i>model turn</span><span><i style="background:rgb(var(--k-lead))"></i>tool</span><span><i style="background:rgb(var(--k-wait));opacity:.5"></i>waiting for you</span><span>dotted: a message</span></div>`;
  };

  const tick = setInterval(() => { if (s && ["queued", "running", "needs_you", "waiting"].includes(s.status)) paintHead(); }, 30000);
  return () => { stopS(); clearInterval(tick); clearTimeout(reloadT); if (insp) insp.close(); };
}
export function update() { return false; }
