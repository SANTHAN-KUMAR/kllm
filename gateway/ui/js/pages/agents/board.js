// Agents sessions list (#agents) and Board (#agents/board). GET /admin/api/agents/sessions, refreshed on the
// all-sessions event stream (GET /admin/api/agents/events) and every 15 s.
import { get, poll } from "../../api.js";
import { html, render, $, icon, dur, when, empty, errorBox, loading, debounce, Fresh } from "../../ui.js";
import { stream } from "../../live.js";

export const STATUS = { queued: ["Queued", "k-wait"], running: ["Running", "k-ok"], needs_you: ["Needs you", "k-wait"], waiting: ["Waiting for you", "k-wait"], done: ["Done", ""], stopped: ["Stopped", ""], failed: ["Failed", "k-risk"] };
export const statusBadge = s => html`<span class="k-badge ${STATUS[s] ? STATUS[s][1] : ""}">${STATUS[s] ? STATUS[s][0] : s}</span>`;
const COLS = [["Working", ["queued", "running"]], ["Needs you", ["needs_you", "waiting"]], ["Done", ["done"]], ["Stopped or failed", ["stopped", "failed"]]];

export async function mount(el, ctx) {
  const board = ctx.route.parts[1] === "board";
  render(el, html`<div class="k-page k-agents-list"><span class="k-kicker">Agents · ${board ? "Board" : "Sessions"}</span><h1 class="k-head" id="b-head">${""}</h1>
    <div class="k-acts"><a class="k-btn k-primary" href="#agents/new">${icon("plus")}New session</a>${board ? html`<a class="k-btn" href="#agents">List</a>` : html`<a class="k-btn" href="#agents/board">Board</a>`}</div>
    <div id="b-body" class="k-staleable" style="margin-top:36px">${loading()}</div><p class="k-fresh" id="b-fresh"></p></div>`);
  const fresh = new Fresh($(".k-agents-list", el), 30000, $("#b-fresh", el));
  const row = s => html`<a class="k-row" href="#agents/s/${s.id}"><span class="k-glyph">${icon(s.mode === "team" ? "layers" : "term")}</span>
    <span class="k-t"><b>${s.title}</b><small>${s.mode === "team" ? "team with a lead" : s.mode === "playbook" ? "playbook" : "one agent"} · ${s.workers} ${s.workers === 1 ? "worker" : "workers"} · ${s.permission_mode === "skip" ? "skips permissions" : "asks first"}${s.attempts > 1 ? ` · best of ${s.attempts}` : ""} · ${when(s.created_at)}</small></span>
    <span class="k-v">${statusBadge(s.status)}${s.approvals_pending ? html`<span class="k-wait-t k-note">${s.approvals_pending} waiting</span>` : ""}</span></a>`;
  const load = async () => {
    const r = await get("/admin/api/agents/sessions");
    if (!r.ok) { render($("#b-body", el), errorBox(r.error)); return; }
    fresh.mark();
    const ss = r.data.sessions, live = ss.filter(s => ["queued", "running", "needs_you", "waiting"].includes(s.status)), wait = ss.filter(s => s.status === "needs_you" || s.status === "waiting").length, active = ss.filter(s => s.status === "queued" || s.status === "running").length;
    render($("#b-head", el), !ss.length ? html`No sessions yet.<br><span>Describe a task to start one.</span>`
      : html`${live.length ? (active ? `${live.length} ${live.length === 1 ? "session is" : "sessions are"} running.` : `${live.length} ${live.length === 1 ? "session is" : "sessions are"} waiting for you.`) : "Nothing is running."}<br><span>${wait ? `${wait} ${wait === 1 ? "needs" : "need"} you.` : `${ss.length} sessions so far.`}</span>`);
    if (!ss.length) { render($("#b-body", el), empty("Agents work in your repo while models run on the Kaggle pool.", "Each session gets its own git worktree, and every change gets a checkpoint you can undo.", "")); return; }
    render($("#b-body", el), board ? html`<div class="k-board">${COLS.map(([name, st]) => { const items = ss.filter(s => st.includes(s.status)); return html`<section class="k-board-col"><h3>${name} <span class="k-dim">${items.length}</span></h3>
        ${items.map(s => html`<a class="k-card k-raised" href="#agents/s/${s.id}"><b>${s.title}</b><small>${s.workers} ${s.workers === 1 ? "worker" : "workers"} · ${when(s.created_at)}</small>${s.approvals_pending ? html`<small class="k-wait-t">${s.approvals_pending} approvals waiting</small>` : ""}</a>`)}
        ${!items.length ? html`<p class="k-note">None</p>` : ""}</section>`; })}</div>`
      : html`${live.length ? html`<div class="k-sect k-sect-s" style="margin-top:0"><h2>Running</h2></div>${live.map(row)}` : ""}
        <div class="k-sect k-sect-s"><h2>Earlier</h2></div>${ss.filter(s => !live.includes(s)).map(row)}`);
  };
  const reload = debounce(load, 400);
  const stopS = stream("/admin/api/agents/events", { onEvent: ev => { if (ev.type === "status" || ev.type.startsWith("approval")) reload(); } });
  const stopP = poll(load, 15000);
  return () => { stopS(); stopP(); fresh.stop(); };
}
export function update() { return false; }
