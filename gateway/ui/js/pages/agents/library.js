// Playbooks, Schedules and Memory: GET/POST/PUT/DELETE /admin/api/agents/{playbooks,schedules,notes}.
import { get, post, put, del } from "../../api.js";
import { html, render, $, $$, icon, when, sheet, act, toast, empty, errorBox, loading } from "../../ui.js";

export async function mount(el, ctx) {
  const view = ctx.route.parts[1];
  if (view === "playbooks") return playbooks(el, ctx);
  if (view === "schedules") return schedules(el, ctx);
  return memory(el, ctx);
}
export function update() { return false; }

// Three-step explainer for empty pages: what the thing is, how it is used, what kllm enforces.
const how = steps => html`<div class="k-how">${steps.map(([t, d], i) => html`<div><i>${i + 1}</i><b>${t}</b><p>${d}</p></div>`)}</div>`;
const head = (kick, title, sub, action) => html`<span class="k-kicker">Agents · ${kick}</span><h1 class="k-head">${title}<br><span>${sub}</span></h1>${action ? html`<div class="k-acts">${action}</div>` : ""}`;

// ---- playbooks -----------------------------------------------------------------------------------------------
async function playbooks(el, ctx) {
  render(el, html`<div class="k-page k-narrow-page"><div id="p-head"></div><div id="p-list" style="margin-top:34px">${loading()}</div></div>`);
  const hs = await get("/admin/api/agents/harnesses");
  const usable = hs.ok ? hs.data.harnesses.filter(h => h.usable) : [];
  const load = async () => {
    const r = await get("/admin/api/agents/playbooks");
    if (!r.ok) { render($("#p-list", el), errorBox(r.error)); return; }
    const bs = r.data.playbooks;
    render($("#p-head", el), head("Playbooks", bs.length ? `${bs.length} ${bs.length === 1 ? "playbook" : "playbooks"}.` : "No playbooks yet.", "Saved teams, one click to start.", html`<button class="k-btn k-primary" data-new>${icon("plus")}New playbook</button>`));
    render($("#p-list", el), bs.length ? bs.map(b => html`<div class="k-row k-hov"><span class="k-glyph">${icon("page")}</span><span class="k-t"><b>${b.name}</b>
      <small>${b.lead ? `lead in ${b.lead.harness} · ` : ""}${(b.workers || []).map(w => w.harness).join(", ") || "no workers"} · ${b.permission_mode === "ask" ? "asks first" : "skips permissions"} · used ${b.uses}</small></span>
      <span class="k-row-acts"><a class="k-btn k-xs" href="#agents/new?playbook=${b.id}">Start</a><button class="k-btn k-xs" data-edit="${b.id}">Edit</button></span></div>`)
      : html`${empty("Save a setup you use often.", "For example a lead in Antigravity with two opencode workers, one on code and one on tests.", "")}${how([["Describe the team", "Pick a lead, the workers, their harnesses and models."], ["Set the rules", "Permission mode, attempts and the paths no agent may touch."], ["Start it anywhere", "From New session, the palette, or on a schedule."]])}`);
    el.onclick = e => {
      if (e.target.closest("[data-new]")) edit(null);
      const ed = e.target.closest("[data-edit]"); if (ed) edit(bs.find(b => b.id === ed.dataset.edit));
    };
  };
  const edit = b => {
    const workers = (b && b.workers) || [{ harness: (usable[0] || {}).id, objective: "" }];
    const opts = sel => usable.map(h => html`<option value="${h.id}" ${h.id === sel ? "selected" : ""}>${h.name}</option>`);
    const s = sheet({ title: b ? b.name : "New playbook", wide: true, body: html`<div class="k-form">
      <label><span>Name</span><input class="k-field" id="pb-name" value="${b ? b.name : ""}"></label>
      <label><span>Description</span><input class="k-field" id="pb-desc" value="${b ? b.description || "" : ""}"></label>
      <label><span>Default goal</span><textarea class="k-textarea" id="pb-goal" rows="2">${b ? b.goal || "" : ""}</textarea></label>
      <label><span>Folder</span><input class="k-field" id="pb-cwd" value="${b ? b.cwd || "" : ""}" placeholder="kllm's own folder when empty"></label>
      <label><span>Lead</span><select class="k-select" id="pb-lead"><option value="">No lead: workers start at once</option>${opts(b && b.lead ? b.lead.harness : "")}</select></label>
      <label><span>Permissions</span><select class="k-select" id="pb-perm"><option value="skip">Skip permissions</option><option value="ask" ${b && b.permission_mode === "ask" ? "selected" : ""}>Ask before risky changes</option></select></label>
      <div><span class="k-chart-title">Workers</span><div id="pb-workers"></div><button type="button" class="k-btn k-sm" data-addw>${icon("plus")}Add worker</button></div><div id="pb-err"></div></div>`,
      foot: html`<button class="k-btn k-primary" data-save>Save playbook</button>${b ? html`<button class="k-btn k-push" data-del>Delete</button>` : ""}` });
    const paintW = () => render($("#pb-workers", s.el), workers.map((w, i) => html`<div class="k-pb-worker"><select class="k-select" data-wh="${i}" aria-label="Harness">${opts(w.harness)}</select>
      <input class="k-field" data-wo="${i}" value="${w.objective || ""}" placeholder="objective" aria-label="Objective"><input class="k-field" data-wm="${i}" value="${(w.may_edit || []).join(", ")}" placeholder="may edit, e.g. gateway/**" aria-label="May edit">
      <button type="button" class="k-icon-btn" data-rmw="${i}" aria-label="Remove worker">${icon("x")}</button></div>`));
    paintW();
    s.el.addEventListener("input", e => {
      const i = +(e.target.dataset.wo || e.target.dataset.wm || e.target.dataset.wh || -1); if (i < 0) return;
      if (e.target.dataset.wo != null) workers[i].objective = e.target.value;
      if (e.target.dataset.wm != null) workers[i].may_edit = e.target.value.split(",").map(x => x.trim()).filter(Boolean);
      if (e.target.dataset.wh != null) workers[i].harness = e.target.value;
    });
    s.el.addEventListener("click", async e => {
      if (e.target.closest("[data-addw]")) { workers.push({ harness: (usable[0] || {}).id, objective: "" }); paintW(); }
      const rm = e.target.closest("[data-rmw]"); if (rm) { workers.splice(+rm.dataset.rmw, 1); paintW(); }
      const sv = e.target.closest("[data-save]");
      if (sv) {
        const lead = $("#pb-lead", s.el).value;
        const body = { name: $("#pb-name", s.el).value.trim(), description: $("#pb-desc", s.el).value.trim(), goal: $("#pb-goal", s.el).value.trim(), cwd: $("#pb-cwd", s.el).value.trim() || null,
          mode: "playbook", lead: lead ? { harness: lead } : null, permission_mode: $("#pb-perm", s.el).value, workers: workers.filter(w => w.harness) };
        const r = await act(sv, () => (b ? put(`/admin/api/agents/playbooks/${b.id}`, body) : post("/admin/api/agents/playbooks", body)));
        if (r.ok) { s.close(); load(); } else render($("#pb-err", s.el), errorBox(r.error));
      }
      const d = e.target.closest("[data-del]"); if (d) { const r = await act(d, () => del(`/admin/api/agents/playbooks/${b.id}`)); if (r.ok) { s.close(); load(); } }
    });
  };
  await load();
}

// ---- schedules ----------------------------------------------------------------------------------------------------
async function schedules(el) {
  render(el, html`<div class="k-page k-narrow-page"><div id="sc-head"></div><div id="sc-list" style="margin-top:34px">${loading()}</div></div>`);
  const load = async () => {
    const [r, p] = await Promise.all([get("/admin/api/agents/schedules"), get("/admin/api/agents/playbooks")]);
    if (!r.ok) { render($("#sc-list", el), errorBox(r.error)); return; }
    const books = p.ok ? p.data.playbooks : [], name = id => (books.find(b => b.id === id) || {}).name || "a deleted playbook";
    const scs = r.data.schedules, on = scs.filter(x => x.enabled).length;
    render($("#sc-head", el), head("Schedules", scs.length ? `${on} of ${scs.length} schedules are on.` : "No schedules yet.", "Playbooks on a timer, quota first.",
      books.length ? html`<button class="k-btn k-primary" data-new>${icon("plus")}New schedule</button>` : html`<a class="k-btn" href="#agents/playbooks">Create a playbook first</a>`));
    render($("#sc-list", el), scs.length ? scs.map(x => html`<div class="k-row k-hov"><button class="k-toggle" role="switch" aria-checked="${x.enabled}" data-tog="${x.id}" aria-label="Schedule on"></button>
      <span class="k-t"><b>${name(x.playbook_id)} <code class="k-dim">${x.cron}</code></b><small>${x.conditions && x.conditions.min_quota_hours != null ? `needs ${x.conditions.min_quota_hours} h of quota · ` : ""}${x.conditions && x.conditions.no_chat_activity ? "only when no chat is active · " : ""}${x.last_run ? `last ${when(x.last_run)}: ${x.last_result}` : "not run yet"}</small></span>
      <span class="k-row-acts"><button class="k-btn k-xs" data-del="${x.id}">Delete</button></span></div>`)
      : html`${empty("Nothing is scheduled.", "A schedule starts a playbook with cron timing, for example every night at 03:00.", "")}${how([["Pick a playbook", "Schedules run saved team setups, never free-form tasks."], ["Choose when", "Cron timing, such as weekdays at 02:00."], ["Quota first", "A run waits when the pool is low or someone is chatting."]])}`);
    el.onclick = async e => {
      if (e.target.closest("[data-new]")) create(books);
      const t = e.target.closest("[data-tog]"); if (t) { const r2 = await put(`/admin/api/agents/schedules/${t.dataset.tog}`, { enabled: t.getAttribute("aria-checked") !== "true" }); if (r2.ok) load(); else toast(r2.error, true); }
      const d = e.target.closest("[data-del]"); if (d) { const r2 = await act(d, () => del(`/admin/api/agents/schedules/${d.dataset.del}`)); if (r2.ok) load(); }
    };
  };
  const create = books => {
    const s = sheet({ title: "New schedule", body: html`<div class="k-form">
      <label><span>Playbook</span><select class="k-select" id="sc-pb">${books.map(b => html`<option value="${b.id}">${b.name}</option>`)}</select></label>
      <label><span>When, as cron (minute hour day month weekday)</span><input class="k-field k-mono" id="sc-cron" value="0 3 * * *"></label>
      <p class="k-note">0 3 * * * runs at 03:00 every day. */30 9-17 * * 1-5 runs every 30 min in working hours. Times are this machine's.</p>
      <label><span>Only when the pool has at least this many GPU hours left</span><input class="k-field" id="sc-q" type="number" min="0" step="1" placeholder="no minimum"></label>
      <label class="k-check"><input type="checkbox" id="sc-chat"> Only when no chat session is active</label><div id="sc-err"></div></div>`,
      foot: html`<button class="k-btn k-primary" data-save>Save schedule</button>` });
    $("[data-save]", s.el).onclick = async e => {
      const cond = {}; const q = $("#sc-q", s.el).value; if (q !== "") cond.min_quota_hours = +q; if ($("#sc-chat", s.el).checked) cond.no_chat_activity = true;
      const r = await act(e.currentTarget, () => post("/admin/api/agents/schedules", { playbook_id: $("#sc-pb", s.el).value, cron: $("#sc-cron", s.el).value.trim(), conditions: cond, enabled: true }));
      if (r.ok) { s.close(); load(); } else render($("#sc-err", s.el), errorBox(r.error));
    };
  };
  await load();
}

// ---- memory ---------------------------------------------------------------------------------------------------------
async function memory(el) {
  render(el, html`<div class="k-page k-narrow-page"><div id="me-head"></div>
    <form class="k-memory-add" id="me-form"><input class="k-field k-wide" id="me-text" placeholder="Add a note, for example: use kg accounts only; tests run with pytest -q" aria-label="New note"><button class="k-btn k-primary" type="submit">Add note</button></form>
    <div id="me-list" style="margin-top:24px">${loading()}</div></div>`);
  const load = async () => {
    const r = await get("/admin/api/agents/notes");
    if (!r.ok) { render($("#me-list", el), errorBox(r.error)); return; }
    const ns = r.data.notes, pinned = ns.filter(n => n.pinned).length;
    render($("#me-head", el), head("Memory", `${ns.length} ${ns.length === 1 ? "note" : "notes"}.`, pinned ? `${pinned} ${pinned === 1 ? "goes" : "go"} into every new prompt.` : "Pin one for every new worker."));
    render($("#me-list", el), ns.length ? ns.map(n => html`<div class="k-row k-hov"><button class="k-toggle" role="switch" aria-checked="${n.pinned}" data-pin="${n.id}" aria-label="Pinned" title="Pinned notes go into every prompt"></button>
      <span class="k-t"><b style="white-space:normal">${n.text}</b><small>${n.source ? n.source + " · " : ""}${when(n.created_at)}</small></span>
      <span class="k-row-acts"><button class="k-btn k-xs" data-del="${n.id}">Delete</button></span></div>`) : html`${empty("No notes yet.", "Notes are facts agents should know: conventions, commands, what never to touch.", "")}${how([["Write a fact", "One line each, such as: run tests with pytest -q."], ["Pin what matters", "Pinned notes go into every new worker's prompt."], ["Keep it short", "Shorter notes leave more context for the task."]])}`);
    el.onclick = async e => {
      const p = e.target.closest("[data-pin]"); if (p) { const r2 = await put(`/admin/api/agents/notes/${p.dataset.pin}`, { pinned: p.getAttribute("aria-checked") !== "true" }); if (r2.ok) load(); else toast(r2.error, true); }
      const d = e.target.closest("[data-del]"); if (d) { const r2 = await act(d, () => del(`/admin/api/agents/notes/${d.dataset.del}`)); if (r2.ok) load(); }
    };
  };
  $("#me-form", el).onsubmit = async e => {
    e.preventDefault();
    const text = $("#me-text", el).value.trim(); if (!text) return;
    const r = await act($("[type=submit]", e.target), () => post("/admin/api/agents/notes", { text, pinned: true, source: "you" }));
    if (r.ok) { $("#me-text", el).value = ""; load(); }
  };
  await load();
}
