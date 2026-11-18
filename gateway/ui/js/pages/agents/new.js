// New session. GET /admin/api/agents/harnesses, /guardrails, /playbooks, /admin/api/config (repo root), /admin/api/state
// (kllm chat models for the model fields); POST /admin/api/agents/sessions, /admin/api/agents/playbooks (save a flow).
// Workflow mode builds a graph of steps: each step picks its own harness and model and waits for the steps it needs.
import { get, post } from "../../api.js";
import { html, raw, render, $, $$, icon, act, errorBox, loading, pref, setPref, toast } from "../../ui.js";

// a starting point people can edit: plan with one harness, build on the pool, review with another
const TEMPLATE = (h, pool) => [
  { id: "plan", title: "Plan", harness: h[0], model: "", after: [], objective: "Read the code involved and write PLAN.md: the files to change, the steps, and how to check the result. Do not edit code." },
  { id: "build", title: "Build", harness: h.includes("opencode") ? "opencode" : h[0], model: pool ? "kllm/" + pool : "", after: ["plan"], objective: "Follow PLAN.md from the plan step. Make the change and run the checks it names." },
  { id: "review", title: "Review", harness: h[1] || h[0], model: "", after: ["build"], objective: "Review the diff against PLAN.md. Fix real bugs you find; list anything left in REVIEW.md." },
];

export async function mount(el, ctx) {
  const [h, g, p, c, stt] = await Promise.all([get("/admin/api/agents/harnesses"), get("/admin/api/agents/guardrails"), get("/admin/api/agents/playbooks"), get("/admin/api/config"), get("/admin/api/state")]);
  if (!h.ok) { render(el, html`<div class="k-page">${errorBox(h.error)}</div>`); return; }
  const usable = h.data.harnesses.filter(x => x.usable);
  const st = { mode: pref("agMode", "one"), harness: pref("agHarness", (usable[0] || {}).id), perm: "skip", attempts: 1, playbook: ctx.route.query.playbook || null };
  if (!usable.find(x => x.id === st.harness)) st.harness = (usable[0] || {}).id;
  const gr = g.ok ? g.data.guardrails : null;
  const books = p.ok ? p.data.playbooks : [];
  if (st.playbook) st.mode = "playbook";
  const root = c.ok ? c.data.root : "";
  const chat = stt.ok ? stt.data.models.filter(m => m.kind === "chat").map(m => m.key) : [];
  const serving = stt.ok ? [...new Set(stt.data.pool.filter(x => x.healthy).map(x => x.model_key))] : [];
  let steps = pref("agFlow", null) || TEMPLATE(usable.map(x => x.id), serving[0]);
  // each harness's own models (asked from the harness), grouped by provider; "Other" takes any id
  const MODELS = {};
  const loadModels = async h => {
    if (!h || MODELS[h]) return;
    MODELS[h] = "loading";
    const r = await get(`/admin/api/agents/harnesses/${h}/models`);
    MODELS[h] = r.ok ? r.data.models : [];
    paint();
  };
  const modelSel = (h, val, attrs, custom) => {
    // "Other model id" turns the picker into a text field (no browser dialogs in the panel)
    if (custom) return html`<input class="k-field k-model-sel" ${attrs} value="${val || ""}" placeholder="model id, as ${h} names it" autocomplete="off">`;
    const list = MODELS[h];
    if (!Array.isArray(list)) { loadModels(h); return html`<select class="k-select k-model-sel" ${attrs} disabled><option>Loading models</option></select>`; }
    const groups = [...new Set(list.map(m => m.group))];
    const known = !val || list.some(m => m.id === val);
    return html`<select class="k-select k-model-sel" ${attrs}>
      <option value="" ${!val ? "selected" : ""}>Harness default</option>
      ${groups.map(g => html`<optgroup label="${g === "kllm pool" ? "kllm pool (Kaggle)" : g}">${list.filter(m => m.group === g).map(m => html`<option value="${m.id}" ${m.id === val ? "selected" : ""}>${m.label}${g === "kllm pool" && serving.includes(m.label) ? " · serving" : ""}</option>`)}</optgroup>`)}
      ${!known ? html`<option value="${val}" selected>${val}</option>` : ""}
      <option value="__other">Other model id</option></select>`;
  };
  render(el, html`<div class="k-page k-narrow-page k-new"><span class="k-kicker">Agents work on your machine, models run where the harness sends them</span>
    <h1 class="k-head">What should the agents do?</h1>
    <form id="n-form" class="k-new-box k-glow">
      <textarea id="n-goal" rows="3" aria-label="Goal" placeholder="Describe the task, the files involved and what done looks like."></textarea>
      <div class="k-new-opts">
        <div class="k-seg" role="group" aria-label="Mode">${[["one", "One agent"], ["team", "Team with a lead"], ["flow", "Workflow"], ["playbook", "Playbook"]].map(([k, n]) => html`<button type="button" data-mode="${k}" aria-pressed="${k === st.mode}">${n}</button>`)}</div>
        <input class="k-field" id="n-cwd" value="${pref("agCwd", root)}" aria-label="Folder (a git checkout)" title="Folder (a git checkout)">
        <select class="k-select" id="n-harness" aria-label="Harness">${usable.map(x => html`<option value="${x.id}" ${x.id === st.harness ? "selected" : ""}>${x.name}${x.driver === "stream" ? " (no live thinking)" : ""}</option>`)}</select>
        <span id="n-model-box"></span>
        <select class="k-select" id="n-book" aria-label="Playbook" hidden>${books.map(b => html`<option value="${b.id}" ${b.id === st.playbook ? "selected" : ""}>${b.name}</option>`)}</select>
        <span class="k-note" id="n-team-note"></span>
        <button class="k-btn k-primary k-new-go" type="submit" id="n-go">Start</button></div>
      <div id="n-flow" class="k-flowedit" hidden></div></form>

    <div class="k-new-grid">
      <div><p class="k-chart-title">Permissions</p><div class="k-seg" role="group" aria-label="Permissions">
        <button type="button" data-perm="ask" aria-pressed="false">Ask before risky changes</button><button type="button" data-perm="skip" aria-pressed="true">Skip permissions</button></div>
        <p class="k-note" id="n-perm-note"></p></div>
      <div><p class="k-chart-title">Attempts</p><div class="k-seg" role="group" aria-label="Attempts">${[1, 2, 3].map(n => html`<button type="button" data-att="${n}" aria-pressed="${n === 1}">${n}</button>`)}</div>
        <p class="k-note">More attempts run the same task in parallel worktrees so you can compare and keep one.</p></div></div>
    ${gr ? html`<p class="k-guard-line"><span>Own worktree: ${gr.worktree ? "on" : "off"}</span><span>Max ${gr.max_turns} turns per worker</span><span>Up to ${gr.max_workers} workers</span><span>Never touch ${gr.never_touch.slice(0, 3).join(", ")}${gr.never_touch.length > 3 ? " and more" : ""}</span><a class="k-link" href="#agents/guardrails">Change</a></p>` : ""}
    <div id="n-err"></div>
    <div class="k-sect"><h2>Start from a playbook</h2><span>saved setups</span><a class="k-link" href="#agents/playbooks">Manage</a></div>
    ${books.length ? books.map(b => html`<button class="k-row" data-use="${b.id}"><span class="k-glyph">${icon("page")}</span><span class="k-t"><b>${b.name}</b><small>${b.description || b.goal || ""}</small></span>
      <span class="k-v k-dim">${(b.workers || []).length + (b.lead ? 1 : 0)} agents · used ${b.uses}</span></button>`)
      : html`<p class="k-note">No playbooks yet. Save a team setup on the Playbooks page to start it in one click.</p>`}
    ${!usable.length ? html`<div class="k-error-box" style="margin-top:20px">No harness is usable on this machine. Install opencode or Antigravity, then check the Harnesses page.</div>` : ""}</div>`);

  const paint = () => {
    $$("[data-mode]", el).forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === st.mode));
    $$("[data-perm]", el).forEach(b => b.setAttribute("aria-pressed", b.dataset.perm === st.perm));
    $$("[data-att]", el).forEach(b => b.setAttribute("aria-pressed", +b.dataset.att === st.attempts));
    $("#n-book", el).hidden = st.mode !== "playbook";
    $("#n-harness", el).hidden = st.mode === "playbook" || st.mode === "flow";
    $("#n-flow", el).hidden = st.mode !== "flow";
    if (st.mode === "flow") paintFlow();
    const hh = usable.find(x => x.id === $("#n-harness", el).value);
    const mb = $("#n-model-box", el), cur = st.model || "";
    mb.hidden = st.mode === "playbook" || st.mode === "flow";
    if (!mb.hidden && !(st.custom && $("#n-model", el) && document.activeElement === $("#n-model", el))) render(mb, modelSel(hh && hh.id, cur, raw('id="n-model" aria-label="Model"'), st.custom));
    $("#n-team-note", el).textContent = st.mode === "team" ? "The lead plans and starts workers through kllm's tools." : st.mode === "flow" ? "kllm runs the steps in order; steps that wait for nothing run side by side." : "";
    const askOk = st.mode === "playbook" || (st.mode === "flow" ? steps.every(x => (usable.find(u => u.id === x.harness) || {}).ask) : hh && hh.ask);
    const askBtn = $('[data-perm="ask"]', el);
    askBtn.disabled = !askOk;
    if (!askOk && st.perm === "ask") { st.perm = "skip"; paint(); return; }
    $("#n-perm-note", el).textContent = st.perm === "skip"
      ? "Skip lets agents act without asking. They still work in their own worktree, kllm refuses never-touch paths, and you can undo to any checkpoint."
      : `Permission requests from the harness wait in your Inbox. ${hh ? hh.name : "The harness"} decides what it asks about, from its own settings.`;
    if (!askOk) $("#n-perm-note", el).textContent += ` Ask needs an ACP harness; ${hh ? hh.name : "this one"} runs without one here.`;
  };
  // ---- workflow builder ----
  const save = () => setPref("agFlow", steps);
  const paintFlow = () => {
    render($("#n-flow", el), html`<div class="k-flowedit-head"><b>Steps</b><span class="k-note">Each step is one agent. Pick its harness and model, and the steps it waits for.</span>
        <button type="button" class="k-btn k-xs" data-fl="template">Use the plan, build, review example</button></div>
      ${steps.map((x, i) => html`<div class="k-flowstep k-raised" data-i="${i}">
        <div class="k-flowstep-top"><span class="k-glyph">${i + 1}</span><input class="k-field" data-f="title" value="${x.title}" aria-label="Step name" placeholder="Step name">
          <select class="k-select" data-f="harness" aria-label="Harness">${usable.map(u => html`<option value="${u.id}" ${u.id === x.harness ? "selected" : ""}>${u.name}</option>`)}</select>
          ${modelSel(x.harness, x.model || "", raw('data-f="model" aria-label="Model"'), x.custom)}
          <button type="button" class="k-icon-btn" data-fl="up" title="Move up" aria-label="Move up" ${i === 0 ? "disabled" : ""}>${icon("arrowdown")}</button>
          <button type="button" class="k-icon-btn" data-fl="rm" title="Remove step" aria-label="Remove step" ${steps.length === 1 ? "disabled" : ""}>${icon("x")}</button></div>
        <textarea class="k-textarea" data-f="objective" rows="2" aria-label="What this step does" placeholder="What this step does and what done looks like">${x.objective}</textarea>
        ${i ? html`<div class="k-flowstep-after"><span class="k-dim">Waits for</span>${steps.slice(0, i).map(y => html`<button type="button" class="k-chip" data-after="${y.id}" aria-pressed="${x.after.includes(y.id)}">${y.title || y.id}</button>`)}
          ${!x.after.length ? html`<span class="k-note">nothing: starts at once</span>` : ""}</div>` : ""}</div>`)}
      <div class="k-acts"><button type="button" class="k-btn k-sm" data-fl="add">${icon("plus")}Add a step</button><button type="button" class="k-btn k-sm" data-fl="savebook">Save as a playbook</button></div>`);
  };
  $("#n-flow", el).addEventListener("change", e => {
    const f = e.target.dataset.f, box = e.target.closest("[data-i]"); if (!f || !box) return;
    const s0 = steps[+box.dataset.i];
    if (f === "model" && e.target.value === "__other") { s0.custom = true; s0.model = ""; save(); paintFlow(); const i = $(`[data-i="${box.dataset.i}"] [data-f="model"]`, el); if (i) i.focus(); return; }
    if (f === "harness") { s0.model = ""; s0.custom = false; }
    if (f === "model" || f === "harness") { s0[f] = e.target.value; save(); paintFlow(); paint(); }
  });
  $("#n-flow", el).addEventListener("input", e => {
    const f = e.target.dataset.f, box = e.target.closest("[data-i]"); if (!f || !box || f === "harness" || (f === "model" && e.target.tagName === "SELECT")) return;
    steps[+box.dataset.i][f] = e.target.value; save();
  });
  $("#n-flow", el).addEventListener("click", async e => {
    const box = e.target.closest("[data-i]"), i = box ? +box.dataset.i : -1;
    const af = e.target.closest("[data-after]");
    if (af) { const s0 = steps[i], id = af.dataset.after; s0.after = s0.after.includes(id) ? s0.after.filter(x => x !== id) : [...s0.after, id]; save(); paintFlow(); return; }
    const b = e.target.closest("[data-fl]"); if (!b) return;
    const k = b.dataset.fl;
    if (k === "add") { const n = steps.length + 1; steps.push({ id: "s" + Date.now().toString(36), title: `Step ${n}`, harness: (usable[0] || {}).id, model: "", after: steps.length ? [steps[steps.length - 1].id] : [], objective: "" }); }
    if (k === "rm") { const gone = steps[i].id; steps.splice(i, 1); steps.forEach(x => { x.after = x.after.filter(a => a !== gone); }); }
    if (k === "up" && i > 0) { const [x] = steps.splice(i, 1); steps.splice(i - 1, 0, x); steps.forEach((y, j) => { y.after = y.after.filter(a => steps.slice(0, j).some(z => z.id === a)); }); }
    if (k === "template") steps = TEMPLATE(usable.map(x => x.id), serving[0]);
    if (k === "savebook") {
      const name = ($("#n-goal", el).value.trim().split("\n")[0] || "Workflow").slice(0, 60);
      const r = await act(b, () => post("/admin/api/agents/playbooks", { name, mode: "flow", goal: $("#n-goal", el).value.trim(), permission_mode: st.perm, flow: { steps: flowBody() } }), { quiet: true });
      if (r.ok) toast(`Saved the playbook "${name}".`); else render($("#n-err", el), errorBox(r.error));
      return;
    }
    save(); paintFlow(); paint();
  });
  const flowBody = () => steps.map(x => ({ id: x.id, title: x.title, harness: x.harness, model: x.model || null, objective: x.objective, after: x.after }));

  el.addEventListener("click", e => {
    const m = e.target.closest("[data-mode]"); if (m) { st.mode = m.dataset.mode; setPref("agMode", st.mode === "playbook" ? "one" : st.mode); paint(); }
    const pp = e.target.closest("[data-perm]"); if (pp && !pp.disabled) { st.perm = pp.dataset.perm; paint(); }
    const a = e.target.closest("[data-att]"); if (a) { st.attempts = +a.dataset.att; paint(); }
    const u = e.target.closest("[data-use]");
    if (u) { st.mode = "playbook"; $("#n-book", el).value = u.dataset.use; const b = books.find(x => x.id === u.dataset.use); if (b && !$("#n-goal", el).value) $("#n-goal", el).value = b.goal || ""; if (b && b.permission_mode) st.perm = b.permission_mode; paint(); $("#n-goal", el).focus(); }
  });
  $("#n-harness", el).onchange = e => { setPref("agHarness", e.target.value); st.model = ""; st.custom = false; paint(); };
  $("#n-model-box", el).addEventListener("change", e => {
    if (e.target.value === "__other") { st.custom = true; st.model = ""; paint(); $("#n-model", el).focus(); return; }
    st.model = e.target.value.trim();
  });
  $("#n-model-box", el).addEventListener("input", e => { if (e.target.tagName === "INPUT") st.model = e.target.value.trim(); });
  $("#n-form", el).onsubmit = async e => {
    e.preventDefault();
    const body = { goal: $("#n-goal", el).value.trim(), mode: st.mode, permission_mode: st.perm, attempts: st.attempts, cwd: $("#n-cwd", el).value.trim() };
    if (!body.goal && st.mode !== "playbook") { render($("#n-err", el), errorBox("Describe what the agents should do first: the task, the files involved and what done looks like.")); $("#n-goal", el).focus(); return; }
    if (st.mode === "playbook" && !$("#n-book", el).value) { render($("#n-err", el), errorBox("Pick a playbook first, or switch to One agent.")); return; }
    render($("#n-err", el), "");
    if (st.mode === "flow") {
      const empty = steps.findIndex(x => !x.objective.trim());
      if (empty >= 0) { render($("#n-err", el), errorBox(`Step ${empty + 1} needs a description of what it does.`)); return; }
      body.steps = flowBody();
    } else if (st.mode === "playbook") body.playbook_id = $("#n-book", el).value;
    else { body.harness = $("#n-harness", el).value; if (st.model) body.model = st.model; }
    if (st.mode === "team") body.lead = { harness: body.harness, model: body.model };
    setPref("agCwd", body.cwd);
    const r = await act($("#n-go", el), () => post("/admin/api/agents/sessions", body), { quiet: true });
    if (r.ok) { ctx.shared.refreshAgents(); ctx.go(`#agents/s/${r.data.id}`); } else render($("#n-err", el), errorBox(r.error));
  };
  paint();
  $("#n-goal", el).focus();
}
