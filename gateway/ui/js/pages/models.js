// Models: the registry in models.yaml. GET /admin/api/models; POST /admin/api/models/{search,update,default,delete},
// POST /admin/api/inspect and /admin/api/add-from-hf.
import { get, post } from "../api.js";
import { html, render, $, $$, icon, esc, num, sheet, act, toast, empty, errorBox, loading, debounce } from "../ui.js";

const KINDS = [["all", "All"], ["chat", "Chat"], ["image", "Image"], ["video", "Video"], ["running", "Running"]];
const isRunning = m => (m.running || []).length > 0 || (m.launching || []).length > 0;

export async function mount(el, ctx) {
  let data = null, kind = ctx.route.query.kind || "all", q = "";
  render(el, html`<div class="k-page"><span class="k-kicker" id="m-kick">Models</span><h1 class="k-head" id="m-head">${""}</h1>
    <div class="k-acts"><button class="k-btn k-primary" id="m-add">${icon("plus")}Add from Hugging Face</button><button class="k-btn" id="m-inspect">${icon("search")}Inspect a repo</button></div>
    <div class="k-models-bar"><div class="k-chips" id="m-kinds" role="group" aria-label="Kind"></div>
      <input class="k-field" id="m-q" placeholder="Filter models" aria-label="Filter models"></div>
    <div id="m-list">${loading()}</div></div>`);
  const load = async () => {
    const r = await get("/admin/api/models");
    if (!r.ok) { render($("#m-list", el), errorBox(r.error)); return; }
    data = r.data; paint();
  };
  const spec = m => [m.ctx ? Math.round(m.ctx / 1024) + "K" : "", m.quant, m.vram_gb ? m.vram_gb + " GB" : "", m.runtime !== "llama.cpp" ? m.runtime : "", ...(m.tags || []).slice(0, 2)].filter(Boolean).join(" · ");
  const paint = () => {
    const ms = data.models;
    const def = data.default_model;
    render($("#m-head", el), html`${num(ms.length)} models.<br><span>${["chat", "image", "video"].map(k => [ms.filter(m => m.kind === k).length, k]).filter(x => x[0]).map(([n, k]) => `${n} ${k}`).join(", ")}.</span>`);
    const count = k => k === "all" ? ms.length : k === "running" ? ms.filter(isRunning).length : ms.filter(m => m.kind === k).length;
    render($("#m-kinds", el), KINDS.map(([k, n]) => html`<button class="k-chip" data-kind="${k}" aria-pressed="${k === kind}">${n}<em>${count(k)}</em></button>`));
    const inKind = m => kind === "all" ? true : kind === "running" ? isRunning(m) : m.kind === kind;
    const rows = ms.filter(m => inKind(m) && (!q || (m.key + " " + (m.hf || "") + " " + (m.desc || "")).toLowerCase().includes(q)));
    render($("#m-list", el), rows.length ? html`<div class="k-list-head"><span>Model</span><span>State</span></div>${rows.map(m => html`
      <div class="k-row k-hov"><span class="k-glyph">${icon(m.kind === "chat" ? "chip" : m.kind === "video" ? "film" : "image")}</span>
        <span class="k-t"><b>${m.key}${m.default || (m.default_for || []).length ? html` <span class="k-badge">default${(m.default_for || []).length ? " " + m.default_for.join(", ") : ""}</span>` : ""}</b>
          <small class="k-spec">${spec(m)}</small>${(m.problems || []).length ? html`<small class="k-risk-t">${m.problems.join("; ")}</small>` : ""}</span>
        <span class="k-row-acts">${!(m.problems || []).length ? html`<a class="k-btn k-xs" href="#launch?model=${encodeURIComponent(m.key)}">Launch</a>` : ""}<button class="k-btn k-xs" data-edit="${m.key}">Edit</button></span>
        <span class="k-v"><span class="${m.failing ? "k-risk-t" : m.running.length ? "k-ok-t" : "k-dim"}">${m.failing ? "paused by circuit breaker" : m.running.length ? `running on ${m.running.join(", ")}` : m.launching.length ? "starting" : m.verified ? "verified" : "unverified"}</span></span></div>`)}`
      : empty(q ? "No model matches." : kind === "running" ? "Nothing is running." : kind === "all" ? "No models." : `No ${kind} models.`, q ? "Clear the filter, or add one from Hugging Face." : "Add one from Hugging Face.", ""));
  };
  el.addEventListener("click", e => {
    // the whole row opens its sheet; buttons and links inside keep their own action
    const row = e.target.closest(".k-row.k-hov");
    if (row && !e.target.closest("a, button, input, select")) { const b = row.querySelector("[data-edit]"); if (b) { b.click(); return; } }
    const k = e.target.closest("[data-kind]"); if (k) { kind = k.dataset.kind; paint(); }
    const ed = e.target.closest("[data-edit]"); if (ed) edit(data.models.find(m => m.key === ed.dataset.edit));
  });
  $("#m-q", el).oninput = debounce(e => { q = e.target.value.trim().toLowerCase(); paint(); }, 120);
  $("#m-add", el).onclick = () => addSheet(false);
  $("#m-inspect", el).onclick = () => addSheet(true);

  // ---- edit sheet: description, context, advanced llama-server args (validated JSON list) ----
  const edit = m => {
    const s = sheet({ title: m.key, sub: m.hf || "", body: html`<div class="k-form">
      <label><span>Description</span><input class="k-field" id="e-desc" value="${m.desc || ""}"></label>
      ${m.kind === "chat" ? html`<label><span>Context, tokens</span><input class="k-field" id="e-ctx" type="number" min="2048" step="1024" value="${m.ctx || ""}"></label>` : ""}
      <details class="k-adv"><summary>Advanced</summary><label><span>llama-server arguments, one per line</span><textarea class="k-textarea k-mono" id="e-args" rows="6">${(m.args || []).join("\n")}</textarea></label>
        <p class="k-note">Each line is one argument, for example <code>--temp</code> then <code>0.6</code>. kllm checks them before saving.</p></details>
      <div id="e-err"></div></div>`,
      foot: html`<button class="k-btn k-primary" data-save>Save</button>${m.kind === "chat" && !m.default ? html`<button class="k-btn" data-def>Make default</button>` : ""}<button class="k-btn k-push" data-del>Remove</button>` });
    $("[data-save]", s.el).onclick = async e => {
      const changes = {};
      const desc = $("#e-desc", s.el).value.trim(); if (desc !== (m.desc || "")) changes.desc = desc;
      const c = $("#e-ctx", s.el); if (c && c.value && +c.value !== m.ctx) changes.ctx = +c.value;
      const args = $("#e-args", s.el).value.split("\n").map(x => x.trim()).filter(Boolean);
      if (JSON.stringify(args) !== JSON.stringify(m.args || [])) {
        if (args.some(a => /\s/.test(a) && !a.startsWith("-"))) { render($("#e-err", s.el), errorBox("Put each argument on its own line: a flag, then its value.")); return; }
        changes.args = args;
      }
      if (!Object.keys(changes).length) { toast("Nothing changed."); return; }
      const r = await act(e.currentTarget, () => post("/admin/api/models/update", { key: m.key, changes }));
      if (r.ok) { s.close(); load(); } else render($("#e-err", s.el), errorBox(r.error));
    };
    const d = $("[data-def]", s.el);
    if (d) d.onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/models/default", { key: m.key })); if (r.ok) { s.close(); load(); ctx.shared.refresh(); } };
    $("[data-del]", s.el).onclick = () => {
      render(s.foot, html`<span class="k-sec" style="align-self:center">Remove ${m.key} from models.yaml?</span><button class="k-btn k-risk k-push" data-yes>Remove</button><button class="k-btn" data-no>Keep</button>`);
      $("[data-no]", s.el).onclick = () => { s.close(); edit(m); };
      $("[data-yes]", s.el).onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/models/delete", { key: m.key })); if (r.ok) { s.close(); load(); } };
    };
  };

  // ---- add from Hugging Face: search, inspect, pick a file, confirm ----
  const addSheet = (inspectOnly = false) => {
    const s = sheet({ title: inspectOnly ? "Inspect a repo" : "Add a model", sub: inspectOnly ? "Paste a Hugging Face repo. kllm checks how it fits two T4s, lists its GGUF files and shows what would land in models.yaml." : "Search Hugging Face, or paste a repo such as org/name. kllm checks it fits two T4s.", wide: true,
      body: html`<div class="k-form"><input class="k-field k-wide" id="a-q" placeholder="${inspectOnly ? "org/name, or https://huggingface.co/..." : "qwen3 27b, or org/name"}" aria-label="Search or repo">
        ${inspectOnly ? "" : html`<label class="k-check"><input type="checkbox" id="a-abl" checked> Prefer abliterated GGUF builds</label>`}</div>
        <div id="a-res" style="margin-top:16px"></div>`, foot: html`` });
    const res = $("#a-res", s.el);
    const search = async () => {
      const v = $("#a-q", s.el).value.trim();
      if (v.length < 2) { render(res, ""); return; }
      if (/^[\w.-]+\/[\w.-]+/.test(v) || v.includes("huggingface.co/")) return inspect(v);
      render(res, loading());
      const r = await post("/admin/api/models/search", { q: v, abliterated: ($("#a-abl", s.el) || {}).checked });
      if (!r.ok) { render(res, errorBox(r.error)); return; }
      render(res, r.data.results.length ? r.data.results.map(x => html`<button class="k-row" data-repo="${x.repo}"><span class="k-t"><b>${x.repo}</b><small>${num(x.downloads || 0)} downloads${x.likes ? ` · ${num(x.likes)} likes` : ""}${x.updated ? ` · updated ${String(x.updated).slice(0, 10)}` : ""}</small></span>${icon("arrow")}</button>`)
        : html`<p class="k-note">Nothing found. Paste the exact repo name instead.</p>`);
    };
    const inspect = async (source, file) => {
      render(res, loading()); render(s.foot, html``);
      const r = await post("/admin/api/inspect", { source, file });
      if (!r.ok) { render(res, errorBox(r.error)); return; }
      const x = r.data;
      render(res, html`<h3 class="k-sub-title">${x.repo}</h3><p class="k-spec">${x.kind} · ${x.runtime} · ${x.weights_gb ? x.weights_gb + " GB weights · " : ""}${x.vram_gb ? "about " + x.vram_gb + " GB VRAM · " : ""}${x.ctx ? num(x.ctx) + " context" : ""}</p>
        ${x.warnings && x.warnings.length ? html`<div class="k-banner k-warn" style="margin-top:12px"><i class="k-dot"></i><span>${x.warnings.join(" ")}</span></div>` : ""}
        ${!x.abliterated ? html`<p class="k-note k-wait-t" style="margin-top:10px">This build is not abliterated. Every chat model kllm ships is.</p>` : ""}
        ${x.files && x.files.length > 1 ? html`<div class="k-form" style="margin-top:14px"><label><span>File</span><select class="k-select" id="a-file">${x.files.map(f => html`<option value="${f.file}" ${f.file === x.chosen ? "selected" : ""}>${f.quant || f.file} · ${f.size_gb} GB${f.parts > 1 ? ` · ${f.parts} parts` : ""}</option>`)}</select></label></div>` : ""}
        <div class="k-form" style="margin-top:14px"><label><span>Name in models.yaml</span><input class="k-field" id="a-name" value="${x.key}"></label>
          ${x.kind === "chat" ? html`<label class="k-check"><input type="checkbox" id="a-def"> Make it the default chat model</label>` : ""}</div><div id="a-err"></div>`);
      const f = $("#a-file", s.el); if (f) f.onchange = () => inspect(source, f.value);
      render(s.foot, html`<button class="k-btn k-primary" data-add>${icon("plus")}Add ${x.key}</button>${!x.fits ? html`<span class="k-note k-wait-t" style="align-self:center">Does not look like it fits</span>` : ""}`);
      $("[data-add]", s.el).onclick = async e => {
        const body = { source, file: f ? f.value : x.chosen, name: $("#a-name", s.el).value.trim(), default: ($("#a-def", s.el) || {}).checked || false, force: !x.fits };
        const r2 = await act(e.currentTarget, () => post("/admin/api/add-from-hf", body));
        if (r2.ok) { s.close(); kind = x.kind; load(); ctx.shared.refresh(); } else render($("#a-err", s.el), errorBox(r2.error));
      };
    };
    $("#a-q", s.el).addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); search(); } });
    $("#a-q", s.el).oninput = debounce(search, 450);
    res.onclick = e => { const b = e.target.closest("[data-repo]"); if (b) inspect(b.dataset.repo); };
  };
  await load();
}
