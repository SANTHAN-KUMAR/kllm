// First run: add a Kaggle account, pick a model, connect a tool, say hello.
// POST /admin/api/accounts/add, GET /admin/api/models, POST /admin/api/backends/launch.
import { get, post } from "../api.js";
import { html, render, $, icon, act, errorBox, loading } from "../ui.js";

export async function mount(el, ctx) {
  let step = ctx.shared.backends && ctx.shared.backends.accounts.length ? 2 : 1;
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">First run</span><h1 class="k-head">Run models on your Kaggle GPUs.<br><span>Four steps, about ten minutes.</span></h1>
    <div class="k-rail" id="w-rail"></div><div id="w-body" style="margin-top:28px"></div></div>`);
  const STEPS = ["Add a Kaggle account", "Pick a model", "Connect a tool", "Say hello"];
  const paint = async () => {
    render($("#w-rail", el), html`<ol class="k-rail-list">${STEPS.map((s, i) => html`<li class="${i + 1 < step ? "k-done" : i + 1 === step ? "k-now" : ""}"><i>${i + 1 < step ? icon("check") : i + 1}</i><span>${s}</span></li>`)}</ol>`);
    const box = $("#w-body", el);
    if (step === 1) {
      render(box, html`<div class="k-form k-narrow">
        <ol class="k-howto">
          <li><b>Open your Kaggle settings.</b> <button class="k-btn k-sm" id="w-open" type="button">Open kaggle.com/settings</button><br>
            <small class="k-note">No account yet? Sign up there; it is free.</small></li>
          <li><b>Verify your phone number</b> on the same page, if you have not. Kaggle gives GPUs only to phone-verified accounts.</li>
          <li><b>Under API, click Create New Token</b>, then paste the token below, or drop the <code>kaggle.json</code> file it downloads.</li>
        </ol>
        <label><span>Kaggle API token or kaggle.json</span><input class="k-field k-wide" id="w-token" type="password" autocomplete="off" placeholder="KGAT_... or {&quot;username&quot;: ..., &quot;key&quot;: ...}"></label>
        <div class="k-drop" id="w-drop" tabindex="0" role="button" aria-label="Choose or drop kaggle.json">Drop <code>kaggle.json</code> here, or click to choose it<input type="file" id="w-file" accept=".json,application/json" hidden></div>
        <p class="k-note">kllm checks the token with Kaggle, shows your username, and keeps it on this computer only. Each account gets about 30 free GPU hours a week.</p>
        <div><button class="k-btn k-primary" id="w-add">Add account</button></div><div id="w-err"></div></div>`);
      const add = async (btn, token) => {
        const r = await act(btn, () => post("/admin/api/accounts/add", { token: (token || "").trim() }));
        if (r.ok) { step = 2; await ctx.shared.refresh(); paint(); } else render($("#w-err", el), errorBox(r.error));
      };
      const fromFile = async f => { if (f) await add($("#w-add", el), await f.text()); };
      $("#w-open", el).onclick = () => window.open("https://www.kaggle.com/settings", "_blank", "noopener");
      $("#w-add", el).onclick = e => add(e.currentTarget, $("#w-token", el).value);
      $("#w-token", el).onkeydown = e => { if (e.key === "Enter") add($("#w-add", el), e.currentTarget.value); };
      const drop = $("#w-drop", el), file = $("#w-file", el);
      drop.onclick = () => file.click();
      drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } };
      file.onchange = () => fromFile(file.files[0]);
      drop.ondragover = e => { e.preventDefault(); drop.classList.add("k-over"); };
      drop.ondragleave = () => drop.classList.remove("k-over");
      drop.ondrop = e => { e.preventDefault(); drop.classList.remove("k-over"); fromFile(e.dataTransfer.files[0]); };
    } else if (step === 2) {
      render(box, loading());
      const r = await get("/admin/api/models");
      if (!r.ok) { render(box, errorBox(r.error)); return; }
      const ms = r.data.models.filter(m => m.kind === "chat" && !(m.problems || []).length).sort((a, b) => (b.verified - a.verified) || ((b.key === r.data.default_model) - (a.key === r.data.default_model)));
      render(box, html`<p class="k-sec">Verified models are tested on two T4s. The default is used when a client asks for <code>kllm</code>.</p>
        <div style="margin-top:14px">${ms.slice(0, 8).map(m => html`<div class="k-row k-hov"><span class="k-glyph">${icon("chip")}</span><span class="k-t"><b>${m.key}${m.key === r.data.default_model ? html` <span class="k-badge">default</span>` : ""}</b>
          <small class="k-spec">${[m.ctx ? Math.round(m.ctx / 1024) + "K" : "", m.quant, m.vram_gb ? m.vram_gb + " GB" : "", m.verified ? "verified" : "unverified"].filter(Boolean).join(" · ")}</small></span>
          <button class="k-btn k-sm" data-launch="${m.key}">Launch</button></div>`)}</div>
        <div class="k-acts"><button class="k-btn" data-skip>Skip, autoscale starts it on first use</button></div><div id="w-err"></div>`);
      box.onclick = async e => {
        const b = e.target.closest("[data-launch]");
        if (b) { const r2 = await act(b, () => post("/admin/api/backends/launch", { model: b.dataset.launch })); if (r2.ok) { step = 3; ctx.shared.refresh(); paint(); } }
        if (e.target.closest("[data-skip]")) { step = 3; paint(); }
      };
    } else if (step === 3) {
      render(box, html`<p class="k-sec">Claude Code, opencode, Cursor and any OpenAI client can use kllm. The Connect page has the exact settings with your key filled in.</p>
        <div class="k-acts"><a class="k-btn k-primary" href="#connect">Open Connect</a><button class="k-btn" data-next>Next</button></div>`);
      box.onclick = e => { if (e.target.closest("[data-next]")) { step = 4; paint(); } };
    } else {
      render(box, html`<p class="k-sec">Send a first message. If the model is still loading, Chat waits for it.</p>
        <div class="k-acts"><a class="k-btn k-primary" href="#chat">Open Chat</a><a class="k-btn" href="#home">Go to Home</a></div>`);
    }
  };
  paint();
}
