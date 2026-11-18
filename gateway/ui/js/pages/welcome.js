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
      render(box, html`<div class="k-form k-narrow"><label><span>Kaggle API token</span><input class="k-field k-wide" id="w-token" type="password" autocomplete="off" placeholder="kaggle.com/settings, API, Create New Token"></label>
        <p class="k-note">kllm checks the token with Kaggle and shows the username. Kaggle gives GPUs only to phone-verified accounts. Each account gets about 30 GPU hours a week.</p>
        <div><button class="k-btn k-primary" id="w-add">Add account</button></div><div id="w-err"></div></div>`);
      $("#w-add", el).onclick = async e => {
        const r = await act(e.currentTarget, () => post("/admin/api/accounts/add", { token: $("#w-token", el).value.trim() }));
        if (r.ok) { step = 2; await ctx.shared.refresh(); paint(); } else render($("#w-err", el), errorBox(r.error));
      };
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
