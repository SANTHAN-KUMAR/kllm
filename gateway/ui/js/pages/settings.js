// Settings: one sentence and one control per setting, saved on change. GET /admin/api/config, POST /admin/api/config/set,
// GET /admin/api/config/key, POST /admin/api/config/rotate-key, GET /admin/api/config/raw, GET /admin/api/backup.
import { get, post } from "../api.js";
import { html, render, $, $$, icon, copy, act, toast, errorBox, loading, setPref } from "../ui.js";

const TABS = [["general", "General"], ["keys", "Keys and tunnel"], ["connect", "Connect a client"], ["files", "Files"], ["look", "Appearance"]];

export async function mount(el, ctx) {
  let tab = ctx.route.parts[1] || "general", cfg = null, models = [];
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">Settings</span><h1 class="k-head" id="se-head">${""}</h1>
    <div class="k-tabs" role="tablist" style="margin-top:34px">${TABS.map(([k, n]) => html`<button role="tab" data-tab="${k}" aria-selected="${k === tab}">${n}</button>`)}</div>
    <div id="se-body">${loading()}</div></div>`);
  const load = async () => {
    const [c, m] = await Promise.all([get("/admin/api/config"), get("/admin/api/models")]);
    if (!c.ok) { render($("#se-body", el), errorBox(c.error)); return; }
    cfg = c.data; models = m.ok ? m.data.models : [];
    render($("#se-head", el), html`Settings.<br><span>Changes save as you go.</span>`);
    paint();
  };
  const sel = (key, kind, allowNone) => html`<select class="k-select k-inline" data-key="${key}" aria-label="${key.replace(/_/g, " ")}">
    ${allowNone ? html`<option value="">none</option>` : ""}${models.filter(x => x.kind === kind).map(x => html`<option value="${x.key}" ${x.key === cfg[key] ? "selected" : ""}>${x.key}</option>`)}</select>`;
  const numIn = (key, val, min, max, step = 1, scale = 1) => html`<input class="k-field k-inline k-inline-num" type="number" data-key="${key}" data-scale="${scale}" value="${+(val / scale).toFixed(2)}" min="${min}" max="${max}" step="${step}" aria-label="${key.replace(/_/g, " ")}">`;
  const tog = (key, on) => html`<button class="k-toggle" role="switch" data-toggle="${key}" aria-checked="${!!on}" aria-label="${key.replace(/_/g, " ")}"></button>`;
  const line = (content, note = "") => html`<div class="k-setting"><p>${content}</p><span class="k-note k-saved" data-note></span>${note ? html`<p class="k-note">${note}</p>` : ""}</div>`;
  const paint = async () => {
    const box = $("#se-body", el);
    if (tab === "general") {
      render(box, html`
        ${line(html`Use ${sel("default_model", "chat")} when a client asks for <code>kllm</code>.`)}
        ${line(html`Make images with ${sel("default_image_model", "image", true)} and videos with ${sel("default_video_model", "video", true)}.`)}
        ${line(html`Check kernel health every ${numIn("health_interval_seconds", cfg.health_interval_seconds, 20, 600)} s.`)}
        ${line(html`Replace a kernel after ${numIn("health_fail_limit", cfg.health_fail_limit, 1, 10)} failed checks in a row.`)}
        ${line(html`Ask Kaggle for ${numIn("session_seconds", cfg.session_seconds, 1, 11.5, 0.5, 3600)} h sessions.`, "Kaggle stops a GPU session at 12 h.")}
        ${line(html`Start a successor ${numIn("prelaunch_lead_seconds", cfg.prelaunch_lead_seconds, 5, 120, 1, 60)} min before a session ends.`)}
        ${line(html`${tog("local_fallback_enabled", cfg.local_fallback_enabled)} Fall back to the local llama-server when no kernel answers.`, cfg.local_fallback_url ? `Local server: ${cfg.local_fallback_url}` : "")}
        <p class="k-note" style="margin-top:28px">Autoscale rules live on the <a href="#autoscale">Autoscale</a> page. Everything else is in <code>${cfg.config_file}</code>.</p>`);
    } else if (tab === "keys") {
      render(box, html`
        <div class="k-setting"><p>Gateway key: <code id="se-key">${cfg.gateway_key_set ? cfg.gateway_key_preview : "not set"}</code></p>
          <p class="k-note">Requests from this machine need no key. Other devices and the tunnel must send it as a Bearer token.</p>
          <div class="k-acts" style="margin-top:12px"><button class="k-btn k-sm" data-reveal>Reveal and copy</button><button class="k-btn k-sm" data-rotate>${cfg.gateway_key_set ? "Rotate key" : "Create a key"}</button></div><div id="se-rot"></div></div>
        ${line(html`${tog("remote_enabled", cfg.remote_enabled)} Reach kllm from anywhere through a Cloudflare tunnel.`,
          cfg.remote_url ? `Tunnel address: ${cfg.remote_url}${cfg.tunnel_running ? "" : " (not running)"}` : cfg.remote_enabled ? "The address shows up a few seconds after the tunnel starts." : "")}
        ${!cfg.gateway_key_set ? html`<p class="k-note k-wait-t">Create a gateway key before turning on the tunnel: without one, remote callers are refused.</p>` : ""}
        <div class="k-setting"><p>Relay listens on <code>${cfg.relay}</code>.</p></div>`);
    } else if (tab === "connect") {
      const base = (cfg.relay || "http://127.0.0.1:8080") + "/v1";
      render(box, html`
        <div class="k-setting"><p>Any client that speaks the OpenAI API can use kllm: point it at <code>${base}</code> and any key from this machine.</p>
          <div class="k-acts" style="margin-top:12px"><button class="k-btn k-sm" data-copy-base="${base}">${icon("copy")}Copy the base URL</button><a class="k-btn k-sm" href="#connect">Open the Connect page</a></div></div>
        <div class="k-setting"><p>Claude Code on your pool: run the launcher. It picks the running model and matches the context.</p>
          <div class="k-copy" style="margin-top:10px"><code>claude-kllm</code><button class="k-btn k-xs" data-copy-cmd="claude-kllm">Copy</button></div></div>
        <div class="k-setting"><p>opencode with kllm as a provider:</p>
          <div class="k-copy" style="margin-top:10px"><code>opencode -m kllm/kllm</code><button class="k-btn k-xs" data-copy-cmd="opencode -m kllm/kllm">Copy</button></div></div>
        <div class="k-setting"><p>Tools over MCP: kllm registers itself as an MCP server in Claude Code, opencode, Antigravity and Gemini CLI, so an agent can start, steer and stop kllm workers. Set it up from the Connect page.</p></div>
        <p class="k-note" style="margin-top:20px">Other devices need the gateway key from the <a href="#settings/keys">Keys and tunnel</a> tab.</p>`);
    } else if (tab === "files") {
      render(box, html`<div class="k-acts" style="margin-top:0"><button class="k-btn k-sm" data-file="config" aria-pressed="true">config.yaml</button><button class="k-btn k-sm" data-file="models">models.yaml</button>
        <button class="k-btn k-sm" data-backup>${icon("down")}Download backup</button></div><div id="se-file" style="margin-top:14px">${loading()}</div>`);
      showFile("config");
    } else {
      const t = ctx.store.theme;
      render(box, html`<div class="k-setting"><p>Theme</p><div class="k-seg" role="group" aria-label="Theme">${[["system", "Follow the system"], ["dark", "Dark"], ["light", "Light"]].map(([k, n]) =>
        html`<button data-theme="${k}" aria-pressed="${k === t}">${n}</button>`)}</div></div>
        <div class="k-setting"><p>${tog("thinking-pref", ctx.store.thinking)} Show model thinking in Chat and Agents.</p></div>`);
    }
  };
  const showFile = async which => {
    const r = await get(`/admin/api/config/raw?file=${which}`);
    render($("#se-file", el), r.ok ? html`<p class="k-note">${r.data.file}${r.data.masked ? " · secrets are hidden" : ""} · read only. Edit it with a text editor; kllm picks up changes.</p>
      <pre class="k-code k-file-view">${r.data.text}</pre>` : errorBox(r.error));
  };
  const save = async (key, value, node) => {
    const note = node && node.closest(".k-setting") && node.closest(".k-setting").querySelector("[data-note]");
    if (note) note.textContent = "Saving";
    const r = await post("/admin/api/config/set", { key, value });
    if (r.ok) { cfg = r.data.config; if (note) { note.textContent = "Saved"; setTimeout(() => { note.textContent = ""; }, 1800); } }
    else { if (note) note.textContent = ""; toast(r.error, true); paint(); }
    return r;
  };
  el.addEventListener("change", e => {
    const n = e.target.closest("[data-key]");
    if (!n) return;
    const scale = +(n.dataset.scale || 1);
    save(n.dataset.key, n.tagName === "SELECT" ? n.value : Math.round(+n.value * scale), n);
  });
  el.addEventListener("click", async e => {
    const t = e.target.closest("[data-tab]");
    if (t) { tab = t.dataset.tab; $$("[data-tab]", el).forEach(x => x.setAttribute("aria-selected", x === t)); history.replaceState(null, "", "#settings" + (tab === "general" ? "" : "/" + tab)); paint(); }
    const g = e.target.closest("[data-toggle]");
    if (g) {
      const on = g.getAttribute("aria-checked") !== "true";
      if (g.dataset.toggle === "thinking-pref") { ctx.store.set("thinking", on); g.setAttribute("aria-checked", on); return; }
      const r = await save(g.dataset.toggle, on, g); if (r.ok) { g.setAttribute("aria-checked", on); if (g.dataset.toggle === "remote_enabled") { toast(r.data.msg); setTimeout(load, 4000); } }
    }
    const th = e.target.closest("[data-theme]");
    if (th) {
      ctx.store.set("theme", th.dataset.theme);
      if (th.dataset.theme === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = th.dataset.theme;
      $$("[data-theme]", el).forEach(x => x.setAttribute("aria-pressed", x === th));
    }
    if (e.target.closest("[data-reveal]")) { const r = await get("/admin/api/config/key"); if (r.ok && r.data.key) { $("#se-key", el).textContent = r.data.key; copy(r.data.key, e.target.closest("[data-reveal]")); } else toast(r.error || "No key is set.", !r.ok); }
    const cb = e.target.closest("[data-copy-base]"); if (cb) { copy(cb.dataset.copyBase, cb); return; }
    const cc = e.target.closest("[data-copy-cmd]"); if (cc) { copy(cc.dataset.copyCmd, cc); return; }
    if (e.target.closest("[data-rotate]")) {
      render($("#se-rot", el), html`<div class="k-confirm k-raised" style="margin-top:12px"><p class="k-sec">Every remote client stops working until you give it the new key.</p>
        <div class="k-acts" style="margin-top:10px"><button class="k-btn k-sm k-risk" data-rot-yes>Rotate now</button><button class="k-btn k-sm" data-rot-no>Cancel</button></div></div>`);
    }
    if (e.target.closest("[data-rot-no]")) render($("#se-rot", el), "");
    if (e.target.closest("[data-rot-yes]")) { const r = await act(e.target.closest("[data-rot-yes]"), () => post("/admin/api/config/rotate-key", {})); if (r.ok) { await load(); $("#se-key", el).textContent = r.data.key; } }
    const f = e.target.closest("[data-file]"); if (f) { $$("[data-file]", el).forEach(x => x.setAttribute("aria-pressed", x === f)); showFile(f.dataset.file); }
    if (e.target.closest("[data-backup]")) {
      const r = await get("/admin/api/backup");
      if (!r.ok) { toast(r.error, true); return; }
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(r.data, null, 2)], { type: "application/json" }));
      a.download = `kllm-backup-${new Date().toISOString().slice(0, 10)}.json`; a.click(); URL.revokeObjectURL(a.href);
      toast("Backup saved: config.yaml and models.yaml in one file.");
    }
  });
  await load();
}
export function update() { return false; }
