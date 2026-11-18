// Connect: point every harness and app at kllm. GET /admin/api/connect (tools and their options), GET /admin/api/state
// (endpoints), GET /admin/api/config/key, POST /admin/api/connect/{tool}/setup|undo {what: "models"|"mcp"}.
import { get, post } from "../api.js";
import { html, render, $, $$, icon, copy, act, errorBox, loading } from "../ui.js";

const STATE = { on: ["Connected", "k-ok"], off: ["Not set up", ""], missing: ["Not installed", ""], unsupported: ["Not available", ""], manual: ["Manual", ""] };

export async function mount(el) {
  render(el, html`<div class="k-page k-narrow-page"><span class="k-kicker">Connect</span><h1 class="k-head" id="cn-head">${""}</h1>
    <p class="k-lede">Two ways to connect a tool. <b>Models</b> runs it on your Kaggle pool through the relay. <b>Tools over MCP</b> lets it start and steer kllm agents. Every change keeps a backup, and Undo puts the file back.</p>
    <div id="cn-list" style="margin-top:34px">${loading()}</div></div>`);
  let open = null;
  const load = async () => {
    const [t, s, k] = await Promise.all([get("/admin/api/connect?refresh=1"), get("/admin/api/state"), get("/admin/api/config/key")]);
    if (!t.ok || !s.ok) { render($("#cn-list", el), errorBox(t.error || s.error)); return; }
    const ep = s.data.endpoints, base = ep.openai_base, key = (k.ok && k.data.key) || s.data.api_key, remote = s.data.remote_url;
    const tools = t.data.tools, on = tools.filter(x => x.options.some(o => o.state === "on")).length;
    render($("#cn-head", el), html`${on ? `${on} ${on === 1 ? "tool uses" : "tools use"} kllm.` : "Connect a tool."}<br><span>${base}</span>`);
    const row = (label, val) => html`<div class="k-copy"><code>${val}</code><button class="k-btn k-xs" data-copy="${val}">Copy</button></div>${label ? html`<p class="k-note">${label}</p>` : ""}`;
    const steps = {
      "claude-code": html`<ol class="k-num-steps"><li>Run Claude Code on the pool with the launcher. It picks the running model and its context size.${row("", "claude-kllm")}</li>
        <li>Or set these yourself before running <code>claude</code>.${row("", `ANTHROPIC_BASE_URL=${s.data.relay_base} ANTHROPIC_AUTH_TOKEN=${key}`)}</li></ol>`,
      opencode: html`<ol class="k-num-steps"><li>Start opencode on the pool.${row("", "opencode -m kllm/kllm")}</li></ol>`,
      codex: html`<ol class="k-num-steps"><li>After setup, start Codex with the kllm profile.${row("", "codex --profile kllm")}</li><li>Or point it at the relay for one run.${row("", `OPENAI_BASE_URL=${base} OPENAI_API_KEY=${key}`)}</li></ol>`,
      agy: html`<ol class="k-num-steps"><li>Restart Antigravity after setup. Its agent can then call <code>spawn_harness_subagent</code> to start kllm workers.</li></ol>`,
      gemini: html`<ol class="k-num-steps"><li>Restart Gemini CLI after setup, then check with <code>/mcp</code>.</li></ol>`,
      cursor: html`<ol class="k-num-steps"><li>Models: Settings, Models, override the OpenAI base URL.${row("", base)}</li><li>API key${row("", key)}</li></ol>`,
      openai: html`<ol class="k-num-steps"><li>Base URL${row("", base)}</li><li>API key${row("From this machine any key works. Other devices need this one.", key)}</li>
        <li>On your network${row("Reachable from other devices only when the relay listens on that address.", ep.openai_base_lan)}</li></ol>`,
      code: html`<ol class="k-num-steps"><li>Python${row("", `from openai import OpenAI; client = OpenAI(base_url="${base}", api_key="${key}")`)}</li>
        <li>JavaScript${row("", `new OpenAI({ baseURL: "${base}", apiKey: "${key}" })`)}</li><li>Use the model name <code>kllm</code> for whatever is running, or a model key.</li></ol>`,
      phone: remote ? html`<ol class="k-num-steps"><li>Use the tunnel address with your gateway key.${row("", remote.replace(/\/$/, "") + "/v1")}</li><li>Key${row("", key)}</li></ol>`
        : html`<ol class="k-num-steps"><li>Turn on the remote tunnel in Settings, Keys and tunnel. Set a gateway key first: remote callers without one are refused.</li><li><a href="#settings/keys">Open Settings</a></li></ol>`,
    };
    const opt = (x, o) => html`<div class="k-conn-opt"><span class="k-t"><b>${o.label}</b><small>${o.detail}</small></span>
      <span class="k-badge ${STATE[o.state] ? STATE[o.state][1] : ""}">${STATE[o.state] ? STATE[o.state][0] : o.state}</span>
      <span class="k-conn-acts">${o.can_setup ? html`<button class="k-btn k-xs k-primary" data-setup="${x.id}" data-what="${o.what}">Set up</button>` : ""}${o.can_undo ? html`<button class="k-btn k-xs" data-undo="${x.id}" data-what="${o.what}">Undo</button>` : ""}</span></div>`;
    render($("#cn-list", el), tools.map(x => html`<section class="k-conn ${open === x.id ? "k-open" : ""}">
      <button class="k-row" data-open="${x.id}" aria-expanded="${open === x.id}"><span class="k-glyph">${icon(x.id === "phone" ? "link" : x.id === "code" ? "page" : "term")}</span>
        <span class="k-t"><b>${x.name}</b><small>${x.installed === false ? "not installed" : x.version || (x.installed === null ? "any app" : "installed")}</small></span>
        <span class="k-v">${x.options.some(o => o.state === "on") ? html`<span class="k-badge k-ok">Connected</span>` : ""}</span>${icon("down")}</button>
      <div class="k-conn-opts">${x.options.map(o => opt(x, o))}</div>
      <div class="k-conn-body" ${open === x.id ? "" : "hidden"}>${steps[x.id] || ""}</div></section>`));
  };
  el.addEventListener("click", async e => {
    const o = e.target.closest("[data-open]");
    if (o) { open = open === o.dataset.open ? null : o.dataset.open; $$(".k-conn", el).forEach(c => { const me = $("[data-open]", c).dataset.open === open; c.classList.toggle("k-open", me); $(".k-conn-body", c).hidden = !me; $("[data-open]", c).setAttribute("aria-expanded", me); }); }
    const c = e.target.closest("[data-copy]"); if (c) copy(c.dataset.copy, c);
    const s = e.target.closest("[data-setup]"); if (s) { const r = await act(s, () => post(`/admin/api/connect/${s.dataset.setup}/setup`, { what: s.dataset.what })); if (r && r.ok) load(); }
    const u = e.target.closest("[data-undo]"); if (u) { const r = await act(u, () => post(`/admin/api/connect/${u.dataset.undo}/undo`, { what: u.dataset.what })); if (r && r.ok) load(); }
  });
  await load();
}
