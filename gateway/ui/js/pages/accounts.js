// Accounts: Kaggle accounts and their weekly GPU quota. GET /admin/api/accounts (30 s);
// POST /admin/api/accounts/{add,disable,remove,clear-error,sync-quota}.
import { get, post, poll } from "../api.js";
import { html, render, $, $$, icon, num, dur, bar, sheet, act, empty, errorBox, loading, debounce, Fresh } from "../ui.js";
import { quotaStrip } from "./home.js";

// "In use" are the accounts kllm may launch on (account_allow_prefixes); the rest stay listed under "Not used".
const FILTERS = [["inuse", "In use"], ["ready", "Ready"], ["busy", "Busy"], ["attention", "Needs attention"], ["unused", "Not used"]];
const used = a => a.allowed !== false;
const busy = a => used(a) && (a.running || a.launching);
const attention = a => used(a) && (a.disabled || a.error_active || (a.credentials && !a.credentials.ok));
const ready = a => used(a) && !busy(a) && !attention(a);
const match = { inuse: used, ready, busy, attention, unused: a => !used(a) };

export async function mount(el, ctx) {
  let data = null, filter = "inuse", q = ctx.route.query.q || "";
  render(el, html`<div class="k-page k-accounts"><span class="k-kicker" id="a-kick">Accounts</span><h1 class="k-head" id="a-head">${""}</h1>
    <div class="k-acts"><button class="k-btn k-primary" id="a-add">${icon("plus")}Add account</button><button class="k-btn" id="a-sync">Sync quota</button></div>
    <div id="a-strip" style="margin-top:40px"></div>
    <div class="k-search k-raised" style="margin-top:28px">${icon("search")}<input id="a-q" placeholder="Find an account" aria-label="Find an account" value="${q}"></div>
    <div class="k-chips" id="a-filters" style="margin:14px 0 6px" role="group" aria-label="Filter"></div>
    <div id="a-list" class="k-staleable">${loading()}</div><p class="k-fresh" id="a-fresh"></p></div>`);
  const fresh = new Fresh($(".k-accounts", el), 30000, $("#a-fresh", el));
  const load = async () => {
    const r = await get("/admin/api/accounts");
    if (!r.ok) { if (!data) render($("#a-list", el), errorBox(r.error)); return; }
    data = r.data; fresh.mark(); paint();
  };
  const status = a => a.running ? `Serving ${a.model_key || "a model"}` : a.launching ? `Starting ${a.model_key || "a kernel"}`
    : a.disabled ? "Disabled: the planner skips it" : a.credentials && !a.credentials.ok ? a.credentials.error
    : a.error_active ? `Cooling down ${dur(a.cooldown_left_s)} after: ${a.last_error || "a failed launch"}` : a.last_error ? `Last error: ${a.last_error}` : "Ready";
  const paint = () => {
    const accts = data.accounts, enabled = accts.filter(a => !a.disabled && a.allowed !== false);
    const left = enabled.reduce((s, a) => s + Math.max(0, a.left_h), 0), limit = enabled.reduce((s, a) => s + a.limit_h, 0);
    $("#a-kick", el).textContent = `Accounts · weekly reset in ${data.week_resets_in_h != null ? dur(data.week_resets_in_h * 3600) : "unknown"}`;
    const inUse = accts.filter(used).length;
    render($("#a-head", el), html`${num(inUse)} ${inUse === 1 ? "account" : "accounts"} in use.<br><span>${num(left, 1)} of ${num(limit)} hours left.</span>`);
    render($("#a-strip", el), enabled.length ? quotaStrip(enabled.map(a => ({ name: a.name, left_h: a.left_h })), new Set(accts.filter(busy).map(a => a.name))) : "");
    render($("#a-filters", el), FILTERS.map(([k, n]) => html`<button class="k-chip" data-f="${k}" aria-pressed="${k === filter}">${n} <em>${accts.filter(match[k]).length}</em></button>`));
    const rows = accts.filter(match[filter]).filter(a => !q || (a.name + " " + (a.user || "")).toLowerCase().includes(q.toLowerCase()));
    render($("#a-list", el), !accts.length ? empty("No accounts yet.", "Add a Kaggle account. Kaggle gives GPUs only to phone-verified accounts.", html`<button class="k-btn k-primary" data-add>${icon("plus")}Add account</button>`)
      : rows.length ? rows.map(a => html`<div class="k-row k-hov"><span class="k-glyph">${a.name}</span>
        <span class="k-t"><b>${a.user ? "@" + a.user : a.name}${a.allowed === false ? html` <span class="k-badge">not used by kllm</span>` : ""}</b><small class="${attention(a) ? "k-wait-t" : ""}">${status(a)}</small></span>
        <span class="k-row-acts"><button class="k-btn k-xs" data-manage="${a.name}">Manage</button></span>
        <span class="k-v"><span>${num(Math.max(0, a.left_h), 1)} h left</span>${bar(Math.max(0, a.left_h) / a.limit_h, busy(a) ? "k-live" : a.left_h <= 0 ? "k-wait" : "")}</span></div>`)
      : empty("No account matches.", "Clear the search or pick another filter.", ""));
  };
  el.addEventListener("click", e => {
    // the whole row opens its sheet; buttons and links inside keep their own action
    const row = e.target.closest(".k-row.k-hov");
    if (row && !e.target.closest("a, button, input, select")) { const b = row.querySelector("[data-manage]"); if (b) { b.click(); return; } }
    const f = e.target.closest("[data-f]"); if (f) { filter = f.dataset.f; paint(); }
    const m = e.target.closest("[data-manage]"); if (m) manage(data.accounts.find(a => a.name === m.dataset.manage));
    if (e.target.closest("[data-add]")) addSheet();
  });
  $("#a-q", el).oninput = debounce(e => { q = e.target.value.trim(); paint(); }, 120);
  $("#a-add", el).onclick = () => addSheet();
  $("#a-sync", el).onclick = async e => { const r = await act(e.currentTarget, () => post("/admin/api/accounts/sync-quota", {})); if (r.ok) load(); };

  const manage = a => {
    const s = sheet({ title: a.name, sub: `${a.user ? "@" + a.user + " · " : ""}${status(a)}`, body: html`
      <div class="k-sheet-big"><b>${num(Math.max(0, a.left_h), 1)} h</b><span>left of ${a.limit_h} h this week</span></div>${bar(a.used_h / a.limit_h, "k-full")}
      <div class="k-kv" style="margin-top:18px">
        <div><span>Used this week</span><span>${num(a.used_h, 1)} h</span></div>
        <div><span>Burning now</span><span>${a.burn_h_per_h ? num(a.burn_h_per_h, 1) + " h per hour" : "nothing"}</span></div>
        ${a.exhaust_in_h ? html`<div><span>Runs out in</span><span>${dur(a.exhaust_in_h * 3600)}</span></div>` : ""}
        <div><span>Launches this week</span><span>${a.launches || 0}${a.fails ? `, ${a.fails} failed` : ""}</span></div>
        <div><span>Credentials</span><span class="${a.credentials && a.credentials.ok ? "" : "k-risk-t"}">${a.credentials && a.credentials.ok ? a.credentials.kind : "missing"}</span></div>
        <div><span>Quota synced</span><span>${a.quota_info && a.quota_info.synced_at ? new Date(a.quota_info.synced_at * 1000).toLocaleString("en-GB") : "estimated from launches"}</span></div></div>
      ${a.last_error ? html`<div class="k-error-box" style="margin-top:14px">${a.last_error}</div>` : ""}
      <div id="mg-confirm"></div>`,
      foot: html`<button class="k-btn k-sm" data-dis>${a.disabled ? "Enable" : "Disable"}</button>${a.error_active || a.fails ? html`<button class="k-btn k-sm" data-clr>Clear error</button>` : ""}
        <button class="k-btn k-sm" data-sync>Sync quota</button><button class="k-btn k-sm k-push" data-rm>Remove</button>` });
    const done = r => { if (r && r.ok) { s.close(); load(); ctx.shared.refresh(); } };
    $("[data-dis]", s.el).onclick = async e => done(await act(e.currentTarget, () => post("/admin/api/accounts/disable", { name: a.name, disabled: !a.disabled })));
    const c = $("[data-clr]", s.el); if (c) c.onclick = async e => done(await act(e.currentTarget, () => post("/admin/api/accounts/clear-error", { account: a.name })));
    $("[data-sync]", s.el).onclick = async e => done(await act(e.currentTarget, () => post("/admin/api/accounts/sync-quota", { account: a.name })));
    $("[data-rm]", s.el).onclick = () => {
      render($("#mg-confirm", s.el), html`<div class="k-confirm k-raised"><p><b>Remove ${a.name} from config.yaml?</b></p>
        <p class="k-sec">kllm stops using it. Its credential files stay unless you also delete them below.</p>
        ${(a.removable_credentials || []).length ? html`<label class="k-check"><input type="checkbox" id="rm-creds"> Also delete ${a.removable_credentials.join(", ")}</label>
          <input class="k-field" id="rm-type" placeholder="Type ${a.name} to delete the files" hidden>` : ""}
        <div class="k-acts" style="margin-top:12px"><button class="k-btn k-sm k-risk" data-yes>Remove ${a.name}</button><button class="k-btn k-sm" data-no>Keep it</button></div></div>`);
      const cb = $("#rm-creds", s.el); if (cb) cb.onchange = () => { $("#rm-type", s.el).hidden = !cb.checked; };
      $("[data-no]", s.el).onclick = () => render($("#mg-confirm", s.el), "");
      $("[data-yes]", s.el).onclick = async e => {
        const del = cb && cb.checked;
        done(await act(e.currentTarget, () => post("/admin/api/accounts/remove", { name: a.name, delete_credentials: !!del, confirm_credentials: del ? $("#rm-type", s.el).value.trim() : undefined })));
      };
    };
  };

  const addSheet = () => {
    const s = sheet({ title: "Add account", sub: "Paste a Kaggle API token. kllm checks it with Kaggle and saves it next to config.yaml.", body: html`<div class="k-form">
      <label><span>Kaggle API token</span><input class="k-field" id="ad-token" type="password" autocomplete="off" placeholder="from kaggle.com/settings, Create New Token"></label>
      <label><span>Name in kllm</span><input class="k-field" id="ad-name" placeholder="${data ? data.next_name : "kgN"}"></label>
      <p class="k-note">Kaggle gives GPUs only to phone-verified accounts. Names must match account_allow_prefixes in config.yaml.</p><div id="ad-err"></div></div>`,
      foot: html`<button class="k-btn k-primary" data-go>Add account</button>` });
    $("[data-go]", s.el).onclick = async e => {
      const r = await act(e.currentTarget, () => post("/admin/api/accounts/add", { token: $("#ad-token", s.el).value.trim(), name: $("#ad-name", s.el).value.trim() || undefined }));
      if (r.ok) { s.close(); load(); ctx.shared.refresh(); } else render($("#ad-err", s.el), errorBox(r.error));
    };
  };
  const stop = poll(load, 30000);
  return () => { stop(); fresh.stop(); };
}
