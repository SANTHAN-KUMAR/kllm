// Wall: every running kernel at a glance · the same numbers the Fleet slide-over Resources tab shows,
// one card per backend, fed by the kernel telemetry stream. Separate page on purpose: Fleet stays untouched.
import { get } from "../api.js";
import { html, render, $, dur, num, bytes, gb, bar, empty, loading, Fresh } from "../ui.js";
import { spark } from "../charts.js";
import { stream } from "../live.js";
import { speedRows, avgOf, byBackend } from "../speed.js";

function wallGrid(backends, pending, samples, statuses, speeds, prefill) {
  const cards = [];
  for (const x of backends) {
    const rows = samples[x.account] || [];
    const last = rows[rows.length - 1];
    const st = statuses[x.account] || "waiting";
    const spd = speeds["kaggle:" + x.account];
    const spdPre = prefill["kaggle:" + x.account];
    const dec = last && last.decode_tps != null ? { v: last.decode_tps, live: true }
              : spd != null ? { v: spd, live: false } : null;
    const pre = last && last.prefill_tps != null ? { v: last.prefill_tps, live: true }
              : spdPre != null ? { v: spdPre, live: false } : null;
    const head = html`<div class="k-wall-head"><b class="k-mono">${x.account}</b><i class="k-dot ${x.healthy ? "k-ok" : "k-risk"}"></i><span class="k-wall-model" title="${x.model_key}">${x.model_key}</span></div>
      <div class="k-wall-sub">${x.user ? "@" + x.user + " · " : ""}${dur(x.left_s)} left · up ${dur(x.up_s)}</div>`;
    if (st === "unsupported") {
      cards.push(html`<div class="k-wall-card k-raised">${head}<p class="k-note">This kernel has no resource telemetry (older template, or it was turned off). Relaunch with the current template to see GPU, memory, context and speeds here.</p><a class="k-btn k-sm" href="#fleet/${x.account}">Open kernel</a></div>`);
      continue;
    }
    if (!last) {
      cards.push(html`<div class="k-wall-card k-raised">${head}<p class="k-note">${st.startsWith("error") ? "Cannot reach the kernel: " + st.slice(7) + "." : "Waiting for the first sample. The kernel reports every 5 s."}</p><a class="k-btn k-sm" href="#fleet/${x.account}">Open kernel</a></div>`);
      continue;
    }
    const gpus = last.gpu || [];
    const memPct = last.cg_limit ? last.cg_used / last.cg_limit : null;
    const disk = (last.disk || {})["/kaggle/working"];
    const slots = last.slots || null;
    const ctxUsed = slots ? slots.reduce((a, q) => a + (q.used || 0), 0) : null;
    const ctxMax = slots && slots.length ? slots[0].n_ctx : null;
    cards.push(html`<div class="k-wall-card k-raised">${head}
      <div class="k-kv" style="margin-top:2px">
        ${gpus.map(g => html`<div><span>GPU ${g.i}${g.temp != null ? " · " + g.temp + "°C" : ""}</span><span>${(g.used_mib / 1024).toFixed(1)} of ${(g.total_mib / 1024).toFixed(1)} GB · ${g.util}% · ${g.power_w != null ? Math.round(g.power_w) + " W" : ""}</span></div>`)}
        <div><span>Memory</span><span><b>${last.cg_used ? gb(last.cg_used) : "?"}</b> of ${last.cg_limit ? gb(last.cg_limit) : "?"} GB</span></div>
      </div>
      ${memPct != null ? bar(memPct, `k-full ${memPct > 0.9 ? "k-wait" : ""}`) : ""}
      <div class="k-kv">
        ${disk ? html`<div><span>Disk, /kaggle/working</span><span>${gb(disk.used)} of ${gb(disk.total)} GB</span></div>` : ""}
        <div><span>CPU</span><span>${last.cpu_pct != null ? last.cpu_pct + "%" : "-"}${last.load1 != null ? " · load " + last.load1.toFixed(1) : ""}</span></div>
        <div><span>Network</span><span>${last.net_rx_bps != null ? `in ${bytes(last.net_rx_bps)}/s · out ${bytes(last.net_tx_bps)}/s` : "-"}</span></div>
        <div><span>Context</span><span>${slots ? `${num(ctxUsed)} of ${num(ctxMax)} · ${slots.length} slot${slots.length === 1 ? "" : "s"}` : "not reported"}</span></div>
        <div><span>Decode speed</span><span>${dec ? `${dec.v.toFixed(1)} t/s${dec.live ? "" : " est"}` : "-"}${pre ? ` · prefill ${pre.v.toFixed(1)} t/s${pre.live ? "" : " est"}` : ""}</span></div>
      </div>
      ${gpus.length ? spark(samples[x.account].slice(-120).map(q => ((q.gpu || [])[0] || {}).util || 0), { w: 320, h: 30, label: "GPU 0 utilisation" }) : ""}
      <a class="k-btn k-sm" href="#fleet/${x.account}">Open kernel</a>
    </div>`);
  }
  for (const p of pending) {
    cards.push(html`<div class="k-wall-card k-raised">
      <div class="k-wall-head"><b class="k-mono">${p.account}</b><i class="k-dot k-wait"></i><span class="k-wall-model">${p.model_key}</span></div>
      <div class="k-wall-sub">launching · ${p.age_s != null ? dur(p.age_s) : "now"} so far</div>
      <p class="k-note">${p.stage || "queued"}${p.failed ? " · failed" : ""}</p>
      <a class="k-btn k-sm" href="#starting/${p.account}">Open Starting</a>
    </div>`);
  }
  return cards.join("");
}

export async function mount(root, ctx) {
  let el = root;
  render(el, html`<div class="k-page k-wall">
    <div class="k-fleet-top"><div><span class="k-kicker">Wall · all kernels</span><h1 class="k-head k-s" id="w-head">Loading…</h1></div>
      <p class="k-note" style="margin:0;max-width:440px">One card per running kernel: GPU, memory, CPU, disk, network, context and speed. The same data as Fleet → kernel → Resources, refreshed live.</p></div>
    <div id="w-body" class="k-wall-grid"></div>
    <p class="k-fresh" id="w-fresh"></p></div>`);

  const samples = {}, statuses = {};
  const fresh = new Fresh($(".k-wall", el), 15000, $("#w-fresh", el));
  const renderAll = async () => {
    const b = ctx.shared.backends;
    if (!b) { render($("#w-body", el), loading()); return; }
    $("#w-head", el).innerHTML = `${b.backends.length} running<br><span>${b.pending.length} loading</span>`;
    if (!b.backends.length && !b.pending.length) {
      render($("#w-body", el), empty("Nothing is running.", "Launch a model from Fleet or send a client request with autoscale on."));
      return;
    }
    const rows = await speedRows();
    const by = byBackend(rows);
    const spd = {}, spdPre = {};
    for (const k in by) {
      const d = avgOf(by[k], "decode", 5); if (d != null) spd[k] = d;
      const p = avgOf(by[k], "prefill", 5); if (p != null) spdPre[k] = p;
    }
    $("#w-body", el).innerHTML = wallGrid(b.backends, b.pending || [], samples, statuses, spd, spdPre);
  };
  const onSample = d => {
    if (d.account && d.sample) {
      (samples[d.account] = samples[d.account] || []).push(d.sample);
      if (samples[d.account].length > 180) samples[d.account].shift();
      statuses[d.account] = d.status;
      fresh.mark();
      renderAll();
    }
  };
  // seed: latest samples per backend, then the stream keeps them live
  const seed = async () => {
    const b = ctx.shared.backends;
    if (!b) return;
    await Promise.all(b.backends.map(x =>
      get(`/admin/api/backends/${x.account}/telemetry?since=0&limit=120`).then(r => {
        if (r.ok) { samples[x.account] = r.data.samples || []; statuses[x.account] = r.data.status; }
      }).catch(() => {})));
  };
  const off = ctx.shared.on(renderAll);
  const st = stream("/admin/api/telemetry/stream", { named: ["sample"], onEvent: onSample });
  await seed();
  renderAll();
  return () => { off(); st(); fresh.stop(); };
}

export const screen = () => "wall";
