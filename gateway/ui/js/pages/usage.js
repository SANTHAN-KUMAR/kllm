// Usage: requests, tokens, speed and compute. GET /admin/api/metrics?range= (30 s), GET /admin/api/compute.
import { get, poll } from "../api.js";
import { html, render, $, $$, num, compact, dur, errorBox, loading, Fresh } from "../ui.js";
import { area, stacked, bars } from "../charts.js";

const RANGES = [["1h", "1 hour"], ["6h", "6 hours"], ["24h", "24 hours"], ["7d", "7 days"], ["14d", "14 days"], ["30d", "30 days"]];

export async function mount(el, ctx) {
  let range = ctx.route.query.range || "24h";
  render(el, html`<div class="k-page k-usage"><span class="k-kicker">Usage</span><h1 class="k-head" id="u-head">${""}</h1>
    <div class="k-chips" id="u-ranges" style="margin-top:26px" role="group" aria-label="Range"></div>
    <div id="u-body" class="k-staleable">${loading()}</div><p class="k-fresh" id="u-fresh"></p>
    <div class="k-sect"><h2>GPU compute this week</h2><span id="u-c-note"></span></div><div id="u-compute">${loading()}</div></div>`);
  const fresh = new Fresh($(".k-usage", el), 30000, $("#u-fresh", el));
  const paintRanges = () => render($("#u-ranges", el), RANGES.map(([k, n]) => html`<button class="k-chip" data-r="${k}" aria-pressed="${k === range}">${n}</button>`));
  paintRanges();
  const load = async () => {
    const r = await get(`/admin/api/metrics?range=${range}&tz=${-new Date().getTimezoneOffset()}`);
    const box = $("#u-body", el);
    if (!r.ok) { render(box, errorBox(r.error)); return; }
    fresh.mark();
    const m = r.data, t = m.totals;
    const series = ["1h", "6h", "24h"].includes(range) ? m.hourly : m.daily;
    const label = x => { const d = new Date(x.t * 1000); return series === m.hourly ? d.getHours().toString().padStart(2, "0") + ":00" : d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }); };
    const xl = series.length ? [label(series[0]), label(series[Math.floor(series.length / 2)]), label(series[series.length - 1])] : [];
    render($("#u-head", el), t.n ? html`${num(t.n)} ${t.n === 1 ? "request" : "requests"}.<br><span>${Math.round((t.cache_hit || 0) * 100)}% came from cache.</span>` : html`No requests yet.<br><span>Send one from Chat or any client.</span>`);
    render(box, html`<div class="k-stats" style="margin-top:34px">
        <div><b>${num(t.req_per_hour || 0, 1)}</b><small>requests per hour</small></div>
        <div><b>${t.ttft_avg != null ? num(t.ttft_avg, 1) : "none"}<small>${t.ttft_avg != null ? "s" : ""}</small></b><small>average first token</small></div>
        <div><b>${t.gen_tps_avg != null ? num(t.gen_tps_avg, 1) : "none"}</b><small>tokens/s generated</small></div>
        <div><b>${compact(t.out || 0)}</b><small>tokens out</small></div>
        <div><b>${t.err || 0}</b><small>errors</small></div></div>
      <div class="k-two" style="margin-top:36px">
        <div><p class="k-chart-title">Requests per ${series === m.hourly ? "hour" : "day"}</p>${area(series.map(x => x.n), { label: "Requests over time", xlabels: xl })}</div>
        <div><p class="k-chart-title">Prompt tokens in, new versus cached</p>${stacked(series.map(x => ({ a: x.new_in, b: x.cached_in })), { label: "New and cached prompt tokens", xlabels: xl })}
          <div class="k-legend"><span><i style="background:var(--k-text)"></i>new ${compact(t.new_in)}</span><span><i style="background:rgb(var(--k-lead))"></i>cached ${compact(t.cached_in)}</span></div></div></div>
      <div class="k-two" style="margin-top:28px">
        <div><p class="k-chart-title">Generation speed by prompt size, median tokens/s</p>${bars((m.ctx_buckets || []).map(b => ({ label: b.label, v: b.tps_p50 || 0 })), { label: "Median generation speed by prompt size" })}</div>
        <div><p class="k-chart-title">Latency</p><div class="k-kv">
          <div><span>First token, median</span><span>${m.latency.ttft.p50 != null ? m.latency.ttft.p50 + " s" : "none"}</span></div>
          <div><span>First token, 90th percentile</span><span>${m.latency.ttft.p90 != null ? m.latency.ttft.p90 + " s" : "none"}</span></div>
          <div><span>First token, 99th percentile</span><span>${m.latency.ttft.p99 != null ? m.latency.ttft.p99 + " s" : "none"}</span></div>
          <div><span>Generation, median</span><span>${m.latency.gen_tps.p50 != null ? m.latency.gen_tps.p50 + " tokens/s" : "none"}</span></div>
          <div><span>Thinking share of output</span><span>${t.think_share != null ? Math.round(t.think_share * 100) + "%" : "none"}</span></div></div></div></div>
      <div class="k-sect"><h2>By model</h2></div>
      ${m.by_model.length ? html`<div class="k-scroll-x"><table class="k-table"><thead><tr><th>Model</th><th class="k-r">Requests</th><th class="k-r">Cached in</th><th class="k-r">New in</th><th class="k-r">Out</th><th class="k-r">First token</th><th class="k-r">Speed</th></tr></thead><tbody>
        ${m.by_model.map(x => html`<tr><td>${x.key}</td><td class="k-r">${num(x.n)}</td><td class="k-r">${compact(x.cached_in)}</td><td class="k-r">${compact(x.new_in)}</td><td class="k-r">${compact(x.out)}</td>
          <td class="k-r">${x.ttft_n ? (x.ttft_sum / x.ttft_n).toFixed(1) + " s" : ""}</td><td class="k-r">${x.gen_n ? (x.gen_sum / x.gen_n).toFixed(1) + " tokens/s" : ""}</td></tr>`)}</tbody></table></div>`
        : html`<p class="k-note">No model has served a request in this range.</p>`}`);
  };
  el.addEventListener("click", e => { const b = e.target.closest("[data-r]"); if (b) { range = b.dataset.r; paintRanges(); history.replaceState(null, "", `#usage?range=${range}`); load(); } });
  const stop = poll(load, 30000);
  get("/admin/api/compute").then(r => {
    const box = $("#u-compute", el);
    if (!r.ok) { render(box, errorBox(r.error)); return; }
    const p = r.data.pool;
    $("#u-c-note", el).textContent = `resets in ${dur((r.data.week_resets_in_h || 0) * 3600)}`;
    render(box, html`<div class="k-stats"><div><b>${num(p.used_h, 1)}<small>h</small></b><small>used of ${num(p.limit_h)} h</small></div>
      <div><b>${num(p.burn_now || 0, 1)}</b><small>hours burning per hour now</small></div>
      <div><b>${p.exhaust_in_h != null ? dur(p.exhaust_in_h * 3600) : "never"}</b><small>until the pool runs out at this rate</small></div>
      <div><b>${p.launches || 0}</b><small>launches this week</small></div><div><b>${num(p.standby_h || 0, 1)}<small>h</small></b><small>idle on standby</small></div></div>`);
  });
  return () => { stop(); fresh.stop(); };
}
