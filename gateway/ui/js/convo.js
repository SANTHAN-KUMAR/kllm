// Assistant-style conversation used by Chat, Image and Video: history column, model menu, a centred thread,
// a composer docked at the bottom (centred when the thread is empty), suggestions, hover actions, edit, retry,
// stop, scroll-to-end. Every value shown comes from the relay or the stream (FRONTEND_GUIDE "Is everything live?").
// Threads live in IndexedDB in this browser; generated media also live on the relay (GET /admin/api/media).
import { get, post } from "./api.js";
import { html, raw, render, $, $$, icon, esc, md, num, dur, when, bytes, toast, copy, pref, setPref, sheet, act, errorBox } from "./ui.js";

// ---- IndexedDB: threads {id, kind, title, model, messages[], updated, system} ---------------------------------------
const DB = { db: null };
function idb() {
  if (DB.db) return Promise.resolve(DB.db);
  return new Promise((ok, bad) => {
    let req;
    try { req = indexedDB.open("kllm-chat", 1); } catch (e) { bad(e); return; }
    req.onupgradeneeded = () => req.result.createObjectStore("chats", { keyPath: "id" });
    req.onsuccess = () => { DB.db = req.result; ok(DB.db); };
    req.onerror = () => bad(req.error);
  });
}
async function tx(mode, fn) {
  try {
    const db = await idb();
    return await new Promise((ok, bad) => {
      const t = db.transaction("chats", mode), s = t.objectStore("chats");
      const r = fn(s);
      t.oncomplete = () => ok(r && r.result);
      t.onerror = () => bad(t.error);
    });
  } catch { return null; }
}
const listThreads = async kind => ((await tx("readonly", s => s.getAll())) || []).filter(c => (c.kind || "chat") === kind).sort((a, b) => b.updated - a.updated);
const saveThread = c => tx("readwrite", s => s.put(c));
const deleteThread = id => tx("readwrite", s => s.delete(id));
const newId = () => Date.now().toString(36) + Math.floor(performance.now() * 1000).toString(36).slice(-4);

const KIND = {
  chat: { title: "Chat", noun: "chat", nouns: "Chats", hello: ["What should we work on?", "Every answer runs on your Kaggle pool."],
    placeholder: "Message kllm", suggestions: ["Explain what this error means and how to fix it", "Write a Python script that renames files by date",
      "Summarise the trade-offs between Q4_K_M and IQ3 quants", "Review this function for bugs"] },
  image: { title: "Image", noun: "image", nouns: "Images", hello: ["Describe an image.", "It is drawn on your Kaggle pool."],
    placeholder: "Describe the image you want", suggestions: ["A lighthouse on a cliff at dawn, film photo", "A server rack lit by one cold blue light, long exposure",
      "An isometric map of a tiny harbour town", "A close portrait in soft window light, 85 mm"] },
  video: { title: "Video", noun: "video", nouns: "Videos", hello: ["Describe a shot.", "It renders on your Kaggle pool."],
    placeholder: "Describe the shot: subject, motion, camera", suggestions: ["Slow drone push over a harbour at sunrise, mist lifting",
      "A rainy street at night, slow dolly, neon reflections", "Waves rolling onto black sand, low angle, golden hour", "A paper boat drifting down a gutter stream, macro"] },
};
const SHAPES = { image: [["Default", ""], ["Square", "1024x1024"], ["Wide", "1344x768"], ["Tall", "768x1344"]],
  video: [["Default", ""], ["Wide", "832x480"], ["Tall", "480x832"], ["Square", "640x640"]] };

// One running instance per kind (the pages are kept alive by app.js): studio.js asks it to open another thread.
export const INST = {};
const STRENGTH = [["Subtle", 0.35], ["Balanced", 0.6], ["Strong", 0.85]];

// Phone photos are 3 to 12 MB: send at most 1536 px on the long side (JPEG), which every model downsizes to anyway.
function shrink(file, max = 1536) {
  return new Promise(ok => {
    const r = new FileReader();
    r.onload = () => {
      const img = new Image();
      img.onload = () => {
        const s = Math.min(1, max / Math.max(img.width, img.height));
        if (s === 1 && file.size < 1.5e6) { ok(r.result); return; }
        const c = document.createElement("canvas"); c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        ok(c.toDataURL("image/jpeg", 0.9));
      };
      img.onerror = () => ok(r.result);
      img.src = r.result;
    };
    r.readAsDataURL(file);
  });
}

export async function mountConvo(el, ctx, kind) {
  const K = KIND[kind];
  let threads = await listThreads(kind);
  let thread = null, models = [], running = [], pending = [], defaultModel = null, attachments = [];
  const jobs = new Map(), queues = new Map();          // per thread: the running job, and messages waiting for it
  const busyHere = () => (thread && jobs.get(thread.id)) || null;
  const opt = {
    chat: { thinking: pref("thinking", true), temp: pref("chatTemp", 0.7), max: pref("chatMax", 4096) },
    image: { size: pref("imgSize", ""), n: pref("imgN", 1), steps: "", seed: "", strength: pref("imgStrength", 0.6) },
    video: { size: pref("vidSize", ""), frames: "", fps: "", steps: "", seed: "" },
  }[kind];
  // on a narrow screen the history is a drawer: start closed whatever the saved preference says
  const narrow = matchMedia("(max-width: 1000px)");
  const ui = { hist: narrow.matches ? false : pref(kind + "Hist", true), full: false, useLast: true };

  render(el, html`<div class="k-page k-tool k-cv ${ui.hist ? "" : "k-cv-nohist"} ${kind !== "chat" ? "k-cv-studio" : ""}" data-kind="${kind}">
    <aside class="k-cv-side" aria-label="${K.nouns}">
      <div class="k-cv-side-top"><b>${K.nouns}</b><button class="k-icon-btn" id="cv-new" title="New ${K.noun} (Ctrl Shift O)" aria-label="New ${K.noun}">${icon("compose")}</button></div>
      <label class="k-cv-search">${icon("search")}<input id="cv-q" placeholder="Search" aria-label="Search ${K.nouns.toLowerCase()}"></label>
      <nav id="cv-list" class="k-cv-list"></nav>
      <p class="k-cv-side-note">${kind === "chat" ? "Kept in this browser." : "Threads are kept in this browser. Every output is also saved on the relay."}</p>
    </aside>
    ${kind !== "chat" ? html`<aside class="k-st-opts" id="cv-opts" aria-label="${kind === "image" ? "Image" : "Video"} settings"></aside>` : ""}
    <section class="k-cv-main">
      <header class="k-cv-head">
        <button class="k-icon-btn" id="cv-hist" aria-label="Show or hide ${K.nouns.toLowerCase()}" title="${K.nouns}">${icon("panel")}</button>
        <div class="k-cv-model"><button class="k-cv-model-btn" id="cv-model-btn" aria-haspopup="listbox" aria-expanded="false"><i class="k-dot" id="cv-dot"></i><span id="cv-model-name">Loading models</span>${icon("down")}</button>
          <div class="k-menu k-float" id="cv-menu" role="listbox" hidden></div></div>
        <span class="k-cv-head-meta" id="cv-meta"></span>
        <button class="k-icon-btn" id="cv-new2" aria-label="New ${K.noun}" title="New ${K.noun}">${icon("compose")}</button>
        <button class="k-icon-btn" id="cv-full" aria-pressed="false" aria-label="Full screen" title="Full screen">${icon("max")}</button>
      </header>
      <div class="k-cv-scroll" id="cv-scroll"><div class="k-cv-thread" id="cv-thread" aria-live="polite"></div></div>
      <button class="k-cv-tobottom k-float" id="cv-down" aria-label="Scroll to the latest message" hidden>${icon("arrowdown")}</button>
      <div class="k-cv-dock" id="cv-dock">
        <form class="k-cv-comp k-raised" id="cv-form">
          <div class="k-cv-queue" id="cv-queue"></div>
          <div class="k-cv-att" id="cv-att"></div>
          <textarea id="cv-in" rows="1" aria-label="${K.placeholder}" placeholder="${K.placeholder}"></textarea>
          <div class="k-cv-tools">
            <label class="k-cv-tool" id="cv-attach" title="${kind === "video" ? "Start from an image" : kind === "image" ? "Reference image" : "Add an image"}">${icon("plus")}<input type="file" id="cv-file" accept="image/*" ${kind === "chat" ? "multiple" : ""} hidden><span class="k-sr">Add image</span></label>
            ${kind === "chat" ? html`<button type="button" class="k-cv-tool" id="cv-opt-btn" aria-expanded="false" title="Options">${icon("sliders")}<span>Options</span></button>` : ""}
            <span class="k-cv-chips" id="cv-chips"></span>
            <button class="k-send" id="cv-send" type="submit" aria-label="Send" title="Send (Enter)">${icon("send")}</button>
          </div>
          <div class="k-cv-pop k-float" id="cv-pop" hidden></div>
        </form>
        <p class="k-cv-foot" id="cv-foot"></p>
      </div>
    </section></div>`);

  const root = $(".k-cv", el), scroller = $("#cv-scroll", el), threadEl = $("#cv-thread", el), input = $("#cv-in", el);

  // ---- models -----------------------------------------------------------------------------------------------------
  const loadModels = async () => {
    if (kind === "chat") {
      const r = await get("/admin/api/state");
      if (!r.ok) { render($("#cv-foot", el), errorBox(r.error)); return; }
      running = r.data.pool.filter(p => p.healthy).map(p => p.model_key);
      pending = (r.data.pending || []).map(p => p.model_key);
      models = r.data.models.filter(m => m.kind === "chat");
      defaultModel = r.data.default_model;
    } else {
      const r = await get("/admin/api/models");
      if (!r.ok) { render($("#cv-foot", el), errorBox(r.error)); return; }
      models = r.data.models.filter(m => m.kind === kind && !(m.problems || []).length);
      defaultModel = r.data[`default_${kind}_model`] || (models[0] && models[0].key);
      const b = ctx.shared.backends;
      running = b ? b.backends.filter(x => x.healthy).map(x => x.model_key) : [];
      pending = b ? b.pending.map(x => x.model_key) : [];
    }
    const want = (thread && thread.model) || pref(kind + "Model", null);
    const pick = [want, running.find(k => models.some(m => m.key === k)), defaultModel, models[0] && models[0].key].find(k => k && models.some(m => m.key === k));
    current = pick || null;
    paintModel();
  };
  let current = null;
  const model = () => models.find(m => m.key === current);
  const isVision = m => m && ((m.tags || []).some(t => /vision|vl/i.test(t)) || /vision|vl\b/i.test(m.key));
  const where = key => {
    const b = ctx.shared.backends;
    const on = b && b.backends.find(x => x.model_key === key && x.healthy);
    const pend = b && b.pending.find(x => x.model_key === key);
    return { on, pend };
  };
  // what a waiting render is actually waiting for: a serving kernel, a launch in progress, or autoscale
  const genState = key => {
    const { on, pend } = where(key);
    if (on) return `Generating with ${key} on ${on.account}`;
    if (pend) { const st = (pend.steps || []).find(s => s.state === "active" || s.state === "failed"); return `Starting ${key} on ${pend.account}: ${st ? st.label.toLowerCase() : "launching"}. Runs when ready`; }
    const b = ctx.shared.backends;
    return b && b.autoscale ? `Waiting for autoscale to start ${key}` : `Nothing serves ${key} and autoscale is off`;
  };
  const paintModel = () => {
    const m = model();
    $("#cv-model-name", el).textContent = m ? m.key : models.length ? "Pick a model" : `No ${kind} models`;
    const st = m ? where(m.key) : {};
    $("#cv-dot", el).className = "k-dot " + (m && (running.includes(m.key) || st.on) ? "k-ok" : m && (pending.includes(m.key) || st.pend) ? "k-wait" : "");
    $("#cv-attach", el).hidden = kind === "chat" ? !isVision(m) : false;
    paintFoot(); paintChips();
  };
  const paintFoot = () => {
    const m = model(), foot = $("#cv-foot", el), b = ctx.shared.backends;
    if (!models.length) { render(foot, html`No ${kind} model is set up. <a href="#models">Add one on the Models page</a>.`); return; }
    if (!m) { foot.textContent = ""; return; }
    const { on, pend } = where(m.key);
    const ctxNote = kind === "chat" && m.ctx ? ` · ${thread && thread.lastPrompt ? `${(thread.lastPrompt / 1000).toFixed(1)}K of ` : ""}${Math.round(m.ctx / 1024)}K context` : "";
    if (on) render(foot, html`Runs on <b>${on.account}</b> through your relay${ctxNote}.`);
    else if (pend) render(foot, html`${m.key} is starting on <b>${pend.account}</b>. Your message waits for it${ctxNote}.`);
    else if (b && b.autoscale) render(foot, html`Nothing serves ${m.key} yet. Sending starts it on a free account, about 6 to 10 min${ctxNote}.`);
    else render(foot, html`Nothing serves ${m.key} and autoscale is off. <a href="#launch?model=${encodeURIComponent(m.key)}">Launch it</a> first.`);
    $("#cv-meta", el).textContent = thread && thread.messages.length ? thread.title : "";
  };
  const menu = () => {
    const box = $("#cv-menu", el), btn = $("#cv-model-btn", el);
    const open = box.hidden;
    box.hidden = !open; btn.setAttribute("aria-expanded", open);
    if (!open) return;
    const on = models.filter(m => running.includes(m.key) || where(m.key).on), rest = models.filter(m => !on.includes(m));
    const item = m => html`<button role="option" data-model="${m.key}" aria-selected="${m.key === current}"><b>${m.key}</b>
      <small>${[m.ctx ? Math.round(m.ctx / 1024) + "K" : "", m.quant, m.vram_gb ? "~" + m.vram_gb + " GB" : "", isVision(m) ? "vision" : "", m.key === defaultModel ? "default" : ""].filter(Boolean).join(" · ")}</small></button>`;
    render(box, html`${on.length ? html`<span>Running</span>${on.map(item)}` : ""}${rest.length ? html`<span>${on.length ? "Starts when you send" : "None running. Sending starts one"}</span>${rest.map(item)}` : ""}
      ${!models.length ? html`<p class="k-note" style="padding:12px">No ${kind} models. Add one on the Models page.</p>` : ""}`);
    const s = $("[aria-selected=true]", box); if (s) s.scrollIntoView({ block: "nearest" });
  };

  // ---- options popover and chips ---------------------------------------------------------------------------------------
  const paintChips = () => {
    const c = [];
    if (kind === "chat") { if (!opt.thinking) c.push("Thinking off"); if (thread && thread.system) c.push("System prompt"); if (+opt.temp !== 0.7) c.push(`Temperature ${opt.temp}`); }
    else {
      const sh = SHAPES[kind].find(s => s[1] === opt.size); if (opt.size) c.push(sh ? sh[0] : opt.size);
      if (kind === "image" && +opt.n > 1) c.push(`${opt.n} images`);
      if (kind === "image" && (attachments.length || startsFromLast())) c.push(STRENGTH.find(s => s[1] === +opt.strength)?.[0] + " change");
      if (opt.steps) c.push(`${opt.steps} steps`); if (opt.seed !== "") c.push(`seed ${opt.seed}`);
      if (kind === "video" && opt.frames) c.push(`${opt.frames} frames`);
    }
    const last = startsFromLast();
    render($("#cv-chips", el), html`${last ? html`<span class="k-cv-chip k-cv-chip-x"><img src="${last}" alt="">Starts from the last image<button type="button" data-nolast aria-label="Start fresh instead" title="Start fresh instead">${icon("x")}</button></span>` : ""}${c.map(x => html`<span class="k-cv-chip">${x}</span>`)}`);
  };
  // Image threads: a follow-up without an attachment edits the last image (img2img), the way an assistant would
  const lastImage = th => {
    if (!th) return null;
    for (let i = th.messages.length - 1; i >= 0; i--) {
      const m = th.messages[i];
      if (m.role === "assistant" && m.items && m.items.length) return m.items[0];
      if (m.role === "user" && m.image) return m.image;
    }
    return null;
  };
  const startsFromLast = () => kind === "image" && ui.useLast && !attachments.length ? lastImage(thread) : null;
  const field = (id, label, value, attrs = "") => html`<label class="k-cv-f"><span>${label}</span><input class="k-field" id="${id}" type="number" value="${value}" ${raw(attrs)}></label>`;
  // the generation controls for image and video, shared by the studio column (always visible) and the popover
  const studioControlsHtml = () => html`<div class="k-cv-f k-cv-wide"><span>Shape</span><div class="k-seg" id="o-shape">${SHAPES[kind].map(([n, v]) => html`<button type="button" data-size="${v}" aria-pressed="${opt.size === v}" title="${v || "model default"}">${n}</button>`)}</div></div>
      ${kind === "image" ? html`<div class="k-cv-f k-cv-wide"><span>How much to change a starting image</span><div class="k-seg" id="o-str">${STRENGTH.map(([n, v]) => html`<button type="button" data-str="${v}" aria-pressed="${+opt.strength === v}">${n}</button>`)}</div></div>` : ""}
      ${kind === "image" ? html`<div class="k-cv-f k-cv-wide"><span>Images per prompt</span><div class="k-seg" id="o-n">${[1, 2, 3, 4].map(v => html`<button type="button" data-n="${v}" aria-pressed="${+opt.n === v}">${v}</button>`)}</div></div>` : ""}
      <div class="k-cv-pgrid">${field("o-steps", "Steps", opt.steps, 'min="1" max="100" placeholder="model default"')}${field("o-seed", "Seed", opt.seed, 'min="0" placeholder="random"')}
        ${kind === "video" ? html`${field("o-frames", "Frames", opt.frames, 'min="9" max="241" placeholder="default"')}${field("o-fps", "Frames a second", opt.fps, 'min="4" max="60" placeholder="default"')}` : ""}</div>`;
  const popHtml = () => kind === "chat" ? html`
      <div class="k-cv-prow"><span><b>Thinking</b><small>Show the model's reasoning while it works</small></span><button type="button" class="k-toggle" id="o-think" role="switch" aria-checked="${opt.thinking}" aria-label="Thinking"></button></div>
      <label class="k-cv-f k-cv-wide"><span>System prompt for this chat</span><textarea class="k-textarea" id="o-sys" rows="3" placeholder="For example: answer in short paragraphs">${thread ? thread.system || "" : pref("chatSystem", "")}</textarea></label>
      <div class="k-cv-pgrid">${field("o-temp", "Temperature", opt.temp, 'min="0" max="2" step="0.1"')}${field("o-max", "Max tokens", opt.max, 'min="64" max="131072" step="256"')}</div>`
    : studioControlsHtml();
  const togglePop = force => {
    const pop = $("#cv-pop", el), btn = $("#cv-opt-btn", el);
    if (kind !== "chat") return;
    const open = force != null ? force : pop.hidden;
    if (open) render(pop, popHtml());
    pop.hidden = !open; btn.setAttribute("aria-expanded", open);
  };
  // image and video: the same controls, always visible in the studio column beside the results
  const paintStudio = () => {
    if (kind === "chat") return;
    const box = $("#cv-opts", el); if (!box) return;
    render(box, html`<div class="k-st-opts-head"><b>${kind === "image" ? "Image" : "Video"} settings</b><span class="k-note">Saved as you change them</span></div>
      <div class="k-st-opts-body">${studioControlsHtml()}
        ${kind === "image" ? html`<div class="k-cv-f k-cv-wide"><span>Start from an image, optional</span><div class="k-st-drop" id="o-drop" role="button" tabindex="0" aria-label="Drop a starting image, or click to pick one"></div></div>` : ""}
      </div>
      <p class="k-note k-st-opts-foot">${kind === "image" ? "Describe the image and press Send. A reference image makes it edit that image instead of starting from noise." : "Describe the clip and press Send. Frames and seed fix the motion to the frames you pick."}</p>`);
    paintDrop();
  };
  const paintDrop = () => {
    if (kind === "chat") return;
    const drop = $("#o-drop", el); if (!drop) return;
    if (attachments.length) {
      render(drop, html`<div class="k-st-drop-has"><img src="${attachments[0]}" alt="Starting image"><span><b>Starting from this image.</b><small>Describe the change you want, then press Send.</small><br><button type="button" class="k-link" data-rm="0">Remove it</button></span></div>`);
    } else {
      render(drop, html`<span class="k-st-drop-empty">${icon("plus")}<b>Drop a starting image</b><small>paste one, or click to pick</small></span>`);
    }
  };
  const bindOpts = container => {
    if (!container) return;
    container.addEventListener("click", e => {
      const t = e.target;
      if (t.id === "o-think") { opt.thinking = !opt.thinking; t.setAttribute("aria-checked", opt.thinking); setPref("thinking", opt.thinking); ctx.store.thinking = opt.thinking; $$(".k-cv-think", el).forEach(x => { x.hidden = !opt.thinking; }); }
      const sz = t.closest("[data-size]"); if (sz) { opt.size = sz.dataset.size; setPref(kind === "image" ? "imgSize" : "vidSize", opt.size); $$("[data-size]", el).forEach(x => x.setAttribute("aria-pressed", x === sz)); }
      const st = t.closest("[data-str]"); if (st) { opt.strength = +st.dataset.str; setPref("imgStrength", opt.strength); $$("[data-str]", el).forEach(x => x.setAttribute("aria-pressed", x === st)); }
      const nn = t.closest("[data-n]"); if (nn) { opt.n = +nn.dataset.n; setPref("imgN", opt.n); $$("[data-n]", el).forEach(x => x.setAttribute("aria-pressed", x === nn)); }
      const dr = t.closest("#o-drop"); if (dr && !t.closest("[data-rm]")) { const fi = $("#cv-file", el); if (fi) fi.click(); }
      paintChips();
    });
    container.addEventListener("input", e => {
      const t = e.target, v = t.value;
      if (t.id === "o-sys") { if (thread) thread.system = v; else setPref("chatSystem", v); }
      if (t.id === "o-temp") { opt.temp = v === "" ? 0.7 : +v; setPref("chatTemp", opt.temp); }
      if (t.id === "o-max") { opt.max = v === "" ? 4096 : +v; setPref("chatMax", opt.max); }
      if (t.id === "o-steps") opt.steps = v; if (t.id === "o-seed") opt.seed = v;
      if (t.id === "o-frames") opt.frames = v; if (t.id === "o-fps") opt.fps = v;
      paintChips();
    });
    container.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { const dr = e.target.closest("#o-drop"); if (dr) { e.preventDefault(); const fi = $("#cv-file", el); if (fi) fi.click(); } } });
  };
  bindOpts($("#cv-pop", el));
  bindOpts($("#cv-opts", el));
  paintStudio();

  // ---- sidebar ----------------------------------------------------------------------------------------------------------
  const paintList = () => {
    const q = $("#cv-q", el).value.trim().toLowerCase();
    const list = threads.filter(c => !q || c.title.toLowerCase().includes(q));
    const groups = [];
    const now = new Date(); const day = d => new Date(d).toDateString();
    list.forEach(c => {
      const age = (now - c.updated) / 86400000;
      const g = day(c.updated) === now.toDateString() ? "Today" : age < 2 ? "Yesterday" : age < 7 ? "This week" : "Earlier";
      let grp = groups.find(x => x.g === g); if (!grp) groups.push(grp = { g, items: [] }); grp.items.push(c);
    });
    render($("#cv-list", el), list.length ? groups.map(({ g, items }) => html`<span class="k-cv-g">${g}</span>${items.map(c => html`
      <div class="k-cv-item ${thread && thread.id === c.id ? "k-on" : ""}"><a href="#${kind}/${c.id}" title="${c.title}">${c.title}</a>
        <button class="k-icon-btn" data-del="${c.id}" aria-label="Delete ${c.title}" title="Delete">${icon("trash")}</button></div>`)}`)
      : html`<p class="k-cv-side-note">${q ? "No match." : `Your ${K.nouns.toLowerCase()} appear here.`}</p>`);
  };

  // ---- thread rendering ------------------------------------------------------------------------------------------------
  const userText = m => Array.isArray(m.content) ? m.content.filter(c => c.type === "text").map(c => c.text).join("\n") : (m.prompt || m.content || "");
  const userImgs = m => Array.isArray(m.content) ? m.content.filter(c => c.type === "image_url").map(c => c.image_url.url) : (m.image ? [m.image] : []);
  const statLine = s => [s.tokens != null ? `${num(s.tokens)} tokens` : "", s.tps ? `${s.tps.toFixed(1)} tokens/s` : "", s.ttft != null ? `${s.ttft.toFixed(s.ttft < 1 ? 2 : 1)} s to first token` : "",
    s.cached != null && s.prompt ? `${Math.round((100 * s.cached) / s.prompt)}% cached` : "", s.took ? `${dur(s.took)}` : "", s.size || "", s.backend ? s.backend.replace("kaggle:", "") : ""].filter(Boolean);
  const msgHtml = (m, i) => {
    if (m.role === "user") return html`<div class="k-cv-u" data-i="${i}">
      <div class="k-cv-bub">${userImgs(m).map(u => html`<img src="${u}" alt="Attached image">`)}<div class="k-cv-text">${userText(m)}</div></div>
      <div class="k-cv-acts"><button class="k-icon-btn" data-edit="${i}" title="Edit" aria-label="Edit message">${icon("edit")}</button><button class="k-icon-btn" data-copyu="${i}" title="Copy" aria-label="Copy message">${icon("copy")}</button></div></div>`;
    if (m.role !== "assistant") return "";
    return html`<div class="k-cv-a" data-i="${i}">${aBody(m, i)}</div>`;
  };
  const aBody = (m, i) => {
    const s = m.stats || {};
    let body;
    if (m.kind === "image" || m.kind === "video") {
      body = m.live ? html`<div class="k-cv-gen ${m.kind}"><div class="k-shimmer"></div><span><i class="k-dot k-wait"></i><span data-genstate="${m.t0}">${genState(m.model)}</span><em data-elapsed="${m.t0}">${dur((Date.now() - m.t0) / 1000)}</em></span></div>`
        : m.items && m.items.length ? html`<div class="k-cv-media k-n${Math.min(m.items.length, 4)}">${m.items.map((u, j) => m.kind === "video"
          ? html`<video src="${u}" controls preload="metadata" ${j === 0 ? "autoplay muted loop" : ""}></video>`
          : html`<button class="k-cv-shot" data-view="${i}:${j}" aria-label="Open image ${j + 1}"><img src="${u}" alt="${m.prompt || "Generated image"}" loading="lazy"></button>`)}</div>` : "";
    } else {
      body = html`${m.reasoning ? html`<details class="k-cv-think" ${m.live && !m.content ? "open" : ""} ${opt.thinking === false ? "hidden" : ""}><summary><i class="k-dot k-lead ${m.live && !m.content ? "" : "k-still"}"></i>${m.live && !m.content ? "Thinking" : `Thought for ${s.think_s >= 1 ? s.think_s.toFixed(0) + " s" : "under a second"}`}</summary><div class="k-cv-think-body">${m.reasoning}</div></details>` : ""}
        <div class="k-md">${m.content ? raw(md(m.content)) : m.live && !m.reasoning ? html`<span class="k-cv-typing"><i></i><i></i><i></i></span>` : ""}</div>`;
    }
    return html`${body}${m.error ? html`<div class="k-error-box">${m.error}</div>` : ""}
      ${!m.live ? html`<div class="k-cv-acts k-cv-acts-a">
        ${m.kind === "chat" || !m.kind ? html`<button class="k-icon-btn" data-copya="${i}" title="Copy" aria-label="Copy answer">${icon("copy")}</button>` : ""}
        <button class="k-icon-btn" data-retry="${i}" title="Regenerate" aria-label="Regenerate">${icon("rotate")}</button>
        <span class="k-cv-stats">${statLine(s).join(" · ")}</span></div>` : ""}`;
  };
  const decorate = node => {
    $$("pre", node).forEach(pre => {
      if (pre.previousElementSibling && pre.previousElementSibling.classList.contains("k-cv-codehead")) return;
      const head = document.createElement("div");
      head.className = "k-cv-codehead";
      render(head, html`<span>${pre.dataset.lang || "code"}</span><button class="k-link" data-copycode>Copy</button>`);
      pre.before(head);
    });
  };
  const nearEnd = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120;
  const toEnd = () => { scroller.scrollTop = scroller.scrollHeight; };
  const paintThread = () => {
    const main = $(".k-cv-main", el), dockEl = $("#cv-dock", el);
    if (dockEl.parentElement !== main) main.append(dockEl);
    const emptyNow = !thread || !thread.messages.length;
    root.classList.toggle("k-cv-empty", emptyNow);
    if (emptyNow) {
      render(threadEl, html`<div class="k-cv-hello"><h1>${K.hello[0]}<br><span>${K.hello[1]}</span></h1></div>`);
      const dock = $("#cv-dock", el);
      threadEl.append(dock);
      const sug = document.createElement("div"); sug.className = "k-cv-sugs";
      render(sug, K.suggestions.map(t => html`<button type="button" class="k-cv-sug" data-sug="${t}">${t}</button>`));
      threadEl.append(sug);
      if (kind !== "chat") { const rec = document.createElement("div"); rec.id = "cv-recent"; threadEl.append(rec); loadRecent(); }
      return;
    }
    render(threadEl, thread.messages.map(msgHtml));
    decorate(threadEl);
    toEnd();
  };
  const paintLive = (th, msg) => {
    if (th !== thread) return;                     // a job in another thread keeps running; it paints when opened
    const i = thread.messages.indexOf(msg);
    const node = $(`.k-cv-a[data-i="${i}"]`, threadEl);
    if (!node) { paintThread(); return; }
    const end = nearEnd();
    const det = $("details", node); const wasOpen = det ? det.open : null;
    render(node, aBody(msg, i));
    const det2 = $("details", node); if (det2 && wasOpen != null && !msg.live) det2.open = wasOpen;
    decorate(node);
    if (end) toEnd();
  };
  const loadRecent = async () => {
    const r = await get(`/admin/api/media?kind=${kind}&limit=12`);
    const box = $("#cv-recent", el); if (!box) return;
    if (!r.ok || !r.data.media.length) { box.innerHTML = ""; return; }
    const rows = r.data.media;
    render(box, html`<div class="k-cv-recent-h"><b>Recent ${K.nouns.toLowerCase()}</b><span>${rows.length} saved on the relay</span></div>
      <div class="k-cv-recent">${rows.map(m => html`<button class="k-cv-shot" data-media="${m.id}" title="${m.prompt || ""}">${kind === "video" ? html`<video src="${m.url}" preload="metadata" muted></video>` : html`<img src="${m.url}" alt="${m.prompt || "Generated image"}" loading="lazy">`}</button>`)}</div>`);
    box.onclick = e => { const b = e.target.closest("[data-media]"); if (b) mediaSheet(rows.find(m => m.id === b.dataset.media)); };
  };
  const mediaSheet = m => {
    const s = sheet({ title: K.title, sub: `${when(m.ts)}${m.model ? " · " + m.model : ""}${m.backend ? " · " + m.backend.replace("kaggle:", "") : ""}`, wide: true,
      body: html`${kind === "video" ? html`<video class="k-media-big" src="${m.url}" controls></video>` : html`<img class="k-media-big" src="${m.url}" alt="${m.prompt || ""}">`}
        <p style="margin-top:14px">${m.prompt || html`<span class="k-dim">No prompt recorded.</span>`}</p>
        <p class="k-spec" style="margin-top:8px">${[m.size, m.seed != null ? "seed " + m.seed : "", bytes(m.bytes)].filter(Boolean).join(" · ")}</p>`,
      foot: html`<button class="k-btn k-sm" data-use>Use prompt</button><a class="k-btn k-sm" href="${m.url}" download>Download</a><button class="k-btn k-sm k-push" data-rm>Delete</button>` });
    $("[data-use]", s.el).onclick = () => { input.value = m.prompt || ""; autosize(); s.close(); input.focus(); };
    $("[data-rm]", s.el).onclick = async e => { const r = await act(e.currentTarget, () => post(`/admin/api/media/${m.id}/delete`)); if (r && r.ok) { s.close(); loadRecent(); } };
  };
  const viewSheet = (i, j) => {
    const m = thread.messages[i], u = m.items[j];
    const s = sheet({ title: K.title, sub: [m.model, m.stats && m.stats.size, m.stats && m.stats.backend && m.stats.backend.replace("kaggle:", "")].filter(Boolean).join(" · "), wide: true,
      body: html`<img class="k-media-big" src="${u}" alt="${m.prompt || ""}"><p style="margin-top:14px">${m.prompt || ""}</p>`,
      foot: html`<button class="k-btn k-sm" data-use>Use prompt</button><a class="k-btn k-sm" href="${u}" download="kllm-${Date.now()}.png">Download</a>` });
    $("[data-use]", s.el).onclick = () => { input.value = m.prompt || ""; autosize(); s.close(); input.focus(); };
  };

  const open = id => {
    thread = id ? (threads.find(c => c.id === id) || null) : null;
    if (id && !thread) { history.replaceState(null, "", "#" + kind); }
    if (thread && thread.model && models.some(x => x.key === thread.model)) current = thread.model;
    ui.useLast = true;
    paintList(); paintThread(); paintModel(); paintBusy(); paintQueue();
  };

  // ---- send: one job per thread; a message sent while a job runs waits in that thread's queue ----------------------
  const paintBusy = () => {
    const b = $("#cv-send", el), on = !!busyHere(), typed = !!input.value.trim() || attachments.length > 0;
    const stop = on && !typed;
    b.setAttribute("aria-label", stop ? "Stop" : on ? "Queue" : "Send"); b.title = stop ? "Stop (Esc)" : on ? "Queue: runs when this one finishes" : "Send (Enter)";
    b.classList.toggle("k-stop", stop);
    render(b, icon(stop ? "stop" : "send"));
  };
  const paintQueue = () => {
    const q = (thread && queues.get(thread.id)) || [];
    render($("#cv-queue", el), q.map((x, i) => html`<span class="k-cv-queued"><span>Queued</span><b>${x.text}</b><button type="button" class="k-icon-btn" data-unq="${i}" aria-label="Remove from queue" title="Remove">${icon("x")}</button></span>`));
  };
  const ensureThread = text => {
    if (thread) return;
    thread = { id: newId(), kind, title: (text || K.title).slice(0, 70), model: current, messages: [], updated: Date.now(), system: kind === "chat" ? pref("chatSystem", "") : "" };
    threads.unshift(thread);
    history.replaceState(null, "", `#${kind}/${thread.id}`);
  };
  const userMsg = (text, att, from) => kind === "chat"
    ? { role: "user", content: att.length ? [{ type: "text", text }, ...att.map(u => ({ type: "image_url", image_url: { url: u } }))] : text }
    : { role: "user", prompt: text, image: att[0] || from || null, ...(from && !att.length ? { fromLast: true } : {}) };
  const send = async (fromIndex = null) => {
    const m = model();
    if (!m) { toast(models.length ? "Pick a model first." : `Add a ${kind} model on the Models page first.`, true); return; }
    togglePop(false);
    if (fromIndex == null) {
      const text = input.value.trim();
      if (!text && !attachments.length) return;
      if (kind !== "chat" && !text) { toast("Write a prompt first.", true); return; }
      const att = attachments.slice(), from = startsFromLast();
      input.value = ""; attachments = []; paintAtt(); autosize();
      if (busyHere()) {
        const q = queues.get(thread.id) || []; q.push({ text, att, from }); queues.set(thread.id, q);
        paintQueue(); paintBusy(); paintChips();
        return;
      }
      ensureThread(text);
      thread.messages.push(userMsg(text, att, from));
    } else {
      if (busyHere()) return;
      thread.messages.splice(fromIndex);
    }
    run(thread, m);
  };
  const run = async (th, m) => {
    th.model = m.key; setPref(kind + "Model", m.key);
    th.updated = Date.now();
    paintList(); if (th === thread) paintThread();
    const ctrl = new AbortController();
    jobs.set(th.id, ctrl); paintBusy();
    try { if (kind === "chat") await streamChat(th, m, ctrl); else await generate(th, m, ctrl); }
    finally {
      jobs.delete(th.id);
      th.updated = Date.now();
      paintList(); if (th === thread) { paintFoot(); paintChips(); }
      paintBusy();
      await saveThread(JSON.parse(JSON.stringify(th)));
      const q = queues.get(th.id);
      if (q && q.length && !ctrl.signal.aborted) {
        const nx = q.shift(); if (th === thread) paintQueue();
        th.messages.push(userMsg(nx.text, nx.att, nx.att.length ? null : (kind === "image" ? lastImage(th) : null)));
        run(th, models.find(x => x.key === th.model) || m);
      }
    }
  };

  const streamChat = async (th, m, ctrl) => {
    const reply = { role: "assistant", kind: "chat", content: "", reasoning: "", live: true, stats: {}, model: m.key };
    th.messages.push(reply);
    if (th === thread) { threadEl.insertAdjacentHTML("beforeend", String(msgHtml(reply, th.messages.length - 1))); toEnd(); }
    const msgs = [...(th.system ? [{ role: "system", content: th.system }] : []), ...th.messages.slice(0, -1).filter(x => x.role === "user" || x.role === "assistant").map(x => ({ role: x.role, content: x.content }))];
    const body = { model: m.key, messages: msgs, stream: true, stream_options: { include_usage: true }, temperature: +opt.temp, max_tokens: +opt.max };
    if (!opt.thinking) body.chat_template_kwargs = { enable_thinking: false };
    const t0 = performance.now();
    let tFirst = null, tThinkEnd = null, lastPaint = 0;
    try {
      const res = await fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      reply.stats.backend = res.headers.get("x-kllm-backend");
      if (!res.ok) { let t = await res.text(); try { const j = JSON.parse(t); t = (j.error && j.error.message) || j.detail || t; } catch { /* text */ } throw new Error(`HTTP ${res.status}: ${String(t).slice(0, 400)}`); }
      const reader = res.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const d = line.slice(5).trim(); if (d === "[DONE]") continue;
          let j; try { j = JSON.parse(d); } catch { continue; }
          const delta = (j.choices && j.choices[0] && j.choices[0].delta) || {};
          if (delta.reasoning_content) { reply.reasoning += delta.reasoning_content; if (tFirst == null) tFirst = performance.now(); }
          if (delta.content) { if (tFirst == null) tFirst = performance.now(); if (tThinkEnd == null && reply.reasoning) tThinkEnd = performance.now(); reply.content += delta.content; }
          if (j.usage) { reply.stats.tokens = j.usage.completion_tokens; reply.stats.prompt = j.usage.prompt_tokens; const c = (j.usage.prompt_tokens_details || {}).cached_tokens; if (c != null) reply.stats.cached = c; }
          if (j.timings) {
            if (j.timings.predicted_per_second) reply.stats.tps = j.timings.predicted_per_second;
            if (j.timings.cache_n != null && reply.stats.cached == null) reply.stats.cached = j.timings.cache_n;
            if (j.timings.prompt_n != null && reply.stats.prompt == null) reply.stats.prompt = j.timings.prompt_n + (j.timings.cache_n || 0);
          }
        }
        if (performance.now() - lastPaint > 60) { lastPaint = performance.now(); paintLive(th, reply); }
      }
    } catch (e) {
      reply.error = e.name === "AbortError" ? "Stopped." : String(e.message || e).startsWith("Failed to fetch") ? "The relay is not running. Start it with `kllm serve`." : String(e.message || e);
    }
    const tEnd = performance.now();
    reply.live = false;
    if (tFirst != null) reply.stats.ttft = (tFirst - t0) / 1000;
    if (reply.reasoning) reply.stats.think_s = ((tThinkEnd || tEnd) - (tFirst || t0)) / 1000;
    if (!reply.stats.tps && reply.stats.tokens && tFirst != null && tEnd > tFirst) reply.stats.tps = reply.stats.tokens / ((tEnd - tFirst) / 1000);
    if (reply.stats.prompt) th.lastPrompt = reply.stats.prompt + (reply.stats.tokens || 0);
    paintLive(th, reply);
  };

  const generate = async (th, m, ctrl) => {
    const u = th.messages[th.messages.length - 1];
    const reply = { role: "assistant", kind, live: true, model: m.key, prompt: u.prompt, t0: Date.now(), stats: {}, items: [] };
    th.messages.push(reply);
    if (th === thread) { threadEl.insertAdjacentHTML("beforeend", String(msgHtml(reply, th.messages.length - 1))); toEnd(); }
    const tick = setInterval(() => {
      const e = $(`[data-elapsed="${reply.t0}"]`, threadEl); if (e) e.textContent = dur((Date.now() - reply.t0) / 1000);
      const g = $(`[data-genstate="${reply.t0}"]`, threadEl); if (g) g.textContent = genState(reply.model);
    }, 1000);
    const body = { model: m.key, prompt: u.prompt, response_format: "b64_json" };
    if (opt.size) body.size = opt.size;
    if (opt.steps !== "") body.steps = +opt.steps;
    if (opt.seed !== "") body.seed = +opt.seed;
    if (kind === "image" && +opt.n > 1) body.n = +opt.n;
    if (kind === "video") { if (opt.frames !== "") body.num_frames = +opt.frames; if (opt.fps !== "") body.fps = +opt.fps; }
    if (u.image) {
      body.image = u.image.startsWith("data:") ? u.image : await toDataUrl(u.image);
      if (kind === "image") body.strength = +opt.strength;
    }
    try {
      const res = await fetch(`/v1/${kind === "video" ? "videos" : "images"}/generations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      const text = await res.text(); let data;
      try { data = JSON.parse(text); } catch { data = { error: { message: text.slice(0, 300) } }; }
      if (!res.ok) throw new Error((data.error && data.error.message) || data.detail || `HTTP ${res.status}`);
      const local = (data.data || []).map(d => d.b64_mp4 ? `data:video/mp4;base64,${d.b64_mp4}` : d.b64_json ? `data:image/png;base64,${d.b64_json}` : d.url).filter(Boolean);
      reply.stats = { took: (Date.now() - reply.t0) / 1000, backend: res.headers.get("x-kllm-backend"), size: body.size || "" };
      // prefer the copies the relay saved, so the thread keeps small URLs and survives reloads
      // the relay writes its copy just after answering: look twice before falling back to the inline data
      let saved = [];
      for (const wait of [300, 1200]) {
        await new Promise(res => setTimeout(res, wait));
        const r = await get(`/admin/api/media?kind=${kind}&limit=${Math.max(4, local.length + 2)}`);
        saved = r.ok ? r.data.media.filter(x => x.prompt === u.prompt && x.ts * 1000 >= reply.t0 - 5000).map(x => x.url) : [];
        if (saved.length >= local.length) break;
      }
      reply.items = saved.length >= local.length && saved.length ? saved.slice(0, local.length).reverse() : local;
      if (!reply.items.length) reply.error = "The kernel answered without any output.";
    } catch (e) {
      reply.error = e.name === "AbortError" ? "Stopped. The kernel may still finish this request, and the result will appear under Recent." : String(e.message || e).startsWith("Failed to fetch") ? "The relay is not running. Start it with `kllm serve`." : String(e.message || e);
    } finally { clearInterval(tick); }
    reply.live = false;
    paintLive(th, reply);
  };
  // a saved output (/admin/api/media/...) used as the next starting image: the kernel needs the bytes
  const toDataUrl = async url => {
    const b = await (await fetch(url)).blob();
    return await new Promise(ok => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(b); });
  };

  // ---- edit a user message: rewrite it and run again from there ----------------------------------------------------------
  const edit = i => {
    const m = thread.messages[i], node = $(`.k-cv-u[data-i="${i}"]`, threadEl);
    if (!node || busyHere()) return;
    render(node, html`<form class="k-cv-editbox k-raised"><textarea class="k-cv-edit" rows="3" aria-label="Edit message">${userText(m)}</textarea>
      <div class="k-cv-edit-acts"><button type="button" class="k-btn k-sm" data-cancel>Cancel</button><button class="k-btn k-sm k-primary" type="submit">Send</button></div></form>`);
    const ta = $("textarea", node); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
    $("[data-cancel]", node).onclick = () => paintThread();
    $("form", node).onsubmit = e => {
      e.preventDefault();
      const t = ta.value.trim(); if (!t) return;
      if (kind === "chat") m.content = Array.isArray(m.content) ? [{ type: "text", text: t }, ...m.content.filter(c => c.type === "image_url")] : t;
      else m.prompt = t;
      if (i === 0) thread.title = t.slice(0, 70);
      send(i + 1);
    };
    ta.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form", node).requestSubmit(); } if (e.key === "Escape") paintThread(); });
  };

  // ---- attachments -----------------------------------------------------------------------------------------------------------
  const paintAtt = () => { render($("#cv-att", el), attachments.map((u, i) => html`<span class="k-att"><img src="${u}" alt="Attachment ${i + 1}"><button type="button" class="k-icon-btn" data-rm="${i}" aria-label="Remove image">${icon("x")}</button></span>`)); paintDrop(); };
  const addFiles = files => Array.from(files).filter(f => f.type.startsWith("image/")).forEach(async f => {
    if (f.size > 40 * 1024 * 1024) { toast(`${f.name} is over 40 MB.`, true); return; }
    const url = await shrink(f);
    if (kind !== "chat") attachments = [];
    attachments.push(url); paintAtt(); paintChips(); paintBusy();
  });
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 220) + "px"; };

  // ---- events ------------------------------------------------------------------------------------------------------------------
  const submit = () => { const j = busyHere(); if (j && !input.value.trim() && !attachments.length) { j.abort(); return; } send(); };
  $("#cv-form", el).onsubmit = e => { e.preventDefault(); submit(); };
  input.addEventListener("input", () => { autosize(); paintBusy(); });
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); } });
  input.addEventListener("paste", e => { const f = Array.from(e.clipboardData && e.clipboardData.files || []); if (f.length && (kind !== "chat" || isVision(model()))) { e.preventDefault(); addFiles(f); } });
  $("#cv-file", el).onchange = e => { addFiles(e.target.files); e.target.value = ""; };
  el.addEventListener("dragover", e => { if (kind !== "chat" || isVision(model())) e.preventDefault(); });
  el.addEventListener("drop", e => { if (kind !== "chat" || isVision(model())) { e.preventDefault(); addFiles(e.dataTransfer.files); } });
  $("#cv-model-btn", el).onclick = e => { e.stopPropagation(); menu(); };
  $("#cv-menu", el).onclick = e => { const b = e.target.closest("[data-model]"); if (!b) return; current = b.dataset.model; setPref(kind + "Model", current); if (thread) thread.model = current; $("#cv-menu", el).hidden = true; $("#cv-model-btn", el).setAttribute("aria-expanded", "false"); paintModel(); input.focus(); };
  const optBtn = $("#cv-opt-btn", el);
  if (optBtn) optBtn.onclick = e => { e.stopPropagation(); togglePop(); };
  $("#cv-q", el).oninput = paintList;
  const fresh = () => ctx.go("#" + kind);
  $("#cv-new", el).onclick = fresh; $("#cv-new2", el).onclick = fresh;
  $("#cv-hist", el).onclick = () => { ui.hist = !ui.hist; if (!narrow.matches) setPref(kind + "Hist", ui.hist); root.classList.toggle("k-cv-nohist", !ui.hist); };
  // picking a thread in the drawer closes it on a narrow screen
  $("#cv-list", el).addEventListener("click", e => { if (narrow.matches && e.target.closest("a")) { ui.hist = false; root.classList.add("k-cv-nohist"); } });
  $("#cv-full", el).onclick = e => {
    ui.full = !ui.full; e.currentTarget.setAttribute("aria-pressed", ui.full);
    document.body.dataset.side = ui.full ? "hidden" : ctx.store.side;
    root.classList.toggle("k-cv-nohist", ui.full || !ui.hist);
  };
  scroller.addEventListener("scroll", () => { $("#cv-down", el).hidden = nearEnd() || root.classList.contains("k-cv-empty"); });
  $("#cv-down", el).onclick = () => scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
  const outside = e => {
    if (!e.target.closest("#cv-menu, #cv-model-btn")) { $("#cv-menu", el).hidden = true; $("#cv-model-btn", el).setAttribute("aria-expanded", "false"); }
    if (!e.target.closest("#cv-pop, #cv-opt-btn")) togglePop(false);
  };
  document.addEventListener("click", outside);
  const keys = e => {
    if (el.closest("[hidden]")) return;            // kept-alive page in the background: not ours
    if (e.key === "Escape") { const j = busyHere(); if (j) { j.abort(); e.preventDefault(); } togglePop(false); $("#cv-menu", el).hidden = true; }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); fresh(); }
  };
  document.addEventListener("keydown", keys);
  el.addEventListener("click", async e => {
    const d = e.target.closest("[data-del]");
    if (d) { e.preventDefault(); await deleteThread(d.dataset.del); threads = threads.filter(c => c.id !== d.dataset.del); if (thread && thread.id === d.dataset.del) fresh(); else paintList(); return; }
    const sg = e.target.closest("[data-sug]"); if (sg) { input.value = sg.dataset.sug; autosize(); input.focus(); return; }
    const ed = e.target.closest("[data-edit]"); if (ed) { edit(+ed.dataset.edit); return; }
    const cu = e.target.closest("[data-copyu]"); if (cu) { if (await copy(userText(thread.messages[+cu.dataset.copyu]), null)) toast("Copied."); return; }
    const ca = e.target.closest("[data-copya]"); if (ca) { if (await copy(thread.messages[+ca.dataset.copya].content, null)) toast("Copied."); return; }
    const cc = e.target.closest("[data-copycode]"); if (cc) { const pre = cc.parentElement.nextElementSibling; copy(pre ? pre.innerText : "", cc); return; }
    const rt = e.target.closest("[data-retry]"); if (rt && !busyHere()) { send(+rt.dataset.retry); return; }
    const rm = e.target.closest("[data-rm]"); if (rm) { attachments.splice(+rm.dataset.rm, 1); paintAtt(); paintChips(); paintBusy(); return; }
    const nl = e.target.closest("[data-nolast]"); if (nl) { ui.useLast = false; paintChips(); return; }
    const uq = e.target.closest("[data-unq]"); if (uq) { const q = queues.get(thread.id) || []; q.splice(+uq.dataset.unq, 1); paintQueue(); paintBusy(); return; }
    const v = e.target.closest("[data-view]"); if (v) { const [i, j] = v.dataset.view.split(":").map(Number); viewSheet(i, j); }
  });
  const off = ctx.shared.on(() => { if (kind !== "chat") { const b = ctx.shared.backends; running = b ? b.backends.filter(x => x.healthy).map(x => x.model_key) : []; pending = b ? b.pending.map(x => x.model_key) : []; } paintModel(); });
  const modelPoll = kind === "chat" ? setInterval(loadModels, 15000) : null;

  await loadModels();
  open(ctx.route.parts[1]);
  input.focus();
  INST[kind] = {
    open: id => { if ((thread ? thread.id : undefined) !== id || !id) open(id); input.focus(); },
    hide: () => { togglePop(false); if (ui.full) { ui.full = false; document.body.dataset.side = ctx.store.side; root.classList.toggle("k-cv-nohist", !ui.hist); } },
  };
  return () => {
    off(); clearInterval(modelPoll);
    document.removeEventListener("click", outside); document.removeEventListener("keydown", keys);
    for (const j of jobs.values()) j.abort();
    if (ui.full) document.body.dataset.side = ctx.store.side;
  };
}
