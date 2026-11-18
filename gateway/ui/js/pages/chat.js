// Chat: llama.cpp's own web UI (MIT, vendored under ui/vendor/llama-ui), served by the relay at /chat-ui/ in
// router mode. It lists kllm's chat models, streams through the relay's normal /v1/chat/completions route and
// keeps its conversations in this browser. The page is kept alive (app.js), so a reply keeps streaming while you
// use the rest of the panel.
import { html, render, $ } from "../ui.js";

export const keepAlive = true;
let frame = null;

export async function mount(el, ctx) {
  render(el, html`<div class="k-page k-tool k-chatui">
    <iframe id="chat-ui" title="Chat" src="/chat-ui/${chatHash(ctx.route)}" allow="clipboard-read; clipboard-write"></iframe></div>`);
  frame = $("#chat-ui", el);
}

// #chat opens the UI's start page, #chat/<id> one of its conversations (the UI routes with #/chat/<id>)
const chatHash = r => r.parts[1] ? `#/chat/${encodeURIComponent(r.parts[1])}` : "";

export function update(r) {
  if (frame && r.parts[1]) frame.contentWindow.location.hash = `/chat/${encodeURIComponent(r.parts[1])}`;
  return true;
}
