// Image and Video: prompts in a conversation, results as images or clips. POST /v1/images/generations and
// /v1/videos/generations through the relay; saved outputs from GET /admin/api/media. Shared layout in convo.js.
// Kept alive (app.js): a render keeps running while you use other pages, and switching threads never cancels it.
import { mountConvo, INST } from "../convo.js";

const kindOf = r => (r.parts[0] === "video" ? "video" : "image");
let shown = null;

export const keepAlive = true;
export const keepKey = kindOf;
export function mount(el, ctx) { shown = kindOf(ctx.route); return mountConvo(el, ctx, shown); }
export function update(r) {
  const k = kindOf(r), changed = k !== shown;
  if (!INST[k]) return false;                 // not mounted yet: app.js mounts it in its own kept box
  shown = k; INST[k].open(r.parts[1]);
  return !changed;                            // other kind: app.js swaps the visible box
}
export function hide() { Object.values(INST).forEach(i => i.hide()); }
