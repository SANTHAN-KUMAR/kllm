"""kllm helper nodes for Wan2.2-Animate replacement in multi-person footage.

KllmPoseFromMask: run kijai's PoseAndFaceDetection, but take each frame's person box from a
mask (SAM3 text-prompted track of e.g. "standing barber") instead of YOLO's single top hit.
That makes the pose skeleton, face crops and the replacement mask provably the same person;
stock YOLO picks the highest-confidence human per frame, which in a barber clip can be the
client or a bystander and can flip between frames.
"""
import sys

import numpy as np
import torch


def _wap():
    for name, mod in list(sys.modules.items()):
        if "WanAnimatePreprocess" in name and hasattr(mod, "PoseAndFaceDetection"):
            return mod
    raise RuntimeError("ComfyUI-WanAnimatePreprocess is not loaded")


class _MaskBoxDetector:
    """Duck-types the Yolo wrapper: returns the next precomputed xyxy+score box per call."""

    def __init__(self, boxes):
        self.boxes, self.i = boxes, 0

    def reinit(self):
        self.i = 0

    def cleanup(self):
        pass

    def __call__(self, *_a, **_k):
        b = self.boxes[min(self.i, len(self.boxes) - 1)]
        self.i += 1
        return [[{"bbox": b}]]


def mask_boxes(masks, pad=0.06):
    n, h, w = masks.shape
    boxes, last = [], None
    for i in range(n):
        ys, xs = torch.nonzero(masks[i] > 0.5, as_tuple=True)
        if len(xs) < 50:
            boxes.append(last)
            continue
        x1, x2, y1, y2 = xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item()
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        last = np.array([max(0, x1 - px), max(0, y1 - py), min(w - 1, x2 + px), min(h - 1, y2 + py), 1.0], dtype=np.float32)
        boxes.append(last)
    # frames before the first detection borrow the first good box
    first = next((b for b in boxes if b is not None), np.array([0, 0, w - 1, h - 1, 1.0], dtype=np.float32))
    return [first if b is None else b for b in boxes]


class KllmPoseFromMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("POSEMODEL",),
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "width": ("INT", {"default": 480, "min": 64, "max": 2048}),
            "height": ("INT", {"default": 832, "min": 64, "max": 2048}),
            "face_padding": ("INT", {"default": 0, "min": 0, "max": 512}),
        }}

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING")
    RETURN_NAMES = ("pose_data", "face_images", "boxes_json")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, model, images, masks, width, height, face_padding=0):
        import json
        boxes = mask_boxes(masks)
        m2 = {"vitpose": model["vitpose"], "yolo": _MaskBoxDetector(boxes)}
        pose_data, faces, _pts, _bb, _fb = _wap().PoseAndFaceDetection().process(m2, images, width, height, None, face_padding)
        return (pose_data, faces, json.dumps([[float(v) for v in b[:4]] for b in boxes]))


class KllmMaskStats:
    """Fraction of frames with a non-empty mask + mean coverage, surfaced in the log for QA."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK", "STRING")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, masks):
        cov = (masks > 0.5).float().mean(dim=(1, 2))
        s = "mask frames_nonempty=%.2f mean_cov=%.3f min_cov=%.3f max_cov=%.3f" % (
            (cov > 0.002).float().mean().item(), cov.mean().item(), cov.min().item(), cov.max().item())
        print("[kllm] " + s, flush=True)
        return (masks, s)


_LIMBS = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (8, 11)]


class KllmCompleteMask:
    """Union the SAM3 track with the selected person's skeleton: a head box (face keypoints expanded,
    mostly upward to leave room for hair/bun) plus thick limb capsules. Text masks like
    'man in blue shirt' segment the shirt only; anything outside the mask is copied from the plate,
    so the original head would leak into the replacement."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"masks": ("MASK",), "pose_data": ("POSEDATA",),
                             "head_up": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 4.0}),
                             "head_side": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 4.0}),
                             "limb_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0}),
                             "min_score": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0})}}

    RETURN_TYPES = ("MASK", "STRING")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, masks, pose_data, head_up=1.4, head_side=0.9, limb_scale=0.5, min_score=0.3):
        import cv2
        metas = pose_data.get("pose_metas_original") or pose_data["pose_metas"]
        n, h, w = masks.shape
        out = (masks > 0.5).cpu().numpy().astype(np.uint8)
        heads = 0
        for i in range(min(n, len(metas))):
            m = metas[i]
            canvas = np.zeros((h, w), np.uint8)
            face = np.asarray(m["keypoints_face"], dtype=np.float32)
            body = np.asarray(m["keypoints_body"], dtype=np.float32)
            fpts = face[face[:, 2] > min_score][:, :2] * (w, h) if face.shape[1] > 2 else face[:, :2] * (w, h)
            if len(fpts) < 5:   # fall back to nose/eyes/ears from the body set
                sel = body[[0, 14, 15, 16, 17]]
                fpts = sel[sel[:, 2] > min_score][:, :2] * (w, h)
            fw = 0
            if len(fpts) >= 2:
                x1, y1 = fpts.min(0)
                x2, y2 = fpts.max(0)
                fw, fh = max(x2 - x1, 8), max(y2 - y1, 8)
                cv2.rectangle(canvas, (int(x1 - fw * head_side), int(y1 - fh * head_up)), (int(x2 + fw * head_side), int(y2 + fh * 0.35)), 1, -1)
                heads += 1
            thick = int(max(10, (fw or 0.06 * w) * limb_scale * 2))
            for a, b in _LIMBS:
                if body[a, 2] > min_score and body[b, 2] > min_score:
                    pa = (int(body[a, 0] * w), int(body[a, 1] * h))
                    pb = (int(body[b, 0] * w), int(body[b, 1] * h))
                    cv2.line(canvas, pa, pb, 1, thick)
            out[i] = np.maximum(out[i], canvas)
        res = torch.from_numpy(out.astype(np.float32))
        s = "complete-mask frames=%d with_head=%d mean_cov_before=%.3f after=%.3f" % (
            n, heads, (masks > 0.5).float().mean().item(), res.mean().item())
        print("[kllm] " + s, flush=True)
        return (res, s)


class KllmSkirtMask:
    """Widen the edit mask from the hips to below the ankles into a flared skirt region so a long garment
    (saree / lehenga / long dress) from the reference has room to be drawn. Without it the mask hugs the source
    person's trouser silhouette and Animate renders trousers (e22n/e25b). Uses the target's pose keypoints:
    body idx 8/11 = hips, 9/12 = knees, 10/13 = ankles (kijai's 20-point layout)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",), "pose_data": ("POSEDATA",),
                             "hip_scale": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0}),
                             "hem_scale": ("FLOAT", {"default": 2.6, "min": 0.5, "max": 6.0}),
                             "min_score": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0})}}

    RETURN_TYPES = ("MASK", "STRING")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, mask, pose_data, hip_scale=1.5, hem_scale=2.6, min_score=0.3):
        import cv2
        metas = pose_data.get("pose_metas_original") or pose_data["pose_metas"]
        n, h, w = mask.shape
        out = (mask > 0.5).cpu().numpy().astype(np.uint8)
        done = 0
        for i in range(min(n, len(metas))):
            b = np.asarray(metas[i]["keypoints_body"], dtype=np.float32)
            ok = lambda k: b[k, 2] > min_score
            if not (ok(8) and ok(11)):
                continue
            lh, rh = b[8, :2] * (w, h), b[11, :2] * (w, h)
            hip_c, hip_w = (lh + rh) / 2, max(np.linalg.norm(lh - rh), 0.04 * w)
            feet = [b[k, :2] * (w, h) for k in (10, 13, 9, 12) if ok(k)]
            if feet:
                low = np.array(feet)
                hem_y = low[:, 1].max() + 0.05 * h
                hem_c = np.array([low[:, 0].mean(), hem_y])
            else:                                   # legs not visible: extend a body-length below the hips
                hem_c = hip_c + np.array([0, 2.2 * hip_w])
            # trapezoid hips -> hem, oriented along the hip->hem axis (works for seated, bent legs)
            axis = hem_c - hip_c
            nrm = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-6)
            top_half, hem_half = hip_w * hip_scale / 2, hip_w * hem_scale / 2
            poly = np.array([hip_c + nrm * top_half, hip_c - nrm * top_half, hem_c - nrm * hem_half, hem_c + nrm * hem_half], np.int32)
            canvas = np.zeros((h, w), np.uint8)
            cv2.fillPoly(canvas, [poly], 1)
            out[i] = np.maximum(out[i], canvas)
            done += 1
        res = torch.from_numpy(out.astype(np.float32))
        s = "skirt-mask frames=%d/%d cov %.3f -> %.3f" % (done, n, (mask > 0.5).float().mean().item(), res.mean().item())
        print("[kllm] " + s, flush=True)
        return (res, s)


class KllmHybridControl:
    """VACE control video for 'recreate': depth everywhere (locks the scene layout + camera + other people) but
    inside the target person's mask the depth is replaced by that person's pose skeleton, so their body shape,
    clothing and face come from the reference image + prompt instead of the source silhouette."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"depth": ("IMAGE",), "pose": ("IMAGE",), "mask": ("MASK",),
                             "feather": ("INT", {"default": 6, "min": 0, "max": 64})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, depth, pose, mask, feather=6):
        import torch.nn.functional as F
        n = min(depth.shape[0], pose.shape[0], mask.shape[0])
        d, p = depth[:n], pose[:n]
        if p.shape[1:3] != d.shape[1:3]:
            p = F.interpolate(p.movedim(-1, 1), size=d.shape[1:3], mode="bilinear", align_corners=False).movedim(1, -1)
        m = mask[:n].unsqueeze(1).float()
        if m.shape[2:] != d.shape[1:3]:
            m = F.interpolate(m, size=d.shape[1:3], mode="nearest")
        if feather > 0:
            k = feather * 2 + 1
            m = F.avg_pool2d(F.max_pool2d(m, k, 1, feather), k, 1, feather)
        m = m.movedim(1, -1)
        return (d * (1 - m) + p[..., :3] * m,)


class KllmProtectOthers:
    """Remove every *other* tracked person from the edit mask. `others_track` is a SAM3 track of generic
    'person' objects; objects whose masks overlap the target track by less than `iou_max` (mean IoU over frames)
    are other people (client, bystanders). Their pixels are subtracted from `mask`, except where the target's
    own SAM mask is (the barber's hands on the client's head stay editable)."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",), "target_mask": ("MASK",), "others_track": ("SAM3_TRACK_DATA",),
                             "iou_max": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                             "protect_grow": ("INT", {"default": 6, "min": 0, "max": 64})}}

    RETURN_TYPES = ("MASK", "STRING")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, mask, target_mask, others_track, iou_max=0.3, protect_grow=6):
        import torch.nn.functional as F
        from comfy.ldm.sam3.tracker import unpack_masks
        packed = others_track.get("packed_masks")
        n, h, w = mask.shape
        if packed is None:
            return (mask, "protect: no other people tracked")
        tgt = (target_mask > 0.5).float()
        protect = torch.zeros_like(mask)
        kept, dropped = 0, 0
        for o in range(packed.shape[1]):
            om = unpack_masks(packed[:, o].clone()).unsqueeze(1).float()
            om = F.interpolate(om, size=(h, w), mode="bilinear", align_corners=False)[:, 0][:n]
            om = (om > 0.5).float()
            inter = (om * tgt[:om.shape[0]]).sum(dim=(1, 2))
            union = ((om + tgt[:om.shape[0]]) > 0).float().sum(dim=(1, 2)).clamp(min=1)
            iou = (inter / union).mean().item()
            if iou < iou_max:
                protect[:om.shape[0]] = torch.maximum(protect[:om.shape[0]], om)
                kept += 1
            else:
                dropped += 1
        if protect_grow > 0:
            k = protect_grow * 2 + 1
            protect = F.max_pool2d(protect.unsqueeze(1), k, stride=1, padding=protect_grow)[:, 0]
        protect = protect * (1 - tgt)
        out = mask * (1 - protect)
        s = "protect: others=%d target-like=%d cov %.3f -> %.3f" % (kept, dropped, mask.mean().item(), out.mean().item())
        print("[kllm] " + s, flush=True)
        return (out, s)


def _gguf_mod():
    for name, mod in list(sys.modules.items()):
        if "GGUF" in name and hasattr(mod, "UnetLoaderGGUF") and hasattr(mod, "GGUFModelPatcher"):
            return mod
    raise RuntimeError("ComfyUI-GGUF is not loaded")


def _load_gguf_unet(unet_name, **_kw):
    return _gguf_mod().UnetLoaderGGUF().load_unet(unet_name)[0]


class KllmUnetLoaderGGUF:
    """UnetLoaderGGUF that registers a reload factory (cached_patcher_init), which ComfyUI's native
    multi-GPU nodes (MultiGPU_WorkUnits CFG split) require to build per-device clones."""

    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        return {"required": {"unet_name": (folder_paths.get_filename_list("unet_gguf"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "kllm"

    def load(self, unet_name):
        m = _load_gguf_unet(unet_name)
        m.cached_patcher_init = (_load_gguf_unet, (unet_name,))
        return (m,)


class KllmMaskBBoxCrop:
    """Crop image[0] to the bbox of mask[0] (>0.5) expanded by `margin` x bbox size, rounded to multiples of 16.
    Lets an image editor see only the target person (e31d/e31e H8: Qwen edited the wrong person in the full frame)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crop", "bbox")
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, image, mask, margin):
        import torch
        H, W = image.shape[1], image.shape[2]
        m = mask[0] if mask.dim() == 3 else mask
        if m.shape[-2:] != (H, W):
            m = torch.nn.functional.interpolate(m[None, None].float(), size=(H, W), mode="nearest")[0, 0]
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(ys) == 0:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
            x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)
        w, h = max(16, (x1 - x0) // 16 * 16), max(16, (y1 - y0) // 16 * 16)
        x0, y0 = min(x0, W - w), min(y0, H - h)
        return (image[:1, y0:y0 + h, x0:x0 + w, :].clone(), "%d,%d,%d,%d" % (x0, y0, w, h))


class KllmPasteCrop:
    """Resize `src` to the bbox size, paste it into `dest` at the bbox, blended only inside `mask` (full-frame)."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"dest": ("IMAGE",), "src": ("IMAGE",), "bbox": ("STRING", {"forceInput": True}), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "kllm"

    def process(self, dest, src, bbox, mask):
        import torch
        x0, y0, w, h = [int(v) for v in bbox.split(",")]
        out = dest[:1].clone()
        H, W = out.shape[1], out.shape[2]
        s = torch.nn.functional.interpolate(src[:1].movedim(-1, 1).float(), size=(h, w), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)
        m = mask[0] if mask.dim() == 3 else mask
        if m.shape[-2:] != (H, W):
            m = torch.nn.functional.interpolate(m[None, None].float(), size=(H, W), mode="bilinear")[0, 0]
        a = m[y0:y0 + h, x0:x0 + w, None].to(out.dtype)
        out[0, y0:y0 + h, x0:x0 + w, :] = s[0].to(out.dtype) * a + out[0, y0:y0 + h, x0:x0 + w, :] * (1 - a)
        return (out,)


NODE_CLASS_MAPPINGS = {"KllmMaskBBoxCrop": KllmMaskBBoxCrop, "KllmPasteCrop": KllmPasteCrop, "KllmPoseFromMask": KllmPoseFromMask, "KllmMaskStats": KllmMaskStats, "KllmUnetLoaderGGUF": KllmUnetLoaderGGUF, "KllmCompleteMask": KllmCompleteMask, "KllmProtectOthers": KllmProtectOthers, "KllmSkirtMask": KllmSkirtMask, "KllmHybridControl": KllmHybridControl}
