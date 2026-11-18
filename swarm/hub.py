"""kllm-hub: Lightweight Low-Latency Userspace Binary Packet Switch.

Designed to run on a free-tier GCP e2-micro instance (us-central1) or local coordinator.
Routes pipeline activation frames stage-by-stage across Kaggle nodes over persistent WebSockets.
Zero disk I/O, <40 MB RAM footprint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from .protocol import (
    HEADER_SIZE,
    SwarmFrameHeader,
    unpack_activation_frame,
)

logger = logging.getLogger("kllm.hub")
router = APIRouter(prefix="/swarm", tags=["swarm"])


class SwarmPacketSwitch:
    """Manages stage-to-stage pipeline routing for active swarm rings."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.stages: Dict[int, WebSocket] = {}
        self.node_metadata: Dict[int, Dict[str, Any]] = {}
        self.total_stages: int = 6
        self.frames_routed: int = 0
        self.bytes_routed: int = 0
        self.start_time: float = time.time()
        self.coordinator_ws: Optional[WebSocket] = None
        self.stage0_url: Optional[str] = None

    async def register_stage(
        self, stage_idx: int, total_stages: int, ws: WebSocket, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        async with self.lock:
            self.stages[stage_idx] = ws
            self.total_stages = total_stages
            meta = dict(metadata or {})
            meta["registered_at"] = time.time()
            if stage_idx == 0 and meta.get("url"):
                self.stage0_url = meta["url"]
            self.node_metadata[stage_idx] = meta
            logger.info("Stage %d/%d registered (zone=%s, rtt=%.1fms)",
                        stage_idx, total_stages, meta.get("zone", "unknown"), meta.get("rtt_ms", 0.0))

    async def unregister_stage(self, stage_idx: int) -> None:
        ws_to_close: Optional[WebSocket] = None
        async with self.lock:
            if stage_idx in self.stages:
                ws_to_close = self.stages.pop(stage_idx)
            if stage_idx in self.node_metadata:
                del self.node_metadata[stage_idx]
            logger.info("Stage %d unregistered", stage_idx)
        if ws_to_close is not None:
            try:
                await ws_to_close.close()
            except Exception:
                pass

    async def forward_frame(self, raw_frame: bytes) -> Optional[int]:
        """Route a binary activation frame to the next stage in the pipeline.

        Isolates downstream network drops so a disconnection in stage i+1
        never crashes stage i's connection loop.
        """
        if len(raw_frame) < HEADER_SIZE:
            logger.warning("Dropped undersized frame: %d bytes (min %d)", len(raw_frame), HEADER_SIZE)
            return None

        try:
            header, _ = unpack_activation_frame(raw_frame)
        except ValueError as exc:
            logger.warning("Dropped malformed activation frame: %s", exc)
            return None

        next_stage = (header.stage_idx + 1) % header.total_stages

        target_ws: Optional[WebSocket] = None
        async with self.lock:
            target_ws = self.stages.get(next_stage)
            self.frames_routed += 1
            self.bytes_routed += len(raw_frame)

        # If tail stage completed and coordinator is connected, copy frame to coordinator
        if next_stage == 0 and self.coordinator_ws is not None:
            try:
                await self.coordinator_ws.send_bytes(raw_frame)
            except Exception as exc:
                logger.debug("Coordinator ws write failed: %s", exc)
                self.coordinator_ws = None

        if target_ws is not None:
            try:
                await target_ws.send_bytes(raw_frame)
                return next_stage
            except Exception as exc:
                logger.error("Failed to forward frame to stage %d (%s); unregistering dead stage", next_stage, exc)
                await self.unregister_stage(next_stage)
                return None
        else:
            logger.warning("Target stage %d not connected; frame held/dropped", next_stage)
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "total_stages": self.total_stages,
            "connected_stages": sorted(list(self.stages.keys())),
            "all_stages_ready": len(self.stages) == self.total_stages and self.total_stages > 0,
            "frames_routed": self.frames_routed,
            "bytes_routed": self.bytes_routed,
            "nodes": self.node_metadata,
            "coordinator_connected": self.coordinator_ws is not None,
            "stage0_url": self.stage0_url,
        }


hub = SwarmPacketSwitch()


@router.get("/ping")
async def ping() -> Dict[str, Any]:
    """Ultra-low overhead ping endpoint for node latency calibration."""
    return {"status": "ok", "timestamp": time.time()}


@router.get("/status")
async def status() -> Dict[str, Any]:
    """Cluster health and connected stages status."""
    return hub.get_status()


@router.post("/tunnel")
async def register_tunnel(request: Request) -> Dict[str, Any]:
    """Register Stage 0 tunnel URL directly to bypass GitHub Gist rate limits."""
    try:
        body = await request.json()
        stage = int(body.get("stage", 0))
        url = body.get("url")
        if stage == 0 and url:
            hub.stage0_url = str(url)
            logger.info("Stage 0 tunnel URL registered directly on hub: %s", url)
            return {"status": "ok", "stage0_url": url}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ignored"}


@router.post("/route")
async def route_http(request: Request) -> Dict[str, Any]:
    """HTTP fallback for binary frame forwarding."""
    body = await request.body()
    next_stage = await hub.forward_frame(body)
    if next_stage is None:
        raise HTTPException(status_code=503, detail="Next stage not reachable or frame dropped")
    return {"status": "routed", "next_stage": next_stage, "bytes": len(body)}


@router.post("/dispatch")
async def dispatch_request(request: Request) -> Dict[str, Any]:
    """Dispatch a prompt or inference task to Stage 0 (Head Node)."""
    payload = await request.json()
    stage0 = hub.stages.get(0)
    if stage0 is None:
        raise HTTPException(status_code=503, detail="Stage 0 (Head Node) not connected")
    try:
        await stage0.send_text(json.dumps(payload))
        return {"status": "dispatched", "stage": 0}
    except Exception as exc:
        logger.error("Failed to dispatch to Stage 0: %s", exc)
        await hub.unregister_stage(0)
        raise HTTPException(status_code=502, detail="Failed to send to Stage 0")


@router.websocket("/ws/coordinator")
async def websocket_coordinator_endpoint(websocket: WebSocket) -> None:
    """WebSocket connection for local PC coordinator to stream tokens and manage ring."""
    await websocket.accept()
    hub.coordinator_ws = websocket
    try:
        while True:
            msg = await websocket.receive_text()
            # Forward command or prompt to stage 0
            stage0 = hub.stages.get(0)
            if stage0 is not None:
                await stage0.send_text(msg)
    except WebSocketDisconnect:
        if hub.coordinator_ws is websocket:
            hub.coordinator_ws = None
    except Exception as exc:
        logger.debug("Coordinator websocket exited: %s", exc)
        if hub.coordinator_ws is websocket:
            hub.coordinator_ws = None


@router.websocket("/ws/{stage_idx}")
async def websocket_stage_endpoint(websocket: WebSocket, stage_idx: int) -> None:
    """Persistent bidirectional WebSocket connection for pipeline stage workers."""
    await websocket.accept()
    total_stages = int(websocket.query_params.get("total_stages", "6"))
    zone = websocket.query_params.get("zone", "unknown")
    try:
        rtt_ms = float(websocket.query_params.get("rtt_ms", "0.0"))
    except ValueError:
        rtt_ms = 0.0

    await hub.register_stage(
        stage_idx,
        total_stages,
        websocket,
        metadata={"zone": zone, "rtt_ms": rtt_ms},
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            await hub.forward_frame(data)
    except WebSocketDisconnect:
        await hub.unregister_stage(stage_idx)
    except Exception as exc:
        logger.error("Error in stage %d websocket: %s", stage_idx, exc)
        await hub.unregister_stage(stage_idx)
