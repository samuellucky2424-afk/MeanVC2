#!/usr/bin/env python3
"""RunPod load-balanced WebSocket server for MeanVC2 + Morphly.

Protocol
--------
1. Connect to /ws.
2. Server sends a JSON hello message.
3. Client sends one JSON start message:
     {"type":"start", "voice_id":"voice-name"}
   or
     {"type":"start", "upload_target":true}
   followed by one binary message containing a WAV reference voice.
4. Server sends {"type":"ready"}.
5. Client streams mono 16 kHz signed 16-bit little-endian PCM as binary frames.
6. Server returns converted mono 16 kHz signed 16-bit PCM as binary frames.

This baseline intentionally allows one active live session per worker because
MeanVC2 keeps mutable ASR/VC/vocoder streaming caches. RunPod should scale
workers horizontally for concurrent users.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / "runtime"
# run_rt.py expects `src` to resolve to runtime/src when executed directly.
sys.path.insert(0, str(RUNTIME_DIR))
import run_rt  # noqa: E402

MODEL_NAME = os.getenv("MEANVC_MODEL", "40ms")
DEVICE = os.getenv("MEANVC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
VOICE_DIR = Path(os.getenv("VOICE_DIR", str(ROOT / "voices")))
DEFAULT_TARGET_WAV = os.getenv("DEFAULT_TARGET_WAV", "")
MAX_TARGET_BYTES = int(os.getenv("MAX_TARGET_BYTES", str(15 * 1024 * 1024)))

if MODEL_NAME not in run_rt.MODEL_PATHS:
    raise RuntimeError(f"Unsupported MEANVC_MODEL={MODEL_NAME!r}; use 40ms or 120ms")
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("MEANVC_DEVICE=cuda but CUDA is not available")


class ServerVCRunner(run_rt.VCRunner):
    """VCRunner variant that loads the engine before selecting a target voice."""

    def __init__(self, device: str = "cuda", model: str = "40ms"):
        self.device = device
        torch.set_num_threads(1)
        torch.backends.cudnn.enabled = False

        paths = run_rt.MODEL_PATHS[model]
        vc_ckpt = paths["ckpt"]
        vc_config = paths["config"]
        self._asr_ckpt = paths["asr_ckpt"]
        self._bn_window = paths["bn_window"]
        self._bn_stride = paths["bn_stride"]
        self._required_cache_size = paths["required_cache_size"]
        self._asr_offset_init = paths["asr_offset_init"]
        self._asr_offset_step = paths["asr_offset_step"]

        print("[Init] Loading ASR encoder (JIT)...", flush=True)
        self.asr = torch.jit.load(self._asr_ckpt)
        self.asr.eval()

        print("[Init] Loading VC model...", flush=True)
        self.vc = run_rt._load_vc_model(vc_config, vc_ckpt, device)

        print("[Init] Loading vocoder...", flush=True)
        self.vocoder = torch.jit.load(run_rt.VOCODER_PATH)
        if device == "cuda":
            self.vocoder = self.vocoder.to(device)

        print("[Init] Loading speaker model...", flush=True)
        self.spk_model = run_rt.init_speaker_model(
            run_rt.SPEAKER_MODEL_PATH,
            device,
            wavlm_config=run_rt.WAVLM_CONFIG_PATH,
        )

        # Set by set_target_wav()/set_target_embedding() before streaming.
        self.vc_spk_emb = None
        self.vc_gtm_kv = None

        self.chunk_size = 12 if model == "120ms" else 4
        self.block_size = 4
        # The 40+40 model uses 80 ms ASR windows; feed one window per call.
        self.CHUNK = 1280 if model == "40ms" else 2560

        self.vocoder_overlap = 2
        self.upsample_factor = 160
        self.vocoder_wav_overlap = self.upsample_factor
        self.vocoder_wav_pad = (
            (self.vocoder_overlap - 1) * self.upsample_factor
            - self.vocoder_wav_overlap
        )
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap)
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap)
        self._init_cache()

    def set_target_embedding(self, embedding: torch.Tensor) -> None:
        embedding = embedding.to(self.device)
        self.vc_spk_emb = embedding
        with torch.no_grad():
            self.vc_gtm_kv = self.vc.gtm(self.vc_spk_emb)
        self._init_cache()

    def set_target_wav(self, target_wav: str) -> torch.Tensor:
        print(f"[Voice] Extracting target embedding: {target_wav}", flush=True)
        embedding = run_rt.extract_embedding(
            self.spk_model,
            target_wav,
            device=self.device,
        )
        self.set_target_embedding(embedding)
        return embedding.detach().cpu()

    def process_chunk(self, samples: np.ndarray) -> np.ndarray | None:
        if self.vc_spk_emb is None or self.vc_gtm_kv is None:
            raise RuntimeError("Target speaker has not been selected")
        return super().process_chunk(samples)


app = FastAPI(title="MeanVC2 Morphly Streaming API", version="0.1.0")
engine: ServerVCRunner | None = None
engine_error: str | None = None
engine_loading = True
session_lock = threading.Lock()
speaker_cache: dict[str, torch.Tensor] = {}


def _load_engine() -> None:
    global engine, engine_error, engine_loading
    try:
        print(
            f"[Server] Loading MeanVC2 model={MODEL_NAME} device={DEVICE}",
            flush=True,
        )
        engine = ServerVCRunner(device=DEVICE, model=MODEL_NAME)
        print("[Server] MeanVC2 engine ready", flush=True)
    except Exception:
        engine_error = traceback.format_exc()
        print(engine_error, flush=True)
    finally:
        engine_loading = False


@app.on_event("startup")
async def startup_event() -> None:
    # Start listening immediately so RunPod /ping can return 204 while the
    # model initializes, then 200 when it is actually ready.
    threading.Thread(target=_load_engine, name="meanvc2-loader", daemon=True).start()


@app.get("/ping")
async def ping():
    if engine_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": engine_error[-2000:]},
        )
    if engine_loading or engine is None:
        return Response(status_code=204)
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
    }


@app.get("/info")
async def info():
    return {
        "service": "MeanVC2 Morphly Streaming API",
        "ready": engine is not None and not engine_loading and not engine_error,
        "model": MODEL_NAME,
        "device": DEVICE,
        "sample_rate": 16000,
        "input_format": "pcm_s16le_mono",
        "output_format": "pcm_s16le_mono",
        "model_window_ms": 80 if MODEL_NAME == "40ms" else 160,
        "upstream_input_block_samples": 1280 if MODEL_NAME == "40ms" else 2560,
        "upstream_input_block_ms": 80 if MODEL_NAME == "40ms" else 160,
        "concurrency_per_worker": 1,
    }


def _safe_voice_path(voice_id: str) -> Path:
    clean = Path(voice_id).name
    if not clean.lower().endswith(".wav"):
        clean += ".wav"
    return VOICE_DIR / clean


def _pcm16_bytes_to_float32(payload: bytes) -> np.ndarray:
    if len(payload) % 2:
        raise ValueError("PCM16 payload length must be divisible by 2")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def _float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


async def _apply_target_from_path(runner: ServerVCRunner, path: Path, cache_key: str) -> None:
    if cache_key in speaker_cache:
        await asyncio.to_thread(runner.set_target_embedding, speaker_cache[cache_key])
        return
    embedding = await asyncio.to_thread(runner.set_target_wav, str(path))
    speaker_cache[cache_key] = embedding


async def _apply_uploaded_target(runner: ServerVCRunner, wav_bytes: bytes) -> str:
    if not wav_bytes:
        raise ValueError("Target WAV is empty")
    if len(wav_bytes) > MAX_TARGET_BYTES:
        raise ValueError(f"Target WAV exceeds {MAX_TARGET_BYTES} bytes")

    digest = hashlib.sha256(wav_bytes).hexdigest()
    cache_key = f"sha256:{digest}"
    if cache_key in speaker_cache:
        await asyncio.to_thread(runner.set_target_embedding, speaker_cache[cache_key])
        return cache_key

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp.write(wav_bytes)
            temp_path = temp.name
        embedding = await asyncio.to_thread(runner.set_target_wav, temp_path)
        speaker_cache[cache_key] = embedding
        return cache_key
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


async def _warm_stream(runner: ServerVCRunner) -> None:
    # Warm the streaming path after selecting the target voice, then clear all
    # mutable caches so user audio starts from a clean session.
    zeros = np.zeros(runner.CHUNK, dtype=np.float32)
    for _ in range(3):
        await asyncio.to_thread(runner.process_chunk, zeros)
    runner._init_cache()


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    if engine_error:
        await websocket.send_json({"type": "error", "error": "engine_failed"})
        await websocket.close(code=1011)
        return
    if engine_loading or engine is None:
        await websocket.send_json({"type": "error", "error": "engine_initializing"})
        await websocket.close(code=1013)
        return
    if not session_lock.acquire(blocking=False):
        await websocket.send_json({"type": "error", "error": "worker_busy"})
        await websocket.close(code=1013)
        return

    runner = engine
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "sample_rate": 16000,
                "format": "pcm_s16le_mono",
                "recommended_frame_samples": runner.CHUNK,
                "recommended_frame_ms": runner.CHUNK / 16,
                "model": MODEL_NAME,
            }
        )

        first = await websocket.receive()
        if first.get("type") == "websocket.disconnect":
            return
        start_text = first.get("text")
        if not start_text:
            await websocket.send_json(
                {"type": "error", "error": "first_message_must_be_start_json"}
            )
            await websocket.close(code=1003)
            return

        try:
            start = json.loads(start_text)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "error": "invalid_start_json"})
            await websocket.close(code=1003)
            return

        if start.get("type") != "start":
            await websocket.send_json({"type": "error", "error": "expected_start"})
            await websocket.close(code=1008)
            return

        target_label = None
        voice_id = start.get("voice_id")
        if voice_id:
            voice_path = _safe_voice_path(str(voice_id))
            if not voice_path.is_file():
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "voice_not_found",
                        "voice_id": str(voice_id),
                    }
                )
                await websocket.close(code=1008)
                return
            await _apply_target_from_path(runner, voice_path, f"voice:{voice_path.name}")
            target_label = voice_path.name
        elif start.get("upload_target"):
            await websocket.send_json({"type": "need_target_wav"})
            target_message = await websocket.receive()
            wav_bytes = target_message.get("bytes")
            if wav_bytes is None:
                await websocket.send_json(
                    {"type": "error", "error": "target_wav_must_be_binary"}
                )
                await websocket.close(code=1003)
                return
            target_label = await _apply_uploaded_target(runner, wav_bytes)
        elif DEFAULT_TARGET_WAV:
            default_path = Path(DEFAULT_TARGET_WAV)
            if not default_path.is_file():
                await websocket.send_json(
                    {"type": "error", "error": "default_target_missing"}
                )
                await websocket.close(code=1011)
                return
            await _apply_target_from_path(
                runner,
                default_path,
                f"default:{default_path.resolve()}",
            )
            target_label = default_path.name
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "target_required",
                    "detail": "Send voice_id or upload_target=true",
                }
            )
            await websocket.close(code=1008)
            return

        await _warm_stream(runner)
        await websocket.send_json(
            {
                "type": "ready",
                "target": target_label,
                "sample_rate": 16000,
                "format": "pcm_s16le_mono",
            }
        )

        input_buffer = np.empty(0, dtype=np.float32)
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "invalid_control_json"})
                    continue
                if control.get("type") == "stop":
                    await websocket.send_json({"type": "stopped"})
                    break
                if control.get("type") == "reset":
                    runner._init_cache()
                    input_buffer = np.empty(0, dtype=np.float32)
                    await websocket.send_json({"type": "reset_ok"})
                continue

            payload = message.get("bytes")
            if payload is None:
                continue

            try:
                incoming = _pcm16_bytes_to_float32(payload)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
                continue

            if input_buffer.size == 0:
                input_buffer = incoming
            else:
                input_buffer = np.concatenate((input_buffer, incoming))

            while input_buffer.size >= runner.CHUNK:
                chunk = input_buffer[: runner.CHUNK]
                input_buffer = input_buffer[runner.CHUNK :]
                converted = await asyncio.to_thread(runner.process_chunk, chunk)
                if converted is not None and converted.size:
                    await websocket.send_bytes(_float32_to_pcm16_bytes(converted))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        try:
            await websocket.send_json(
                {"type": "error", "error": "stream_failure", "detail": str(exc)}
            )
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        runner._init_cache()
        session_lock.release()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
