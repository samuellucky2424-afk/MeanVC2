# Morphly + MeanVC2 on RunPod Serverless

This branch adds a production-oriented baseline for running MeanVC2 behind a RunPod **Load Balancer** endpoint and streaming live audio from Morphly over WebSocket.

## Why Load Balancer instead of `/run` or `/runsync`

Live voice conversion needs a persistent low-latency connection. RunPod load-balanced endpoints route HTTP/WebSocket traffic directly to a worker without the queue used by traditional Serverless endpoints.

## Files added

- `Dockerfile` — CUDA/PyTorch image, dependencies and MeanVC2 model download.
- `server.py` — FastAPI health/info routes plus a binary WebSocket audio stream.
- `server_requirements.txt` — API/container-only dependencies.
- `.dockerignore` — keeps local caches/checkpoints out of Docker build context.

## Deploy from GitHub on RunPod

1. In RunPod, create a new Serverless endpoint.
2. Choose **Import Git Repository** and select this repository/branch.
3. Dockerfile path: `Dockerfile`.
4. Endpoint type: **Load Balancer**.
5. Start with a 16 GB or 24 GB NVIDIA GPU. An L4/24 GB class GPU is a sensible production starting point.
6. Keep `PORT=80` and `PORT_HEALTH=80` (the Dockerfile already sets both).
7. Recommended variables:
   - `MEANVC_MODEL=40ms`
   - `MEANVC_DEVICE=cuda`
   - `VOICE_DIR=/app/voices`
8. For the first real-time test, set at least one active worker to avoid cold-start delay during a live call. Scale/max-worker settings can be tuned after latency measurements.

RunPod health-checks `GET /ping`:

- `204` while the model is initializing.
- `200` after the engine is ready.
- `500` if model initialization failed.

`GET /info` returns the stream format and engine configuration.

## WebSocket URL

After deployment:

```text
wss://ENDPOINT_ID.api.runpod.ai/ws
```

Morphly should send its normal RunPod authorization header when opening the socket.

## Stream protocol

The baseline uses **mono, 16 kHz, signed 16-bit little-endian PCM** for live audio.

### 1. Connect

Server replies with JSON similar to:

```json
{
  "type": "hello",
  "sample_rate": 16000,
  "format": "pcm_s16le_mono",
  "recommended_frame_samples": 2560,
  "recommended_frame_ms": 160,
  "model": "40ms"
}
```

### 2A. Select a server-side voice

Place consented reference WAV files under `/app/voices`, then send:

```json
{"type":"start","voice_id":"sophia"}
```

The server resolves this to `/app/voices/sophia.wav`.

### 2B. Or upload a reference WAV for the session

Send:

```json
{"type":"start","upload_target":true}
```

The server answers:

```json
{"type":"need_target_wav"}
```

Send the complete reference WAV as the next **binary** WebSocket frame. The worker extracts and caches its speaker embedding by SHA-256.

### 3. Wait for ready

```json
{"type":"ready", "sample_rate":16000, "format":"pcm_s16le_mono"}
```

### 4. Stream microphone PCM

Send binary PCM16LE microphone frames. The server buffers input to the upstream driver's current 2560-sample block size, runs MeanVC2, and returns converted PCM16LE as binary frames.

Morphly can route returned PCM to its playback/virtual-microphone path (for example VB-CABLE).

### 5. Stop/reset controls

```json
{"type":"reset"}
```

clears the streaming caches without disconnecting.

```json
{"type":"stop"}
```

ends the session cleanly.

## Concurrency

This first deployment intentionally permits **one active WebSocket voice session per worker**. MeanVC2's current runtime keeps mutable ASR, VC KV, noise, BN and vocoder caches on the runner, so sharing one runner between users would mix session state. RunPod can scale multiple workers for concurrent users.

After the baseline is proven, the next optimization is to split immutable model weights from per-user stream state so one GPU worker can safely host multiple simultaneous Morphly sessions.

## Latency note

Although the repository includes a 40ms MeanVC2 model, the current upstream `runtime/run_rt.py` microphone driver sets `CHUNK = 2560`, which is 160ms of 16 kHz audio per host input block. This server preserves that behavior for correctness on the first deployment. Reducing transport/inference buffering is a separate optimization and should be benchmarked against the upstream 40ms streaming schedule before changing production behavior.

## Model files in the Docker build

`initialization.py` downloads the ASR, MeanVC2 and Vocos checkpoints, but upstream requires the fine-tuned WavLM/ECAPA speaker checkpoint to be downloaded manually from Google Drive. The Dockerfile automates that step.

The official initialization process also downloads the 1.2 GB WavLM base checkpoint to extract `wavlm_large_cfg.pt`. Runtime reconstructs WavLM from that small config plus the fine-tuned checkpoint, so the Dockerfile removes the redundant base checkpoint after initialization to reduce image size.

## Security / voice consent

Only expose reference voices that Morphly is authorized to use. Do not accept arbitrary server file paths from clients. The API therefore uses sanitized `voice_id` names or an explicit reference-WAV upload rather than accepting a filesystem path from the network.
