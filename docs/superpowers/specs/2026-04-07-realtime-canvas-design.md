# Real-Time Generative Canvas - Design Spec

## Overview

A Krea.ai-style real-time generative canvas where users draw on one side and see AI-generated photorealistic images on the other, updating in near real-time. The system also supports sketch-to-3D model generation (Phase 2).

## Decisions

- **Backend-first build order**: Validate StreamDiffusion pipeline on RunPod before building the frontend.
- **Architecture**: FastAPI + GPU thread pool (Approach C). Migrate to split worker architecture (Approach B with Redis) when scaling demands it.
- **Starting model**: SD 1.5 + LCM-LoRA + ControlNet Scribble + TinyVAE (best FPS, most mature ControlNet support).
- **GPU**: RunPod Pod with NVIDIA A40 (48GB VRAM).
- **No fal.ai**: Direct WebSocket from Next.js to FastAPI on RunPod.
- **3D generation**: Designed for from the start, built in Phase 2.

## System Architecture

```
┌──────────────────────┐       WebSocket (binary JPEG)       ┌─────────────────────────────┐
│   Next.js 14 Frontend│◄──────────────────────────────────► │   RunPod Pod (A40, 48GB)    │
│                      │                                     │                             │
│  ┌────────────────┐  │       REST (JSON + binary)          │  ┌───────────────────────┐  │
│  │ tldraw Canvas  │  │◄──────────────────────────────────► │  │  FastAPI Server        │  │
│  │ (drawing)      │  │                                     │  │                       │  │
│  ├────────────────┤  │                                     │  │  WS /ws/generate      │  │
│  │ AI Output View │  │                                     │  │  POST /api/generate-3d│  │
│  │ (generated img)│  │                                     │  │  GET /api/health      │  │
│  ├────────────────┤  │                                     │  ├───────────────────────┤  │
│  │ 3D Viewer      │  │                                     │  │  GPU Thread Pool      │  │
│  │ (Three.js)     │  │                                     │  │  ┌─────────────────┐  │  │
│  ├────────────────┤  │                                     │  │  │ StreamDiffusion  │  │  │
│  │ Controls       │  │                                     │  │  │ SD1.5+LCM+CN    │  │  │
│  │ - Prompt input │  │                                     │  │  │ + TinyVAE        │  │  │
│  │ - AI strength  │  │                                     │  │  ├─────────────────┤  │  │
│  │ - Model select │  │                                     │  │  │ SF3D / TripoSR  │  │  │
│  └────────────────┘  │                                     │  │  │ (Phase 2)       │  │  │
│                      │                                     │  │  └─────────────────┘  │  │
│  Deployed: Vercel    │                                     │  └───────────────────────┘  │
└──────────────────────┘                                     └─────────────────────────────┘
```

### Data Flow: Real-Time Canvas

1. User draws on tldraw canvas.
2. Canvas change triggers 150ms debounced export to JPEG (70% quality).
3. Binary JPEG sent over WebSocket to FastAPI.
4. FastAPI dispatches to GPU thread: StreamDiffusion (SD1.5 + LCM-LoRA + ControlNet Scribble + TinyVAE).
5. Generated image returned as binary JPEG over same WebSocket.
6. Frontend displays result in AI Output View.

### Data Flow: 3D Generation (Phase 2)

1. User clicks "Generate 3D" on a generated image.
2. POST request with the generated image to `/api/generate-3d`.
3. SF3D/TripoSR produces a `.glb` mesh (~0.5s).
4. GLB returned, loaded into Three.js viewer.

## Backend Design

### Project Structure

```
server/
├── main.py                 # FastAPI app, WebSocket + REST endpoints
├── pipeline.py             # StreamDiffusion pipeline wrapper
├── pipeline_3d.py          # SF3D/TripoSR wrapper (Phase 2)
├── similarity_filter.py    # Cosine similarity skip filter
├── inference_queue.py      # Request cancellation (latest-wins)
├── config.py               # Model paths, server settings
├── requirements.txt
└── Dockerfile
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ws/generate` | WebSocket | Real-time sketch-to-image stream |
| `/api/generate-3d` | POST | One-shot image-to-3D mesh (Phase 2) |
| `/api/health` | GET | Health check + GPU status |
| `/api/config` | GET | Available models, current settings |

### WebSocket Protocol

- Client sends binary JPEG frames (sketch images).
- Client sends JSON messages for config updates (prompt, strength, model params).
- Server sends binary JPEG frames (generated images).
- Message discrimination: first byte `{` (0x7B) = JSON config, otherwise = binary image.

### Pipeline Initialization (Warm-Up on Startup)

1. Load SD1.5 base model (float16).
2. Load + fuse LCM-LoRA weights.
3. Load ControlNet Scribble model.
4. Replace default VAE with TinyVAE (madebyollin/taesd).
5. Wrap in StreamDiffusion with 4-step denoising schedule `[0, 16, 32, 45]`.
6. Run `torch.compile` on UNet (`mode="reduce-overhead", fullgraph=True`).
7. Run dummy inference with blank 512x512 image to trigger compilation and warm caches.
8. Enable similarity filter.
9. Server reports `{"status": "ready"}` on health endpoint and begins accepting WebSocket connections.

Health endpoint returns `{"status": "warming_up"}` during steps 1-8. WebSocket connections are rejected until ready.

### Key Optimizations

1. **Similarity filter**: Cosine similarity on 64x64 thumbnails (threshold 0.98). Skips inference when canvas hasn't changed meaningfully. Saves 50-80% GPU cycles during idle/slow drawing.
2. **Latest-wins queue**: When user draws fast, only the most recent frame is processed. Stale frames are dropped.
3. **TinyVAE**: Replaces default VAE decoder for ~2x faster image decode.
4. **torch.compile**: Applied to UNet on startup for ~1.5x speedup. TensorRT INT8 as future upgrade.
5. **Warm-up on startup**: Dummy inference at launch so first real request is fast.

## Frontend Design

### Project Structure

```
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── components/
│   │   ├── Canvas.tsx          # tldraw drawing surface
│   │   ├── AIOutputView.tsx    # Generated image display
│   │   ├── ThreeDViewer.tsx    # Three.js GLB viewer (Phase 2)
│   │   ├── ControlPanel.tsx    # Prompt, strength slider, settings
│   │   └── ConnectionStatus.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts     # WebSocket connection + reconnect
│   │   └── useCanvasExport.ts  # tldraw -> JPEG export with debounce
│   └── lib/
│       └── config.ts
├── public/
├── package.json
├── tailwind.config.ts
└── next.config.ts
```

### Layout

Split-view: left half = tldraw drawing canvas, right half = AI-generated output. Controls bar at top or bottom.

### Canvas Export Pipeline

1. tldraw `onChange` fires on every stroke.
2. 150ms debounce batches rapid changes.
3. Export tldraw content to SVG -> rasterize to offscreen canvas -> `toBlob('image/jpeg', 0.7)`.
4. Send binary blob over WebSocket.

### WebSocket Client Behavior

- Auto-connect on mount.
- Auto-reconnect with exponential backoff on disconnect.
- Wait for server `ready` status before sending frames.
- Display connection state (connecting / connected / reconnecting).

### Controls

- Text prompt input (style guidance, e.g., "photorealistic landscape").
- AI strength slider (0.1 - 1.0, maps to denoising strength).
- Connection status indicator.
- "Generate 3D" button on AI output (Phase 2, disabled until implemented).

### State Management

React state + refs. No external state library needed. State is simple: prompt string, strength number, WebSocket ref, latest generated image URL.

## RunPod Deployment

### Pod Configuration

- GPU: NVIDIA A40 (48GB VRAM).
- Base image: RunPod official PyTorch template.
- Exposed port: 8000 (FastAPI via RunPod HTTPS proxy).
- Persistent volume at `/workspace` for model weights and torch.compile cache.

### Model Storage

- First launch: download models from HuggingFace to `/workspace/models/`.
- Subsequent launches: load from local persistent volume (fast).
- torch.compile cache persists to `/workspace/.cache/` so warm-up is only slow on the very first pod start (~5 min). Subsequent restarts: ~30s.

### Networking

- RunPod HTTPS proxy URL: `https://{pod-id}-8000.proxy.runpod.net`
- WebSocket: `wss://{pod-id}-8000.proxy.runpod.net/ws/generate`
- Frontend env var: `NEXT_PUBLIC_BACKEND_URL`

### Health Check

1. Pod starts -> FastAPI starts -> pipeline warm-up begins.
2. `/api/health` returns `{"status": "warming_up"}`.
3. Warm-up completes -> returns `{"status": "ready", "gpu": "A40", "model": "sd1.5-lcm-controlnet"}`.
4. Frontend polls health before establishing WebSocket.

## Upgrade Path

### Model Upgrades

| Phase | Model | Quality | Est. FPS (A40) | VRAM |
|-------|-------|---------|----------------|------|
| Phase 1 | SD 1.5 + LCM-LoRA + ControlNet Scribble | Good | 20-50 | ~6GB |
| Upgrade A | SDXL Turbo (single-step) | Better | 10-20 | ~10GB |
| Upgrade B | SDXL + LCM-LoRA + ControlNet Scribble SDXL | Best SD | 8-15 | ~14GB |
| Upgrade C | FLUX.1 Schnell + ControlNet Union Pro 2.0 | Best overall | 4-8 | ~20GB |

Model presets defined in `config.py` so switching is a config change. A40's 48GB VRAM fits FLUX without quantization.

### 3D Generation (Phase 2)

- SF3D or TripoSR loaded as a second pipeline on demand.
- REST endpoint `POST /api/generate-3d` accepts generated 2D image, returns GLB binary.
- Coexists on same A40: SD 1.5 (~6GB) + SF3D (~4GB) = ~10GB, well within 48GB.
- Frontend: Three.js viewer with orbit controls.
- Could later move to RunPod Serverless (one-shot requests, not streaming).

### Scaling to Multi-User (Approach B Migration)

- Trigger: concurrent users exceed ~5-10 at 3 FPS each.
- Add Redis as message broker between FastAPI and GPU workers.
- FastAPI becomes stateless connection handler.
- GPU workers become separate processes, can span multiple pods.
- NGINX with sticky sessions (WebSocket affinity) in front of multiple FastAPI instances.

### TensorRT Upgrade

- Replace `torch.compile` with full TensorRT INT8 compilation.
- ~2x additional speedup over torch.compile.
- Worth doing once pipeline is stable and model choice is finalized.

## Build Order

1. **Backend pipeline**: StreamDiffusion + SD1.5 + LCM-LoRA + ControlNet + TinyVAE on RunPod A40. Includes similarity filter, latest-wins queue, torch.compile warm-up, and health endpoint — these are part of the core server, not separate steps.
2. **Backend server**: FastAPI with WebSocket endpoint, REST endpoints, warm-up lifecycle.
3. **Frontend**: Next.js + tldraw canvas, WebSocket client, split-view layout, controls.
4. **Integration + deploy**: Connect frontend to RunPod backend, end-to-end testing, deploy frontend to Vercel.
5. **3D generation** (Phase 2): SF3D/TripoSR endpoint + Three.js viewer.
6. **Model upgrades**: SDXL / FLUX when quality needs increase.
7. **Scaling** (Phase 3): Approach B migration with Redis + NGINX.
