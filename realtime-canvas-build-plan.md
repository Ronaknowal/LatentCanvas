# Build Plan: Real-Time Generative Canvas (Open Source)

## Project Overview

Build a Krea.ai-style real-time generative canvas where users draw on one side and see AI-generated photorealistic images on the other side, updating in near real-time (<500ms). The system also supports sketch-to-3D model generation.

**Target Stack (All Open Source / Apache 2.0 where possible):**

- **Base Model**: SDXL Turbo (single-step, fastest) or FLUX.1 [schnell] (highest quality, Apache 2.0)
- **Accelerator**: LCM-LoRA (universal adapter for any SD model)
- **Sketch Conditioning**: ControlNet Scribble (SDXL) or FLUX ControlNet Union Pro 2.0
- **Pipeline Optimizer**: StreamDiffusion (91 FPS on RTX 4090)
- **3D Generation**: Stable Fast 3D (SF3D) — 0.5s per mesh
- **Frontend**: Next.js 14 + tldraw SDK (or React Konva)
- **Backend**: FastAPI + WebSockets
- **GPU Hosting**: RunPod / Modal (dev), self-hosted (production)

---

## Phase 1: Weekend Prototype (2-3 days)

**Goal**: Get a working sketch → AI image pipeline running end-to-end using a hosted API, proving the concept before investing in self-hosted infrastructure.

### Step 1.1: Set Up the Frontend Canvas

```
Tech: Next.js 14 (App Router) + tldraw SDK
```

1. **Initialize the project:**
   ```bash
   npx create-next-app@latest realtime-canvas --typescript --tailwind --app
   cd realtime-canvas
   npm install tldraw @fal-ai/client
   ```

2. **Create the canvas component** (`app/components/Canvas.tsx`):
   - Use tldraw's `<Tldraw />` component for the drawing surface
   - Support freehand drawing, shapes, colors, eraser
   - Add a split-view layout: left = drawing canvas, right = AI output
   - Implement a **150ms debounce** on canvas changes to throttle API calls
   - On each canvas change, export the canvas as a base64 JPEG (70% quality) using tldraw's `editor.getSvg()` → rasterize to canvas → `toDataURL('image/jpeg', 0.7)`

3. **Add the AI Strength slider** (maps to `denoising_strength` 0.1–1.0):
   - Low values (0.2–0.4) = preserve sketch structure
   - High values (0.7–1.0) = creative interpretation

4. **Add a text prompt input** for style guidance (e.g., "photorealistic landscape", "anime style portrait")

### Step 1.2: Connect to fal.ai Real-Time API

```
Tech: fal.ai WebSocket API (fastest hosted option)
```

1. **Create a Next.js API route** as a proxy to protect your API key:
   ```
   app/api/fal/route.ts → uses @fal-ai/serverless-proxy
   ```

2. **Use fal.ai's real-time WebSocket connection** in the frontend:
   ```typescript
   import { fal } from "@fal-ai/client";

   const connection = fal.realtime.connect("fal-ai/lcm-sd15-i2i", {
     onResult: (result) => {
       setGeneratedImage(result.images[0].url);
     },
   });

   // On canvas change (throttled):
   connection.send({
     prompt: promptText,
     image_url: canvasBase64DataUrl,
     strength: aiStrength,
     num_inference_steps: 4,
     guidance_scale: 1.0,
     seed: 42, // Fixed seed for consistency
     enable_safety_checker: false,
   });
   ```

3. **Model options on fal.ai** (pick one):

   | Model | Endpoint | Speed | Quality | Cost |
   |-------|----------|-------|---------|------|
   | SD 1.5 + LCM | `fal-ai/lcm-sd15-i2i` | ~120ms GPU | Good | ~$0.002/img |
   | SDXL Turbo | `fal-ai/fast-turbo-diffusion` | ~200ms GPU | Better | ~$0.003/img |
   | SDXL Lightning | `fal-ai/fast-lightning-sdxl` | ~250ms GPU | Best SD | ~$0.003/img |
   | FLUX.1 Schnell | `fal-ai/flux/schnell` | ~400ms GPU | Best overall | ~$0.003/mp |

   **Recommendation for prototype**: Start with `fal-ai/lcm-sd15-i2i` (fastest, cheapest), then upgrade to SDXL Lightning or FLUX Schnell once the pipeline works.

### Step 1.3: Add ControlNet Scribble Conditioning

The key difference between img2img and a true "generative canvas" is **ControlNet** — it preserves the *structure* of your sketch while generating detail, rather than just using the sketch as a noisy starting point.

1. **On fal.ai**, use the SDXL ControlNet endpoint:
   ```
   fal-ai/fast-sdxl-controlnet-canny
   ```
   - Preprocess the canvas export with a client-side Canny edge detector (use OpenCV.js or a simple Sobel filter in a Web Worker)
   - Or use the scribble variant which accepts raw hand-drawn lines directly

2. **For FLUX models**, use the unified ControlNet endpoint:
   ```
   fal-ai/flux-general (supports ControlNet Union Pro 2.0)
   ```

### Step 1.4: Deploy the Prototype

1. Deploy frontend to **Vercel** (free tier works fine)
2. Environment variable: `FAL_KEY` in Vercel dashboard
3. Share and test with real users

**Expected result**: A working real-time canvas with 3-5 FPS visual updates, ~500ms round-trip latency.

---

## Phase 2: Self-Hosted Inference Backend (1-2 weeks)

**Goal**: Replace the API with a self-hosted StreamDiffusion pipeline for 10x lower latency, 250x lower per-image cost, and full control over the model stack.

### Step 2.1: Set Up the GPU Server

1. **Choose a GPU provider:**

   | Provider | GPU | Cost/hr | Best For |
   |----------|-----|---------|----------|
   | RunPod Community | RTX 4090 (24GB) | ~$0.44 | Dev/testing |
   | RunPod Secure | A100 80GB | ~$2.17 | Production |
   | Modal | A100 80GB | ~$2.78 | Serverless (scales to zero) |
   | Vast.ai | RTX 4090 | ~$0.30 | Budget dev |

   **Recommendation**: Start with RunPod RTX 4090 ($0.44/hr) for development.

2. **Base environment setup:**
   ```bash
   # On your GPU server
   conda create -n canvas python=3.10
   conda activate canvas

   # PyTorch with CUDA 12.4
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

   # Core dependencies
   pip install diffusers transformers accelerate safetensors
   pip install fastapi uvicorn websockets pillow
   ```

### Step 2.2: Install StreamDiffusion (V1 for Image Canvas)

StreamDiffusion is the key optimization layer that takes inference from ~3 FPS to ~30+ FPS through batched denoising, similarity filtering, and TensorRT compilation.

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
pip install -e .

# Install TensorRT acceleration (critical for max speed)
pip install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

**Note**: StreamDiffusionV2 (released Oct 2025, MLSys 2026) is for *video* diffusion models (Wan2.1). For an image-based canvas like Krea's, StreamDiffusion V1 is the correct choice and more mature.

### Step 2.3: Download the Models

```bash
# Option A: SDXL Turbo (fastest, single-step, 512×512)
# Best for: Maximum FPS real-time canvas
huggingface-cli download stabilityai/sdxl-turbo --local-dir models/sdxl-turbo

# Option B: SD 1.5 + LCM-LoRA (most flexible, 512×512)
# Best for: Compatibility with most ControlNets and LoRAs
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd15
huggingface-cli download latent-consistency/lcm-lora-sdv1-5 --local-dir models/lcm-lora-sd15

# Option C: SDXL + LCM-LoRA (best quality at speed, 1024×1024)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir models/sdxl
huggingface-cli download latent-consistency/lcm-lora-sdxl --local-dir models/lcm-lora-sdxl

# ControlNet Scribble (choose matching version)
huggingface-cli download lllyasviel/sd-controlnet-scribble --local-dir models/controlnet-scribble-sd15
# OR for SDXL:
huggingface-cli download xinsir/controlnet-scribble-sdxl-1.0 --local-dir models/controlnet-scribble-sdxl

# Tiny VAE (critical for speed — replaces default VAE decoder)
huggingface-cli download madebyollin/taesd --local-dir models/taesd
# OR for SDXL:
huggingface-cli download madebyollin/taesdxl --local-dir models/taesdxl
```

**Recommended starting config**: SD 1.5 + LCM-LoRA + ControlNet Scribble + Tiny VAE. This gives the best FPS on a single RTX 4090 and has the most battle-tested ControlNet support.

### Step 2.4: Build the StreamDiffusion Inference Pipeline

Create `server/pipeline.py`:

```python
import torch
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from diffusers import AutoencoderTiny, StableDiffusionPipeline, ControlNetModel

class RealtimeCanvasPipeline:
    def __init__(self, device="cuda"):
        # Load base model
        controlnet = ControlNetModel.from_pretrained(
            "models/controlnet-scribble-sd15",
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            "models/sd15",
            torch_dtype=torch.float16,
        ).to(device)

        # Load LCM-LoRA for few-step inference
        pipe.load_lora_weights("models/lcm-lora-sd15")
        pipe.fuse_lora()

        # Initialize StreamDiffusion wrapper
        self.stream = StreamDiffusion(
            pipe,
            t_index_list=[0, 16, 32, 45],  # 4-step denoising
            torch_dtype=torch.float16,
            cfg_type="self",  # RCFG Self-Negative (fastest)
        )

        # Load Tiny VAE for fast decoding
        self.stream.vae = AutoencoderTiny.from_pretrained(
            "models/taesd"
        ).to(device=device, dtype=torch.float16)

        # Prepare with default prompt
        self.stream.prepare(
            prompt="high quality, detailed",
            guidance_scale=1.2,
            delta=0.5,
        )

        # Enable TensorRT acceleration
        # (first call compiles — takes ~5 min, then cached)
        self.stream.enable_similar_image_filter(
            similar_image_filter_threshold=0.98
        )

    def generate(self, sketch_image, prompt=None, strength=0.5):
        """
        sketch_image: PIL.Image (from canvas export)
        prompt: str (style guidance)
        strength: float 0-1 (AI interpretation level)
        """
        if prompt:
            self.stream.prepare(
                prompt=prompt,
                guidance_scale=1.0 + strength,
            )

        # Feed sketch through img2img with ControlNet
        output = self.stream(sketch_image)
        return postprocess_image(output, output_type="pil")[0]
```

### Step 2.5: Build the WebSocket Server

Create `server/main.py`:

```python
import asyncio
import io
import base64
from fastapi import FastAPI, WebSocket
from PIL import Image
from pipeline import RealtimeCanvasPipeline

app = FastAPI()
pipeline = RealtimeCanvasPipeline()

@app.websocket("/ws/generate")
async def generate_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive binary JPEG frame from canvas
            data = await websocket.receive_bytes()

            # Decode the sketch image
            sketch = Image.open(io.BytesIO(data)).convert("RGB")
            sketch = sketch.resize((512, 512))

            # Generate (runs on GPU)
            result = await asyncio.to_thread(
                pipeline.generate, sketch
            )

            # Send result back as binary JPEG
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=85)
            await websocket.send_bytes(buffer.getvalue())

    except Exception as e:
        print(f"Connection closed: {e}")

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 2.6: Update Frontend to Use Self-Hosted Backend

Replace the fal.ai connection with a direct WebSocket to your server:

```typescript
// In your Canvas component
const ws = useRef<WebSocket | null>(null);

useEffect(() => {
  ws.current = new WebSocket("wss://your-gpu-server:8000/ws/generate");
  ws.current.binaryType = "arraybuffer";

  ws.current.onmessage = (event) => {
    const blob = new Blob([event.data], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    setGeneratedImage(url);
  };

  return () => ws.current?.close();
}, []);

// On canvas change (throttled to 150ms):
const sendSketch = useCallback(
  throttle(async () => {
    const canvas = canvasRef.current;
    canvas.toBlob(
      (blob) => {
        if (ws.current?.readyState === WebSocket.OPEN && blob) {
          blob.arrayBuffer().then((buf) => ws.current!.send(buf));
        }
      },
      "image/jpeg",
      0.7
    );
  }, 150),
  []
);
```

**Expected result**: 10-30+ FPS on RTX 4090, <100ms GPU inference, ~200ms round-trip.

---

## Phase 3: Production Polish & Advanced Features (2-4 weeks)

### Step 3.1: Add ControlNet Properly with Diffusers

For production, integrate ControlNet directly rather than relying on StreamDiffusion's basic img2img:

```python
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
)

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
)

# Build pipeline with ControlNet
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# Swap to LCM scheduler for few-step inference
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()

# Tiny VAE
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=torch.float16
).to("cuda")

# Compile with torch.compile for extra speed
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Generate
result = pipe(
    prompt="photorealistic landscape, high detail",
    image=sketch_image,         # The canvas export
    control_image=sketch_image, # Same image as control signal
    strength=0.6,               # AI interpretation level
    controlnet_conditioning_scale=0.7,  # How much to follow the sketch
    num_inference_steps=4,
    guidance_scale=1.5,
).images[0]
```

### Step 3.2: Upgrade to FLUX.1 (Highest Quality Path)

For maximum quality (at the cost of needing more VRAM), switch to FLUX:

```python
from diffusers import FluxControlNetPipeline, FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    torch_dtype=torch.bfloat16
)

pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",  # Apache 2.0 license!
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
).to("cuda")

# FLUX.1 Schnell only needs 4 steps natively (no LCM needed!)
result = pipe(
    prompt="photorealistic, detailed",
    control_image=sketch_image,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,
    num_inference_steps=4,
    guidance_scale=3.5,
    width=1024,
    height=1024,
).images[0]
```

**VRAM Requirements**: FLUX needs 16-24GB. Use Nunchaku/SVDQuant 4-bit quantization to fit on 12GB:
```bash
pip install nunchaku
# Reduces FLUX VRAM from 24GB → 8-12GB with minimal quality loss
```

**Licensing note**: FLUX.1 [schnell] = Apache 2.0 (commercial OK). FLUX.1 [dev] = non-commercial research only. For a commercial product, use schnell.

### Step 3.3: Add Sketch-to-3D Generation

Pipeline: Canvas sketch → 2D image (from Step 3.1) → 3D mesh

1. **Install SF3D (Stable Fast 3D)**:
   ```bash
   git clone https://github.com/Stability-AI/stable-fast-3d.git
   cd stable-fast-3d
   pip install -e .
   ```

2. **Or use TripoSR** (simpler, also sub-second):
   ```bash
   git clone https://github.com/VAST-AI-Research/TripoSR.git
   cd TripoSR
   pip install -e .
   ```

3. **Integrate into the pipeline:**
   ```python
   from sf3d.system import SF3D

   sf3d_model = SF3D.from_pretrained("stabilityai/stable-fast-3d").to("cuda")

   def sketch_to_3d(generated_2d_image):
       """Takes the AI-generated 2D image and creates a 3D mesh"""
       mesh = sf3d_model.generate(
           generated_2d_image,
           remesh="quad",  # Clean quad topology
           texture_resolution=1024,
       )
       mesh.export("output.glb")  # GLB format for web viewing
       return mesh
   ```

4. **Frontend 3D viewer** — Use Three.js or `@react-three/fiber`:
   ```bash
   npm install three @react-three/fiber @react-three/drei
   ```
   Load the `.glb` mesh and display it in an interactive 3D viewport next to the canvas.

5. **Alternative: SPAR3D** (interactive point cloud editing):
   - Users can edit a sparse 3D point cloud before final mesh generation
   - Two-stage: 0.4s point cloud + 0.3s mesh = 0.7s total

### Step 3.4: Add TensorRT Compilation (2x Speedup)

```python
# After loading the pipeline, compile the UNet to TensorRT
# This takes ~5-10 minutes on first run, then caches

# Option A: torch.compile (easier, ~1.5x speedup)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Option B: TensorRT (harder to set up, ~2x speedup)
# Use the NVIDIA TensorRT Model Optimizer
pip install nvidia-modelopt
# Then convert UNet to TensorRT INT8:
# See: github.com/NVIDIA/TensorRT-Model-Optimizer
```

### Step 3.5: Implement the Stochastic Similarity Filter

This is one of StreamDiffusion's most impactful optimizations — skip GPU inference entirely when the canvas hasn't changed significantly:

```python
import numpy as np
from PIL import Image

class SimilarityFilter:
    def __init__(self, threshold=0.98):
        self.threshold = threshold
        self.prev_image = None

    def should_generate(self, new_image: Image.Image) -> bool:
        """Returns False if the image is too similar to skip generation"""
        arr = np.array(new_image.resize((64, 64))).flatten().astype(float)

        if self.prev_image is None:
            self.prev_image = arr
            return True

        # Cosine similarity
        similarity = np.dot(arr, self.prev_image) / (
            np.linalg.norm(arr) * np.linalg.norm(self.prev_image) + 1e-8
        )

        if similarity > self.threshold:
            return False  # Skip — canvas barely changed

        self.prev_image = arr
        return True
```

This saves 50-80% of GPU cycles during idle moments or slow drawing.

### Step 3.6: Build the Request Cancellation System

When a user draws fast, many frames queue up. Only the latest matters:

```python
import asyncio

class InferenceQueue:
    def __init__(self):
        self.latest_request = None
        self.lock = asyncio.Lock()

    async def submit(self, image_data):
        """Replace any pending request with the newest one"""
        async with self.lock:
            self.latest_request = image_data

    async def get_latest(self):
        """Get and clear the latest request"""
        async with self.lock:
            request = self.latest_request
            self.latest_request = None
            return request
```

---

## Phase 4: Scale & Deploy (Ongoing)

### Step 4.1: Multi-User Architecture

```
                    ┌─────────────┐
  Users ──WSS──→    │  NGINX      │
                    │  (SSL +     │──→  FastAPI Server 1 (GPU: RTX 4090)
                    │   Load      │──→  FastAPI Server 2 (GPU: RTX 4090)
                    │   Balancer) │──→  FastAPI Server 3 (GPU: A100)
                    └─────────────┘
```

- Each GPU server handles ~5-10 concurrent users at 3 FPS each
- Use sticky sessions (WebSocket affinity) in NGINX
- Each server loads the model once, shares across connections

### Step 4.2: Cost Optimization Table

| Scale | Setup | Monthly Cost | Per-Image Cost |
|-------|-------|-------------|----------------|
| Prototype | fal.ai API | ~$50-200 (usage) | $0.002-0.003 |
| 10 users | 1× RTX 4090 RunPod | ~$320 | $0.00004 |
| 50 users | 3× RTX 4090 RunPod | ~$960 | $0.00003 |
| 100+ users | 2× A100 RunPod | ~$3,120 | $0.00002 |

Self-hosting becomes dramatically cheaper past ~10 concurrent users.

### Step 4.3: Model Selection Decision Tree

```
Need commercial license?
├── Yes → FLUX.1 [schnell] (Apache 2.0) + 4 steps
│         └── VRAM: 16-24GB (or 8-12GB with Nunchaku quantization)
│
└── No → What's your priority?
    ├── Maximum FPS → SD 1.5 + LCM-LoRA + StreamDiffusion
    │               └── VRAM: 4-6GB, Speed: 30-90 FPS on 4090
    │
    ├── Best quality/speed balance → SDXL Turbo (1 step)
    │                               └── VRAM: 8-10GB, Speed: 15-30 FPS
    │
    └── Best quality → SDXL + Hyper-SD LoRA (1-4 steps)
                       └── VRAM: 12-16GB, Speed: 8-15 FPS
```

---

## Key Open-Source Repos Reference

| Component | Repo | Stars | Purpose |
|-----------|------|-------|---------|
| StreamDiffusion | `cumulo-autumn/StreamDiffusion` | 10.7k | Pipeline optimizer (91 FPS) |
| StreamDiffusionV2 | `chenfengxu714/StreamDiffusionV2` | New | Video streaming pipeline |
| LCM-LoRA | `luosiallen/latent-consistency-model` | 7k+ | Universal acceleration adapter |
| ControlNet | `lllyasviel/ControlNet` | 30k+ | Sketch/structure conditioning |
| FLUX ControlNet Union | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | — | Multi-mode FLUX control |
| TripoSR | `VAST-AI-Research/TripoSR` | 5k+ | Single-image to 3D (<0.5s) |
| SF3D | `Stability-AI/stable-fast-3d` | 2k+ | Fast 3D with PBR textures |
| SPAR3D | `Stability-AI/spar3d` | 1k+ | Interactive 3D with point editing |
| TRELLIS.2 | `microsoft/TRELLIS` | 8k+ | Production-quality 3D meshes |
| Nunchaku | `mit-han-lab/nunchaku` | 3k+ | 4-bit quantization for FLUX |
| tldraw | `tldraw/tldraw` | 38k+ | Open-source canvas SDK |
| draw-fast | `tldraw/draw-fast` | 1k+ | tldraw + fal.ai real-time demo |
| Infinite Kanvas | `fal-ai-community/infinite-kanvas` | — | Next.js + React Konva + fal.ai |

---

## Key Papers to Read

1. **LCM** — "Latent Consistency Models" (Luo et al., 2023) — The distillation breakthrough
2. **SDXL Turbo** — "Adversarial Diffusion Distillation" (Stability AI, 2023) — Single-step generation
3. **SDXL Lightning** — "Progressive Adversarial Diffusion Distillation" (ByteDance, 2024) — Best quality/speed
4. **StreamDiffusion** — arXiv:2312.12491 — Pipeline-level optimizations for real-time
5. **Hyper-SD** — arXiv:2404.13686 — Trajectory Segmented Consistency Distillation
6. **ControlNet** — "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al.)
7. **StreamDiffusionV2** — arXiv:2511.07399 — Video streaming pipeline (MLSys 2026)

---

## Milestone Checklist

- [ ] **M1 (Day 1-2)**: Frontend canvas with tldraw, split-view layout, prompt input, AI strength slider
- [ ] **M2 (Day 2-3)**: fal.ai WebSocket integration working, seeing real-time AI output from sketches
- [ ] **M3 (Day 3)**: Deploy prototype to Vercel, test with friends
- [ ] **M4 (Week 2)**: Set up RunPod GPU, install StreamDiffusion + SD 1.5 + LCM-LoRA + ControlNet
- [ ] **M5 (Week 2)**: FastAPI WebSocket server running, replace fal.ai with self-hosted
- [ ] **M6 (Week 3)**: TensorRT compilation, similarity filter, request cancellation
- [ ] **M7 (Week 3)**: Upgrade to SDXL or FLUX for higher quality
- [ ] **M8 (Week 4)**: Integrate SF3D/TripoSR for sketch-to-3D
- [ ] **M9 (Week 4)**: Three.js 3D model viewer in frontend
- [ ] **M10 (Week 5+)**: Multi-user support, NGINX load balancing, production deployment
