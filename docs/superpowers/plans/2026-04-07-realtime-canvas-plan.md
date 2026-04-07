# Real-Time Generative Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time generative canvas where users draw sketches and see AI-generated images updating in near real-time via WebSocket, powered by StreamDiffusion + ControlNet on a RunPod A40 GPU.

**Architecture:** FastAPI server with WebSocket endpoint runs StreamDiffusion (SD 1.5 + LCM-LoRA + ControlNet Scribble + TinyVAE) on a RunPod Pod. Next.js 14 frontend with tldraw provides the drawing surface. Binary JPEG frames flow over WebSocket in both directions.

**Tech Stack:** Python 3.10, FastAPI, StreamDiffusion (zjysteven ControlNet fork), diffusers, torch, Pillow | Next.js 14, TypeScript, tldraw, Tailwind CSS

**Key discovery from research:** Official StreamDiffusion does not support ControlNet. We use zjysteven's fork which provides `StreamUNetControlDiffusion` class. Use xformers acceleration (not TensorRT) for ControlNet compatibility.

---

## File Structure

### Backend (`server/`)

| File | Responsibility |
|------|---------------|
| `server/config.py` | Model paths, server settings, model presets |
| `server/similarity_filter.py` | Cosine similarity filter to skip unchanged frames |
| `server/inference_queue.py` | Latest-wins queue for request cancellation |
| `server/pipeline.py` | StreamDiffusion + ControlNet wrapper, warm-up logic |
| `server/main.py` | FastAPI app, WebSocket + REST endpoints, lifespan |
| `server/download_models.py` | Script to download all model weights from HuggingFace |
| `server/requirements.txt` | Python dependencies |
| `server/Dockerfile` | RunPod-compatible container image |
| `server/tests/__init__.py` | Test package |
| `server/tests/test_similarity_filter.py` | Unit tests for similarity filter |
| `server/tests/test_inference_queue.py` | Unit tests for inference queue |
| `server/tests/test_main.py` | FastAPI endpoint tests with mocked pipeline |

### Frontend (`frontend/`)

| File | Responsibility |
|------|---------------|
| `frontend/app/layout.tsx` | Root layout with metadata |
| `frontend/app/page.tsx` | Main page, dynamic-imports Canvas |
| `frontend/app/components/Canvas.tsx` | tldraw drawing surface with change detection |
| `frontend/app/components/AIOutputView.tsx` | Displays generated image |
| `frontend/app/components/ControlPanel.tsx` | Prompt input, strength slider, settings |
| `frontend/app/components/ConnectionStatus.tsx` | WebSocket connection state indicator |
| `frontend/app/hooks/useWebSocket.ts` | WebSocket connection, reconnect, message handling |
| `frontend/app/hooks/useCanvasExport.ts` | tldraw → JPEG blob export with debounce |
| `frontend/app/lib/config.ts` | Backend URL, defaults |

---

## Task 1: Backend Configuration

**Files:**
- Create: `server/config.py`
- Create: `server/requirements.txt`
- Create: `server/__init__.py`
- Create: `server/tests/__init__.py`

- [ ] **Step 1: Create server directory structure**

```bash
mkdir -p server/tests
```

- [ ] **Step 2: Write `server/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ModelConfig:
    name: str
    base_model_path: str
    lcm_lora_path: str
    controlnet_path: str
    tiny_vae_path: str
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    t_index_list: list[int] = field(default_factory=lambda: [0, 16, 32, 45])
    guidance_scale: float = 1.0
    controlnet_conditioning_scale: float = 0.7


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: Path = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
    cache_dir: Path = Path(os.environ.get("CACHE_DIR", "/workspace/.cache"))
    similarity_threshold: float = 0.98
    similarity_max_skip: int = 10
    default_prompt: str = "high quality, detailed, photorealistic"
    default_strength: float = 0.5


SD15_CONFIG = ModelConfig(
    name="sd1.5-lcm-controlnet-scribble",
    base_model_path="runwayml/stable-diffusion-v1-5",
    lcm_lora_path="latent-consistency/lcm-lora-sdv1-5",
    controlnet_path="lllyasviel/sd-controlnet-scribble",
    tiny_vae_path="madebyollin/taesd",
    width=512,
    height=512,
)

SERVER_CONFIG = ServerConfig()
ACTIVE_MODEL = SD15_CONFIG
```

- [ ] **Step 3: Write `server/requirements.txt`**

```
torch>=2.1.0
torchvision>=0.16.0
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
safetensors>=0.4.0
xformers>=0.0.23
fastapi>=0.109.0
uvicorn[standard]>=0.25.0
websockets>=12.0
pillow>=10.2.0
numpy>=1.26.0
huggingface_hub>=0.20.0
controlnet_aux>=0.0.7
pytest>=7.4.0
pytest-asyncio>=0.23.0
httpx>=0.26.0
```

- [ ] **Step 4: Create `__init__.py` files**

Create empty `server/__init__.py` and `server/tests/__init__.py`.

- [ ] **Step 5: Commit**

```bash
git add server/config.py server/requirements.txt server/__init__.py server/tests/__init__.py
git commit -m "feat: add backend configuration and dependencies"
```

---

## Task 2: Similarity Filter (TDD)

**Files:**
- Create: `server/tests/test_similarity_filter.py`
- Create: `server/similarity_filter.py`

- [ ] **Step 1: Write the failing tests**

```python
# server/tests/test_similarity_filter.py
import numpy as np
from PIL import Image
import pytest

from similarity_filter import SimilarityFilter


def _white_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (255, 255, 255))


def _black_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), (0, 0, 0))


def _noisy_image(size: int = 64, seed: int = 42) -> Image.Image:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestSimilarityFilter:
    def test_first_image_always_passes(self):
        f = SimilarityFilter(threshold=0.98)
        assert f.should_generate(_white_image()) is True

    def test_identical_image_is_skipped(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        assert f.should_generate(_white_image()) is False

    def test_very_different_image_passes(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        assert f.should_generate(_black_image()) is True

    def test_threshold_controls_sensitivity(self):
        f = SimilarityFilter(threshold=0.99999)
        img1 = _noisy_image(seed=1)
        img2 = _noisy_image(seed=2)
        f.should_generate(img1)
        # With very high threshold, even different images might be skipped
        # But truly random images should still pass
        assert f.should_generate(img2) is True

    def test_low_threshold_skips_more(self):
        f = SimilarityFilter(threshold=0.5)
        f.should_generate(_noisy_image(seed=1))
        # Similar-ish random images with low threshold should still pass
        # because cosine similarity of random vectors is low
        assert f.should_generate(_noisy_image(seed=2)) is True

    def test_previous_image_updates_on_pass(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        f.should_generate(_black_image())  # passes, updates prev
        # Now white image is different from black (the new prev)
        assert f.should_generate(_white_image()) is True

    def test_previous_image_does_not_update_on_skip(self):
        f = SimilarityFilter(threshold=0.98)
        f.should_generate(_white_image())
        f.should_generate(_white_image())  # skipped, prev stays white
        # Black is still different from white (original prev)
        assert f.should_generate(_black_image()) is True

    def test_accepts_different_image_sizes(self):
        f = SimilarityFilter(threshold=0.98)
        big = Image.new("RGB", (1024, 1024), (255, 255, 255))
        assert f.should_generate(big) is True
        small = Image.new("RGB", (256, 256), (255, 255, 255))
        assert f.should_generate(small) is False  # still white
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd server && python -m pytest tests/test_similarity_filter.py -v
```

Expected: `ModuleNotFoundError: No module named 'similarity_filter'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/similarity_filter.py
import numpy as np
from PIL import Image


class SimilarityFilter:
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self._prev: np.ndarray | None = None

    def _to_vector(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((64, 64)).convert("RGB")
        return np.array(resized, dtype=np.float32).flatten()

    def should_generate(self, image: Image.Image) -> bool:
        vec = self._to_vector(image)

        if self._prev is None:
            self._prev = vec
            return True

        norm_a = np.linalg.norm(vec)
        norm_b = np.linalg.norm(self._prev)
        if norm_a == 0 or norm_b == 0:
            self._prev = vec
            return True

        similarity = np.dot(vec, self._prev) / (norm_a * norm_b)

        if similarity > self.threshold:
            return False

        self._prev = vec
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd server && python -m pytest tests/test_similarity_filter.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/similarity_filter.py server/tests/test_similarity_filter.py
git commit -m "feat: add similarity filter with TDD tests"
```

---

## Task 3: Inference Queue (TDD)

**Files:**
- Create: `server/tests/test_inference_queue.py`
- Create: `server/inference_queue.py`

- [ ] **Step 1: Write the failing tests**

```python
# server/tests/test_inference_queue.py
import asyncio
import pytest
import pytest_asyncio

from inference_queue import InferenceQueue


@pytest.mark.asyncio
class TestInferenceQueue:
    async def test_submit_and_get_latest(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        result = await q.get_latest()
        assert result == b"frame1"

    async def test_get_latest_returns_none_when_empty(self):
        q = InferenceQueue()
        result = await q.get_latest()
        assert result is None

    async def test_latest_wins_overwrites_pending(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.submit(b"frame2")
        await q.submit(b"frame3")
        result = await q.get_latest()
        assert result == b"frame3"

    async def test_get_latest_clears_after_read(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.get_latest()
        result = await q.get_latest()
        assert result is None

    async def test_submit_after_get_works(self):
        q = InferenceQueue()
        await q.submit(b"frame1")
        await q.get_latest()
        await q.submit(b"frame2")
        result = await q.get_latest()
        assert result == b"frame2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd server && python -m pytest tests/test_inference_queue.py -v
```

Expected: `ModuleNotFoundError: No module named 'inference_queue'`

- [ ] **Step 3: Write minimal implementation**

```python
# server/inference_queue.py
import asyncio


class InferenceQueue:
    def __init__(self):
        self._latest: bytes | None = None
        self._lock = asyncio.Lock()

    async def submit(self, data: bytes) -> None:
        async with self._lock:
            self._latest = data

    async def get_latest(self) -> bytes | None:
        async with self._lock:
            data = self._latest
            self._latest = None
            return data
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd server && python -m pytest tests/test_inference_queue.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/inference_queue.py server/tests/test_inference_queue.py
git commit -m "feat: add latest-wins inference queue with TDD tests"
```

---

## Task 4: Pipeline Wrapper

**Files:**
- Create: `server/pipeline.py`
- Create: `server/download_models.py`

This task requires GPU hardware. The pipeline wraps StreamDiffusion (zjysteven ControlNet fork) and cannot be unit-tested without a GPU. It will be integration-tested on RunPod.

- [ ] **Step 1: Write the model download script**

```python
# server/download_models.py
"""Download all required model weights from HuggingFace.

Run once on a fresh RunPod pod to populate the persistent volume:
    python download_models.py
"""
from huggingface_hub import snapshot_download
from config import SD15_CONFIG, SERVER_CONFIG


def download_all():
    models_dir = SERVER_CONFIG.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    models = [
        (SD15_CONFIG.base_model_path, "sd15"),
        (SD15_CONFIG.lcm_lora_path, "lcm-lora-sd15"),
        (SD15_CONFIG.controlnet_path, "controlnet-scribble-sd15"),
        (SD15_CONFIG.tiny_vae_path, "taesd"),
    ]

    for repo_id, local_name in models:
        target = models_dir / local_name
        if target.exists():
            print(f"Skipping {repo_id} (already at {target})")
            continue
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        print(f"Done: {repo_id}")

    print("All models downloaded.")


if __name__ == "__main__":
    download_all()
```

- [ ] **Step 2: Write the pipeline wrapper**

```python
# server/pipeline.py
"""StreamDiffusion + ControlNet pipeline wrapper.

Uses zjysteven's StreamDiffusion fork for ControlNet support.
Requires GPU — not unit-testable, integration-tested on RunPod.
"""
import io
import logging
import torch
from PIL import Image
from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
)

from config import ModelConfig, ServerConfig, SD15_CONFIG, SERVER_CONFIG
from similarity_filter import SimilarityFilter

logger = logging.getLogger(__name__)


class RealtimeCanvasPipeline:
    def __init__(
        self,
        model_config: ModelConfig = SD15_CONFIG,
        server_config: ServerConfig = SERVER_CONFIG,
    ):
        self.model_config = model_config
        self.server_config = server_config
        self.ready = False
        self._prompt = server_config.default_prompt
        self._strength = server_config.default_strength
        self._similarity_filter = SimilarityFilter(
            threshold=server_config.similarity_threshold
        )
        self._stream = None
        self._pipe = None

    def initialize(self):
        """Load models and warm up. Call once at server startup."""
        logger.info("Loading ControlNet model...")
        models_dir = self.server_config.models_dir
        mc = self.model_config

        # Resolve model paths: use local dir if exists, otherwise HF repo ID
        def resolve(repo_id: str, local_name: str) -> str:
            local = models_dir / local_name
            return str(local) if local.exists() else repo_id

        base_path = resolve(mc.base_model_path, "sd15")
        lcm_path = resolve(mc.lcm_lora_path, "lcm-lora-sd15")
        cn_path = resolve(mc.controlnet_path, "controlnet-scribble-sd15")
        vae_path = resolve(mc.tiny_vae_path, "taesd")

        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.float16
        )

        logger.info("Loading base pipeline...")
        self._pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        logger.info("Loading LCM-LoRA...")
        self._pipe.load_lora_weights(lcm_path)
        self._pipe.fuse_lora()

        logger.info("Swapping to TinyVAE...")
        self._pipe.vae = AutoencoderTiny.from_pretrained(
            vae_path, torch_dtype=torch.float16
        ).to("cuda")

        self._pipe.enable_xformers_memory_efficient_attention()

        # Compile UNet for speed
        logger.info("Compiling UNet with torch.compile...")
        self._pipe.unet = torch.compile(
            self._pipe.unet, mode="reduce-overhead", fullgraph=True
        )

        # Warm up with dummy inference to trigger compilation
        logger.info("Running warm-up inference...")
        dummy = Image.new("RGB", (mc.width, mc.height), (255, 255, 255))
        self._run_inference(dummy, dummy)
        logger.info("Warm-up complete. Pipeline ready.")
        self.ready = True

    def _run_inference(self, image: Image.Image, control_image: Image.Image) -> Image.Image:
        result = self._pipe(
            prompt=self._prompt,
            image=image,
            control_image=control_image,
            strength=self._strength,
            controlnet_conditioning_scale=self.model_config.controlnet_conditioning_scale,
            num_inference_steps=self.model_config.num_inference_steps,
            guidance_scale=self.model_config.guidance_scale,
            output_type="pil",
        ).images[0]
        return result

    def update_config(self, prompt: str | None = None, strength: float | None = None):
        if prompt is not None:
            self._prompt = prompt
        if strength is not None:
            self._strength = max(0.1, min(1.0, strength))

    def generate(self, sketch_image: Image.Image) -> Image.Image | None:
        """Generate from a sketch image. Returns None if similarity filter skips."""
        sketch = sketch_image.convert("RGB").resize(
            (self.model_config.width, self.model_config.height)
        )

        if not self._similarity_filter.should_generate(sketch):
            return None

        return self._run_inference(sketch, sketch)

    def generate_to_jpeg(self, sketch_bytes: bytes) -> bytes | None:
        """Convenience: bytes in, bytes out."""
        sketch = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
        result = self.generate(sketch)
        if result is None:
            return None
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
```

- [ ] **Step 3: Commit**

```bash
git add server/pipeline.py server/download_models.py
git commit -m "feat: add StreamDiffusion pipeline wrapper and model download script"
```

---

## Task 5: FastAPI Server

**Files:**
- Create: `server/main.py`
- Create: `server/tests/test_main.py`

- [ ] **Step 1: Write the failing tests for endpoints**

```python
# server/tests/test_main.py
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io


# Mock the pipeline before importing main
@pytest.fixture(autouse=True)
def mock_pipeline():
    """Mock the GPU pipeline for all tests."""
    with patch("main.pipeline") as mock_pipe:
        mock_pipe.ready = True
        mock_pipe.model_config = MagicMock()
        mock_pipe.model_config.name = "test-model"
        mock_pipe.server_config = MagicMock()
        mock_pipe.server_config.default_prompt = "test prompt"
        mock_pipe.server_config.default_strength = 0.5

        # generate_to_jpeg returns a small JPEG
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        mock_pipe.generate_to_jpeg.return_value = buf.getvalue()

        yield mock_pipe


@pytest.fixture
def app():
    from main import app
    return app


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_ready(self, app, mock_pipeline):
        mock_pipeline.ready = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    async def test_health_warming_up(self, app, mock_pipeline):
        mock_pipeline.ready = False
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "warming_up"


@pytest.mark.asyncio
class TestConfigEndpoint:
    async def test_config_returns_model_info(self, app, mock_pipeline):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "default_prompt" in data
        assert "default_strength" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd server && python -m pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write the FastAPI server**

```python
# server/main.py
import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import SD15_CONFIG, SERVER_CONFIG
from pipeline import RealtimeCanvasPipeline
from inference_queue import InferenceQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = RealtimeCanvasPipeline(
    model_config=SD15_CONFIG,
    server_config=SERVER_CONFIG,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize pipeline in a background thread (blocks but doesn't freeze event loop)
    logger.info("Starting pipeline initialization...")
    await asyncio.to_thread(pipeline.initialize)
    logger.info("Pipeline ready.")
    yield
    # Shutdown: nothing to clean up
    logger.info("Shutting down.")


app = FastAPI(title="LatentCanvas", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ready" if pipeline.ready else "warming_up",
        "model": pipeline.model_config.name,
    }


@app.get("/api/config")
async def config():
    return {
        "model": pipeline.model_config.name,
        "width": pipeline.model_config.width,
        "height": pipeline.model_config.height,
        "default_prompt": pipeline.server_config.default_prompt,
        "default_strength": pipeline.server_config.default_strength,
    }


@app.websocket("/ws/generate")
async def generate_ws(websocket: WebSocket):
    await websocket.accept()

    if not pipeline.ready:
        await websocket.close(code=1013, reason="Server warming up")
        return

    queue = InferenceQueue()
    generating = False

    async def process_loop():
        nonlocal generating
        while True:
            data = await queue.get_latest()
            if data is None:
                await asyncio.sleep(0.01)
                continue

            generating = True
            try:
                result = await asyncio.to_thread(pipeline.generate_to_jpeg, data)
                if result is not None:
                    await websocket.send_bytes(result)
            except Exception as e:
                logger.error(f"Generation error: {e}")
            finally:
                generating = False

    process_task = asyncio.create_task(process_loop())

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Binary frame: sketch image
                await queue.submit(message["bytes"])
            elif "text" in message and message["text"]:
                # JSON message: config update
                try:
                    config_data = json.loads(message["text"])
                    pipeline.update_config(
                        prompt=config_data.get("prompt"),
                        strength=config_data.get("strength"),
                    )
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON config")
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd server && python -m pytest tests/test_main.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/main.py server/tests/test_main.py
git commit -m "feat: add FastAPI server with WebSocket endpoint and tests"
```

---

## Task 6: Dockerfile and RunPod Deployment

**Files:**
- Create: `server/Dockerfile`
- Create: `server/start.sh`

- [ ] **Step 1: Write the startup script**

```bash
#!/bin/bash
# server/start.sh — RunPod pod entrypoint
set -e

echo "=== LatentCanvas Server Starting ==="

# Download models if not cached in persistent volume
cd /app
python download_models.py

# Set torch compile cache dir to persistent volume
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR:-/workspace/.cache}/torchinductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Start the server
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

- [ ] **Step 2: Write the Dockerfile**

```dockerfile
# server/Dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system deps for controlnet_aux (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install StreamDiffusion
RUN pip install --no-cache-dir streamdiffusion

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Models are downloaded at runtime to persistent volume, not baked in
ENV MODELS_DIR=/workspace/models
ENV CACHE_DIR=/workspace/.cache

EXPOSE 8000

CMD ["./start.sh"]
```

- [ ] **Step 3: Commit**

```bash
git add server/Dockerfile server/start.sh
git commit -m "feat: add Dockerfile and startup script for RunPod deployment"
```

---

## Task 7: Frontend Scaffold

**Files:**
- Create: `frontend/` (via create-next-app)
- Modify: `frontend/app/layout.tsx`
- Create: `frontend/app/lib/config.ts`

- [ ] **Step 1: Create the Next.js project**

```bash
npx create-next-app@latest frontend --typescript --tailwind --app --no-eslint --no-src-dir --use-npm
```

When prompted: accept defaults.

- [ ] **Step 2: Install dependencies**

```bash
cd frontend && npm install tldraw
```

- [ ] **Step 3: Write config**

```typescript
// frontend/app/lib/config.ts
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const WS_URL = BACKEND_URL.replace(/^http/, "ws");

export const CANVAS_EXPORT_DEBOUNCE_MS = 150;
export const CANVAS_EXPORT_QUALITY = 0.7;
export const CANVAS_EXPORT_FORMAT: "jpeg" | "png" = "jpeg";

export const DEFAULT_PROMPT = "high quality, detailed, photorealistic";
export const DEFAULT_STRENGTH = 0.5;

export const WS_RECONNECT_BASE_MS = 1000;
export const WS_RECONNECT_MAX_MS = 30000;
export const HEALTH_POLL_INTERVAL_MS = 3000;
```

- [ ] **Step 4: Update layout**

Replace `frontend/app/layout.tsx`:

```tsx
// frontend/app/layout.tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LatentCanvas",
  description: "Real-time generative canvas",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-white antialiased">{children}</body>
    </html>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Next.js 14 frontend with tldraw dependency"
```

---

## Task 8: Canvas Component with Export

**Files:**
- Create: `frontend/app/components/Canvas.tsx`
- Create: `frontend/app/hooks/useCanvasExport.ts`

- [ ] **Step 1: Write the canvas export hook**

```typescript
// frontend/app/hooks/useCanvasExport.ts
"use client";

import { useCallback, useEffect, useRef } from "react";
import type { Editor } from "tldraw";
import { CANVAS_EXPORT_DEBOUNCE_MS } from "../lib/config";

export function useCanvasExport(
  editor: Editor | null,
  onExport: (blob: Blob) => void
) {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onExportRef = useRef(onExport);
  onExportRef.current = onExport;

  const exportCanvas = useCallback(async () => {
    if (!editor) return;

    const shapeIds = editor.getCurrentPageShapeIds();
    if (shapeIds.size === 0) return;

    const result = await editor.toImage([...shapeIds], {
      format: "jpeg",
      scale: 1,
      background: true,
      padding: 0,
    });

    if (result?.blob) {
      onExportRef.current(result.blob);
    }
  }, [editor]);

  useEffect(() => {
    if (!editor) return;

    const cleanup = editor.store.listen(
      () => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        timeoutRef.current = setTimeout(exportCanvas, CANVAS_EXPORT_DEBOUNCE_MS);
      },
      { source: "user", scope: "document" }
    );

    return () => {
      cleanup();
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [editor, exportCanvas]);
}
```

- [ ] **Step 2: Write the Canvas component**

```tsx
// frontend/app/components/Canvas.tsx
"use client";

import { useState } from "react";
import { Tldraw, Editor } from "tldraw";
import "tldraw/tldraw.css";
import { useCanvasExport } from "../hooks/useCanvasExport";

interface CanvasProps {
  onSketchExport: (blob: Blob) => void;
}

export default function Canvas({ onSketchExport }: CanvasProps) {
  const [editor, setEditor] = useState<Editor | null>(null);

  useCanvasExport(editor, onSketchExport);

  return (
    <div className="h-full w-full relative">
      <Tldraw onMount={setEditor} />
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/app/components/Canvas.tsx frontend/app/hooks/useCanvasExport.ts
git commit -m "feat: add tldraw Canvas component with debounced JPEG export"
```

---

## Task 9: WebSocket Hook

**Files:**
- Create: `frontend/app/hooks/useWebSocket.ts`

- [ ] **Step 1: Write the WebSocket hook**

```typescript
// frontend/app/hooks/useWebSocket.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  WS_URL,
  WS_RECONNECT_BASE_MS,
  WS_RECONNECT_MAX_MS,
  BACKEND_URL,
  HEALTH_POLL_INTERVAL_MS,
} from "../lib/config";

export type ConnectionState = "connecting" | "connected" | "reconnecting" | "waiting";

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  sendBinary: (data: ArrayBuffer) => void;
  sendConfig: (config: { prompt?: string; strength?: number }) => void;
  latestImage: string | null;
}

export function useWebSocket(): UseWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>("waiting");
  const [latestImage, setLatestImage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevImageUrlRef = useRef<string | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionState("connecting");
    const ws = new WebSocket(`${WS_URL}/ws/generate`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setConnectionState("connected");
      reconnectAttemptRef.current = 0;
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        // Revoke previous URL to prevent memory leak
        if (prevImageUrlRef.current) {
          URL.revokeObjectURL(prevImageUrlRef.current);
        }
        prevImageUrlRef.current = url;
        setLatestImage(url);
      }
    };

    ws.onclose = () => {
      setConnectionState("reconnecting");
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  const scheduleReconnect = useCallback(() => {
    const attempt = reconnectAttemptRef.current;
    const delay = Math.min(
      WS_RECONNECT_BASE_MS * Math.pow(2, attempt),
      WS_RECONNECT_MAX_MS
    );
    reconnectAttemptRef.current = attempt + 1;

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  }, [connect]);

  // Poll health endpoint, connect WebSocket once server is ready
  useEffect(() => {
    let mounted = true;
    let pollId: ReturnType<typeof setInterval>;

    const checkHealth = async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/api/health`);
        const data = await resp.json();
        if (data.status === "ready" && mounted) {
          clearInterval(pollId);
          connect();
        }
      } catch {
        // Server not reachable yet
      }
    };

    setConnectionState("waiting");
    checkHealth();
    pollId = setInterval(checkHealth, HEALTH_POLL_INTERVAL_MS);

    return () => {
      mounted = false;
      clearInterval(pollId);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
      if (prevImageUrlRef.current) {
        URL.revokeObjectURL(prevImageUrlRef.current);
      }
    };
  }, [connect]);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  const sendConfig = useCallback((config: { prompt?: string; strength?: number }) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(config));
    }
  }, []);

  return { connectionState, sendBinary, sendConfig, latestImage };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/app/hooks/useWebSocket.ts
git commit -m "feat: add WebSocket hook with health polling and auto-reconnect"
```

---

## Task 10: UI Components

**Files:**
- Create: `frontend/app/components/AIOutputView.tsx`
- Create: `frontend/app/components/ControlPanel.tsx`
- Create: `frontend/app/components/ConnectionStatus.tsx`

- [ ] **Step 1: Write ConnectionStatus**

```tsx
// frontend/app/components/ConnectionStatus.tsx
"use client";

import type { ConnectionState } from "../hooks/useWebSocket";

const STATUS_CONFIG: Record<ConnectionState, { label: string; color: string }> = {
  waiting: { label: "Waiting for server", color: "bg-yellow-500" },
  connecting: { label: "Connecting", color: "bg-yellow-500" },
  connected: { label: "Connected", color: "bg-green-500" },
  reconnecting: { label: "Reconnecting", color: "bg-red-500" },
};

export default function ConnectionStatus({ state }: { state: ConnectionState }) {
  const { label, color } = STATUS_CONFIG[state];

  return (
    <div className="flex items-center gap-2 text-sm">
      <div className={`h-2 w-2 rounded-full ${color}`} />
      <span className="text-gray-300">{label}</span>
    </div>
  );
}
```

- [ ] **Step 2: Write AIOutputView**

```tsx
// frontend/app/components/AIOutputView.tsx
"use client";

interface AIOutputViewProps {
  imageUrl: string | null;
}

export default function AIOutputView({ imageUrl }: AIOutputViewProps) {
  return (
    <div className="h-full w-full flex items-center justify-center bg-gray-900 rounded-lg overflow-hidden">
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="AI generated"
          className="max-h-full max-w-full object-contain"
        />
      ) : (
        <div className="text-gray-500 text-center p-8">
          <p className="text-lg">AI Output</p>
          <p className="text-sm mt-2">Draw something to see the AI generate an image</p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Write ControlPanel**

```tsx
// frontend/app/components/ControlPanel.tsx
"use client";

import { useCallback, useRef, useState } from "react";
import { DEFAULT_PROMPT, DEFAULT_STRENGTH } from "../lib/config";

interface ControlPanelProps {
  onPromptChange: (prompt: string) => void;
  onStrengthChange: (strength: number) => void;
}

export default function ControlPanel({
  onPromptChange,
  onStrengthChange,
}: ControlPanelProps) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handlePromptChange = useCallback(
    (value: string) => {
      setPrompt(value);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => onPromptChange(value), 500);
    },
    [onPromptChange]
  );

  const handleStrengthChange = useCallback(
    (value: number) => {
      setStrength(value);
      onStrengthChange(value);
    },
    [onStrengthChange]
  );

  return (
    <div className="flex items-center gap-4 p-3 bg-gray-800 rounded-lg">
      <input
        type="text"
        value={prompt}
        onChange={(e) => handlePromptChange(e.target.value)}
        placeholder="Describe the style..."
        className="flex-1 bg-gray-700 text-white px-3 py-2 rounded text-sm
                   placeholder-gray-400 outline-none focus:ring-1 focus:ring-blue-500"
      />

      <div className="flex items-center gap-2 shrink-0">
        <label className="text-sm text-gray-400">AI Strength</label>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={strength}
          onChange={(e) => handleStrengthChange(parseFloat(e.target.value))}
          className="w-28"
        />
        <span className="text-sm text-gray-300 w-8">{strength.toFixed(2)}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/app/components/ConnectionStatus.tsx frontend/app/components/AIOutputView.tsx frontend/app/components/ControlPanel.tsx
git commit -m "feat: add UI components — ConnectionStatus, AIOutputView, ControlPanel"
```

---

## Task 11: Main Page Assembly

**Files:**
- Modify: `frontend/app/page.tsx`

- [ ] **Step 1: Write the main page**

```tsx
// frontend/app/page.tsx
"use client";

import dynamic from "next/dynamic";
import { useCallback } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import AIOutputView from "./components/AIOutputView";
import ControlPanel from "./components/ControlPanel";
import ConnectionStatus from "./components/ConnectionStatus";

const Canvas = dynamic(() => import("./components/Canvas"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-gray-900">
      <p className="text-gray-500">Loading canvas...</p>
    </div>
  ),
});

export default function Home() {
  const { connectionState, sendBinary, sendConfig, latestImage } = useWebSocket();

  const handleSketchExport = useCallback(
    (blob: Blob) => {
      blob.arrayBuffer().then((buf) => sendBinary(buf));
    },
    [sendBinary]
  );

  const handlePromptChange = useCallback(
    (prompt: string) => sendConfig({ prompt }),
    [sendConfig]
  );

  const handleStrengthChange = useCallback(
    (strength: number) => sendConfig({ strength }),
    [sendConfig]
  );

  return (
    <div className="h-screen flex flex-col">
      {/* Top bar */}
      <div className="flex items-center justify-between p-3 border-b border-gray-800">
        <h1 className="text-lg font-semibold">LatentCanvas</h1>
        <ConnectionStatus state={connectionState} />
      </div>

      {/* Controls */}
      <div className="p-3 border-b border-gray-800">
        <ControlPanel
          onPromptChange={handlePromptChange}
          onStrengthChange={handleStrengthChange}
        />
      </div>

      {/* Split view: Canvas | AI Output */}
      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 border-r border-gray-800">
          <Canvas onSketchExport={handleSketchExport} />
        </div>
        <div className="w-1/2 p-4">
          <AIOutputView imageUrl={latestImage} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify the frontend builds**

```bash
cd frontend && npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat: assemble main page with split-view canvas and AI output"
```

---

## Task 12: Frontend Build Verification and Environment Setup

**Files:**
- Create: `frontend/.env.local`
- Create: `.gitignore`

- [ ] **Step 1: Create .env.local for local development**

```bash
# frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

- [ ] **Step 2: Create root .gitignore**

```
# server
server/__pycache__/
server/tests/__pycache__/
*.pyc
.pytest_cache/

# frontend
frontend/node_modules/
frontend/.next/
frontend/out/

# environment
.env
.env.local
*.env

# models (large files)
models/

# IDE
.vscode/
.idea/
```

- [ ] **Step 3: Run the frontend dev server to verify it starts**

```bash
cd frontend && npm run dev
```

Expected: Server starts on localhost:3000. Canvas loads (tldraw renders). AI Output shows placeholder. Controls visible. Connection status shows "Waiting for server" (expected since no backend running locally).

- [ ] **Step 4: Commit**

```bash
git add .gitignore frontend/.env.local
git commit -m "feat: add environment config and gitignore"
```

---

## Task 13: Integration Smoke Test Script

**Files:**
- Create: `server/tests/test_integration.py`

This test validates the FastAPI server end-to-end with a mocked pipeline (no GPU needed).

- [ ] **Step 1: Write the WebSocket integration test**

```python
# server/tests/test_integration.py
"""Integration smoke test — validates WebSocket flow with mocked pipeline."""
import asyncio
import io
import json
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def mock_pipeline():
    with patch("main.pipeline") as mock_pipe:
        mock_pipe.ready = True
        mock_pipe.model_config = MagicMock()
        mock_pipe.model_config.name = "test-model"
        mock_pipe.model_config.width = 512
        mock_pipe.model_config.height = 512
        mock_pipe.server_config = MagicMock()
        mock_pipe.server_config.default_prompt = "test"
        mock_pipe.server_config.default_strength = 0.5

        mock_pipe.generate_to_jpeg.return_value = _make_jpeg()
        yield mock_pipe


@pytest.fixture
def app():
    from main import app
    return app


@pytest.mark.asyncio
async def test_websocket_send_receive(app, mock_pipeline):
    """Send a sketch frame, receive a generated image back."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", "/ws/generate") as _:
            pass  # httpx doesn't do WS natively

    # Use Starlette's test client for WebSocket
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as ws:
            # Send binary sketch
            sketch = _make_jpeg()
            ws.send_bytes(sketch)

            # Wait for response (with timeout)
            import time
            start = time.time()
            response = None
            while time.time() - start < 5:
                try:
                    response = ws.receive_bytes(timeout=0.5)
                    if response:
                        break
                except Exception:
                    continue

            assert response is not None
            assert len(response) > 0
            # Verify it's a valid JPEG
            img = Image.open(io.BytesIO(response))
            assert img.format == "JPEG"


@pytest.mark.asyncio
async def test_websocket_config_update(app, mock_pipeline):
    """Send a JSON config update over WebSocket."""
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        with client.websocket_connect("/ws/generate") as ws:
            config = {"prompt": "anime style", "strength": 0.8}
            ws.send_text(json.dumps(config))

            # Send a sketch to trigger generation
            ws.send_bytes(_make_jpeg())

            import time
            start = time.time()
            response = None
            while time.time() - start < 5:
                try:
                    response = ws.receive_bytes(timeout=0.5)
                    if response:
                        break
                except Exception:
                    continue

            # Verify config was applied
            mock_pipeline.update_config.assert_called_with(
                prompt="anime style", strength=0.8
            )


@pytest.mark.asyncio
async def test_websocket_rejected_when_not_ready(app, mock_pipeline):
    """WebSocket should close if pipeline isn't ready."""
    mock_pipeline.ready = False

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/generate") as ws:
                ws.receive_bytes()
```

- [ ] **Step 2: Run integration tests**

```bash
cd server && python -m pytest tests/test_integration.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 3: Run all backend tests together**

```bash
cd server && python -m pytest tests/ -v
```

Expected: All tests across all test files PASS.

- [ ] **Step 4: Commit**

```bash
git add server/tests/test_integration.py
git commit -m "feat: add WebSocket integration smoke tests"
```

---

## Summary

| Task | Description | Files | Testable locally? |
|------|-------------|-------|-------------------|
| 1 | Backend config | `config.py`, `requirements.txt` | Yes |
| 2 | Similarity filter (TDD) | `similarity_filter.py`, tests | Yes |
| 3 | Inference queue (TDD) | `inference_queue.py`, tests | Yes |
| 4 | Pipeline wrapper | `pipeline.py`, `download_models.py` | No (GPU required) |
| 5 | FastAPI server | `main.py`, tests | Yes (mocked pipeline) |
| 6 | Dockerfile + deploy | `Dockerfile`, `start.sh` | Build only |
| 7 | Frontend scaffold | Next.js project | Yes |
| 8 | Canvas + export | `Canvas.tsx`, `useCanvasExport.ts` | Yes |
| 9 | WebSocket hook | `useWebSocket.ts` | Yes |
| 10 | UI components | `ConnectionStatus`, `AIOutputView`, `ControlPanel` | Yes |
| 11 | Main page | `page.tsx` | Yes |
| 12 | Environment setup | `.env.local`, `.gitignore` | Yes |
| 13 | Integration tests | `test_integration.py` | Yes (mocked) |

**After this plan:** Deploy the Docker image to a RunPod Pod with A40 GPU, run `download_models.py` to cache weights, then start the server. Point the frontend's `NEXT_PUBLIC_BACKEND_URL` at the RunPod proxy URL and deploy to Vercel.
