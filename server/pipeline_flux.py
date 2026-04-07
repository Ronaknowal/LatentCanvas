"""FLUX.1 + ControlNet HED pipeline wrapper.

Supports both FLUX.1-dev (native ControlNet) and FLUX.1-schnell (workaround).
Requires GPU with ~40GB VRAM (A40 48GB works without quantization).
"""
import io
import logging
import torch
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

from config import ModelConfig, ServerConfig, FLUX_DEV_CONFIG, SERVER_CONFIG
from similarity_filter import SimilarityFilter

logger = logging.getLogger(__name__)


class FluxCanvasPipeline:
    def __init__(
        self,
        model_config: ModelConfig = FLUX_DEV_CONFIG,
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
        self._pipe = None

    def initialize(self):
        """Load models and warm up. Call once at server startup."""
        models_dir = self.server_config.models_dir
        mc = self.model_config
        is_schnell = mc.pipeline_type == "flux_schnell"
        dtype = torch.bfloat16 if mc.torch_dtype == "bfloat16" else torch.float16

        def resolve(repo_id: str, local_name: str) -> str:
            local = models_dir / local_name
            return str(local) if local.exists() else repo_id

        base_path = resolve(mc.base_model_path, mc.pipeline_type)
        cn_path = resolve(mc.controlnet_path, "flux-controlnet-hed")

        logger.info(f"Loading FLUX ControlNet HED from {cn_path}...")
        cn_kwargs = {"torch_dtype": dtype}
        if is_schnell:
            # Workaround: schnell doesn't have guidance embeddings
            cn_kwargs["guidance_embeds"] = False

        controlnet = FluxControlNetModel.from_pretrained(cn_path, **cn_kwargs)

        logger.info(f"Loading FLUX base model from {base_path}...")
        self._pipe = FluxControlNetPipeline.from_pretrained(
            base_path,
            controlnet=controlnet,
            torch_dtype=dtype,
        ).to("cuda")

        # Compile transformer for speed (~30-50% faster after warmup)
        logger.info("Compiling transformer with torch.compile...")
        self._pipe.transformer = torch.compile(
            self._pipe.transformer, mode="max-autotune", fullgraph=True
        )

        logger.info("Running warm-up inference...")
        dummy = Image.new("RGB", (mc.width, mc.height), (255, 255, 255))
        self._run_inference(dummy)
        logger.info("Warm-up complete. FLUX pipeline ready.")
        self.ready = True

    def _run_inference(self, control_image: Image.Image) -> Image.Image:
        mc = self.model_config
        result = self._pipe(
            prompt=self._prompt,
            control_image=control_image,
            controlnet_conditioning_scale=mc.controlnet_conditioning_scale,
            num_inference_steps=mc.num_inference_steps,
            guidance_scale=mc.guidance_scale,
            width=mc.width,
            height=mc.height,
            output_type="pil",
        ).images[0]
        return result

    def update_config(self, prompt: str | None = None, strength: float | None = None):
        if prompt is not None:
            self._prompt = prompt
        if strength is not None:
            self._strength = max(0.1, min(1.0, strength))
            # Map strength to controlnet_conditioning_scale:
            # Low strength (0.1) = high CN influence (1.0) — stick to sketch
            # High strength (1.0) = low CN influence (0.3) — creative freedom
            self.model_config.controlnet_conditioning_scale = 1.0 - (strength * 0.7)

    def generate(self, sketch_image: Image.Image) -> Image.Image | None:
        """Generate from a sketch. Returns None if similarity filter skips."""
        sketch = sketch_image.convert("RGB").resize(
            (self.model_config.width, self.model_config.height)
        )

        if not self._similarity_filter.should_generate(sketch):
            return None

        return self._run_inference(sketch)

    def generate_to_jpeg(self, sketch_bytes: bytes) -> bytes | None:
        """Convenience: bytes in, bytes out."""
        sketch = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
        result = self.generate(sketch)
        if result is None:
            return None
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
