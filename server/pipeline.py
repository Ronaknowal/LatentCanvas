"""StreamDiffusion + ControlNet pipeline wrapper.

Uses diffusers' StableDiffusionControlNetImg2ImgPipeline with LCM-LoRA
for few-step inference and ControlNet Scribble for sketch conditioning.
Requires GPU — not unit-testable, integration-tested on RunPod.
"""
import io
import logging
import torch
from PIL import Image
from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    LCMScheduler,
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
        self._pipe = None

    def initialize(self):
        """Load models and warm up. Call once at server startup."""
        models_dir = self.server_config.models_dir
        mc = self.model_config

        def resolve(repo_id: str, local_name: str) -> str:
            local = models_dir / local_name
            return str(local) if local.exists() else repo_id

        base_path = resolve(mc.base_model_path, "sd15")
        lcm_path = resolve(mc.lcm_lora_path, "lcm-lora-sd15")
        cn_path = resolve(mc.controlnet_path, "controlnet-scribble-sd15")
        vae_path = resolve(mc.tiny_vae_path, "taesd")

        logger.info("Loading ControlNet model...")
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

        logger.info("Switching to LCM scheduler...")
        self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config)

        logger.info("Loading LCM-LoRA...")
        self._pipe.load_lora_weights(lcm_path)
        self._pipe.fuse_lora()

        logger.info("Swapping to TinyVAE...")
        self._pipe.vae = AutoencoderTiny.from_pretrained(
            vae_path, torch_dtype=torch.float16
        ).to("cuda")

        self._pipe.enable_xformers_memory_efficient_attention()

        logger.info("Compiling UNet with torch.compile...")
        self._pipe.unet = torch.compile(
            self._pipe.unet, mode="max-autotune", fullgraph=True
        )

        logger.info("Running warm-up inference...")
        dummy = Image.new("RGB", (mc.width, mc.height), (255, 255, 255))
        self._run_inference(dummy, dummy)
        logger.info("Warm-up complete. Pipeline ready.")
        self.ready = True

    def _run_inference(self, image: Image.Image, control_image: Image.Image) -> Image.Image:
        # strength: how much of the sketch to replace (0=keep, 1=fully regenerate)
        # controlnet_conditioning_scale: how strongly to follow sketch structure
        # Both should be high for sketch→photorealistic: replace pixels but keep shape
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
