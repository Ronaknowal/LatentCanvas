from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os


@dataclass
class ModelConfig:
    """Base config shared by all model pipelines."""
    name: str
    pipeline_type: Literal["sd15", "flux_dev", "flux_schnell"]
    base_model_path: str
    controlnet_path: str
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    controlnet_conditioning_scale: float = 0.7
    # SD1.5-specific
    lcm_lora_path: str = ""
    tiny_vae_path: str = ""
    # FLUX-specific
    torch_dtype: str = "float16"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: Path = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
    cache_dir: Path = Path(os.environ.get("CACHE_DIR", "/workspace/.cache"))
    similarity_threshold: float = 0.98
    similarity_max_skip: int = 10
    default_prompt: str = "high quality, detailed, photorealistic"
    default_strength: float = 0.7
    active_model: str = os.environ.get("ACTIVE_MODEL", "flux_dev")


# --- Model Presets ---

SD15_CONFIG = ModelConfig(
    name="sd1.5-lcm-controlnet-scribble",
    pipeline_type="sd15",
    base_model_path="runwayml/stable-diffusion-v1-5",
    controlnet_path="lllyasviel/sd-controlnet-scribble",
    lcm_lora_path="latent-consistency/lcm-lora-sdv1-5",
    tiny_vae_path="madebyollin/taesd",
    width=512,
    height=512,
    num_inference_steps=6,
    guidance_scale=1.5,
    controlnet_conditioning_scale=1.2,
    torch_dtype="float16",
)

FLUX_DEV_CONFIG = ModelConfig(
    name="flux-dev-controlnet-hed",
    pipeline_type="flux_dev",
    base_model_path="black-forest-labs/FLUX.1-dev",
    controlnet_path="XLabs-AI/flux-controlnet-hed-diffusers",
    width=1024,
    height=1024,
    num_inference_steps=15,
    guidance_scale=3.5,
    controlnet_conditioning_scale=0.7,
    torch_dtype="bfloat16",
)

FLUX_SCHNELL_CONFIG = ModelConfig(
    name="flux-schnell-controlnet-hed",
    pipeline_type="flux_schnell",
    base_model_path="black-forest-labs/FLUX.1-schnell",
    controlnet_path="XLabs-AI/flux-controlnet-hed-diffusers",
    width=1024,
    height=1024,
    num_inference_steps=4,
    guidance_scale=0.0,  # schnell does not use CFG
    controlnet_conditioning_scale=0.7,
    torch_dtype="bfloat16",
)

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "sd15": SD15_CONFIG,
    "flux_dev": FLUX_DEV_CONFIG,
    "flux_schnell": FLUX_SCHNELL_CONFIG,
}

SERVER_CONFIG = ServerConfig()
ACTIVE_MODEL = MODEL_REGISTRY.get(SERVER_CONFIG.active_model, FLUX_DEV_CONFIG)
