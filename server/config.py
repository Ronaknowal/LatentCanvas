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
    num_inference_steps: int = 6
    t_index_list: list[int] = field(default_factory=lambda: [0, 10, 20, 30, 40, 49])
    guidance_scale: float = 1.5
    controlnet_conditioning_scale: float = 1.2


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
