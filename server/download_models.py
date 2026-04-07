"""Download model weights from HuggingFace.

Usage:
    python download_models.py              # Download active model only
    python download_models.py --all        # Download all model presets
    python download_models.py --model sd15 # Download specific model
"""
import argparse
from huggingface_hub import snapshot_download
from config import (
    SD15_CONFIG, FLUX_DEV_CONFIG, FLUX_SCHNELL_CONFIG,
    MODEL_REGISTRY, SERVER_CONFIG, ACTIVE_MODEL,
)

# Map model configs to their local directory names
DOWNLOAD_MAP = {
    "sd15": [
        (SD15_CONFIG.base_model_path, "sd15"),
        (SD15_CONFIG.lcm_lora_path, "lcm-lora-sd15"),
        (SD15_CONFIG.controlnet_path, "controlnet-scribble-sd15"),
        (SD15_CONFIG.tiny_vae_path, "taesd"),
    ],
    "flux_dev": [
        (FLUX_DEV_CONFIG.base_model_path, "flux_dev"),
        (FLUX_DEV_CONFIG.controlnet_path, "flux-controlnet-hed"),
    ],
    "flux_schnell": [
        (FLUX_SCHNELL_CONFIG.base_model_path, "flux_schnell"),
        (FLUX_SCHNELL_CONFIG.controlnet_path, "flux-controlnet-hed"),
    ],
}


def download_models(model_keys: list[str]):
    models_dir = SERVER_CONFIG.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for key in model_keys:
        entries = DOWNLOAD_MAP.get(key, [])
        if not entries:
            print(f"Unknown model key: {key}")
            continue

        print(f"\n=== Downloading models for: {key} ===")
        for repo_id, local_name in entries:
            if local_name in seen:
                continue
            seen.add(local_name)

            target = models_dir / local_name
            if target.exists():
                print(f"  Skipping {repo_id} (already at {target})")
                continue
            print(f"  Downloading {repo_id} -> {target}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
            print(f"  Done: {repo_id}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--all", action="store_true", help="Download all model presets")
    parser.add_argument("--model", type=str, help="Download specific model (sd15, flux_dev, flux_schnell)")
    args = parser.parse_args()

    if args.all:
        download_models(list(DOWNLOAD_MAP.keys()))
    elif args.model:
        download_models([args.model])
    else:
        # Download active model only
        print(f"Downloading active model: {ACTIVE_MODEL.name} ({ACTIVE_MODEL.pipeline_type})")
        download_models([ACTIVE_MODEL.pipeline_type])
