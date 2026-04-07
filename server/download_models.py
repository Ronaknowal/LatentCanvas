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
