"""
MeanVC2 Checkpoint Download Script

Downloads required checkpoints from HuggingFace.

Usage:
    python initialization.py --task preprocess      # BN + SpkEmb extraction
    python initialization.py --task train_120ms      # preprocess + 120ms_40ms VC model + vocoder
    python initialization.py --task train_40ms       # preprocess + 40ms_40ms VC model + vocoder
    python initialization.py --task all              # everything
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ASLP-lab/MeanVC2"

# ---- file mapping: (hf_filename, local_path) ----
# HF repo has flat structure; local code expects nested directories.

# ASR encoder (Fast-U2++)
ASR_FILES = [
    ("fastu2pp_80ms.pt",  "preprocess/ckpts/fastu2pp_80ms.pt"),
    ("fastu2pp_160ms.pt", "preprocess/ckpts/fastu2pp_160ms.pt"),
]

# Speaker embedding: WavLM + ECAPA-TDNN
# These are existing public models — NOT in our HF repo.
# wavlm_large.pt          → WavLM-Large from s3prl's pinned checkpoint mirror
# wavlm_large_cfg.pt      → extracted from wavlm_large.pt (tiny config, ~10 KB)
# wavlm_large_finetune.pth → Google Drive (ECAPA-TDNN fine-tuned weights)
SPK_FILES = [
    # download handled separately by _download_wavlm()
]

# VC model: 120ms chunk + 40ms future (recommended for quality)
VC_MODEL_120MS = [
    ("meanvc2_120ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors"),
]

# VC model: 40ms chunk + 40ms future (lower latency)
VC_MODEL_40MS = [
    ("meanvc2_40ms_40ms.safetensors", "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors"),
]

# Vocoder (Vocos)
VOCODER_FILES = [
    ("vocos.pt",      "ckpts/vocos/vocos.pt"),
]

TASK_FILES = {
    "preprocess":  ASR_FILES + SPK_FILES,
    "train_120ms": ASR_FILES + SPK_FILES + VC_MODEL_120MS + VOCODER_FILES,
    "train_40ms":  ASR_FILES + SPK_FILES + VC_MODEL_40MS + VOCODER_FILES,
    "all":         ASR_FILES + SPK_FILES + VC_MODEL_120MS + VC_MODEL_40MS + VOCODER_FILES,
}


# ---------------------------------------------------------------------------
# WavLM download (existing public models, NOT in our HF repo)
# ---------------------------------------------------------------------------

# Same source used by s3prl.upstream.wavlm.hubconf.wavlm_large.
# Pin both the repository revision and file digest for reproducible builds.
WAVLM_REPO = "s3prl/converted_ckpts"
WAVLM_REVISION = "8cad0b370e7e35f8d56951d95d2be036ea85510c"
WAVLM_SHA256 = "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"
WAVLM_CFG_PATH = "preprocess/ckpts/wavlm_large_cfg.pt"
WAVLM_FINETUNED_PATH = "preprocess/ckpts/wavlm_large_finetune.pth"
WAVLM_FINETUNED_URL = "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"


def _validate_wavlm_config(cfg):
    # A base/small config would build a different speaker encoder.
    if not isinstance(cfg, dict) or any(
        cfg.get(key) != value for key, value in {
            "encoder_layers": 24,
            "encoder_embed_dim": 1024,
            "encoder_attention_heads": 16,
        }.items()
    ):
        raise ValueError("Expected the WavLM Large configuration dictionary")
    return cfg


def _download_wavlm():
    """Prepare the required speaker config; any failure must fail the build."""
    import hashlib
    import torch

    cfg_path = Path(WAVLM_CFG_PATH)
    if cfg_path.is_file():
        _validate_wavlm_config(torch.load(cfg_path, map_location="cpu", weights_only=True))
        print(f"  [verified] {cfg_path}")
        return

    print("  [download] pinned WavLM Large checkpoint for config extraction ...")
    checkpoint_path = hf_hub_download(
        repo_id=WAVLM_REPO, filename="wavlm_large.pt", revision=WAVLM_REVISION,
    )
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WAVLM_SHA256:
        raise ValueError("WavLM Large checkpoint SHA256 mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = _validate_wavlm_config(checkpoint.get("cfg", checkpoint.get("config")))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cfg_path.with_suffix(".tmp")
    try:
        torch.save(cfg, temporary)
        _validate_wavlm_config(torch.load(temporary, map_location="cpu", weights_only=True))
        os.replace(temporary, cfg_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"  [done] {cfg_path}")


def verify_files(task: str):
    """Fail before publishing an image with absent checkpoints or invalid config."""
    import torch

    required = [path for _, path in TASK_FILES[task]] + [
        WAVLM_CFG_PATH, WAVLM_FINETUNED_PATH,
    ]
    missing = [path for path in required if not Path(path).is_file() or Path(path).stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required checkpoints: " + ", ".join(missing))
    _validate_wavlm_config(torch.load(WAVLM_CFG_PATH, map_location="cpu", weights_only=True))
    print(f"  [verified] {len(required)} required checkpoint files for {task}")


# ---------------------------------------------------------------------------

def download_files(task: str):
    files = TASK_FILES[task]
    print(f"Task: {task} | {len(files)} file(s) from HF")

    for hf_filename, local_path in files:
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(local_path):
            print(f"  [skip] {local_path}")
            continue

        print(f"  [download] {hf_filename} -> {local_path} ...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_filename,
        )
        shutil.copy(downloaded, local_path)
        print(f"  [done]  {local_path}")

    print("\n--- HF downloads complete ---\n")

    # WavLM (existing public models, from external sources)
    _download_wavlm()

    verify_files(task)

    # FunASR note
    if task != "preprocess":
        print("\nNote: FunASR models (Paraformer, VAD, punctuation) auto-download from ModelScope at first use.")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MeanVC2 checkpoints from HuggingFace"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_FILES.keys()),
        help="'preprocess' for BN+SpkEmb extraction; "
             "'train_120ms' adds 120ms VC model + vocoder; "
             "'train_40ms' adds 40ms VC model + vocoder; "
             "'all' for everything",
    )
    parser.add_argument("--verify-only", action="store_true", help="Check packaged files without downloading")
    args = parser.parse_args()
    if args.verify_only:
        verify_files(args.task)
    else:
        download_files(args.task)
