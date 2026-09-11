FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG MEANVC_INIT_TASK=train_40ms
ARG WAVLM_FINETUNE_URL=https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=80 \
    PORT_HEALTH=80 \
    MEANVC_MODEL=40ms \
    MEANVC_DEVICE=cuda \
    VOICE_DIR=/app/voices

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsox-fmt-all \
        sox \
    && rm -rf /var/lib/apt/lists/*

# Install only the packages required by the real-time inference/server path.
# PyTorch 2.5.1 + CUDA 12.1 already come from the base image.
COPY server_requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r server_requirements.txt

# MeanVC2's upstream initialization script cannot automatically fetch the
# fine-tuned WavLM/ECAPA speaker checkpoint. Fetch it explicitly, then let the
# official script download the ASR, 40ms VC and Vocos checkpoints.
COPY initialization.py ./
RUN mkdir -p preprocess/ckpts \
    && gdown --fuzzy "${WAVLM_FINETUNE_URL}" -O preprocess/ckpts/wavlm_large_finetune.pth \
    && python initialization.py --task "${MEANVC_INIT_TASK}" \
    && rm -f preprocess/ckpts/wavlm_large.pt \
    && rm -rf /root/.cache/huggingface /root/.cache/torch

# The WavLM base file above is only needed during build to extract
# preprocess/ckpts/wavlm_large_cfg.pt. Runtime rebuilds WavLM from that config
# and the fine-tuned checkpoint, so removing the base file saves ~1.2 GB.
COPY . .
RUN mkdir -p /app/voices

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT_HEALTH}/ping >/dev/null || exit 1

CMD ["python", "server.py"]
