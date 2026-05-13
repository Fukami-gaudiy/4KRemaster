#!/usr/bin/env bash
# Mac M系（Apple Silicon）向け環境構築スクリプト
# 想定: macOS 14+ / Python 3.10+

set -euo pipefail

echo "==> Anime Upscale Experiment - 環境構築開始"

# -------- 0. 前提チェック --------
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew が必要です。https://brew.sh/ からインストールしてください。" >&2
  exit 1
fi

# -------- 1. システム依存（ffmpeg） --------
echo "==> ffmpeg をインストール"
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg

# -------- 2. Python venv --------
echo "==> Python 仮想環境を作成"
if [ ! -d "../.venv" ]; then
  python3 -m venv ../.venv
fi
# shellcheck disable=SC1091
source ../.venv/bin/activate

# -------- 3. Python パッケージ --------
echo "==> Python パッケージをインストール"
pip install --upgrade pip wheel

# PyTorch (MPS対応版)
pip install torch torchvision torchaudio

# 画像処理・ユーティリティ
pip install \
  numpy pillow opencv-python tqdm einops \
  scipy scikit-image lpips

# Real-ESRGAN (アニメ用)
pip install basicsr facexlib gfpgan
pip install realesrgan

# Stable Diffusion / Diffusers (Outpainting用)
pip install diffusers transformers accelerate safetensors

# CLIP (シーン分類用)
pip install open_clip_torch

# Optical Flow (時間整合性)
pip install pytorch-lightning  # RAFT 依存

# -------- 4. モデル重みを事前ダウンロード（任意） --------
echo "==> モデル重み準備"
MODELS_DIR="../models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

if [ ! -f "RealESRGAN_x4plus_anime_6B.pth" ]; then
  curl -L -o RealESRGAN_x4plus_anime_6B.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
fi

cd - >/dev/null

echo "==> 完了。venv 有効化: source ../.venv/bin/activate"
echo "==> 動作確認: python -c 'import torch; print(\"MPS:\", torch.backends.mps.is_available())'"
