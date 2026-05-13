"""手法C: Outpaint - Stable Diffusion Inpainting で左右を生成補完。

4:3 → 16:9 に拡張するため、元画像を中央に配置し、左右を SD で生成する。
アニメ調を保つため、対象作品のスタイルを反映する LoRA を後段で追加可能。

注意:
- Mac M系 (MPS) では1枚あたり10〜40秒程度かかる想定。
- 同一シードで前後フレームを生成しても完全な時間一貫性は得られない。
  → temporal_check.py でフロー一致を検証する。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Diffusers は遅延 import（環境がない時に download_sample.py 等が壊れないように）


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_outpaint_canvas(
    img: Image.Image, target_aspect: float = 16 / 9
) -> tuple[Image.Image, Image.Image]:
    """中央に元画像、左右が空白の canvas と、生成対象を示す mask を返す。"""
    w, h = img.size
    target_w = int(round(h * target_aspect))
    if target_w <= w:
        return img.convert("RGB"), Image.new("L", img.size, 0)

    canvas = Image.new("RGB", (target_w, h), (128, 128, 128))
    offset = (target_w - w) // 2
    canvas.paste(img, (offset, 0))

    # マスク: 白=生成対象、黒=保持
    mask = Image.new("L", (target_w, h), 255)
    # 中央領域を黒（=保持）にする。境界に少しのフェザリングを入れると馴染みが良い。
    feather = 16
    inner = Image.new("L", (w - feather * 2, h), 0)
    mask.paste(inner, (offset + feather, 0))
    return canvas, mask


def outpaint(
    img: Image.Image,
    prompt: str = (
        "vintage hand-drawn animation cel, anime, consistent style, "
        "cinematic composition, soft cel shading, period-appropriate background"
    ),
    negative_prompt: str = (
        "modern, 3d, photorealistic, watermark, text, low quality, blurry, "
        "extra characters, duplicate"
    ),
    steps: int = 30,
    guidance: float = 7.5,
    seed: int | None = 42,
    model_id: str = "stabilityai/stable-diffusion-xl-inpainting-0.1",
) -> Image.Image:
    """SD-XL Inpaint で 4:3 → 16:9 にアウトペイント。

    対象作品でファインチューニングした LoRA があれば、呼び出し側で
    `pipe.load_lora_weights(...)` を実行してから本関数を使うのが望ましい。
    """
    from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore

    device = _device()
    dtype = torch.float16 if device != "cpu" else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    canvas, mask = _make_outpaint_canvas(img)

    # SDXL は 8 の倍数の解像度を要求するため調整
    w, h = canvas.size
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    canvas = canvas.resize((w8, h8), Image.LANCZOS)
    mask = mask.resize((w8, h8), Image.LANCZOS)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canvas,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Image.open(args.input).convert("RGB")
    kwargs = {"seed": args.seed}
    if args.prompt:
        kwargs["prompt"] = args.prompt
    dst = outpaint(src, **kwargs)
    dst.save(args.output)
    print(f"Outpainted: {src.size} -> {dst.size}")
