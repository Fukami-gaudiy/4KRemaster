"""比較・評価モジュール。

(1) 4手法のサイドバイサイド動画を作る（2x2 グリッド or 横並び）
(2) PSNR/SSIM/LPIPS のような客観指標を中央領域（元の4:3に存在する部分）で算出
(3) 結果を JSON / CSV にダンプしてレポート用に整形

使い方:
    python compare.py --grid 2x2
    python compare.py --grid 1x4 --label-bar
    python compare.py --metrics-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

METHODS = ["pillarbox", "crop", "outpaint", "hybrid"]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_grid_video(out_root: Path, output: Path, grid: str, label_bar: bool, fps: int) -> None:
    """ffmpeg の filter_complex で N手法を1動画に。"""
    inputs: list[str] = []
    for m in METHODS:
        v = out_root / m / "result.mp4"
        if not v.exists():
            print(f"  WARN: {v} が見つかりません。生成スキップ。")
            return
        inputs += ["-i", str(v)]

    if grid == "2x2":
        # 左上 pillarbox / 右上 crop / 左下 outpaint / 右下 hybrid
        filt = (
            "[0:v]scale=1920:1080,drawtext=text='Pillarbox':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5[a];"
            "[1:v]scale=1920:1080,drawtext=text='Crop':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5[b];"
            "[2:v]scale=1920:1080,drawtext=text='Outpaint':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5[c];"
            "[3:v]scale=1920:1080,drawtext=text='Hybrid':x=20:y=20:fontsize=48:fontcolor=white:box=1:boxcolor=black@0.5[d];"
            "[a][b]hstack[top];[c][d]hstack[bot];[top][bot]vstack"
        )
    elif grid == "1x4":
        filt = (
            "[0:v]scale=960:540,drawtext=text='Pillarbox':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5[a];"
            "[1:v]scale=960:540,drawtext=text='Crop':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5[b];"
            "[2:v]scale=960:540,drawtext=text='Outpaint':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5[c];"
            "[3:v]scale=960:540,drawtext=text='Hybrid':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5[d];"
            "[a][b][c][d]hstack=4"
        )
    else:
        raise ValueError(grid)

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filt, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)]
    _run(cmd)


def _center_crop(img: np.ndarray, target_ratio: float = 4 / 3) -> np.ndarray:
    """16:9 画像から中央の 4:3 領域だけ取り出す（元画像と比較するため）。"""
    h, w = img.shape[:2]
    target_w = int(round(h * target_ratio))
    if target_w >= w:
        return img
    left = (w - target_w) // 2
    return img[:, left:left + target_w]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_metrics(work_root: Path, out_csv: Path, sample_every: int = 30) -> None:
    """各手法 vs 元フレーム（中央領域）で PSNR/SSIM を計測。

    Outpaint や Pillarbox は左右が変わるだけなので、中央領域だけで比較すれば
    "原画忠実度" の指標になる。
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError as e:
        raise SystemExit("scikit-image が必要です: pip install scikit-image") from e

    try:
        import lpips
        import torch
        lpips_model = lpips.LPIPS(net="alex")
        if torch.backends.mps.is_available():
            lpips_model = lpips_model.to("mps")
        lpips_available = True
    except Exception:
        lpips_available = False
        print("LPIPS 計算スキップ（lpips パッケージなし）")

    raw_dir_any = work_root / METHODS[0] / "raw"
    frame_paths = sorted(raw_dir_any.glob("*.png"))[::sample_every]

    rows: list[dict] = []
    for fp in tqdm(frame_paths, desc="metrics"):
        ref = np.array(Image.open(fp).convert("RGB"))
        for m in METHODS:
            # 4K化済み画像を読み、4:3 領域に戻して比較
            cand_path = work_root / m / "upscaled" / fp.name
            if not cand_path.exists():
                cand_path = work_root / m / "converted" / fp.name
            if not cand_path.exists():
                continue
            cand = np.array(Image.open(cand_path).convert("RGB"))
            cand_4_3 = _center_crop(cand, 4 / 3)
            # サイズ合わせ
            cand_resized = np.array(
                Image.fromarray(cand_4_3).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
            )
            row = {
                "frame": fp.stem,
                "method": m,
                "psnr": _psnr(ref, cand_resized),
                "ssim": float(ssim_fn(ref, cand_resized, channel_axis=2, data_range=255)),
            }
            if lpips_available:
                import torch  # noqa: WPS433
                t = lambda a: (torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0)
                with torch.no_grad():
                    dev = next(lpips_model.parameters()).device
                    d = lpips_model(t(ref).to(dev), t(cand_resized).to(dev)).item()
                row["lpips"] = float(d)
            rows.append(row)

    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"指標を {out_csv} に出力")
        # 集計
        summary: dict[str, dict[str, float]] = {}
        for m in METHODS:
            ms = [r for r in rows if r["method"] == m]
            if not ms:
                continue
            summary[m] = {
                "psnr_mean": float(np.mean([r["psnr"] for r in ms])),
                "ssim_mean": float(np.mean([r["ssim"] for r in ms])),
            }
            if "lpips" in ms[0]:
                summary[m]["lpips_mean"] = float(np.mean([r["lpips"] for r in ms]))
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path(__file__).parent.parent / "outputs")
    parser.add_argument("--work-root", type=Path, default=Path(__file__).parent.parent / "outputs" / "_work")
    parser.add_argument("--grid", choices=["2x2", "1x4"], default="2x2")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample-every", type=int, default=30, help="N フレームごとに指標計算")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only and not args.no_video:
        make_grid_video(args.out_root, args.out_root / f"comparison_{args.grid}.mp4", args.grid, True, args.fps)

    compute_metrics(args.work_root, args.out_root / "metrics.csv", sample_every=args.sample_every)


if __name__ == "__main__":
    main()
