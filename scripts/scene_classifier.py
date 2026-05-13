"""シーン分類器 (Hybrid 手法のコア独自性)。

CLIP のゼロショット分類で各フレームを以下のいずれかに分類し、
パイプライン側で適切なアスペクト変換手法を選ぶ。

  closeup   → Pillarbox（顔・キャラのアップは Outpaint で歪みやすい）
  landscape → Outpaint （広い背景は左右生成で自然に拡張可能）
  dialogue  → Outpaint （2人の会話シーン等。横方向に意味のある空白）
  action   → Crop or Pillarbox（速い動きは Outpaint のフレーム間揺れが目立つ）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

SceneType = Literal["closeup", "landscape", "dialogue", "action"]

# ゼロショット分類用のプロンプト。CLIP が学習データで見ているテキスト分布に合わせる。
PROMPTS: dict[SceneType, list[str]] = {
    "closeup": [
        "a close-up shot of a character's face in anime",
        "a portrait of a single anime character filling the frame",
    ],
    "landscape": [
        "a wide landscape or scenery shot in anime",
        "an establishing shot of a town or nature in anime",
    ],
    "dialogue": [
        "two characters talking to each other in an anime scene",
        "a medium shot of multiple anime characters in conversation",
    ],
    "action": [
        "an action scene with fast motion in anime",
        "characters fighting or running in an anime scene",
    ],
}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class SceneResult:
    scene: SceneType
    confidence: float
    scores: dict[SceneType, float]


class SceneClassifier:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        import open_clip  # type: ignore

        self.device = _device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # 全プロンプトをまとめてエンコードしておく
        all_prompts: list[str] = []
        prompt_owner: list[SceneType] = []
        for scene, prompts in PROMPTS.items():
            for p in prompts:
                all_prompts.append(p)
                prompt_owner.append(scene)
        with torch.no_grad():
            text_tokens = self.tokenizer(all_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features
        self._prompt_owner = prompt_owner

    @torch.no_grad()
    def classify(self, img: Image.Image) -> SceneResult:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sims = (feat @ self._text_features.T).squeeze(0).cpu().tolist()

        # シーンごとに最大スコアを取って正規化
        scene_scores: dict[SceneType, float] = {}
        for scene, score in zip(self._prompt_owner, sims):
            scene_scores[scene] = max(scene_scores.get(scene, -1e9), score)
        # softmax 風に正規化
        max_s = max(scene_scores.values())
        exp_s = {k: pow(2.71828, (v - max_s) * 10) for k, v in scene_scores.items()}
        total = sum(exp_s.values())
        probs: dict[SceneType, float] = {k: v / total for k, v in exp_s.items()}

        best = max(probs, key=lambda k: probs[k])
        return SceneResult(scene=best, confidence=probs[best], scores=probs)


def recommended_method(scene: SceneType) -> str:
    """シーン → 推奨アスペクト変換手法。"""
    return {
        "closeup": "pillarbox",
        "landscape": "outpaint",
        "dialogue": "outpaint",
        "action": "pillarbox",  # crop でもよいが頭切れリスクを避ける
    }[scene]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    clf = SceneClassifier()
    res = clf.classify(img)
    print(f"Scene: {res.scene} (conf={res.confidence:.2%})")
    for k, v in sorted(res.scores.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v:.2%}")
    print(f"Recommended method: {recommended_method(res.scene)}")
