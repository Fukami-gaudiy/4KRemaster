# 実験結果テンプレ

実験を回したらこのファイルに結果を記入する。

## 使用素材
- 作品名:
- 取得元 URL:
- 元解像度 / fps:
- 切り出し範囲（秒）:

## 実行環境
- マシン: MacBook Pro (M__ / __GB)
- macOS バージョン:
- PyTorch: torch.__version__ =
- MPS 利用可否:

## 各手法の所要時間 (10秒抜粋)
| 手法 | フレーム変換 | 4K化 | 合計 |
|------|----:|----:|----:|
| Pillarbox | | | |
| Crop | | | |
| Outpaint | | | |
| Hybrid | | | |

## 客観指標 (compare.py 出力)
中央 4:3 領域 vs 元画像。値が高いほど原画忠実度が高い（LPIPS のみ低いほうが良い）。

| 手法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|----:|----:|----:|
| Pillarbox | | | |
| Crop | | | |
| Outpaint | | | |
| Hybrid | | | |

## 定性評価 (5段階)
| 手法 | 違和感のなさ | 線画品質 | 時間安定性 | 旧作の味 | 総合 |
|------|----:|----:|----:|----:|----:|
| Pillarbox | | | | | |
| Crop | | | | | |
| Outpaint | | | | | |
| Hybrid | | | | | |

## サンプル静止画
- フレーム N (closeup): outputs/_work/{method}/upscaled/000NNN.png
- フレーム N (landscape): ...

## 観察された課題
-
-
-

## 次の打ち手
-
-

## CEO への報告サマリ
- 一言結論:
- 推奨する次フェーズ投資:
  - 例: クラウド GPU での Outpaint LoRA 学習 / 別作品でのトライアル / 商用化検討の判断
