# SUTD 5G：学習・検証・評価区間の前後半入れ替え

既存の `results/figures/direct_z_conv_delta_episode_v4` と比較するため、4シナリオすべてで時系列の前後半を入れ替えて、新規モデルを学習・評価した。

| 実験 | 学習区間 | 検証区間 | 評価区間 |
|---|---|---|---|
| 既存 | 先頭～約40% | 約40～50% | 後半約50% |
| 今回 | 約50～90% | 末尾約10% | 前半約50% |

割合は各シナリオ内の行位置。`split_positions.png` に比較図、`split_manifest.json` にファイルのSHA256・時刻範囲・行数・窓数を保存。既存CSVを変更せず設定のtrain_pathとevaluation_pathを交換した。新設定でtrain_pathが従来のevaluationディレクトリを指すのは意図した動作。

学習3,337窓、検証720窓、評価4,211窓。40点の窓はシナリオ境界・学習/検証/評価の境界をまたがない。正規化の平均・標準偏差は新しい学習区間だけから計算。モデル・ハイパーパラメータ・学習seed=42・検証seed=12346を維持した。最大150エポック、元と同じ早期終了条件で117エポックにて終了し、検証合計損失が最小の113エポックを採用（検証損失0.14175）。元のモデル重みを初期値として利用していない。

評価seed=12346、各窓32本、全4,211窓から134,752本を生成。元結果・元モデル・元データは上書きしていない。

## 比較結果

| 指標 | 既存 | 今回 |
|---|---:|---:|
| 再構成CRPS [dBm]（小さいほど良い） | 1.3862 | 1.0954 |
| 再構成MAE [dBm]（小さいほど良い） | 1.8729 | 1.5065 |
| 90%区間被覆率 | 76.14% | 79.86% |
| 90%区間幅の平均 [dBm] | 5.3851 | 4.9529 |
| 固定z内の生成標準偏差の平均 [dBm] | 1.7441 | 1.6067 |
| 隣接差分の標準偏差：実データ [dB] | 2.4036 | 2.1814 |
| 隣接差分の標準偏差：生成 [dB] | 1.9233 | 1.8939 |
| 二階差分の標準偏差：実データ [dB] | 3.6418 | 3.2837 |
| 二階差分の標準偏差：生成 [dB] | 3.0059 | 2.9205 |

今回の入れ替えでは再構成誤差は小さくなり、被覆率も上がった。生成系列の変化量が実データより小さい傾向と、90%区間の被覆率が90%に達しない傾向は共通している。

評価元系列の平均RSRPは既存−103.60 dBm、今回−98.01 dBmであり、評価対象の分布も変わっている。数値の改善をモデル自体の優位性と解釈することはできない。今回は1通りの入れ替え・1つの学習seedでの比較で、任意の区間変更への安定性や統計的同等性を示すものではない。窓は重なっており独立標本ではない。

本評価は元エピソード全体をEncoderへ入力する条件付き再構成であり、未来予測ではない。今回は後方区間で学習して前方区間を評価している。

## 保存物

- `comparison_metrics.png`：主要評価指標の比較
- `comparison_training.png`：各分割での検証損失推移
- `comparison.csv` / `comparison.json`：比較値
- `episode_boxplot.png`：RSRP分布
- `episode_delta_distribution.png` / `episode_second_delta_distribution.png`：一階・二階差分分布
- `episode_autocorrelation.png` / `episode_mean_squared_change.png`：時間方向の特徴
- `episodes.png` / `episodes_02.png`～`episodes_10.png`：元系列10本と対応する生成系列の比較
- `episode_metrics.json` / `episode_examples.json`：評価指標・例の選択記録
- `config.yaml` / `split_manifest.json` / `provenance.json`：設定・分割・実行環境
- `train.log` / `evaluate.log`：実行ログ

モデルとエポックごとの学習指標は `results/checkpoints/direct_z_conv_delta_episode_v4_swapped_halves/`。
生成CSV・元系列CSV・潜在変数は `results/generated/direct_z_conv_delta_episode_v4_swapped_halves_*`。

## 再実行

プロジェクト直下、episodicdt環境で実行。同じコマンドは今回の成果物を上書きする。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.train.train --config configs/direct_z_conv_delta_episode_v4_swapped_halves.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.evaluation.evaluate --config configs/direct_z_conv_delta_episode_v4_swapped_halves.yaml --samples 32 --seed 12346
python results/figures/direct_z_conv_delta_episode_v4_swapped_halves/compare.py
```
