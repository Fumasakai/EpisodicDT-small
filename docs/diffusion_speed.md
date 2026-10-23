# 拡散モデルのNS-3評価を高速化

> 設定整理後の実行手順は[最新README](../README.md)を参照してください。学習用YAMLは `training_fast.yaml` と `training_diffusion_scenario.yaml` の2つです。以下には過去の設定名を含む調査・実装時の記録があります。

## 変更内容

- 初期検証・各ラウンドの検証・追加学習用候補を並列評価する。
- `diffusion.simulation_workers` で並列数を指定する。省略時は
  `training.simulation_workers`、両方省略時は1を使う。
- `diffusion.stop_on_danger: true` の場合、追加学習用候補だけ危険イベント確定時に終了する。
  初期検証と各ラウンドの検証は常に全期間を実行する。
- 検証候補のrun 101/102は候補内で順次実行し、NS-3の同時実行数は上限を超えない。
- 結果は元の候補順に戻す。潜在状態・条件・報酬の対応と重み付け方法は維持する。
- 候補完了時と待機中の約30秒ごとに完了数・並列数・経過秒数を表示する。
- ノイズ除去学習も開始時・100更新ごと・最終更新に進捗を表示する。

表示例：

```text
diffusion validation_000 candidates=0/8 workers=8 early_stop=False elapsed=0.0s
diffusion validation_000 candidates=1/8 workers=8 early_stop=False elapsed=...
diffusion round_001 candidates=0/24 workers=8 early_stop=True elapsed=0.0s
```

`candidates` は乱数条件をすべて評価し終えた候補の数。無効候補も処理完了数に含む。
待機表示で数が増えていなくても、それだけでは停止を意味しない。

## 現在の通常版モデルを使う実行手順

```bash
OMP_NUM_THREADS=1 python -m src.train.train_diffusion \
  --config data/processed/training_dataset_next/training_diffusion_fast.yaml \
  --checkpoint data/processed/training_dataset_next/mobility_inflight_fixed/mobility.pt
```

新しい保存先は `data/processed/training_dataset_next/diffusion_parallel_fast/`。
既存の `diffusion_inflight_fixed/` は変更しない。これは追加学習途中からのresumeではなく、
既存Encoder・Decoderを固定し、ノイズ除去学習から始める新しい実行である。
旧実行が動いている場合は先に終了し、競合する2つの学習を起動しない。

通常のNS-3バイナリは上書きせず、既に検証済みの別名バイナリ
`NS3_5GLENA_modified/build/scratch/ns3.46-tranData-training-fast` を使う。
バイナリがない環境では、通常ビルドを用意した後に
`python NS3_5GLENA_modified/utils/build_training_fast.py` で作成する。

高速化版Encoder・Decoderの `mobility_fast/mobility.pt` を使う場合は、
`training_fast.yaml` を指定する。その設定にも拡散評価8並列・候補早期終了を追加した。
モデルの入力・潜在空間・通信条件などの互換性検査は維持する。

## 確認結果と限界

Pythonテスト58件が通過。結果の順序保持、無効候補の保持、訓練だけの早期終了、
検証時の全期間実行指定、先頭候補が遅い場合の進捗通知を確認した。
実際のNS-3でも同じ陽性経路の2候補を並列評価し、両方が約4.7秒で
イベントを再確認した正の報酬を返した。
記録は `data/processed/diffusion_parallel_verification/`。
これは候補評価の動作確認であり、学習全体の8倍高速化を保証する比較ではない。

生成数や更新数を減らしていないため、予定実行数は引き続き216回。
危険イベントが遅い候補や負例は時間がかかる。並列数を増やすとCPU・メモリ・I/O負荷も増える。
この変更では長時間の拡散学習全体は自動開始していない。
