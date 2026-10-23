# UE番号に依存しない拡散モデル

## 条件の定義

UEごとの144次元から、シナリオ全体の6次元へ変更した。
条件の並びは `[HOF, pingpong, RLF, HOF_known, pingpong_known, RLF_known]`。
各イベントについて、既知の陽性が1台でもあれば陽性・既知とする。
陽性がなく全UEで判定できれば既知の陰性、一部が未判定なら未知にする。
RLFは引き続き学習対象外で、値・マスクとも0。

UE1にHOFが発生した場合と、UE20にHOFが発生した場合は同じ条件になる。
HOFを起こすUEが1台から複数台になっても同じ条件になる。

訓練潜在状態の条件、ノイズ除去の条件、検証条件、訓練データの条件均等サンプリング、
アンカー選択、追加学習時の報酬重みのグループ分け、生成時の条件選択に同じ集約を使う。
Encoder・Decoderの学習方法・データ形式・モデルは変更していない。

報酬は引き続き任意のUEのHOFまたはピンポンで+1。
イベント種類まで指定どおりだったことを要求する報酬への変更は含まない。

## 再学習

条件の次元が変わるため、旧拡散モデルは継続学習・生成に使用できない。
形式を `ns3_latent_diffusion_v5`、条件形式を `scenario_event_or_v1` として記録し、
生成時には旧モデルを明示的に拒否する。既存のEncoder・Decoderはそのまま使用できる。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_diffusion \
  --config data/processed/training_dataset_next/training_diffusion_scenario.yaml \
  --checkpoint data/processed/training_dataset_next/mobility_inflight_fixed/mobility.pt
```

新規出力先は `data/processed/training_dataset_next/diffusion_scenario/`。
8並列・追加学習候補の危険確定時終了・検証の全期間実行を維持する。
過去の拡散学習結果は上書きしない。既に動いている学習には変更が反映されない。

## 学習後の生成

```bash
OMP_NUM_THREADS=1 python -m src.evaluation.generate_mobility \
  --checkpoint data/processed/training_dataset_next/diffusion_scenario/diffusion.pt \
  --event any_danger --count 4 --runs 201 202 203 --workers 8 \
  --output-dir data/processed/training_dataset_next/generated_scenario
```

`--ue-id` とUE別の `--signature` は廃止した。
`--event any_danger` は学習集合にある危険陽性のシナリオ条件から選ぶ。
`--event ho_failure` / `pingpong` は、そのイベントが任意のUEで陽性の条件から選ぶ。
両方陽性の条件も選ばれ得る。イベント発生はNS-3で確認し、生成条件だけでは保証しない。
`summary.json` の `source_conditions` / `source_masks` は候補ごとに3要素。
UEごとの成功条件を仮に割り当てることはしない。

## 確認

- UEの入れ替え・陽性UE数の増加に対して条件が不変。
- 一部未知を既知の陰性にしない。
- 生成条件から既知の非危険シナリオを除く。
- 旧モデルを読み込むと再学習の案内が出る。
- Pythonテスト63件が通過。
- 実データでは条件は3種類（非危険、ピンポンのみ、HOFとピンポン）。検証9件に未出条件は0件。
- 実モデルと実データで短い学習・保存・読み込み・生成を通して6次元とv5形式を確認。
  この接続テストのNS-3部分はモックであり、再学習後のイベント再現性能を示すものではない。

長時間の拡散モデル再学習は、この修正作業では開始していない。

生成後の評価も候補単位で並列実行します。`--workers` で並列数を指定でき、評価時は常に全期間を実行します。
