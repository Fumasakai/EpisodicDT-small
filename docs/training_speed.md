# 学習時間の短縮

> 設定整理後の実行手順は[最新README](../README.md)を参照してください。学習用YAMLは `training_fast.yaml` と `training_diffusion_scenario.yaml` の2つです。以下には過去の設定名を含む調査・実装時の記録があります。

## 追加した2種類の早期終了

### 学習全体の早期終了

```yaml
training:
  early_stopping:
    patience: 2
    min_delta: 0.0
```

完全な検証で、報酬がそれまでの基準値より `min_delta` を超えて改善しない状態が
`patience` 回続けば学習を終了する。初期検証は基準値を作るだけで、停滞回数には数えない。
不完全な検証は回数に数えない。現在の20更新ごとの検証では、改善がなければ通常40更新目に終了する。
`patience: 0` または設定省略で無効。

停止判定前に最新チェックポイントを保存する。再開時は保存済み履歴から停滞回数を復元し、
既に停止条件を満たす場合は追加のシミュレーションを実行しない。
この設定だけの変更は既存チェックポイントの `--resume` と互換性がある。
`metrics.json` の `updates` は実際の最終更新番号、`requested_updates` は上限、
`stop_reason` は `validation_plateau` または `update_limit`。

`training_inflight_fixed.yaml` と `configs/pipeline.yaml` に上記設定を追加した。
既に起動しているプロセスは変更前の設定を保持するので、自動では反映されない。

### 危険イベント確定時のシミュレーション終了

```yaml
simulation_binary: NS3_5GLENA_modified/build/scratch/ns3.46-tranData-training-fast
training:
  stop_on_danger: true
```

Encoder・Decoderの訓練候補にのみ適用する。対象イベントは現在の報酬定義に含まれる
HOF・ピンポンで、warmup終了以上・エピソード終了未満のイベントを使う。
ピンポンは同じUEの成功した往復HOと時間間隔を確認し、戻りのHOが完了した時点で確定する。
warmup中のHO履歴も保持するが、対象期間前に起きた危険イベントだけでは終了しない。

早期終了のmanifestには `completed: false`、`termination_reason: confirmed_danger`、
実際の終了時刻、イベント・IMSI・判定条件を記録する。
Python側でもイベントCSVを解析し、記録された危険イベントと一致することを確認して+1を与える。
未観測のUE・イベントを「危険なし」とせず、陽性が確認できたラベルだけを既知とする。
全期間の危険UE割合は分からないため `dangerous_ue_fraction` はnullにする。
早期終了ログを通常のエピソード変換に使うことはできない。

危険イベントが発生しない候補は全期間を実行する。初期検証・定期検証、拡散モデルの評価、
生成後の評価も全期間を実行する。通信条件・候補数・報酬定義は変更していない。
早期終了後の時間帯に発生するシミュレータ内部エラーは観測しなくなるため、
全期間を実行する検証・評価を引き続き使用する。

## 実行中の学習との分離

実装時に既存の学習が動いていたため、通常の実行ファイル・共有ライブラリを置き換えず、
高速化用実行ファイルを別名でビルドした。既存の学習プロセスは停止していない。

同じ環境で別名の実行ファイルを作る手順は次のとおり（現在はビルド済み）。

```bash
python NS3_5GLENA_modified/utils/build_training_fast.py
```

既存のNinjaビルド設定を使う。保存先が存在する場合は上書きせず停止する。
`setup_trandata.sh` は通常の実行ファイルも再ビルドするため、既存の学習中には実行しない。

## 利用方法

既存の学習を停止した後、同じビルドで学習全体の早期終了だけを使う場合：

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml --resume
```

候補シミュレーションの早期終了も使う場合は、別ビルドでの新規実行として重みを引き継ぐ。
旧実行と同時に走らせるとCPUを競合するため、先に旧実行を終了する。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_fast.yaml \
  --initialize-from data/processed/training_dataset_next/mobility_inflight_fixed/latest.pt
```

保存先は `mobility_fast/`。初期検証をやり直し、optimizerは初期化する。
更新番号は1から、上限81更新で、検証停滞による早期終了を有効にしている。
これは「元の200更新までの残数」を自動計算する設定ではない。
一度開始してチェックポイントを保存した後は、同じ設定で `--resume` を使う。

拡散モデル学習には `training_fast.yaml` と `mobility_fast/mobility.pt` を指定する。
拡散モデルの保存先は `diffusion_fast/`。

## 検証

Pythonテスト53件が通過。停滞回数・再開・不完全な検証・報酬の再確認・未観測ラベルの扱いを検証した。
実シミュレーションの記録は `data/processed/training_speed_verification/` に保存する。
通常実行と早期終了の報酬比較、HOFだけを対象にした終了、HO無効時の全期間実行を確認した。

同一陽性候補（24UE、13秒、run=102）の実測は通常73.29秒、早期終了4.98秒。
双方の報酬は+1で、早期終了時刻は3.658秒（ピンポン確定）。HO無効の条件は
全期間を実行して−1だった。既存の学習が並行稼働した状態での単発比較であり、
一般的な高速化倍率を示すベンチマークではない。
速度は経路・危険イベントの発生時刻・他の実行負荷によって変わる。
