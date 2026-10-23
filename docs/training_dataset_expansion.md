# 最新の学習データセット

現在使用する設定は `configs/training_dataset_v2.yaml`、経路は `data/scenarios/training_dataset_v2/`、データは `data/processed/training_dataset_v2/` です。実行コマンドは [README](../README.md) にまとめています。

## 条件と採否

基地局2局、A3制御、車両18・歩行者4・静止2の固定環境で、42経路をrun 1・2で各2回実行しました。
経路には、独立した境界通過、出発時刻の変更、境界往復、局所移動、1台だけの境界往復を含みます。
車両1〜65 m/s、歩行者0.3〜2 m/s、静止UEは0 m/sで、停止は許可しています。加速度・旋回角度・道路・衝突回避の制約はありません。

84回中53回が正常終了し、31回を異常終了として除外しました。正常終了した経路は34種類です。
関連経路と乱数違いは同じグループ・データ区分に固定しています。

| 区分 | 同期エピソード | HOF陽性 | ピンポン陽性 | 非危険 |
|---|---:|---:|---:|---:|
| 訓練 | 26 | 21 | 22 | 4 |
| 検証 | 9 | 7 | 7 | 2 |
| 評価 | 18 | 14 | 14 | 4 |

HOFとピンポンのラベルは重複します。ピンポンのみの例は訓練に1件です。
正常終了した実行の3秒以降のHOF603件は、すべて `HandoverFailureNoPreamble` でした。ピンポンは1496件です。イベント件数は独立したエピソード数ではありません。
2回の反復は初期的な再現性確認であり、精密な確率推定ではありません。人工的に危険条件を探索したデータなので、実環境の発生率を表していません。

## ラベルと学習用エピソード

NS-3のHOイベントを用い、エピソードごと・UEごとにHOF、ピンポン、RLFのラベルと観測マスクを保存します。RLFは検出検証未完了のため報酬には使いません。
10秒の入力エピソードに対し、経路復元用のウォームアップ3秒分も保持します。全UEの時系列がそろい、必要な履歴がある窓だけを `ScenarioDataset` が使用します。

`screening.json` の `positive/negative/unknown` は初期区間も含むUE窓全体の集計です。実際に使用する同期エピソードの数は `dataset_summary.json` と `audit.json` を参照してください。
異常終了を非危険例やHOF陽性例に数えることはありません。

## 保存先と整理後の参照

`data/processed/training_dataset_v2/` 内にすべての必要な実験データを集約しています。

- `episodes/`：区分ごとの学習用NPZ。参照する生データは現在の `sources/` に更新済みです。
- `sources/<scenario>/raw/<run>/`：通信KPI、近隣測定、イベント、実行マニフェスト。
- `sources/<scenario>/inputs/`：実行時のCSV。
- `sources/<scenario>/logs/`：異常終了を含む実行ログ。
- `sources/<scenario>/figures/input_routes/`：XY経路と座標の時間変化の図。
- `sources/<scenario>/plan.json`：実行当時の設定・コマンド・ハッシュ。
- `scenario_outcomes.csv`：実行別の採否と、現在の `source_batch`。
- `repeat_outcomes.csv`：経路ごとの反復結果。
- `hof_events.csv`：正常終了した実行のHOF記録。
- `consolidation.json` / `cleanup.json`：参照先移行と削除対象の記録。

生データはバイト単位で保全しており、学習ローダーで全53エピソードのハッシュ検証を通しています。
原計画・生ログ・再実行の要求記録に含まれる旧パスは、実行当時の情報を保つため変更していません。これらは当時の履歴で、再開用の設定ではありません。現在の再実行には `configs/training_dataset_v2.yaml` を使用します。

## 切り出し後のイベント再現

各区分のHOF例1件と非危険例1件、計6件を元CSVから切り出し、ウォームアップ3秒付きで再実行しました。全UEの折れ点時刻の和集合を使い、停止・曲がる位置を保った経路を渡しています。

結果は `replay/hof/replay_summary.json` と `replay/negative/replay_summary.json` にあります。6件とも正常終了し、HOF例はHOFとピンポンが再現、非危険例は非危険のままでした。
これは内部状態のチェックポイント復元ではなく、新規初期化したNS-3での確認です。全エピソードの再現保証ではありません。

新たに切り出し確認を実行する場合：

```bash
OMP_NUM_THREADS=1 python -m src.data.replay_training_episodes \
  --batch-dir data/processed/training_dataset_v2 \
  --output-dir data/processed/training_dataset_replay --workers 3
```

このコマンドは各区分で最初に見つかったHOF例、HOFなしのピンポン例、既知の非危険例を最大1件ずつ選びます。既存の監査と選択例が同一とは限りません。
