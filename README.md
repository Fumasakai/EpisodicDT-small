# EpisodicDT-small

RSRP（Reference Signal Received Power）の時系列を対象にした、小さく実行可能な
Episodic Decision Transformer 風の将来予測プロジェクトです。過去区間を
**TransformerEncoderベースの episode encoder** で潜在ベクトルへ圧縮し、そのコンテキストを条件として
**conditional diffusion model** が将来のRSRP系列を生成します。

## セットアップ

```bash
conda env create -f environment.yml
conda activate episodicdt
```

## 実行

```bash
# 1. SUTD 5Gの4シナリオからTimeとRSRPを抽出し、前半・後半へ分割
python -m src.data.prepare_sutd_5g

# 2. 学習（モデルと正規化統計量を results/checkpoints/ に保存）
python -m src.train.train --config configs/config.yaml

# 3. 将来予測・危険判定・図の出力
python -m src.evaluation.evaluate --config configs/config.yaml
```

`results/figures/forecast.png` に、評価用エピソードから均等に選んだ6件のhistory、将来の実測値、
生成中央値、生成値の50%・90%区間をsmall multiplesとして描画します。`danger_threshold` 未満になる予測割合を危険確率として
出力します。さらに、評価用の**全エピソード**に対する生成RSRP値は
`results/generated/rsrp_forecasts_all_episodes.csv` に保存されます。各行はエピソードID
（`episode_id`）、生成系列ID（`sample_id`）、予測開始からのステップ（`forecast_step`）、
生成RSRP値（`rsrp_generated_dbm`）です。実測futureは
`results/generated/rsrp_actual_all_episodes.csv` に保存されます。
評価用の全エピソード・全future stepを集約した実測futureと生成futureの分布を比較する箱ひげ図は
`results/figures/forecast_boxplot.png` に保存されます。

変換対象は `data/raw` にある次の4ファイルです。

- `Lvl4_AllRRUOn_Anomaly_label.csv`
- `Lvl5_AllRRUOn_Anomaly_label.csv`
- `Lvl6_1RRUOn_Anomaly_label.csv`
- `Lvl6_AllRRUOn_Anomaly_label.csv`

各ファイルから `Time` と `RSRP` だけを抽出し、時刻順に並べた後、前半を
`data/processed/sutd_5g/train/`、後半を `data/processed/sutd_5g/evaluation/` に保存します。
出力列は `time`, `rsrp` です。データは約2 Hzなので、history 20ステップとfuture 20ステップは
それぞれ約10秒、1エピソードは約20秒です。各シナリオ内でのみエピソードを作るため、
シナリオ間や学習・評価境界をまたぐエピソードは作られません。

## 構成

- `src/data`: データ生成と過去・未来ウィンドウの作成
- `src/models`: episode encoder と条件付き拡散モデル
- `src/train`: 学習エントリポイント
- `src/evaluation`: サンプリング、危険判定、可視化

設定値は `configs/config.yaml` で変更できます。CPUでも動くように、初期値は
小さめにしています。学習では各シナリオの学習用データ後半20%を検証用に取り分け、
検証拡散MSEが改善しない状態が15エポック続くとEarly Stoppingで終了します。`epochs` は
最大エポック数で、保存されるモデルは検証MSEが最小だったエポックのものです。
