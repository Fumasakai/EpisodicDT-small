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
ステップごとの比較は `results/figures/forecast_boxplot_by_step.png` に保存されます。
各future stepについて、全評価エピソードの実測値（緑）と、全エピソード・全生成サンプルの
生成値（青）を並べた箱ひげ図です。生成値を平均化せずに集計します。箱は25--75%点、
中央線は中央値、ひげは1.5 IQR以内の最遠の観測値を表し、外れ値の点は非表示です。
横軸0は最初のfuture観測点です。箱の組数は `future_length` に従います。

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
小さめにしています。

## 変化量生成と検証によるモデル選択

モデルは `(future - historyの最終値) / 学習データの標準偏差` を生成します。
`sample()` はhistory最終値を加算して、絶対RSRPの標準化値を返すため、評価CSVの値は
従来どおりdBmです。history自体は絶対RSRPを標準化してエンコーダに渡します。

ノイズスケジュールは終端の信号対雑音比（SNR）がゼロになるよう再スケーリングします。
学習対象はノイズepsilonから `v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * residual`
へ変更しました。逆拡散ではvから残差を推定し、DDPMの事後平均・事後分散を使います。
これにより、純粋なガウスノイズからの生成開始と学習条件が一致し、終端alpha=0での
ゼロ除算も避けられます。表示される `train_v_mse` は旧モデルのノイズMSEとは異なる指標です。

学習用各シナリオの後半20%を検証に使い、境界をまたぐウィンドウは除外します。
標準化統計量も、検証区間を除いた学習用の前方区間だけで計算します。
最終評価データはEarly Stoppingには使いません。

各エポックで検証用historyから32系列を生成し、全エピソード・全future stepの平均CRPS
（小さいほど良い、単位dBm）でモデルを選択します。CRPSは実測との差と生成分布の広がりを
考慮する確率予測の指標です。検証ノイズは毎回固定し、学習側の乱数状態には影響させません。

- `validation_samples: 32`：各検証エピソードの生成数
- `validation_seed: 12345`：検証ノイズのシード
- `early_stopping_min_delta: 0.001`：改善とみなすCRPSの差（dBm）
- `early_stopping_patience: 15`：有意な改善がないエポックの許容数
- `min_epochs: 30`：最低学習回数。`epochs: 150` は最大回数

CRPSが最小になった時点のモデルを毎回保存します（min_delta未満の改善も保存対象）。
`training_metrics.json` に各エポックのv-MSE、CRPS、MAE、90%区間被覆率と、ステップ別の
CRPS・MAE・平均誤差（生成平均−実測）・実測/生成標準偏差・区間被覆率・区間幅を記録します。
50%/90%帯は生成サンプルの経験的分位点であり、実測の被覆率を保証するものではありません。

新しいチェックポイントとログは `results/checkpoints/residual_v1/` に保存します。
旧チェックポイントは新方式と互換性がないため再学習が必要です。
評価時は保存済み設定からモデル構造・history長・future長を復元します。
現在の実装ではノイズスケジュール・v予測・残差生成は一組の固定仕様で、
`model_format: zero_snr_v_residual_v1` によって識別します。
過去に作成したLaTeX資料のepsilon予測や行番号は旧実装の説明です。
現行方式の数式・アルゴリズム・行番号対応は
[日本語LaTeX資料](docs/episodicdt_residual_v1.tex) にまとめています。
LuaLaTeXで次のコマンドを2回実行すると目次・参照を含むPDFを生成できます。

~~~bash
lualatex -interaction=nonstopmode -halt-on-error -output-directory=docs docs/episodicdt_residual_v1.tex
~~~
