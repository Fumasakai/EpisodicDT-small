# EpisodicDT-small

NS-3の通信エピソードから潜在状態を学習し、危険イベントが起きるUE群の移動経路を生成する研究用プログラムです。分類器は使わず、生成経路をNS-3で実行した結果を報酬にしてEncoder・Decoderを同時学習します。その後、イベント条件付き潜在拡散モデルを学習します。

```text
固定基地局・通信条件YAML ＋ UE経路CSV
  → NS-3 → 通信ログ・HOイベント → エピソードとラベル
                                     ↓
                              Encoder → 潜在状態z
                                         ↓
                              Decoder → UE群の移動経路
                                 ↑              ↓
                                 └── 報酬 ← NS-3

学習済みEncoder → イベント条件付き潜在拡散モデル → 新しいz
                                                   ↓
                                          Decoder → NS-3で検証
```

## 現在使用するファイル

`data/processed/` には次の2フォルダを残しています。

```text
data/processed/
  training_dataset_next/          # 現在の学習データ・最新設定・再利用するモデル
    episodes/
    mobility_inflight_fixed/
    training_fast.yaml
    training_diffusion_scenario.yaml
    simulation.yaml
  training_dataset_next_screen/   # 学習データから参照する原ログ・入力経路・経路図
```

`mobility_fast/`、`diffusion_scenario/`、評価結果のフォルダは、対応するコマンドを実行すると作成されます。
原データ内の失敗ログも含め、現在のデータセット作成記録は保持しています。


| 用途 | パス |
|---|---|
| データ生成条件・固定基地局・シナリオ一覧 | `configs/training_dataset_v2.yaml` |
| 入力経路CSV | `data/scenarios/training_dataset_v2/` |
| 現在の学習データセット | `data/processed/training_dataset_next/` |
| データセットが参照する実行原データ | `data/processed/training_dataset_next_screen/` |
| 修正後の学習設定 | `data/processed/training_dataset_next/training_inflight_fixed.yaml` |
| 修正後のEncoder・Decoder保存先 | `data/processed/training_dataset_next/mobility_inflight_fixed/` |
| 修正後の拡散モデル保存先 | `data/processed/training_dataset_next/diffusion_inflight_fixed/` |
| 旧ビルドでの119更新目の重み | `data/processed/training_dataset_next/mobility_fixed/latest.pt` |

`training_fixed.yaml`、`training.yaml`、旧モデルのディレクトリは過去の実行記録です。以下の学習・生成手順では `training_inflight_fixed.yaml` を使用します。`training_dataset_v2` という名前は現在の入力CSVと生成条件にも使われていますが、学習データの参照先は `training_dataset_next` です。

```text
src/data/          経路生成、NS-3実行、エピソード変換、監査
src/models/        Encoder・Decoder、潜在拡散モデル
src/simulation/    生成経路のNS-3評価と報酬
src/train/         モデル学習
src/evaluation/    学習済みモデルからの経路生成と検証
src/tools/         ブラウザ経路エディタ
NS3_5GLENA_modified/  シミュレータ本体と修正パッチ
tests/             Pythonテスト
docs/              調査記録と説明資料
```

## 環境の準備

プロジェクト直下で実行します。

```bash
conda env create -f environment.yml  # 環境が未作成の場合のみ
conda activate episodicdt
bash NS3_5GLENA_modified/utils/setup_trandata.sh
```

セットアップはNS-3への修正パッチ適用とビルドを行います。現在のワークスペースでは内部エラー修正後のビルドを確認済みです。既存データで学習するためにデータ生成をやり直す必要はありません。

## NS-3内部エラーの修正

`Cannot receive DATA while receiving CTRL.` の原因は、ハンドオーバー前から受信中だった旧セルの制御情報が、新セルでの送信指示として処理されることでした。接続リセット時に古い制御情報を無効化しました。異常検知を削除する変更ではありません。

問題の検証条件16実行と過去の訓練条件8実行が全て正常終了しました。HO統合テスト4条件、新規C++テスト、Pythonテスト48件も通過しています。修正は `utils/patches/0006-handover-invalidate-inflight-control.patch` に保存しています（NS-3ディレクトリ内）。

詳細は[原因・修正内容・検証結果](docs/ns3_inflight_control_fix.md)を参照してください。

## Encoder・Decoderの学習

### 開始済みの修正後学習を再開する

現在の `mobility_inflight_fixed/` にはチェックポイントがあります。停止後に再開する場合は次を使います。実行中の学習と同じ保存先に別の学習プロセスを起動しないでください。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml \
  --resume
```

`latest.pt` のモデル・optimizer・更新数などを復元します。同じNS-3ビルドと互換性のある設定が必要です。

### 旧119更新目の重みから初回起動する場合

次は修正後の保存先がまだ存在しない場合の初回用コマンドです。開始済みの学習には上の `--resume` を使います。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml \
  --initialize-from data/processed/training_dataset_next/mobility_fixed/latest.pt
```

NS-3ビルドが変わったため、旧チェックポイントの直接 `--resume` は互換性検査で拒否されます。`--initialize-from` はデータ・モデルの互換性を確認して重みだけを引き継ぎます。optimizerと最良検証スコアは初期化し、修正後のNS-3で初期検証をやり直します。

設定は8並列・追加81更新です。元の200更新から保存済み119更新を除いた数で、新しい実行の表示は `update=1` から始まります。旧結果は上書きしません。引き継ぎ元とハッシュは `initialization.json` に記録します。

重みを引き継がず新規学習する場合は、このYAMLを別名にコピーし、`output_dir` と `diffusion_output_dir` を未使用の保存先に変更して、必要な `training.updates` を設定します。その設定を指定し、`--resume` と `--initialize-from` を付けずに実行してください。

### 学習処理と報酬

1更新では入力エピソード2件をEncoderに渡し、それぞれから4候補、計8候補の経路をDecoderで生成します。通常は候補ごとにNS-3を1回実行します。正常終了した候補が不足した場合は追加試行するため、実行数が増えることがあります。

報酬は、どれか1台でHOFまたはピンポンが確認できれば+1、全UEについて対象イベントなしと確認できれば−1です。元エピソードの全UEとのイベント一致は要求しません。不明な観測やシミュレータ異常終了を負の報酬として扱いません。REINFORCEと同じ入力の他候補によるbaselineでEncoder・Decoderを更新します。

生成経路にはUE属性の速度範囲と初期位置の制約を適用します。車両は移動中1〜65 m/s、歩行者は0.3〜2 m/s、静止UEは0 m/sで、停止を許します。加速度・旋回角度の制約は設けません。入力と出力のUE数・属性構成は同じです。

検証は訓練の更新とは別に8候補×run 101/102の16実行で行い、最良モデルを選びます。不完全な検証では最良モデルを更新しません。`mobility update` は更新番号、`reward` はその更新の有効候補の平均報酬、`danger_success` は危険イベントの成功率です。

並列数は `training.simulation_workers` で変更できます。4並列と8並列の比較では8並列が速かったため、現在は8を使用しています。[比較結果](docs/parallel_simulation.md)を参照してください。

## 学習時間の短縮

検証報酬が2回連続で改善しなければ終了する設定を追加しました。
`training_inflight_fixed.yaml` の `training.early_stopping` で調整できます。
既に起動している学習には自動反映されません。再開時は過去の検証履歴も使って停止を判定します。

さらに、学習候補でHOFまたはピンポンが確定した時点でNS-3を終了する高速化版を用意しました。
検証は全期間を実行します。通常のNS-3実行ファイルは保持し、高速化版を別名でビルド済みです。

既存の学習を停止した後、次のコマンドで重みを引き継げます。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_fast.yaml \
  --initialize-from data/processed/training_dataset_next/mobility_inflight_fixed/latest.pt
```

保存先は `mobility_fast/`、初期検証とoptimizerは新規に作成します。上限81更新、停滞時は早期終了します。
開始後の再開は同じ設定に `--resume` を指定します。
詳細と別名ビルドの手順は[学習時間の短縮](docs/training_speed.md)を参照してください。

## 拡散モデルの学習と危険経路生成

拡散モデルの条件は、**UE番号を含まないシナリオ単位のイベント情報**に変更しました。
HOF・ピンポン・RLFそれぞれの「任意のUEで発生」と「判定済み」を連結した6次元です。
RLFは引き続き対象外です。UE番号が違っても同じシナリオ条件として学習・生成します。
旧144次元の拡散モデルは使用できませんが、Encoder・Decoderと元データは再利用できます。
新しいモデル形式はv5です。詳細は[シナリオ単位の条件](docs/scenario_diffusion.md)を参照してください。


現在の通常版Encoder・Decoderを使い、**拡散評価8並列・追加学習候補の早期終了**を有効にするには次を使います。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_diffusion \
  --config data/processed/training_dataset_next/training_diffusion_scenario.yaml \
  --checkpoint data/processed/training_dataset_next/mobility_inflight_fixed/mobility.pt
```

保存先は `diffusion_scenario/`。ノイズ除去学習から始める別実行です。
既存の学習結果は上書きしません。検証は全期間を実行し、進捗を候補完了時と待機中に表示します。
[高速化設定・表示の読み方](docs/diffusion_speed.md)を参照してください。
このモデルで経路生成する場合は、下記の生成コマンドのチェックポイントを
`data/processed/training_dataset_next/diffusion_scenario/diffusion.pt` に変更してください。



```bash
OMP_NUM_THREADS=1 python -m src.evaluation.generate_mobility \
  --checkpoint data/processed/training_dataset_next/diffusion_scenario/diffusion.pt \
  --event any_danger \
  --count 30 \
  --seed 123 \
  --runs 201 202 203 \
  --workers 8 \
  --output-dir data/processed/training_dataset_next/evaluation_scenario_parallel_001
```

30候補の経路を生成し、各候補を3種類の乱数条件で検証します。
候補単位で最大8並列とし、各候補内のrun 201・202・203は順番に実行します。
評価は早期終了せず全期間を実行します。`--workers` を省略するとチェックポイント内の
`diffusion.simulation_workers`（なければ `training.simulation_workers`、両方なければ1）を使います。
`--workers 1` で逐次実行できます。候補完了時と待機中の約30秒ごとに進捗を表示し、
`summary.json` に並列数を記録します。既に起動中の評価には変更は反映されません。
出力には属性別経路CSV、移動点配列の `routes.json`、潜在状態、経路図、NS-3ログ、評価結果を含みます。成功はシミュレーション結果で確認します。再実行時は未使用の出力先を指定してください。

## 現在の学習データ

UEは車両18・歩行者4・静止2、基地局は固定2局、制御はA3です。データ作成時は84条件を実行し、正常終了53条件を採用、異常終了31条件を除外しました。これは今回のNS-3修正前に作成したデータの集計で、修正後の再生成結果ではありません。

| 区分 | 同期エピソード | 危険イベントあり | 既知の非危険例 |
|---|---:|---:|---:|
| 訓練 | 26 | 22 | 4 |
| 検証 | 9 | 7 | 2 |
| 評価 | 18 | 14 | 4 |
| 合計 | 53 | 43 | 10 |

集計元は `training_dataset_next/dataset_summary.json` です。同一経路の乱数違いや派生経路は同じ区分に固定します。HOFとピンポンは重複する場合があります。既存データのHOF原因は専用プリアンブル不足に偏っています。RLFは検出検証未完了のため学習報酬に使用していません。

## 新しくデータを生成する

修正後のNS-3でデータを作り直す場合は、既存データと分けて新しい保存先に出力します。

```bash
OMP_NUM_THREADS=1 python -m src.data.screen_training_expansion \
  --config configs/training_dataset_v2.yaml \
  --output-dir data/processed/training_dataset_inflight_screen \
  --merged-dir data/processed/training_dataset_inflight --workers 3
```

候補を独立実行し、正常終了した条件を統合します。現在の設定では各候補の `simulation_run` に従って合計84回実行します。新しいデータセットで学習する場合は、その出力先の `training.yaml` を使用します。既存の `training_inflight_fixed.yaml` が自動で新データへ切り替わることはありません。

入力CSV・実行計画・経路図だけを確認するには次を使います。

```bash
python -m src.data.generate_handover_batch \
  --config configs/training_dataset_v2.yaml \
  --batch-dir data/processed/input_preview_inflight --prepare-only
```

既存の保存先は上書きしません。設定・入力・コード・NS-3ビルドが変わったバッチは再開せず、新しい保存先を使用してください。

## CSV経路と経路エディタ

`configs/training_dataset_v2.yaml` の `environment.route_csv_root` が共通フォルダ、`environment.scenarios` が使用するシナリオ一覧です。

```text
data/scenarios/training_dataset_v2/<scenario>/
  car.csv
  walker.csv
  static.csv
```

CSV列は `timestamp,ue_id,location_x,location_y`、単位は秒・メートルです。UE属性ごとにIDを1から付け、各UEの時刻を0からシミュレーション終了時刻まで昇順で記載します。同じ座標を異なる時刻に指定すると停止になります。

```bash
python -m src.tools.route_editor --config configs/training_dataset_v2.yaml
```

ブラウザで固定基地局を確認し、UE属性・経路点・時刻を編集できます。速度範囲外では警告します。保存先は `data/scenarios/manual_routes/revisions/`、保存した経路を指す設定は `configs/routes_editor.yaml` です。終了は起動した端末で `Ctrl+C` を押します。

## 保存先の詳細

以下のデータセット内パスは `data/processed/training_dataset_next/` を基準とします。

| パス | 内容 |
|---|---|
| `training_inflight_fixed.yaml` / `simulation.yaml` | 現在の学習設定／シミュレーション条件 |
| `episodes/{train,validation,evaluation}/*.npz` | 通信特徴量・欠損マスク・イベントラベル |
| `dataset_summary.json` / `screening.json` | 採用件数・区分別集計／候補の実行結果と原データ参照先 |
| `mobility_inflight_fixed/mobility.pt` | 検証報酬が最良のEncoder・Decoder |
| `mobility_inflight_fixed/latest.pt` | 再開用の最新重み・optimizer・更新数等 |
| `mobility_inflight_fixed/history.json` | 更新ごとの報酬・検証履歴 |
| `mobility_inflight_fixed/initialization.json` | 旧チェックポイントからの重み引き継ぎ記録 |
| `mobility_inflight_fixed/rollouts/` | 更新中に生成した経路、NS-3出力、成功・失敗の記録 |
| `mobility_inflight_fixed/validation*/` | 初期・定期検証の経路と実行結果 |
| `mobility_inflight_fixed/metrics.json` | 学習完了時の集計 |
| `diffusion_inflight_fixed/` | 拡散モデル、ノイズ除去学習履歴、NS-3による追加評価 |
| `generated_inflight_fixed/` | 学習後に生成した候補と検証結果 |

訓練データ作成時の生ログは、`screening.json` の `source_batch` が指すディレクトリにあります。現在は主に次の構造です。

```text
data/processed/training_dataset_next_screen/<candidate>/<scenario>/
  raw/<run>/ue_kpi.csv        RSRP・SINR・位置などの通信情報
  raw/<run>/ho_events.csv     HOイベント
  inputs/                    実行時の経路CSV
  figures/                   入力経路図
  logs/                      実行ログ
```

学習中に生成した候補の通信ログは、各候補ディレクトリの `run_<番号>/ue_kpi.csv` にあります。異常終了時は `failure.json`、正常に評価できた場合は `result.json` を確認してください。

学習時に原データのハッシュ検証と経路復元を行うため、`training_dataset_next_screen/` も必要です。NPZだけを残して参照先を削除しないでください。

## 検査と資料

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -q
python NS3_5GLENA_modified/utils/test_trandata_handover.py

OMP_NUM_THREADS=1 python -m src.data.audit_training_expansion \
  --batch-dir data/processed/training_dataset_next \
  --candidates data/scenarios/training_dataset_v2/candidates.json
```

- [NS-3内部エラーの修正と重み引き継ぎ](docs/ns3_inflight_control_fix.md)
- [並列数の比較結果](docs/parallel_simulation.md)
- [データセット作成・監査の記録](docs/training_dataset_expansion.md)
- [ミーティング資料のLaTeXソース](docs/meeting_overleaf/README.md)
