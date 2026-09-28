# EpisodicDT-small

元エピソード全体からEncoderが出力したベクトルをそのまま潜在変数zとし、
zと拡散ノイズから新しいRSRPエピソード全体を生成します。
標準形式は `direct_z_conv_episode_v3` です。

```text
元エピソード [B,40,1] → Transformer Encoder → z [B,32]
                                                 ↓
                                    拡散モデル (x_t, t, z)
                                                 ↓
                                      全体系列 [B,S,40,1]
```

μ・σの推定、潜在変数のサンプリング、KL正則化はありません。
損失は全体系列のv予測MSEと隣接変化量の補助MSEの重み付き和で、両方の勾配がzを通してEncoderへ伝わります。
推論時は `model.eval()` を使い、同じ入力に対して同じzを得ます。
学習時にはEncoderのdropoutが有効です。
同じzからでも、拡散モデルのノイズを変えることで複数の系列を生成できます。

## 実行

```bash
conda activate episodicdt
# 未前処理の場合
python -m src.data.prepare_sutd_5g
python -m src.train.train --config configs/config.yaml
python -m src.evaluation.evaluate --config configs/config.yaml --samples 32 --seed 12346
```

設定は `configs/config.yaml`。`data.episode_length: 40` は全体のステップ数です。
評価と保存済みzからの生成は、CUDAが利用可能なら自動でGPUを使用し、利用できなければCPUで実行します。使用デバイスは起動時に表示します。生成結果とzはバッチごとにCPUへ戻し、CSV・グラフ・潜在変数ファイルの保存形式を維持します。
約2 Hzなら約20秒ですが、実時間は元の計測時刻に依存します。
固定ステップのRSRP系列を扱い、時刻の補間は行いません。

## zだけから再生成

```bash
python -m src.evaluation.generate \
  --checkpoint results/checkpoints/direct_z_conv_delta_episode_v4/episodicdt.pt \
  --latents results/generated/direct_z_conv_delta_generated_episodes.latents.pt \
  --output results/generated/from_direct_z.csv \
  --samples 16 --seed 42
```

元CSVを読むことなく、保存済みzから生成します。潜在ファイルにはz、元窓情報、
チェックポイントのSHA-256を保存します。異なるモデル由来のzは拒否します。
旧方式の `--resample-z` は廃止しました。

```python
model.eval()
z = model.encode(normalized_episode)  # Tensor [B,32]; Encoderの出力そのもの
new_episodes = model.generate(z, num_samples=16)  # [B,16,40,1]
rsrp_dbm = new_episodes * training_std + training_mean
```

## 学習と評価

- 入力はCSVの `rsrp` 列。窓はファイル境界をまたぎません。
- 学習用CSVを時間順に学習・検証へ分割し、境界をまたぐ窓を除外します。
- 標準化は検証を除いた学習区間の平均・標準偏差を使います。
- 絶対RSRPを生成し、元系列の最終値を加算する処理はありません。
- 検証ノイズを固定し、検証合計損失で保存モデルとEarly Stoppingを決めます。
  ログの `loss = diffusion_mse + delta_loss_weight * delta_mse` です。
- 評価は再生成CRPS・MAE・90%区間被覆率・区間幅と、固定z内の生成標準偏差です。
  元系列全体をEncoderへ入力する条件付き再生成であり、未来予測精度ではありません。
- 危険例生成率や危険閾値表示は含みません。標準学習は通常・危険の全窓を対象とします。

チェックポイントは `results/checkpoints/direct_z_conv_delta_episode_v4/`、図は
`results/figures/direct_z_conv_delta_episode_v4/` に保存します。
生成CSVの列は `episode_id, sample_id, episode_step, rsrp_generated_dbm`。
元系列CSV、潜在変数ファイル、評価JSONも保存します。

**旧MLP形式 `direct_z_episode_v2`、旧形式 `latent_episode_v1`、`zero_snr_v_residual_v1` とは非互換で、再学習が必要です。**
旧結果を比較用に残すため、新形式は別の保存先を使います。
旧設定から移行する場合は `kl_beta` を削除し、新しい設定と保存先を使ってください。
旧未来予測用の `EpisodicDiffusion` / `RSRPWindowDataset` は比較用に残しています。

## テストと今後の拡張

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/episodicdt-mpl python -m unittest discover -s tests -v
```

Encoder出力とzの一致、推論時の決定性、Encoderへの勾配、zのみの生成、
学習・保存・評価・元CSVなしの再生成、旧方式の回帰テストを確認します。

同条件・別乱数のペアがある場合、`model.loss_terms(source, target=paired_episode)`
で学習できます。標準のCSVには条件ペア情報がないため、現在は自己再生成です。
潜在変数の物理的意味、生成系列の現実性は未検証です。
環境条件への対応付け、シミュレータへの接続、制御AIの比較評価は今後の課題です。

[2ページ想定のLaTeX概要](docs/latent_episode_overview.tex)

## 隣接ステップの変化量分布

`episode_delta_distribution.png` は各エピソード内で `x[t+1] - x[t]` を計算して比較します。
元系列と個々の生成系列の変化量分布を、1つのパネルで比較します。
横軸はdB（時間当たりの変化率ではない）、縦軸は確率密度です。
共通のビン境界と面積1の正規化で、標本数の違いを考慮します。
全値を含み、別エピソードや別生成サンプルの境界をまたぐ差は取りません。
生成側の分布が広いほど急な変化が多いことを示しますが、これだけでは長時間の相関は評価できません。

## 時間方向の畳み込み生成器（v3）

拡散モデルのv予測器をMLPからConv1dへ変更しました。Encoder、v予測MSE、
ノイズスケジュール、DDPMサンプリングは維持しています。Encoderは従来どおり同時学習します。
固定した学習済みEncoderを用いる比較実験は、この変更には含みません。

- 入力系列を `[B,1,L]` に変換し、位置座標（−1〜1）を追加。
- 1×1畳み込みで64チャネルへ変換。
- 4個の残差ブロック。それぞれkernel=3、dilation=1,2,4,8の畳み込みを2回使用。
- 各ブロックへzと拡散ステップの埋込みをスケール・シフトとして渡す。
- GroupNorm・SiLU・1×1畳み込みで `[B,L,1]` のvを出力。

畳み込み経路の受容野は61ステップです。GroupNormは時間軸も含めて正規化します。
パディングで長さを保ち、全体生成用なので因果マスクはありません。
位置情報は、入力が純粋なノイズの場合もzから時刻固有の特徴を再現できるように与えます。
設定項目は `diffusion.denoiser: conv1d`、`conv_channels: 64`、`conv_blocks: 4` です。
畳み込みに変えただけで変動の改善が保証されるわけではなく、再学習後に個々の生成系列の
変化量分布・自己相関・再生成精度を比較してください。

旧MLPの評価・再学習には `--config configs/mlp_baseline.yaml` を指定できます。
旧v2チェックポイントの読込みにも対応しています。v3の重みとは互換性がありません。
比較時はモデル容量・計算量の違いと学習seedの影響も考慮してください。

## 元エピソードと生成系列の折れ線比較

`episodes.png` はランダムに選んだ元エピソード1本と、それに対応する生成系列から重複なしで選んだ8本を、2列×4行で比較します。各パネルの緑線が同じ元系列、青線が個々の生成系列です。選択には評価の `--seed` を使用し、同じデータとseedなら同じ選択になります。生成本数が8本未満なら、利用可能な全本数を表示します。

## 時間的特徴の比較

評価時に `episode_autocorrelation.png`（自己相関）と
`episode_mean_squared_change.png`（時間間隔ごとの平均二乗変化量）を出力します。
元系列と個々の生成系列でそれぞれ指標を計算し、緑・青の線で平均、帯で系列間の10–90%点を表示します。
帯は信頼区間ではありません。生成値の中央値系列は使用しません。
時間差は0からエピソード長の半分（40ステップなら20）までです。
自己相関は各系列の平均を引き、遅れ積和を系列全体の偏差平方和で割ります。
定数系列は自己相関が未定義なので除外し、除外数を図に表示します。
平均二乗変化量は各時間差kについて `(x[i+k]-x[i])**2` の系列内平均です。
重なる窓から切り出した元系列は独立標本ではありません。

## 隣接変化量の補助損失

標準設定は `training.delta_loss_weight: 0.1` です。0で従来のv予測損失のみになります。
学習中のノイズ付き系列とv予測から `x_hat = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v_hat` を計算し、
`delta_mse = mean((diff(x_hat)-diff(target))**2)` を加えます。差分は各系列内の時間方向です。
標準化した値で計算するため、損失の単位はdB²ではありません。1ステップの系列では差分損失を0とします。
生成を100回繰り返す追加処理はなく、復元値から直接計算します。
学習・検証のログには補助損失も記録し、モデル選択と早期終了には合計損失を使います。

補助損失ありの標準設定は `direct_z_conv_delta_episode_v4` に保存します。
既存v3の評価には `python -m src.evaluation.evaluate --config configs/conv_baseline.yaml` を使えます。
モデル構造は同じなのでチェックポイント形式識別子はv3のままです。
旧チェックポイントに重み設定がない場合は0として読み込みます。
補助損失の効果を見るには再学習が必要です。0.1は比較実験の初期値で、改善を保証するものではありません。

## 学習中の生成評価の軽量化

`training.validation_generation_max_episodes: 256` により、CRPSなどの生成評価だけを
固定した検証エピソード最大256本に限定します。検証seedを使って学習開始時に重複なしで選び、
毎エポック同じ対象を使います。対象が256本以下なら全本数を使います。
検証損失は全検証エピソードで計算し、モデル選択・早期終了の基準を維持します。
生成本数（32本）、評価頻度（毎エポック）、学習終了後の評価コマンドの対象範囲は変更しません。
設定省略時も上限256本です。正の整数を指定してください。
ログの `validation_loss_episodes` と `validation_generation_episodes` で対象本数を確認できます。
選んだデータセット内の窓インデックスはチェックポイントの `validation_generation_indices` に保存します。

## 二階差分の分布

`episode_second_delta_distribution.png` は、各系列内で
`x[t+2] - 2*x[t+1] + x[t]` を計算し、元系列（緑）と個々の生成系列（青）の分布を比較します。
全評価エピソード・全生成サンプルを使い、系列境界をまたぎません。中央値系列は使いません。
共通のビンで確率密度に正規化し、外れ値を含めた全範囲を表示します。
横軸の単位はdBで、時間間隔で割った二階微分ではありません。
絶対値が大きいほど傾きの切り替わりが大きいことを示します。3ステップ未満では計算できない旨を表示します。

## ns-3 / tranDataのRSRPを学習用に変換

EpisodicDT-small直下で実行します。入力にはフォルダー名またはフォルダーのパスを指定できます。

```bash
python -m src.data.prepare_ns3 tranData-1790496613670328420
# パスで指定する場合も同じ
python -m src.data.prepare_ns3 NS3_5GLENA_modified/tranData-1790496613670328420
```

上記2コマンドは同じ変換の別表記です。2回実行すると上書き防止のためエラーになります。
既定の出力先は `data/processed/ns3/train/` で、入力run名・端末ID・連続区間番号ごとに
`rsrp` 1列のCSVを作ります。値はdBmのままで、標準化と40ステップの切り出しは既存Datasetが行います。
UEを混ぜず、`global_ue_id` ごとに `time_ms` を昇順に並べます。
空欄/非有限RSRPは除去し、`run_manifest.json` の `samplePeriod` から想定する時間間隔を外れた箇所は別CSVにします。
補間・ゼロ埋め・他端末との連結は行いません。同一UE・同時刻が重複する入力はエラーにします。

```bash
# 保存先・欠測扱いにする測定の古さを指定する例
python -m src.data.prepare_ns3 tranData-別の実行 \
  --output-dir data/processed/ns3 --max-age-ms 300

# 独立した評価用シミュレーションを変換
python -m src.data.prepare_ns3 tranData-評価用の実行 --partition evaluation

# データを用意してから実行
python -m src.train.train --config configs/ns3.yaml
python -m src.evaluation.evaluate --config configs/ns3.yaml
```

既定は測定の古さによる除去を行わず、有限RSRPを保持します。必要なら `--max-age-ms` を指定してください。
manifestがない場合は `--sample-period-ms 200` などを明示します。0.2秒間隔の40点は約8秒相当で、
従来の約0.5秒間隔データと同じ40点でも時間幅は異なります。この変換はリサンプリングしません。

学習用/最終評価用は `--partition train|evaluation` で指定し、変換時には自動分割しません。
学習コードがtrain内の各連続系列を時間順80%/20%に分け、検証用にも使います。
同じrunを両方に登録しないでください。評価用は独立した軌跡・シナリオを用意してください。
NS3用の `configs/ns3.yaml` は既存の学習設定・結果を上書きしない別設定です。

変換内容・除外行数・系列ごとの時刻範囲・作成可能な窓数は `data/processed/ns3/manifests/` のJSONに記録します。
`--episode-length`（既定40）と `--validation-fraction`（既定0.2）はこの窓数確認に使い、学習設定に合わせます。
短い系列も保存しますが、そのデータだけでは学習できない場合は警告します。
例の `tranData-1790496613670328420` は5端末×9点なので40ステップの窓は0本です。
学習・検証の両方に十分な連続観測を確保するため、まず50〜60秒以上の小規模シミュレーションで確認してください。

## 基地局配置・移動経路を変えた一括データ生成

`src/data/generate_ns3_batch.py` は、配置と移動経路のCSVを作成し、`tranData` を順番に実行して、
RSRPを既存の生成モデル用CSVへ変換します。事前に `NS3_5GLENA_modified/utils/setup_trandata.sh`
でシミュレータをビルドしてください。以下はプロジェクト直下で実行します。

```bash
# 設定・入力CSV・学習設定だけを作成して確認
python -m src.data.generate_ns3_batch --prepare-only
# シミュレーションと変換を一括実行
python -m src.data.generate_ns3_batch
# 作成したデータで学習・評価（バッチ生成自体は学習しません）
python -m src.train.train --config data/processed/ns3_batches/diverse_ho_dt05_mobility_v3/training_config.yaml
python -m src.evaluation.evaluate --config data/processed/ns3_batches/diverse_ho_dt05_mobility_v3/training_config.yaml
```

設定ファイルは `configs/ns3_batch.yaml` です。既定では6シナリオ×送信電力3条件
（24/30/36 dBm）×乱数run番号3条件の **54回** を逐次実行します。
各回は150秒、記録間隔0.5秒、車3台・歩行者1台・静止端末1台です。

| シナリオ | 基地局配置 | 車の移動 | 用途 |
|---|---|---|---|
| approach | 横並び2局 | 基地局へ接近 | 学習 |
| depart | 横並び2局 | 基地局から離れる | 学習 |
| cross_cells | 三角形3局 | セル間を横断 | 学習 |
| stop_and_return | 縦並び2局 | 移動・停止・引き返し | 学習 |
| heldout_turn | 位置を変えた3局 | 折れ曲がる経路 | 独立評価 |
| heldout_diagonal | 斜めに配置した2局 | 斜めに横断 | 独立評価 |

`sites` は `[x, y, 高さ]`（m）、`car_route` / `walker_route` は
`[全実行時間に対する割合, x, y]` です。割合0が開始、1が終了で、点間を線形補間します。
同じ位置を異なる時刻に指定すると停止区間になります。車は `car_offsets` の位置差で3台に展開します。
これらは合成の配置・軌跡であり、実道路や車両の加減速制約を再現する設定ではありません。
既定はUMi・3.5 GHz・20 MHz・Layer 2（RSRP測定、UDP負荷なし）、A3 RSRPハンドオーバー有効です。
既定出力先は `data/processed/ns3_batches/diverse_ho_dt05_mobility_v2/` です。Hysteresisは3 dB、TimeToTriggerは256 ms
（ライブラリ既定値）です。有効化だけで切替が起こるとは限らず、`ho_events.csv` と
`ue_kpi.csv` の `serving_cell_id` を確認してください。60秒のdepart確認実行では切替イベントは0件でした。通信負荷も必要な場合は `layer: 3`
へ変更できますが、RSRPのみのモデルで通信品質全体を評価できるわけではありません。

小規模な確認や条件の絞り込みも可能です。

```bash
# 全6シナリオの短時間動作確認。時間を2秒に縮めるため学習には使わない
python -m src.data.generate_ns3_batch --smoke
# 150秒の学習・独立評価データを各1条件ずつ生成
python -m src.data.generate_ns3_batch \
  --scenarios approach heldout_turn --powers 30 --runs 1 \
  --batch-dir data/processed/ns3_batches/small_v1
```

出力先には `inputs/`（配置・経路）、`raw/`（シミュレーション出力）、`logs/`、
`processed/train/`、`processed/evaluation/`、`training_config.yaml` を保存します。
`--prepare-only` を含む準備処理では、実際に使用する入力CSVからシナリオごとの
配置・経路図 `figures/<シナリオ名>.png` も作成します。黒い三角は基地局（高さを併記）、
色付きの線と矢印は車・歩行者の経路と進行方向、丸は開始点、Xは終了点、四角は静止端末です。
停止区間には停止時刻を表示します。座標はm単位の平面図で、重なる軌跡は同じ位置に描画します。
既存バッチでも、他の条件・入力が一致すれば `--prepare-only` で図を再作成できます。
この場合、以前の実行計画のコードハッシュは保存したままです。コード変更後の本実行には
従来どおり新しい `--batch-dir` を指定してください。
`batch_plan.json` は条件と実行コード等のハッシュ、`batch_summary.json` は各実行の
RSRP範囲・有効値率・変換後の窓数・実行時間を記録します。初期欠測と古い測定値
（既定300 ms超）は変換時に除外し、時間の欠落をまたいで窓を作りません。

同じコマンドを再実行すると、完了済みの出力ハッシュを確認してスキップします。
条件・コード・ビルドを変更した場合は新しい `--batch-dir` を指定してください。
失敗して途中出力が残った実行は自動上書きせず停止するため、ログを確認して新しい出力先で再実行します。

同じ配置・経路の電力違い・乱数違いはすべて同じ用途に所属します。
学習用系列内の前方80%を学習、後方20%を検証とし、独立評価には別の配置・経路を使います。
重なる40ステップ窓は独立した実験ではないため、データ量は窓数に加えてシナリオ数・run数でも判断してください。

現在のバッチ設定は0.5秒間隔・150秒です。車の走行速度を約21～36 km/h、歩行者を1.0～1.4 m/sに設定し、
経路と基地局配置を調整しています。区間ごとの速さは `configs/ns3_batch.yaml` の注釈を参照してください。40点の窓の先頭から末尾までは19.5秒です。
ns-3は離散イベントシミュレータであり、この記録間隔の変更は内部の無線処理周期の変更ではありません。

## ns-3学習モデルをSUTDデータで評価

```bash
python -m src.evaluation.evaluate --config configs/ns3_mobility_v2_on_sutd_5g.yaml
```

この評価専用設定は `diverse_ho_dt05_mobility_v2` の学習済みチェックポイントを読み込み、
`data/processed/sutd_5g/evaluation/` の全40点窓を評価します。再学習は行わず、
正規化の平均・標準偏差もns-3の学習時の値を使います。各窓から32本を生成し、
結果は `results/ns3_mobility_v2_on_sutd_5g/` に保存します。
これはSUTDの元エピソードをエンコーダに入力する条件付き再構成の評価です。
未来予測や制御AIの通信品質評価ではありません。SUTDの実測時刻には揺らぎがありますが、
既存の評価処理は再サンプリングせず連続40点を扱います。

評価時のエピソード比較図は、重複なしでランダムに選んだ最大10個の元エピソードについて、
それぞれ最大8本の生成サンプルを表示します。保存名は `episodes.png`、
`episodes_02.png`～`episodes_10.png` です。最初の図は従来と同じ選択方法で、
選んだ元エピソードIDと生成サンプルIDは `episode_examples.json` に記録します。
同じ評価データとseedでは同じ選択になり、元エピソードが10個未満ならその数だけ作成します。

### ハンドオーバー修正とv3データ（2026-09-28）

`tranData.cc` に基地局ごとのA3判定器の生成、RRCとの双方向SAP接続、
接続前の初期化と終了後の破棄を追加しました。使用中の5G-LENA v4.1.1では
`SetHandoverAlgorithmType` の指定だけでは判定器が接続されません。
修正後は `run_manifest.json` の `handover_controller` が `A3_RSRP_explicit_SAP_v1` になります。

現在の `configs/ns3_batch.yaml` の出力先は `diverse_ho_dt05_mobility_v3` です。
v3の54実行中45実行で計216回の開始・完了イベントを確認し、141本のUE系列で接続セルが変化しました。
v2の既存データは自動ハンドオーバーが動いていない修正前データとして区別してください。

今回の動作確認範囲は **Layer 2（RSRP測定）** です。
`Layer=3` とハンドオーバーを組み合わせると、切替付近でNRのUL CQI処理
`NrMacSchedulerNs3::DoSchedUlCqiInfoReq` 内の異常終了を確認しています。
通信負荷を伴うハンドオーバーの利用には、この別問題の修正が必要です。

```bash
cd NS3_5GLENA_modified
/usr/bin/python3 ns3 build tranData
/usr/bin/python3 utils/test_trandata_handover.py
```

`Completed output missing/changed` は、完了記録と出力ファイルが一致しない場合の保護処理です。
今回v3で欠けていた変換CSVとmanifestは、元のrawデータから復元し、元のハッシュと一致することを確認しました。
