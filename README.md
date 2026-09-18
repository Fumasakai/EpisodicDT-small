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
損失は全体系列のv予測MSEのみで、勾配はzを通してEncoderへ伝わります。
推論時は `model.eval()` を使い、同じ入力に対して同じzを得ます。
学習時にはEncoderのdropoutが有効です。
同じzからでも、拡散モデルのノイズを変えることで複数の系列を生成できます。

## 実行

```bash
conda activate episodicdt
# 未前処理の場合
python -m src.data.prepare_sutd_5g
python -m src.train.train --config configs/config.yaml
python -m src.evaluation.evaluate --config configs/config.yaml --samples 32 --seed 12345
```

設定は `configs/config.yaml`。`data.episode_length: 40` は全体のステップ数です。
評価と保存済みzからの生成は、CUDAが利用可能なら自動でGPUを使用し、利用できなければCPUで実行します。使用デバイスは起動時に表示します。生成結果とzはバッチごとにCPUへ戻し、CSV・グラフ・潜在変数ファイルの保存形式を維持します。
約2 Hzなら約20秒ですが、実時間は元の計測時刻に依存します。
固定ステップのRSRP系列を扱い、時刻の補間は行いません。

## zだけから再生成

```bash
python -m src.evaluation.generate \
  --checkpoint results/checkpoints/direct_z_conv_episode_v3/episodicdt.pt \
  --latents results/generated/direct_z_conv_generated_episodes.latents.pt \
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
- 検証ノイズを固定し、検証v予測MSEで保存モデルとEarly Stoppingを決めます。
  ログの `loss` と `diffusion_mse` は同じ値です。
- 評価は再生成CRPS・MAE・90%区間被覆率・区間幅と、固定z内の生成標準偏差です。
  元系列全体をEncoderへ入力する条件付き再生成であり、未来予測精度ではありません。
- 危険例生成率や危険閾値表示は含みません。標準学習は通常・危険の全窓を対象とします。

チェックポイントは `results/checkpoints/direct_z_conv_episode_v3/`、図は
`results/figures/direct_z_conv_episode_v3/` に保存します。
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
