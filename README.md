# EpisodicDT-small

元エピソード全体からEncoderが出力したベクトルをそのまま潜在変数zとし、
zと拡散ノイズから新しいRSRPエピソード全体を生成します。
標準形式は `direct_z_episode_v2` です。

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
約2 Hzなら約20秒ですが、実時間は元の計測時刻に依存します。
固定ステップのRSRP系列を扱い、時刻の補間は行いません。

## zだけから再生成

```bash
python -m src.evaluation.generate \
  --checkpoint results/checkpoints/direct_z_episode_v2/episodicdt.pt \
  --latents results/generated/direct_z_generated_episodes.latents.pt \
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

チェックポイントは `results/checkpoints/direct_z_episode_v2/`、図は
`results/figures/direct_z_episode_v2/` に保存します。
生成CSVの列は `episode_id, sample_id, episode_step, rsrp_generated_dbm`。
元系列CSV、潜在変数ファイル、評価JSONも保存します。

**旧形式 `latent_episode_v1`、`zero_snr_v_residual_v1` とは非互換で、再学習が必要です。**
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

## 生成中央値のステップ別分布

`episode_median_boxplot_by_step.png` は、各元エピソードについて生成サンプル軸で
中央値を取り、ステップごとに元系列の分布と比較します。
元系列・生成中央値ともに各ステップN個（Nは元エピソード数）の値を使います。
全生成サンプルを集約する既存の `episode_boxplot_by_step.png` も出力します。
中央値比較は生成分布の中心の再現を見るもので、生成例間のばらつきは表示しません。

## 隣接ステップの変化量分布

`episode_delta_distribution.png` は各エピソード内で `x[t+1] - x[t]` を計算して比較します。
左は元系列と個々の生成系列、右は元系列と各元エピソードに対応する生成中央値系列です。
右は「中央値系列の変化量」であり、「変化量の中央値」ではありません。
横軸はdB（時間当たりの変化率ではない）、縦軸は確率密度です。
共通のビン境界と面積1の正規化で、標本数の違いを考慮します。
全値を含み、別エピソードや別生成サンプルの境界をまたぐ差は取りません。
生成側の分布が広いほど急な変化が多いことを示しますが、これだけでは長時間の相関は評価できません。
