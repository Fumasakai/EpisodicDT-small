# EpisodicDT-small

危険エピソードを含むRSRP系列全体を確率的な潜在表現へ変換し、
その潜在変数と拡散ノイズから新しいエピソード全体を生成する研究用モデルです。
現在の標準エントリポイントは `latent_episode_v1` です。

```text
元エピソード [B,L,1] → Encoder → μ, σ → z [B,D]
                                            ↓
                            拡散モデル (x_t, t, z)
                                            ↓
                             新しい全体系列 [B,S,L,1]
```

生成時にhistoryや元系列の最終値を渡す必要はありません。
学習対象は学習区間の統計量で標準化した絶対RSRPで、損失は
`v予測MSE + kl_beta × KL(q(z|episode) || N(0,I))` です。

## 実行

```bash
conda activate episodicdt
# 未前処理の場合
python -m src.data.prepare_sutd_5g
# 新形式で学習
python -m src.train.train --config configs/config.yaml
# 全評価エピソードをencodeして、それぞれから複数生成
python -m src.evaluation.evaluate --config configs/config.yaml --samples 32 --seed 12345
```

設定は `configs/config.yaml`。`data.episode_length: 40` はfutureを含む全体の長さです。
約2 Hzのデータなら約20秒に相当しますが、実時間は元データの時刻に依存します。
本モデルは固定ステップ系列として扱い、時刻や欠測を補間しません。
`training.kl_beta: 0.001` は初期設定であり、最適値を示すものではありません。

## 潜在変数だけから再生成

評価処理は生成CSVと同じ場所に `latent_generated_episodes.latents.pt` を保存します。
元のCSVを読まずに、同じzから新しいノイズで生成できます。

```bash
python -m src.evaluation.generate \
  --checkpoint results/checkpoints/latent_episode_v1/episodicdt.pt \
  --latents results/generated/latent_generated_episodes.latents.pt \
  --output results/generated/from_saved_z.csv \
  --samples 16 --seed 42
```

`--resample-z` を加えると、保存されたμ・σからzを再サンプリングします。
どちらの場合も、一つのzにつき `--samples` 個の系列を生成します。
異なるチェックポイントでの潜在変数の誤使用はSHA-256照合で拒否します。

Python API:

```python
model.eval()
posterior = model.encode(normalized_episode)  # torch.distributions.Normal
z = posterior.sample()                       # [B, latent_dim]
new_episodes = model.generate(z, num_samples=16)  # [B,16,L,1]
rsrp_dbm = new_episodes * training_std + training_mean
```

## 学習・評価の扱い

- 元CSVごとに窓を作成し、ファイル境界をまたぎません。
- 学習用各CSVを時間順に学習・検証へ分割し、境界をまたぐ窓を除外します。
- 平均・標準偏差は検証を除いた学習区間だけから計算します。
- 検証のノイズを固定し、**検証の合計損失**でEarly Stopping・モデル選択します。
- `training_metrics.json` に学習・検証の合計損失、拡散MSE、KLと再生成指標を保存します。
- CRPSやMAEは元エピソードをEncoderに与えた**条件付き再生成**の指標です。
  独立した未来予測や現実性を評価する指標ではありません。

新しいモデル・ログは `results/checkpoints/latent_episode_v1/`、図は
`results/figures/latent_episode_v1/` に保存します。
生成CSVの列は `episode_id, sample_id, episode_step, rsrp_generated_dbm`。
元系列CSV、潜在変数、再生成CRPS・MAE・90%区間被覆率・多様性のJSONも出力します。
潜在変数ファイルには元CSVと窓開始位置も記録します。

現状は通常・危険の全窓で学習します。危険だけへの自動絞り込みや危険性を強制する
追加損失はありません。現段階の評価目的は、潜在変数から元データと似た系列を
生成できるかの確認です。危険例生成率の計算や危険閾値の表示は行いません。
生成例の物理的妥当性や潜在変数の物理的意味は、まだ保証していません。

既存の `EpisodicDiffusion` と `RSRPWindowDataset` は旧方式の比較用として残っています。
旧形式 `zero_snr_v_residual_v1` の重みは新しい学習・評価パイプラインに非互換で、再学習が必要です。
旧設定の `sequence_length` と `future_length` は新設定では `episode_length` に置き換わります。

## テスト

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/episodicdt-mpl python -m unittest discover -s tests -v
```

KLの数式、Encoderへの再パラメータ化勾配、zのみの生成、乱数の再現性、
学習・保存・評価・元CSVなしの再生成、旧方式の回帰テストを確認します。

同じ環境条件から得られた別エピソードを対象に学習したい場合、モデルAPIは
`model.loss_terms(source, beta=..., target=paired_episode)` に対応しています。
通常のCSVには条件ペア情報がないため、標準の学習処理は自己再生成です。
条件推定の補助損失、シミュレータの環境条件生成、制御AIとの接続は今後の拡張です。
