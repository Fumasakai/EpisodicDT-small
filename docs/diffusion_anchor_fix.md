# 拡散モデルの検証時のアンカー不足

> 設定整理後の実行手順は[最新README](../README.md)を参照してください。学習用YAMLは `training_fast.yaml` と `training_diffusion_scenario.yaml` の2つです。以下には過去の設定名を含む調査・実装時の記録があります。

この資料は旧UE別条件での調査記録。現在は[シナリオ単位の条件](scenario_diffusion.md)に変更済み。

`No training anchor for requested per-UE event signature` は、検証条件に完全一致する
訓練例がないため発生した。条件には全UEのイベントと欠損マスクが含まれる。
現在の検証対象8件のうち5件には一致する訓練アンカーがなかった。

修正後は一致する条件には従来のアンカーからの生成を使い、一致しない条件には
その条件を維持したノイズからの生成を使う。別の条件への置き換えや、検証潜在状態の
アンカー利用は行わない。未知の条件でのイベント再現性能はNS-3検証で判断する。

混在する条件の順序、アンカーの選択、ノイズ生成、乱数の再現性をテストした。
Pythonテスト55件が通過。今回の保存済みノイズ除去モデルでも実際の検証8条件から
有限値のUE経路（8候補×24UE×27点×3列）を生成できた。
今回の修正検証では長時間のNS-3追加学習は開始していない。

失敗時の `diffusion_inflight_fixed/` には `denoising.pt` だけが保存されていた。
同じコマンドを再実行できるよう、ディレクトリを
`diffusion_inflight_fixed_failed_anchor_<日時>/` に退避した。モデルは削除していない。
旧 `denoising.pt` は元のEncoderやデータのハッシュを含まないため、これを用いた
自動再開機能は追加していない。再実行ではノイズ除去学習からやり直す。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_diffusion \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml \
  --checkpoint data/processed/training_dataset_next/mobility_inflight_fixed/mobility.pt
```

ノイズ除去学習履歴はNS-3検証前に保存するよう修正した。
`sampling_policy.json` にはフォールバック方針と検証集合の不一致件数を保存する。
