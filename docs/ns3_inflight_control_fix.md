# ハンドオーバー中に受信していた制御情報の修正

> 設定整理後の実行手順は[最新README](../README.md)を参照してください。学習用YAMLは `training_fast.yaml` と `training_diffusion_scenario.yaml` の2つです。以下には過去の設定名を含む調査・実装時の記録があります。

## 原因

120更新目の検証で発生した `Cannot receive DATA while receiving CTRL.` を修正した。
同じ入力・乱数条件では1並列でも再現し、並列数が原因ではなかった。

UE node10は3.856000000秒に旧セル1の制御フレームを受信開始し、
3.856002271秒にハンドオーバーに伴うリセットを行った。
受信完了は3.856035713秒であり、この時には新セル3に接続していた。
旧セルのSRS送信指示のRNTI=30が新セルで割り当てられたRNTIと一致したため、
UEは新セルが予約していないSRSを送信した。
3.857428569秒、新セルが別UEのデータを受信する際にSRS受信と衝突し、停止した。
RNTIはセルごとの識別子なので、番号の一致だけでは古い指示を判別できない。

## 変更

- `NrSpectrumPhy`に接続世代番号と制御受信開始時の世代番号を追加した。
- `NrUePhy::DoReset()`で世代を進め、リセット前から受信中の制御情報を無効にする。
- 受信完了時に世代が一致する制御情報だけをUE PHYへ渡す。
- 物理受信の終了処理、干渉計算、状態遷移は通常どおり完了させる。
- 同じセルIDへ再接続した場合も古い制御情報は無効になる。

異常検知を削除したり、失敗を正常扱いしたりする変更ではない。
修正は `NS3_5GLENA_modified/utils/patches/0006-handover-invalidate-inflight-control.patch`
に保存した。`setup_trandata.sh`は、前のパッチと変更行が重なる場合でも
全パッチ適用済みを検出できるよう更新した。現在の実行ファイルは再ビルド済み。

## 検証

- 問題が発生した検証8候補×run 101/102の16実行が全て正常終了。
  `--packetChecking=1`を有効化した。
  結果: `data/processed/validation120_verified_fixed/summary.json`。
- 過去の訓練失敗6候補と正常2候補の計8実行が全て正常終了。
  結果: `data/processed/ns3_inflight_control_regression/summary.json`。
- 既存のHO統合テスト4条件が通過。HO有効時の切り替え完了と切り替え後の通信を確認。
- 新規C++テストで通常受信、新セル受信、受信中リセット、同じセルIDへのリセットを確認。
  ソース: `NS3_5GLENA_modified/utils/test_nr_handover_control.cc`。
- Python単体テスト48件が通過。
- パッチ適用を未適用・0005まで適用・0006まで適用の各状態で確認。

以上は確認した条件での結果であり、全ての生成経路に対する無障害の保証ではない。

## 119更新目の重みを引き継ぐ

NS-3ビルドが変わったため、旧実行の `--resume` はビルド互換性検査で拒否される。
新しい `--initialize-from` はモデル・データの互換性を確認し、重みだけを新規実行へ引き継ぐ。
optimizerと最良検証スコアは引き継がず、修正後のNS-3で初期検証をやり直す。
引き継ぎ元・SHA256・元の更新数・ビルドは `initialization.json` とチェックポイントに記録する。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml \
  --initialize-from data/processed/training_dataset_next/mobility_fixed/latest.pt
```

設定は8並列・81更新（元の200更新から保存済み119更新を除いた数）。
新規実行の表示はupdate=1から始まる。保存先は
`data/processed/training_dataset_next/mobility_inflight_fixed/`。
元の `mobility_fixed/` は保持する。この修正作業では新しい長時間学習は開始していない。

新規実行がチェックポイントを保存した後の再開は次を使う。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_inflight_fixed.yaml --resume
```

報酬定義やモデル構造は変更していない。クラッシュ修正と学習精度の改善は別に評価する。
