# 学習中の NS-3 異常終了の調査と修正

> 設定整理後の実行手順は[最新README](../README.md)を参照してください。学習用YAMLは `training_fast.yaml` と `training_diffusion_scenario.yaml` の2つです。以下には過去の設定名を含む調査・実装時の記録があります。

現在は追加のNS-3内部修正を適用済み。既存119更新目の重みを引き継ぐ場合は
[最新の修正と実行手順](ns3_inflight_control_fix.md)を参照してください。

対象: `data/processed/training_dataset_next/mobility/rollouts/update_00004/`。
旧ログ、入力 CSV、チェックポイントは変更せず保存している。

## 停止までの流れ

`train_mobility` は入力エピソードを2件選び、各4候補を NS-3 で評価する。
4更新目の8候補のうち6候補で NS-3 が異常終了し、各入力について有効な候補が1件ずつしか残らなかった。
同じ入力の別候補を基準にする leave-one-out baseline には2件以上が必要なため、
Python が `Too few completed simulations for policy gradients` で停止した。
これは CUDA や `OMP_NUM_THREADS` の問題ではない。

| 候補番号 | 旧実行の結果 |
| --- | --- |
| 0, 1, 3, 7 | SIGABRT、`Unknown IP type` |
| 4 | SIGABRT、`Cannot receive DATA while receiving CTRL.` |
| 6 | SIGSEGV、gdbで `NrMacSchedulerNs3::DoSchedDlTriggerReq` を確認 |
| 2, 5 | 正常終了 |

## シミュレータの変更

パッチは `NS3_5GLENA_modified/utils/patches/` に保存した。
`utils/setup_trandata.sh` は従来のパッチと併せて適用する。

### 0003: 解放済み UE の HARQ フィードバック

ハンドオーバーで旧基地局から UE の管理情報が削除された後にも、再送制御の
フィードバックが到着する。DL/UL スケジューラは `find()` が見つからなかった場合にも
イテレータを参照していた。UE と HARQ プロセスの存在を確認してから参照するように変更した。
gNB MAC の ACK 処理も、既に解放された UE・範囲外のプロセスにアクセスしない。
有効なフィードバックやネットワーク上の HOF イベントは除外しない。

### 0004: ハンドオーバー時の PHY 予約の無効化

UE PHY の `DoReset()` は接続フラグ等しか初期化せず、旧基地局の送信予約、パケット、
制御メッセージを残していた。基地局が切り替わる際にこれらを消去する。
予約済みコールバックには接続世代を付け、旧接続の送信、MACへの受け渡し、HARQ応答を実行しない。
スロットの時計と周期的な制御区間は維持する。

パケットの詳細追跡では、2.279秒に送られた RLC SN=22 の断片が4.125秒に届き、
通し番号が一周した後の別パケットの断片と結合されていた。
結果として IPv4 ヘッダーの途中2バイトが欠け、`Unknown IP type` に至ることを確認した。
番号処理の修正だけではこのケースは解消せず、PHY の予約無効化を含めた構成で再検証した。
この観測だけで、すべての遅延パケットの経路を特定できたという意味ではない。

### 0005: RLC の循環通し番号

追加で、10ビットの通し番号の比較に負の剰余が混ざる問題と、1023→0の更新に
通常の整数を使う問題を修正した。再構成区間の走査も循環番号として扱う。
`utils/test_nr_rlc_wrap.cc` は番号比較、および欠落・順序逆転・番号の周回を含む
4,096断片の受信を検証する。古い断片を新しい完全パケットとして受け入れる対処はしていない。

## 学習側の回復処理

- `max_rollout_attempts`（既定3）: 同じ入力で有効候補が2件以上揃わない場合、候補を追加生成する。
  元の成功候補は保持し、異なる入力の候補を baseline として混ぜない。
- それでも不足した更新は、勾配更新を行わず `status: skipped` として記録する。
- `max_consecutive_invalid_updates`（既定5）: 更新不能が連続したら、`latest.pt` と履歴を保存して停止する。
- 異常終了・タイムアウト・イベント判定不能は、+1にも−1にも変換しない。
- `--resume`: 同じデータ・設定・NS-3ビルドで保存済みモデル、optimizer、乱数状態を復元する。
  旧形式チェックポイントに乱数状態がない場合は警告し、新しいサンプリング列を使う。
  シミュレータのバイナリ／ライブラリが変わった場合は再開を拒否する。
- 再開時の実行ログはセッション別ディレクトリに保存し、旧ログを上書きしない。

## 今回の学習の開始方法

NS-3 は今回の修正でビルドが変わったため、旧 `mobility/latest.pt` をそのまま再開しない。
別出力先の設定を作成した。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_fixed.yaml
```

出力先は `data/processed/training_dataset_next/mobility_fixed/`、
対応する拡散モデルの出力先は `diffusion_fixed/`。
200更新の学習自体はこの修正作業では実行していない。
この新しい実行が後で中断した場合、同じビルドのまま次で再開できる。

```bash
OMP_NUM_THREADS=1 python -m src.train.train_mobility \
  --config data/processed/training_dataset_next/training_fixed.yaml --resume
```

既存の入力エピソードは旧ビルドで採取したデータである。上記はそのデータを入力に使い、
生成候補を修正後ビルドで評価する実行になる。修正前後でイベント結果が変わるため、
最終的な研究比較では修正後ビルドで訓練・検証・評価データを再生成して条件を統一する。
旧データのハッシュを書き換えて、新ビルドの実行結果として扱ってはいけない。

## 検証資料

同一の経路 CSV、seed=42、run=1を使い、停止した更新の全8候補を再実行した。
`--packetChecking=1` を付けてパケット整合性検査も有効にした。
詳細は `data/processed/ns3_training_regression_fixed/summary.json`、
各候補の `command.json`、`run.log`、`raw/`、実行ビルドは `build.json` に保存する。
通常の学習では検査フラグは無効で、診断用の大量出力は行わない。

検証結果: 8/8候補が正常終了し、全候補でピンポンイベントを検出した。
Pythonテスト41件、既存のハンドオーバーテスト4条件、RLCテスト（1,992個の完全SDU）も通過した。
今回の8候補では HOF は検出されておらず、HOF 発生率の検証とは区別する。

回帰テスト:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests
/usr/bin/python3 NS3_5GLENA_modified/utils/test_trandata_handover.py
/usr/bin/g++ -std=c++20 -I NS3_5GLENA_modified/build/include \
  NS3_5GLENA_modified/utils/test_nr_rlc_wrap.cc \
  -L NS3_5GLENA_modified/build/lib \
  -Wl,-rpath,"$PWD/NS3_5GLENA_modified/build/lib" \
  -lns3.46-nr -lns3.46-network -lns3.46-core -o /tmp/test_nr_rlc_wrap
/tmp/test_nr_rlc_wrap
```
