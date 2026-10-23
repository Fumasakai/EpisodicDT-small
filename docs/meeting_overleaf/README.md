# ミーティング資料（10ページ）

2026-10-23時点のEpisodicDT-smallの実装に基づく日本語資料です。
本文・図・数式はすべて `main.tex` にあります。外部画像や分割された節ファイルは不要です。

## Overleaf

1. `episodicdt_meeting_overleaf.zip` を「New Project → Upload Project」でアップロード。
2. Main documentを `main.tex` に設定。
3. Compilerを **LaTeX** に設定。
4. Recompileを実行。

`latexmkrc` は `main.tex` と同じ階層に置いてください。
標準のlatex（DVI出力）とdvipdfmxでPDFを生成します。
日本語にはCJKutf8とWadalab Minchoを使い、LuaLaTeX・XeLaTeX・upLaTeXや外部フォント設定は不要です。
TeX Live 2026で10ページのPDF生成を確認しました。Overleafのオンライン環境そのものでは未実行です。

## ローカルでのコンパイル

```bash
cd docs/meeting_overleaf
latexmk -pdfdvi -interaction=nonstopmode -halt-on-error main.tex
```

必要なTeXパッケージ：CJKutf8 / Wadalab、amsmath、amssymb、geometry、booktabs、TikZ、fancyhdr。
生成したPDFは `main.pdf`。配布用コピーは `docs/episodicdt_meeting.pdf`。

## 内容

1. 研究目的と全体構成図
2. シミュレータ条件、入力CSV、イベント定義、データ区分
3. Encoderの入力・構造・潜在状態
4. Decoderの出力、速度・初期位置の制約
5. 同時学習フロー、任意UEの危険イベント報酬、REINFORCE
6. 条件付き潜在拡散のノイズ除去学習
7. NS-3による報酬重み付き追加学習、未知条件への対応
8. 今回の評価結果と、多様性を失った原因の分析
9. 改善案と次の比較実験
10. 現在の実行コマンドと保存先

パラメータは `training_fast.yaml` / `training_diffusion_scenario.yaml` を基準にしています。
性能数値は既存の限定的な検証記録であり、汎化性能の向上や一律の高速化倍率を主張するものではありません。


## 今回の評価結果の出典

- `data/processed/training_dataset_next/evaluation_scenario_001/summary.json` と各候補の `result.json` / `latent.npz`
- `mobility_inflight_fixed/history.json` / `metrics.json` / `mobility.pt`
- `diffusion_scenario/diffusion.pt` / `feedback_history.json`

候補間距離は保存済み潜在状態を対応するDecoderの平均行動で経路へ戻し、
全候補対について同一UE・同一時刻の2次元距離を平均したものです。
成功率19/69は全3乱数条件の評価が成立した23候補についての集計です。
無効7候補は含めていません。原因の各要因の寄与は未確定で、改善案は未実装です。
