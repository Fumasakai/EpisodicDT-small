# プログラム説明資料

- `direct_z_program_guide.tex`：現在の `direct_z_conv_episode_v3` の説明資料
- `direct_z_program_guide.pdf`：コンパイル済みPDF

構成図はTikZで描画しているため、外部画像ファイルは不要です。

## コンパイル

**XeLaTeX** を使用してください（pdfLaTeX / LuaLaTeX用ではありません）。
TeX Liveの `xeCJK`、`fontspec`、`pgf`（TikZ）、`amsmath`、`booktabs`、
`tabularx`、`geometry`、`hyperref` が必要です。
日本語フォントはNoto CJKを優先し、ない場合はTeX Liveの原ノ味フォントを使います。

プロジェクト直下で次を2回実行します。

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=docs docs/direct_z_program_guide.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=docs docs/direct_z_program_guide.tex
```

Overleafの場合は `.tex` をアップロードし、コンパイラを **XeLaTeX** に設定してください。

Tectonicでもコンパイルできます（初回は必要パッケージの取得にネットワーク接続が必要です）。

```bash
tectonic --outdir docs docs/direct_z_program_guide.tex
```

内容は直接出力方式 `z = Encoder(episode)` に対応しています。
旧 `latent_episode_v1` の評価結果を現モデルの結果として掲載していません。

## 最新版（2026-09-23）

7ページの資料を現在のv4実験設定へ更新しました。Transformerによる直接z推定、
Conv1d拡散生成器、v予測と隣接差分の補助損失、固定256本の生成検証、
一階・二階差分、自己相関、平均二乗変化量の評価を説明しています。
モデルの内部形式識別子は構造互換性のためv3のままです。

`episodicdt_overleaf.zip` をOverleafの新規プロジェクトとしてアップロードしてください。
主ファイルは `main.tex`、コンパイラは **XeLaTeX** を指定してください。
外部画像・参考文献ファイル・Python環境は不要です。
ソース単体を使う場合は `direct_z_program_guide.tex` をアップロードして主ファイルに指定できます。
ローカルではTectonic（XeTeX系）でPDF生成を確認済みです。

3ページ目に畳み込み生成器の詳細を追加しました。処理図、時間方向の参照位置、条件付け、残差接続を説明しています。
