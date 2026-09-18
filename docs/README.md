# プログラム説明資料

- `direct_z_program_guide.tex`：現在の `direct_z_episode_v2` の説明資料
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
