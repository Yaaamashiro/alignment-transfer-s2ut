# Alignment and Artifacts

## CTC

`align_speech_text`は固定revisionの`AutoModelForCTC`のlog posteriorから、blankと繰返しlabelを考慮したViterbi pathを計算する。posteriorの値をdurationとして扱わない。spanは元のcodepoint位置へ戻して`[start_sec,end_sec]`で保存する。内部DPはemission frame間でのみ進み、soft attentionから架空の時刻を作らない。

英語モデルの大文字語彙への対応は設定の`text_transform: upper`で明示する。これはCTC語彙lookupだけに使い、T2TTのテキストは変えない。変換後に複数codepointへ広がる文字を単一labelへ無理に対応させない。`unknown_policy: mask`ではOOVとpunctuationのspanを`[-1,-1]`とし、元のindexをmetadataに記録する。`error`なら停止する。全labelが無効なsampleや、繰返しを含めてframe数が不足するsampleは失敗として報告する。

既定モデルは[Japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese/tree/cf031e020336460d15a417eba710bbc5bb43be9a)、[English](https://huggingface.co/facebook/wav2vec2-base-960h/tree/22aad52d435eb6dbaf354bdad9b0da84ce7d6156)。2026-10-20に公式model APIでrevisionを確認した。alignment品質の妥当性は別途実音声で評価する必要がある。

## HuBERT units

公式HuBERT Base固定revisionのhidden_states[6]に、checksum固定の公式100中心を適用する。waveformは16 kHz monoへ変換し、processorの正規化を用いる。unitの値は0–99。

各sampleにoriginal units、reduced units、半開区間のrun `[start,end)`、original frame spans、reduced run spansを保存する。HuBERTのconv_kernel/strideから受容野とhopを計算し、各frame中心のVoronoi幅（1 hop）の時刻区間を使う。unit数を単に音声長へ均等分割してrun timingを推測しない。

## Projection

source token spanと`F`個の音声時間binのoverlapをrow normalizeして`M_src[S,F]`を作る。**Fは実際のforwardから取得する。** source保存artifactにencoder indexを焼き込まない。target reduced run spanとEnglish token spanのoverlapをrow normalizeして`M_tgt[U,T]`を作る。

既定の`--source-mapping frontend`は実際のFBANK frame数とConvのkernel/stride/paddingから、各encoder位置の音響frontend supportを計算する。25 ms窓・10 ms shiftのFBANK、kernel 5/5・stride 2/2では、位置`j`の中心frameは`4j`、supportは元のFBANKの`4j-6`〜`4j+6`。有効なframe indexに切り詰め、各FBANK窓の開始・終了時刻を使う。畳み込みpaddingを観測音声として扱わず、出力長の予測とforwardの実測値が違えば停止する。これはfrontendの局所受容野であり、Transformer self-attention後の情報範囲を制限するものではない。

`--source-mapping uniform`は従来の音声長/Fによる等分割。smoke backendではcentered STFTとstride 4のtest frontendに対応した別の時刻式を使う。各選択はcheckpointの再開契約・実験比較に含まれる。

`A_T = M_tgt @ A_MT @ M_src`を有効rowで再正規化する。source/targetのinvalid span、未知文字を除く。paddingへepsilon質量を加えない。lossではlogに入れる値のみepsilonで安定化する。無効unit rowとEOSではguidance biasを0とする。

Guidance用のuniform混合は投影後に行う。CTC blankの扱いや元のtoken spanを変更せず、raw priorと平滑化後priorを診断で別々に観測できる。

teacher生成CLIは時刻spanとMT attentionを持つ**projection packet**を保存する。denseなU×F行列はforwardと可視化時に生成する。これはencoder長を前処理で固定しないための設計である。

## Integrity

pair_idのSHA256をartifact filenameに使い、任意IDによるpath traversalを防ぐ。NPZはpickleなしで読み、metadata・array shape/dtype/contentの内部hashを検証する。外部`.sha256.json`も保存し、存在すれば検証する。外部hash保存直前に中断しても、内部hashで完成済みNPZを再利用できる。

teacher metadataはpair_id、split、manifest、MT checkpoint、MT/S2UT architecture、tokenizer、CTC artifacts、unit config/contentsのhashを含む。異なるsplit、architecture、checkpoint、tokenizer、unitの混入を拒否する。CTC/unitは入力音声のhashも保持する。

## 品質診断

`diagnose_alignment`は指定splitからseed固定で最大500件を抽出する。`pairs.jsonl`、`summary.json`、先頭20件までのraw MT / 投影 / 平滑化 / 単調prior heatmapとNPZを保存する。

| 指標 | 分子 / 分母 |
|---|---|
| 有効source token率 | frontend overlapがありMTから質量を受ける文字 / 元の日本語文字数 |
| 有効target token率 | unit overlapがありMT rowが有効な文字 / 元の英語文字数 |
| 有効unit row率 | 投影後の質量が正のrow / reduced unit数（EOS除外） |
| prior entropy | 有効unit rowのentropy合計 / 有効unit row総数 |
| zero cell率 | 有効row内の0セル数 / 有効rowの全セル数 |

CTC span自体が有効なsource/target文字数も別途保存する。全体の率はsample別の率の単純平均ではなく、分子・分母を合算する。entropyはnatural log。これらはcoverageと不確実性の診断であり、forced alignmentの時刻精度を測る指標ではない。ASR CERをforced alignment精度の代用とせず、実音声とheatmapを対応させて境界を確認する。
