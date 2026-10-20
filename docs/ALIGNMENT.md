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

`A_T = M_tgt @ A_MT @ M_src`を有効rowで再正規化する。source/targetのinvalid span、未知文字を除く。paddingへepsilon質量を加えない。lossではlogに入れる値のみepsilonで安定化する。無効unit rowとEOSではguidance biasを0とする。

teacher生成CLIは時刻spanとMT attentionを持つ**projection packet**を保存する。denseなU×F行列はforwardと可視化時に生成する。これはencoder長を前処理で固定しないための設計である。

## Integrity

pair_idのSHA256をartifact filenameに使い、任意IDによるpath traversalを防ぐ。NPZはpickleなしで読み、metadata・array shape/dtype/contentの内部hashを検証する。外部`.sha256.json`も保存し、存在すれば検証する。外部hash保存直前に中断しても、内部hashで完成済みNPZを再利用できる。

teacher metadataはpair_id、split、manifest、MT checkpoint、MT/S2UT architecture、tokenizer、CTC artifacts、unit config/contentsのhashを含む。異なるsplit、architecture、checkpoint、tokenizer、unitの混入を拒否する。CTC/unitは入力音声のhashも保持する。
