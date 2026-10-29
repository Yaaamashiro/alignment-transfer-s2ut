# SentencePiece T2TT

CharacterとSentencePieceの違いだけを調べるT2TT実験。旧characterのcheckpoint・teacher・評価結果・保存stageは残す。既存characterモデルをSentencePieceへ変換・warm startしない。

## Colabでの手順

1. [benchmark比較Notebook](../notebooks/colab/benchmark_comparison.ipynb)の設定で`T2TT_TOKENIZER='sentencepiece'`を選ぶ。`RUN_T2TT_SENTENCEPIECE=True`で学習、Falseで完了済みモデルの評価だけを行う。
2. 保存先のセルも再実行し、環境検査・同じmanifest/configの準備まで上から実行する。
3. 6.1a-Sで`ROOT/t2tt-spm-200k`を0から200kまで学習する。characterの20k学習・200k延長はスキップする。fresh SPMの学習にも、固定benchmarkの常駐session・ローカルfreeze・検証付きDrive ZIPを使用する。
4. 6.1bでdev、6.1cで保存済み固定300件、6.1dで曲線と保存済みcharacter/SPMの得点を確認する。固定8Cのサンプルを選び直さない。
5. BLEU・翻訳例・反復診断を確認後、`RUN_TEACHER_ARTIFACTS=True`で6.2以降を実行する。SPM Attentionとteacherを新しく作り、旧character teacherを流用しない。

`RUN_T2TT_EXTENSION`・`USE_EXTENDED_T2TT`はcharacterの設定。SentencePieceを選ぶと、SPM用の新規200k stageと完了確認を使う。CLIのtokenizer既定とNotebookの選択既定は、後方互換のためcharacterのまま。

| 用途 | 独立したstage / ローカル名 |
|---|---|
| T2TT学習 | `t2tt-spm-200k` |
| dev | `t2tt-dev-spm-200k` / `text-evaluation/t2tt-spm-200k/dev-text.json` |
| 固定300件 | `t2tt-flores-fleurs-spm-200k` / `text-evaluation/t2tt-spm-200k/flores-fleurs/` |
| Attention | `t2tt-attention-spm-200k` |
| teacher | `teacher-alignment-spm-200k` |
| audit | `diagnostics-spm-200k` / `audit-spm-200k` |
| Annealed / Distillation / Guidance | 各run名に`spm-200k`を含む |

全層modeのAttention・teacher・audit・S2UTは、さらに`-all-layers`を付ける。CTCは既存`speech-text-alignment-200k`、unitは既存保存先を使い、入力・設定hashが同じ場合だけ再利用する。stageは`PERSISTENT/stages/`以下、ローカルは`ROOT/`以下。characterの`t2tt`・`t2tt-200k`を指定すると混入/上書きを拒否する。

## Tokenizerの仕様

- SentencePiece **0.2.0**、Unigram、source/target別model、各**8000目標**。`hard_vocab_limit=False`なので実語彙数は小さくなる場合がある。coverage=1.0で必要な文字数が上限を超えた場合は、データを削除したりcoverageを下げたりせず停止する。
- `train`だけをmanifestの固定順で入力する。shuffleなし、sentence samplingなし、1 thread。dev/test/固定FLEURS・FLORESは語彙生成へ渡さない。空白以外のtrain全文字をrequired alphabetにも指定し、特殊tokenに似たリテラル文字列の文字も落とさない。
- `normalization_rule_name=identity`に加え、`remove_extra_whitespaces=False`、`add_dummy_prefix=False`。連続・先頭・末尾の空白も保持する。byte fallbackは使わない。
- `<bos>=0, <pad>=1, <eos>=2, <unk>=3`。characterの`<space>`は使わない。tab/LF/CRは固定のuser-defined symbolsとして保持し、train行中の制御文字も削除・変更しない。
- Python immutable protoのsurfaceとoffsetを原文と照合する。offsetがUTF-8 byte単位の実装も明示的にcodepoint境界へ変換する。dummy prefixの0幅tokenや、原文を覆わないspanは受理しない。C++ piece viewを読む間はownerのprotoを保持する。

train/devの全行について元のcodepoint spanを検査する。UNKがない行では`decode(encode(text)) == text`を要求する。**train-only語彙・byte fallbackなしでは、devの未知文字の完全復元は保証できない**。未知文字はUNK spanとして元位置を維持し、未知codepoint数とexact round-trip件数を記録する。未知文字を減らすためにdevから語彙を追加しない。

SentencePieceの空白マーカーそのもの（literal U+2581 `▁`）とNULは、この方針ではlosslessに表現できないため明示エラーとする。削除・置換・文の除外を行わない。実コーパスで該当した場合は、その時点で別の可逆表現方針の検討が必要。

[公式Python API](https://github.com/google/sentencepiece/blob/v0.2.0/python/README.md)と[モデルschema](https://github.com/google/sentencepiece/blob/v0.2.0/src/sentencepiece_model.proto)を参照。protobuf 4.25.3をpreprocess extraへ固定し、モデル内のUnigram/normalizer/special ID等を起動時に検査する。

## 学習条件と保存

Transformerの12 Encoder/6 Decoder、dim 256、FFN 2048、heads 4/8、ReLU、dropout 0.1、pre-norm、sinusoidalを維持する。batch 8・update frequency 4・max tokens 20000・200k、Adam・LR・warmup・clip・label smoothing・validation・best選択を変更しない。語彙に依存するembedding/projectionの行数とtoken系列長は必然的に変わる。backboneを512次元へ増やす等の変更はしない。

per-token lossはcharacter/SPMで単位が異なるため、値だけを性能比較しない。比較の主指標は同じ原文・参照訳・生成設定でのBLEU。`max_chars`という既存CLI名は維持するが、実装上はEOSを含むdecoder token step上限であり、SPMでも既定1000 stepsを使う。

checkpoint/experimentの`tokenizers.source/target`にspecとserialized model protoのbase64を保存する。type・Unigram・identity・requested/actual vocab size・model SHA256・SentencePiece version・normalizer flagsを含み、scientific contractにも含める。外部`.model`が消えてもcheckpointだけから復元できる。

`tokenizers/ja.model`・`ja.vocab`・`en.model`・`en.vocab`も保存し、resume時にcheckpointとbyte一致を検査する。`.vocab`は制御文字を壊さないようJSON-quoted pieceとscoreのTSV。`tokenization.json`にtrain/dev・source/targetごとのmean codepoints、mean tokens、codepoints/token、UNK codepoints、exact round-trip件数を保存する。これらも復旧ZIPへ含める。

SPMの初回は0 updateから。再開は同じSPM stage自身のlast/model/optimizer/RNG/cursorから行い、tokenizerのmodel・語彙要求・spec・contractが変われば拒否する。characterとの切替には新しいstageを要求する。Drive保存は既定10kごと＋50k単位のmilestone＋終了時。最新2＋milestone保持、lossログと`metrics-summary.json`は従来の方式を共用する。

tokenizer metadataがないcheckpointはlegacy CharacterTokenizerとして扱う。旧checkpointのdev/外部評価/抽出は引き続き使える。旧runを学習再開するときは従来どおり保存時のrevisionを使い、既存T2TT extensionはhashが一致する元sourceを隔離する。旧評価JSONは比較のため読み取り、実装hashの異なるコードで再評価する場合は新しい評価保存先を使う。既存cacheの契約を緩めない。

## Attentionと音声時刻

Attentionは`[T_subword,S_subword]`、全層なら`[L,T_subword,S_subword]`。同時にsource/target tokenの`[begin,end)`原文codepoint spanを保存し、tokenizer type・source/target/combined hashを記録する。UNKのAttention row/columnは従来と同様maskする。

CTC forced alignmentは元文字単位のまま。同じsubwordに含まれる有効文字の最初のstart/最後のendへ集約する。空白・句読点等のinvalid文字は無視し、有効文字がなければ`[-1,-1]`。teacherには集約時刻と元CTC時刻・token spanを両方保持する。

projectionは従来の`M_tgt @ A_MT @ M_src`で、最終priorはHuBERT reduced units × speech encoder positions。S2UTのunit語彙・構成・loss・Guidance/Distillation係数は変更しない。対応層転移と全head平均も維持する。native fairseq criterionへの接続も検査する。

legacy character artifactはAttention shapeが原文長と一致した場合だけ1 codepoint/tokenとして受理する。SPMはspanが必須。type/spec/hash/shapeの矛盾、characterとSPMの混合、tokenizerが変わったresumeは拒否する。

## 評価と検証範囲

devは従来のSacreBLEU 13a・case-sensitive・exp smoothing・beam 5・length penalty 1。外部評価は保存済み固定300件・同じnative/benchmark-normalized BLEUを使う。外部JSONには原文codepoint長の0–19/20–39/40–79/80以上のBLEU、空出力・unique出力数・反復4-gram比率も記録する。集計指標の定義・原文・参照訳はtokenizerによって変えない。反復診断は崩壊の証明やcheckpoint選択には使わない。

CPUの小型モデルで実装を検査する。実コーパスでの8000語彙生成・平均系列長・200k学習・実GPUのメモリ・BLEU改善は、Colab実行時の診断/評価で別途測定する。人工データの語彙数・得点を本実験の値として報告しない。

| 変更箇所 | 役割 |
|---|---|
| `tokenization/character.py`, `sentencepiece.py`, `factory.py` | 共通API・identity/offset・checkpoint復元・診断 |
| `tokenization/alignment.py`, `preprocessing.py` | 元文字span検査、subword時刻集約、混入拒否 |
| `training.py`, `cli.py` | tokenizer選択、tokenizer契約、subword Attention抽出 |
| `evaluation/text.py`, `evaluate_t2tt_external.py` | checkpoint factory、codepoint OOV、共通BLEU/追加診断 |
| `spm_runtime.py`, `colab_runtime.py`, `training_metrics.py`, `t2tt_extension.py` | 既存runtimeへの薄いadapter、model/診断の復旧、loss/curve共有 |
| `pyproject.toml`, `environment.py` | 固定依存・import preflight |
| `notebooks/colab/benchmark_comparison.ipynb` | 別stage・新規200k・同条件評価・teacher/S2UTの接続 |
| `diagnose_alignment.py` | subwordにも対応するtext-token軸名 |
| `tests/test_sentencepiece*.py`, `tests/test_notebook_commands.py` | 単体・学習・外部評価・native・復旧・Notebookの検証 |
| `README.md`, `docs/SENTENCEPIECE.md`, `docs/DEVIATIONS.md`, `docs/VALIDATION.md` | 設計・操作・制約・検証記録 |

Python pathは`src/alignment_transfer_s2ut/`からの相対パス。検証結果は[VALIDATION](VALIDATION.md)。
