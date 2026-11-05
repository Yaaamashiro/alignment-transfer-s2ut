# External text-only T2TT

外部で生成済みの日英JSONLを使い、音声を用意する前にT2TT学習・dev BLEU・Attention抽出を行う。JParaCrawlのfiltering、TTS、Whisper QC、S2UT学習、xCOMET/COMETKiwiはこの機能の対象外。

## 入力

```json
{"pair_id":"p0001","split":"train","ja_text":"日本語文","en_text":"English sentence."}
{"pair_id":"p0002","split":"dev","ja_text":"別の文","en_text":"Another sentence."}
```

必要なのは`pair_id / split / ja_text / en_text`の非空文字列。train/devが必要でtestは任意。splitはtrain/dev/testをそのまま使い、再分割しない。重複IDとsplitをまたぐ同じ対訳pairは拒否する。文字列のstrip・Unicode normalization・filteringを行わない。語彙はtrainだけで作り、devでbest checkpointを選択する。

## Manifest API

```python
from alignment_transfer_s2ut.data.manifest import (
    load_manifest, validate, text_manifest_hash, speech_manifest_hash,
)

text_rows = load_manifest("manifest.jsonl", require_audio=False)
speech_rows = load_manifest("manifest-with-audio.jsonl")
# Payloadを開かずに音声metadataだけ検査する既存APIも維持する。
speech_metadata = load_manifest("manifest-with-audio.jsonl", check_audio=False)
```

`TEXT_REQUIRED`は4列、`AUDIO_REQUIRED`は`ja_audio/en_audio`。既存の`REQUIRED`は両者を連結したまま。`require_audio=True`が既定であり、`check_audio=False`だけでは音声列は任意にならない。これは既存S2UTのmetadata検査を弱めないため。

`require_audio=False`では、optionalな音声パスのresolve/stat/hash・音声ファイルの存在検査も行わない。音声path/SHAの重複検査はspeech用途だけで行う。T2TT・dev評価・抽出はこのtext modeを使う。CTC・HuBERT/unit・teacher・S2UT・speech評価・音声を使う可視化は従来のspeech modeで、音声列がなければ明確に停止する。CTC/unitはその検査を外部モデル取得より前に行う。

## Identityと音声追加

`text_manifest_hash(rows)`は、順序を含めて次の4列だけをhashする。

```text
pair_id, split, ja_text, en_text
```

音声path/SHAやその他の列を後から追加・変更しても、このhashは変わらない。ID・原文・split・順序・行集合を変えると変わる。

新しいT2TT checkpoint/experimentは`manifest_identity="text-v1"`を契約へ含め、`train_hash/dev_hash`をtext identityとして保存する。characterとSentencePieceの両方に適用する。新しいdev evaluation contractとAttention metadataにもtext identityの種別を記録する。Attentionには全manifestの`text_manifest_hash`も明記する。

将来、同じ順序・pair_id・split・textへ音声を追加したmanifestでteacherを作る場合、T2TT Attentionだけをtext identityで照合する。CTC・unit・生成するteacher自体は音声を含む従来のidentityで照合する。この境界を明示し、S2UTの音声provenanceをtext-onlyへ弱めない。

`manifest_hash`の従来計算は変更しない。`speech_manifest_hash`はその明示的aliasで、音声pathと既存SHAを含む。S2UT checkpoint/CTC/unit/teacherの全面移行は行わない。pilotのT2TT dev照合には、S2UT experimentへ追加した診断用`dev_text_hash`を使い、S2UTのscientific contractやspeech hash自体は維持する。

## 旧checkpointとartifact

identityの種別がないT2TT checkpointはlegacyのaudio込みidentityとして扱う。評価・抽出は元の音声metadataを使って保存hashを照合し、音声payloadは開かない。relative pathはfilesystemを調べずlexicalなabsolute pathへ戻す。旧実験でsymlinkを解決した音声pathを使っていた場合は、保存時と同じcanonical metadataが必要。

旧checkpointへ4列だけのmanifestを渡しても、昔のaudio込みhashの一致は証明できないため停止する。hashを書き換えて新形式へ昇格させない。元manifestと元revisionを保持して旧学習を再開するか、外部text用の別stageを新規作成する。旧形式Attentionは従来どおりspeech identityを要求し、既存artifactの無変更resumeも維持する。

## CLI

直接、音声列のないJSONLを渡せる。

```bash
python -m alignment_transfer_s2ut.train_t2tt \
  --manifest /data/text.jsonl --config configs/t2tt/fisher.yaml \
  --output outputs/t2tt-external-text --device cuda --max-updates 200000

python -m alignment_transfer_s2ut.evaluate_t2tt \
  --manifest /data/text.jsonl --checkpoint outputs/t2tt-external-text/checkpoint_best.pt \
  --split dev --beam-size 5 --length-penalty 1 --device cuda \
  --output outputs/t2tt-external-dev/dev-text.json

python -m alignment_transfer_s2ut.extract_t2tt_attention \
  --manifest /data/text.jsonl --checkpoint outputs/t2tt-external-text/checkpoint_best.pt \
  --all-layers --device cuda --output outputs/t2tt-external-attention
```

tokenizerのCLI既定はcharacter。SentencePieceでは`--tokenizer sentencepiece --source-vocab-size 8000 --target-vocab-size 8000 --sentencepiece-model-type unigram`を追加する。[SentencePiece仕様](SENTENCEPIECE.md)のidentity/UNK/文字span制約は変わらない。SacreBLEU・prediction JSONL・EOS率・生成metadata・teacher forcing/all-layer抽出を維持する。外部固定300件の評価もcheckpoint-owned tokenizerを使い、音声manifestを要求しない。

## Colabと再開

[training.ipynb](../notebooks/colab/training.ipynb) でT2TTを準備・学習・復元する。

1. GPUランタイムでPREPARE_AUDIO_PACKS=False、PREPARE_S2UT=Falseにする。
2. T2TT_SOURCE_MANIFEST_OVERRIDEへ外部text JSONLを指定し、新しいT2TT_RUN_IDでRUN_T2TT=Trueにする。
3. 「2. T2TT」でcanonical textを作成し、学習を開始する。EVALUATE_T2TT=Trueでdev BLEU・翻訳例・曲線を確認する。
4. 音声やS2UTも使用する場合は、続く前処理・Baseline・Teacher・転移学習へ進む。

T2TT単独でも実行できる。学習の保存先はTEXT_ROOT/t2tt-<tokenizer>、DriveはTEXT_PERSISTENT/stages/t2tt-<tokenizer>。TEXT_ROOTとTEXT_PERSISTENTはT2TT_RUN_IDから決まる。切断後は同じrevision・設定で上から再実行し、同じstageのlast/optimizer/RNG/cursorから再開する。

学習済みT2TTを使う場合はRUN_T2TT=Falseにし、EVALUATE_T2TTまたはTeacherに関するスイッチを有効にする。T2TT_RUN_IDの既定はt2tt-001。再利用するモデルの実際のIDと元text manifestを指定する。旧text-only Notebookは [legacy](../notebooks/legacy/jparacrawl_t2tt.ipynb) に保存している。

## 変更ファイル

| ファイル | 変更 |
|---|---|
| `data/manifest.py`, `data/__init__.py` | 用途別validation、text/speech identity、旧identityの明示的読込み |
| `training.py`, `evaluation/text.py` | audio-free入力、text-v1契約、評価・抽出の互換分岐 |
| `preprocessing.py` | 音声必須の早期検査、text Attentionからspeech teacherへの照合境界 |
| `pilot.py`, `t2tt_extension.py` | pilotのdev text照合、新旧checkpoint契約の検証 |
| `text_manifest.py` | 原本read-onlyのcanonical text staging |
| `text_t2tt_runtime.py`, `spm_runtime.py` | 既存fresh/session/ZIP adapterをcharacterとSPMで共用 |
| `notebooks/legacy/jparacrawl_t2tt.ipynb` | 入力・学習・復元・BLEU・例・任意Attention |
| `tests/test_text_manifest.py`, `test_text_t2tt.py`, `test_text_runtime.py`, `test_text_notebook.py` | API・音声非参照・音声追加・teacher接続・復旧・Notebook |
| `tests/test_t2tt_external.py` | 追加診断表の表示順に依存せず、BLEU比較表と翻訳例の内容を検査 |
| `README.md`, `docs/TEXT_ONLY_T2TT.md`, `docs/DEVIATIONS.md`, `docs/VALIDATION.md` | 手順・identity設計・互換性・検証記録 |

Python pathは`src/alignment_transfer_s2ut/`からの相対パス。本番外部コーパス・実GPUの200k学習やBLEUは別途実行し、CPU小型接続試験を品質結果とは扱わない。
