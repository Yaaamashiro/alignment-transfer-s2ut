# 学習と比較

主な入口は [training.ipynb](../notebooks/colab/training.ipynb) と [evaluate.ipynb](../notebooks/colab/evaluate.ipynb)。

## 入力と実験ID

既定IDはEXPERIMENT_ID=experiment-001、T2TT_RUN_ID=t2tt-001。IDは保存先を識別する名前で、コーパスの種類や件数を指定しない。

DATA_CONFIGにデータ設定YAMLを選ぶ。既定の [colab.yaml](../configs/data/colab.yaml) のcorpus_rootとaccepted_manifest、またはNotebookのCORPUS_ROOT_OVERRIDEとSOURCE_MANIFEST_OVERRIDEに入力先を設定する。T2TTの元テキストはteacher_source_manifestかT2TT_SOURCE_MANIFEST_OVERRIDEへ指定する。

speech manifestは `pair_id`、`split`、`ja_text`、`en_text`、`ja_audio`、`en_audio`、`ja_sha256`、`en_sha256`、`ja_duration`、`en_duration` を持つJSONL。durationは秒、sha256はWAV全体の小文字16進SHA-256。音声は16 kHz PCM16 WAVを使用する。audioはコーパスルートからの相対パス、またはルート配下の絶対パス。ja_wav_16k / en_wav_16k形式も読み込める。splitはtrain/dev/testを維持し、前処理と学習はtrain/devを使う。corpusとsource_idは任意。qc_statusがある場合はacceptedが必要。

release_summaryは任意。QCリリースの場合はcompleteとsplit別accepted件数を検査する。expected_pairsを指定するとtrain/devの合計件数も検査する。未指定なら入力manifestの件数を使う。

既存のデータを指定する例は [設定例](../configs/data/kokoro_jparacrawl.example.yaml)。保存済みrunの再開では実際の保存先に対応する元のIDとrevisionを指定する。

## training.ipynb

1. EXPERIMENT_IDを設定し、CPUでPREPARE_AUDIO_PACKS=Trueにして音声ZIPを作る。AUDIO_PACK_WORKERSは既定2、整数1〜4でCPU作成のみ並列化する。完成済みZIPは再利用する。
2. GPUへ切り替え、Falseにして同じ設定で上から実行する。小規模FP16の更新・保存・再開を合成音声で検査する。この事前検査は実コーパスでの実行確認を代替しない。
3. T2TTは指定した保存先の学習済みSentencePieceを復元できる。新たに学習する場合は新しいT2TT_RUN_IDでRUN_T2TT=Trueにする。EVALUATE_T2TT=Trueでdev BLEU・翻訳例・曲線を確認する。
4. unit・辞書・fairseqデータを準備し、登録事前検査を確認する。RUN_BASELINE=TrueでBaselineを400,000 updates学習する。
5. RUN_TEACHER_ARTIFACTS=Trueで元T2TT manifestと音声manifestの全ID・split・本文を照合し、全層Attention・CTC alignment・projected Teacherを構築する。
6. top-p診断を見てMASK_TOP_Pを明示設定し、λ/τを校正する。
7. RUN_DISTILLATION / RUN_ANNEALED_DISTILLATION / RUN_MASK_DISTILLATIONで転移条件を学習する。RUN_ALIGNMENT_DIAGNOSTICS=Trueで層別診断と可視化を確認する。

学習済みT2TTの再利用時は、T2TT_RUN_IDと元のtext manifestを指定する。Baselineと転移条件は同じ400k予算・初期化・native fairseqレシピを使う。学習済みモデルはそれぞれの検証付きsnapshotへ保存する。

## evaluate.ipynb

1. trainingと同じEXPERIMENT_ID、corpus設定を指定する。
2. 前処理とBaselineを復元する。選択した転移モデルは完成済みの検証付きsnapshotから復元する。
3. 共通のnative fairseq生成器でdevを推論し、UERと任意のASR-BLEUを比較する。

EVALUATE_S2UT=TrueでS2UTを評価する。COMPARE_DISTILLATION / COMPARE_ANNEALED_DISTILLATION / COMPARE_MASK_DISTILLATIONは既定Falseで、Baselineのみを評価する。学習を完了した転移条件だけTrueにして比較する。validation/inferenceはteacher-free。

EVALUATE_T2TT_EXTERNAL=TrueでT2TTの固定300件外部BLEUも確認できる。T2TTだけを評価するときはEVALUATE_S2UT=Falseにする。

## 容量と保存

音声は一括展開し、WAVの必要容量を指定したmanifestのdurationから計算する。ステージング前の容量表示を確認する。checkpointやTeacherにも作業領域が必要。4条件の学習予算は合計1,600,000 updates。

作業はVM、永続化は固定benchmarkのverified ZIP。T2TTとS2UTの保存先は独立しており、同じrevision・実験名・設定でそれぞれ再開できる。旧runの再開は元revisionを使用する。既存の実験用Notebookは [legacy](../notebooks/legacy) に保存している。

詳細は [Method](METHOD.md)、[検証記録](VALIDATION.md)。
