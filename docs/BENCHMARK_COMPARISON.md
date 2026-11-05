# 学習と比較

主な入口は [training.ipynb](../notebooks/colab/training.ipynb) と [evaluate.ipynb](../notebooks/colab/evaluate.ipynb)。

## training.ipynb

1. EXPERIMENT_IDを設定し、CPUでPREPARE_AUDIO_PACKS=TrueにしてQC合格の音声ZIPを作る。
2. GPUへ切り替え、Falseにして同じ設定で上から実行する。小規模FP16の更新・保存・再開を検査する。
3. T2TTは既定の学習済みSentencePieceを復元できる。新たに学習する場合は新しいT2TT_RUN_IDでRUN_T2TT=Trueにする。EVALUATE_T2TT=Trueでdev BLEU・翻訳例・曲線を確認する。
4. unit・辞書・fairseqデータを準備し、登録事前検査を確認する。RUN_BASELINE=TrueでBaselineを400,000 updates学習する。
5. RUN_TEACHER_ARTIFACTS=Trueで元T2TT manifestとKokoroの全ID・split・本文を照合し、全層Attention・CTC alignment・projected Teacherを構築する。
6. top-p診断を見てMASK_TOP_Pを明示設定し、λ/τを校正する。
7. RUN_DISTILLATION / RUN_ANNEALED_DISTILLATION / RUN_MASK_DISTILLATIONで転移条件を学習する。RUN_ALIGNMENT_DIAGNOSTICS=Trueで層別診断と可視化を確認する。

T2TTの既定保存先はJParaCrawl 300,000対で200,000更新を完了したSentencePiece run。Baselineと転移条件は同じ400k予算・初期化・native fairseqレシピを使う。学習済みモデルはそれぞれの検証付きsnapshotへ保存する。

## evaluate.ipynb

1. trainingと同じEXPERIMENT_ID、corpus設定を指定する。
2. 前処理とBaselineを復元する。選択した転移モデルは完成済みの検証付きsnapshotから復元する。
3. 共通のnative fairseq生成器でdevを推論し、UERと任意のASR-BLEUを比較する。

EVALUATE_S2UT=TrueでS2UTを評価する。COMPARE_DISTILLATION / COMPARE_ANNEALED_DISTILLATION / COMPARE_MASK_DISTILLATIONで条件を選ぶ。すべてFalseならBaselineのみを評価できる。validation/inferenceはteacher-free。

EVALUATE_T2TT_EXTERNAL=TrueでT2TTの固定300件外部BLEUも確認できる。T2TTだけを評価するときはEVALUATE_S2UT=Falseにする。

## 容量と保存

Kokoroのtrain/devは216,312 / 1,453対で、testは空。音声は一括展開し、WAVだけで約138.5 GiBを見込む。ステージング前の容量表示を確認する。checkpointやTeacherにも作業領域が必要。4条件の学習予算は合計1,600,000 updates。

作業はVM、永続化は固定benchmarkのverified ZIP。T2TTとS2UTの保存先は独立しており、同じrevision・実験名・設定でそれぞれ再開できる。旧runの再開は元revisionを使用する。既存の実験用Notebookは [legacy](../notebooks/legacy) に保存している。

詳細は [Method](METHOD.md)、[検証記録](VALIDATION.md)。
