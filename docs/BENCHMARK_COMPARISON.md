# Native benchmark comparison

学習済みのbenchmark S2UTをBaselineとして登録し、同じnative fairseq recipeで3種類のloss-only alignment条件を比較する。主Notebookは[benchmark_comparison.ipynb](../notebooks/colab/benchmark_comparison.ipynb)。

## Colabの順序

1. Runtime/repository setupと依存・CUDA preflight。
2. 既存Baseline snapshotを専用VM directoryへ復元し、checkpoint/prepared data/recipe lock/metadataを登録。
3. T2TT best checkpoint・dev text BLEU・翻訳例を確認。延長/SPM学習は明示スイッチを入れた場合だけ実行。
4. T2TT全層AttentionとCTC alignmentを復元/抽出。
5. 既存のprojectionで全層teacherを生成/検査。再利用可能な既存packetは再生成しない。
6. `RUN_MASK_TOPP_DIAGNOSTICS=True`でcoverage表・分布を確認。
7. `MASK_TOP_P`を0.8/0.9/0.95から明示設定し、`RUN_ALIGNMENT_CALIBRATION=True`でλ/τを校正。
8. `RUN_DISTILLATION=True`でKL-based Attention Distillation。
9. `RUN_ANNEALED_DISTILLATION=True`でAnnealed KL-based Attention Distillation。
10. `RUN_MASK_DISTILLATION=True`でMask KL-based Attention Distillation。
11. 共通のteacher-free native生成・dev評価。
12. Baseline→KL-based Attention Distillation→Annealed KL-based Attention Distillation→Attention-level Knowledge Distillationの表、loss/layer curves、独立したdev alignment診断とAttention可視化。

学習スイッチ・校正スイッチは全て既定False。mask pは既定None。校正前に本学習を始めない。

## 揃える条件

登録済みBaselineのnative S2UT本体、3補助task、optimizer/scheduler、FP16、max_tokens/update_frequency、更新予算、dev selectionを維持する。精度fallbackや学習batchの自動変更はしない。別モデルをseedせず、元seedのfresh native initializationで開始する。validation/inferenceはteacher-free。

calibration artifactは同じBaseline登録・teacher signature・prepared data・implementationに固定し、3条件で共有する。違いはmethodだけ。KL-based Attention DistillationとAnnealed KL-based Attention Distillationは同じλKL/τKLで、差はcosine annealingだけ。Maskは独立に校正したλ/τと明示pを使う。

## 保存と互換性

原corpus・Baseline backup・teacher sourceはread-only。頻繁なcheckpoint/ログ/packetはVM、Driveは固定benchmark runtimeのverified ZIP・bounded upload・storage budgetを共用する。新条件は`research-v3-*`の独立stageで、旧Guidance runを上書きしない。

同じrevision・ROOT/PERSISTENT・data・選択したMASK_TOP_Pで上から再実行する。保存済みtop-p/calibration stageはスイッチがFalseでも検証付きで復元する。teacherの既存3 stageも復元できる場合はreview gateで止めずに再利用する。初回の未作成teacherは明示reviewが必要。snapshot以降の更新は再計算される。校正JSONの変更や異なるteacherでのresumeは拒否する。古いrunの再開は元revisionで行う。T2TT artifact形式とprojection自体は変更しない。

詳細は[数式](METHOD.md)、[CLI](USAGE.md)、[検証記録](VALIDATION.md)。

## Microbatch τへの移行

今回の校正は`alignment-calibration-v2`、`tau_calibration_unit=microbatch`。旧logical平均からの校正JSONは受理せず、再校正を要求する。Notebookの派生先は`research-v3-*`へ分離した。旧`research-v2-*`のcheckpoint/校正/ログを上書き・再開しない。Baseline・manifest・units・T2TT全層Attention・projected teacherはそのまま再利用し、新stageでtop-p診断と校正から実行する。既存teacherのZIP復元手順は維持する。
