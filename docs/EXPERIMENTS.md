# Experiments

主要比較は次の順とする。

| Condition | Alignment supervision | Forward | Coefficients |
|---|---|---|---|
| Baseline | なし | raw Attention | なし |
| Distillation | 層別KL + worst-layer LogSumExp | raw Attention | calibrated λKL / τKL |
| Annealed Distillation | 同じKL・集約 | raw Attention | 同じλKL / τKL、0.8Tまでcosine減衰 |
| Mask Distillation | 明示top-p regionのstudent mass + 同じ集約 | raw Attention | calibrated λmask / τmask |

同じ登録Baselineのfresh initialization policy、data、all-head/all-layer teacher、学習予算、optimizer、scheduler、precision、logical batch、checkpoint selection、生成と評価を使う。学習済みBaseline重みを転移条件へseedしない。学習予算Tは登録済みBaselineから取得し、Notebookから変更しない。

1. 全層teacherを再利用/構築する。
2. 固定train subsetのtop-p診断を表示する。
3. ユーザーがMASK_TOP_Pを明示選択する。
4. 同じ初期化でλ/τを校正し、JSONを3条件で共有する。
5. 専用stageで各条件を学習する。
6. 同じsource-only native生成とdev評価へ通す。
7. 独立したteacher-forced dev診断で層別KL・Mask mass・weight分布・Attentionを調べる。

translation評価はASR-BLEU/UERと既存経路を共用する。TERは既存評価が提供する場合だけ報告する。testは校正・top-p選択・checkpoint selectionに使わない。BLEU改善を接続テストから主張しない。

旧GuidanceとAnnealed Guidanceは主比較から除外した。過去の実験を読むためのstandalone legacy経路と歴史的検証記録は残るが、現在のnative研究条件ではない。[手順](BENCHMARK_COMPARISON.md)・[数式](METHOD.md)を参照。
