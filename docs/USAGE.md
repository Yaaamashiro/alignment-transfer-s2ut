# Usage

## ManifestとT2TT

音声なしの外部日英manifestは[text-only T2TT](TEXT_ONLY_T2TT.md)を参照。character/SentencePiece、train-only語彙、dev BLEU、全層Attention抽出を維持する。S2UT/CTC/unit/teacherには音声付きmanifestが必要。

## Native research

初めに`benchmark register`で別学習済みBaselineを登録し、full train/dev manifest・unit・全層teacherを用意する。具体的なBaseline snapshot手順は[Colab手順](BENCHMARK_COMPARISON.md)。

```bash
python -m alignment_transfer_s2ut.alignment_calibration top-p \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --batches 16 --output analysis/top-p/top-p.json
```

coverage診断を見てpを明示選択する。以下の0.9は選択後の例で、既定値ではない。

```bash
python -m alignment_transfer_s2ut.alignment_calibration calibrate \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --batches 16 --diagnostics analysis/top-p/top-p.json --mask-top-p 0.9 \
  --output analysis/calibration/alignment-calibration.json

python -m alignment_transfer_s2ut.benchmark train \
  --baseline baseline.json --manifest manifest.jsonl --units units --teacher teacher \
  --method distillation --all-layers --calibration analysis/calibration/alignment-calibration.json \
  --output runs/distillation --resume
```

methodを`annealed_distillation`/`mask_distillation`に変える場合は出力も別にし、同じ校正JSONを使う。λ/τ/betaをCLIで手入力しない。学習予算・precision等はBaseline登録から取得する。`train_s2ut --trainer fairseq --baseline-record ... --calibration ...`も同じnative経路へ接続する。

## 評価・診断

`benchmark register-transfer`で各best checkpointを登録し、`benchmark infer`→`evaluate`→`benchmark compare`へ通す。生成器にはsource-only入力を渡す。

```bash
python -m alignment_transfer_s2ut.alignment_report diagnose \
  --baseline baseline.json --record registered-distillation.json \
  --manifest manifest.jsonl --units units --teacher teacher \
  --calibration analysis/calibration/alignment-calibration.json --samples 32 \
  --output diagnostics/distillation.json

python -m alignment_transfer_s2ut.alignment_report curves \
  --runs runs/distillation runs/annealed_distillation runs/mask_distillation \
  --output diagnostics/curves
```

層別診断はteacher-forcedの独立分析で、生成結果には使わない。Attention heatmapは`visualize_attention`を使い、native checkpointではteacherを注入しない。

## Historical standalone tools

`--trainer legacy`、smoke backend、旧pilot/pipelineは過去のcontrolled実験・回帰検査用。旧Guidance名が残るのはこの隔離経路だけで、新しい主要比較には使わない。学習済みrunの再開は元source revisionを保持する。研究用の入口はnative benchmark Notebookと上記4条件である。
