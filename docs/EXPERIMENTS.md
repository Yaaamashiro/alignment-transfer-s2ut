# Experiments

## 共通レシピ

| 引数 | 既定値 |
|---|---|
| backend | fairseq |
| seed | 1 |
| optimizer | Adam, betas=(0.9,0.98), eps=1e-8 |
| learning-rate | 0.0005 |
| warmup-updates / warmup-init-lr | 10000 / 1e-7 |
| scheduler | linear warmup → inverse square root |
| batch-size | 8 |
| max-tokens | 20000（source/targetのpadded長×batch数の上限） |
| update-frequency | 4 |
| clip-norm | 10 |
| label-smoothing | 0.2、fairseqのV−1規約 |
| max-updates | 2（明示的に本学習の更新数へ変更する） |
| validation-interval | 1 |
| save-interval-updates | 1 |
| patience | 0（early stoppingなし、正数ならdev非改善回数） |
| checkpoint selection | teacher-free dev/main_loss最小 |
| inference | beam 5, length penalty 1, max units 1000, beta 0 |
| recipe | controlled（strongは明示指定） |
| prior-type / prior-smoothing | t2tt / 0.05 |
| source-mapping / monotonic-sigma | frontend / 0.1 |

同じS2UT source/target frontend、units、seed、全recipe引数、最大更新数、早期停止、vocoder、checkpoint選択、推論設定を4条件で共通化する。変更点はmethodと転移係数のみ。trainとdevの対訳/音声path fingerprint、unit config、初期stateのhashを`experiment.json`へ保存する。

## Strong recipe

`--recipe strong`は[strong.yaml](../configs/recipes/strong.yaml)を既定値とする。独立したrun directoryでBaseline / Annealed Monotonic / Annealed T2TTを比較する。controlledのcheckpointからstrongへresumeすることはできない。

| 要素 | strong設定 |
|---|---|
| source character decoder | encoder layer 6、weight 8.0 |
| target character decoder | encoder layer 8、weight 8.0 |
| decoder target CTC | decoder_layer 3、weight 1.6、blank/BOS 0、zero_infinity=false |
| character aux architecture | native Transformer decoder 2層、dim 256、FFN 2048、heads 4、dropout 0.3 |
| character aux loss | label smoothing 0.2、EOSあり |
| CTC vocabulary | train EnglishのみのUnigram、最大1000 pieces、identity normalization |
| features | waveform×32768、utterance CMVN、trainのみSpecAugment |
| SpecAugment | freq mask 1×27、time mask 1×100、time p=1、time warpなし |

目的関数の補助部分は`8*source_char_CE_sum + 8*target_char_CE_sum + 1.6*decoder_CTC_sum`。main CEと加算後、optimizer更新内のmain unit token総数で割る。補助taskごとに文字数で正規化しない。CTCはEOS/PADなしのSP target、入力長はEOSを含むunit target長。繰返しlabelを含めてCTC targetが収まらないsampleは停止し、lossを黙って0にしない。

参照する層indexはupstream fairseqの規約を保持する。encoder statesはlayer出力のみ、decoder `inner_states[0]`はembedding。従って`decoder_ctc_layer: 3`は`inner_states[2]`を参照する。

`--recipe-config PATH`で完全なstrong YAMLを指定できる。tinyモデルではencoder参照層・補助decoder次元も明示的に下げる。time warpは保存済みalignment時刻を変えるため拒否する。学習済みSP model・train-only文字語彙・recipeはcheckpointへ内包し、推論では音声にCMVNを適用して補助decoderを呼ばない。

`max_tokens`はtoken bucket samplerではなく、決定的shuffleで選んだbatchを必要に応じて縮める上限。1 sampleだけで超える場合は停止する。4条件とも同じsample順とbatchを使う。

長時間の本学習では、たとえば全条件で`--validation-interval 1000 --save-interval-updates 1000`へ変更し、dev評価とcheckpoint I/Oの頻度を抑える。最後の更新と各validation時にもcheckpointを保存する。再開は最後に保存された更新から行う。

## 再開

`--resume`で`checkpoint_last.pt`を読み、モデル、Adam、RNG、epoch/permutation内のcursor、dev最良値、非改善回数を復元する。同一CPU条件では途中終了からの再開と連続学習のparameter一致をtestする。CUDAのkernel差や非決定性についてbit一致は保証しない。

architecture、語彙、train/dev pair、backend、optimizer recipe、method/係数、teacher identityが変わるresumeを拒否する。最大更新数の延長は許可する。annealingは独立した`anneal_steps`で固定され、resumeでresetしない。早期停止済みrunは同じpatienceのまま自動で再始動しない。checkpointは信頼済みローカルファイルのみ読み込む。

実装ソースのhashとPython/Torch/NumPy versionも保存し、異なる実装・runtimeでの再開を拒否する。新しいColabランタイムでも同じrepository commitと環境を復元する。`compare_experiments`で4条件の制御変数を検証できる。

`compare_experiments`はbaselineを含む2条件以上を受け付け、同じmethodでもprior-typeが異なる対照実験に対応する。recipe・補助語彙/SP model・初期重み・timing・平滑化・観測layer/headを照合し、同一methodのprior比較ではbeta scheduleなども照合する。T2TT teacher identityはT2TT priorのrun同士で照合する。単調priorはteacher identityを持たない。

実モデルのGPU統合テストは`AT_S2UT_GPU_MANIFEST=/path/to/tiny-real-manifest.jsonl python -m pytest -m gpu -q`で実行する。少数のtrain/dev/testを含む実音声manifest、CUDA、固定CTC/HuBERT/vocoder取得環境が必要。通常テストでは実行もモデル取得もしない。

## Logging

`metrics.jsonl`に`train/main_loss, train/alignment_loss, train/total_loss, dev/main_loss, dev/alignment_loss, alignment/teacher_entropy, alignment/student_entropy, alignment/kl_teacher_student, guidance/beta, distillation/lambda`とarchitectureを保存する。devのmain lossは全条件でpriorを適用せず計算する。dev teacher artifactが存在する場合はKLも診断用に計算し、存在しなければ`dev/alignment_loss=null`とする。checkpoint選択へKL/total lossを使わない。

strongでは重み適用後の補助lossを`train/auxiliary_source_letter, train/auxiliary_target_letter, train/auxiliary_decoder_target_ctc`へ保存する。KLとteacher entropyは更新内の全有効rowで集約し、student entropyは全非padding target tokenで集約する。

## 評価と可視化

Unit error rateと固定ASRによるASR-BLEUをCLIで計算する。ASR-BLEUはsacrebleuのsignatureとASRのconfig hashを保存する。ID集合、checkpoint、vocoder、推論設定の混入を拒否する。testは選択やearly stoppingへ利用しない。人工smoke出力のASR-BLEUは拒否する。

`visualize_attention`は同一sampleの4種類のmatrixを保存する。matrix軸はT2TTではtext token、それ以外ではreduced unit / encoder frame。metadataにpair_id、split、checkpoint、method、step、beta、lambda、原文/訳文、layer/headを含む。
