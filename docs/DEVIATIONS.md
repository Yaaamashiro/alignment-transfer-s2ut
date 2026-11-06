# Deviations and implementation definitions

1. **Reference training:** 主比較は固定fairseq/benchmarkのnative S2UT・補助loss・optimizer・scheduler・precision・logical batch・学習予算・selectionを維持する。alignment用task/model/criterionはlocal extensionで、reference sourceを変更しない。
2. **Loss-only transfer:** 旧native Guidance/Annealed Guidance・Monotonic control・beta/smoothing設定を主比較CLI/Notebookから除外した。新条件はKL-based Attention Distillation、Annealed KL-based Attention Distillation、Attention-level Knowledge Distillation。standalone legacy経路は歴史的回帰検査用として残し、現在の研究の入口と区別する。
3. **Worst-layer microbatch definition:** LogSumExpは各training microbatchに適用し、native Trainerがsample-sizeでlogical accumulationする。logical batch全graphを保持したLogSumExpとは異なる。この定義を[Method](METHOD.md)へ明記した。
4. **Calibration:** 固定train subsetを同一fresh initializationで2 pass再生してτとgradient normを測る。学習updateは0。KL/Mask別、λの目標比0.5・worst/median比3・16 logical batches。τ候補はlogical平均を作らず各microbatchから求め、λの勾配ノルムはlogical batchで測る。loss scaleが定まらないと停止し、fallbackしない。
5. **Teacher provenance:** 全層抽出、文字/subword span、両側projection、全head平均と既存packet形式を維持する。text-only由来Attentionはtext identity、CTC/unit/teacher/S2UTは従来speech identityで照合する。元artifactは上書きしない。
6. **Persistence:** 作業と頻繁なログはVM。Driveへは固定runtimeの検証付きZIPだけを公開する。新alignmentログもrecovery filesへ含める。calibration artifactのhashはplan/checkpointへ固定し、異なるartifactでresumeしない。
7. **Diagnostics:** per-update layer loss/weightと、独立したteacher-forced dev alignment診断を追加する。translation generation/validationはteacher-freeのまま。Attentionを正解alignmentとは主張しない。
8. **Limits:** CPU tiny接続試験・Windowsのsampler fixtureを実GPU/実コーパスの結果として扱わない。Colabではcompiled native batch samplerのpreflightを必須とし、Python samplerへfallbackしない。長時間trainingとBLEU改善の検証は別途必要。

9. **設定と表現:** annealingのfraction/scheduleはcanonical conditions YAML→policy→loss/logの経路で統一する。tailはtask-only、validation/inferenceはteacher-freeと呼ぶ。公開名称を整理し、内部method IDは維持する。校正v2/派生stage v3は旧校正の暗黙再利用を拒否する。

10. **Corpus input:** コーパスとmanifestは実験IDから独立して明示指定する。split・ID・本文を維持し、train/devでBaselineを学習する。QC情報がある場合はacceptedのみを受け付ける。固定ベンチマークのprepared validatorには空test TSVを扱うアダプターを用いる。音声対とT2TTの元テキストはID・split・本文で部分集合照合する。既存Kokoroリリースの設定は任意の例とし、旧CLI名は互換入口として残す。旧runの再開では元ID・revisionを使う。

11. **S2UT audio loading:** 学習前にprepared TSVの音声をZIP索引と照合し、ZIPとWAVの実体は使用時に固定benchmarkのDownloaderで検証する。fairseqローダーが同じサンプル順序で次のバッチを取得する。I/O変更を受けたソースのSHA-256が承認済みの組と完全に一致する場合だけ、S2UT Baselineのアダプター識別子と転移学習の実装識別子に旧版74c5efaのハッシュを用いる。 [検証記録](VALIDATION.md)
