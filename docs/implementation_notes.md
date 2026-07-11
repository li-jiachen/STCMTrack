# STCMTrack Implementation Notes

This document records the released STCMTrack configuration corresponding to the manuscript `STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking`. The implementation inherits and extends the SPMTrack codebase.

Module mapping:

- Paper `LTCP`: local-enhanced temporal context propagation. In this project the checkpoint-compatible internal config key remains `model.ltcp`; public mixin name is `ltcp`.
- Paper `CTR`: confidence-triggered re-localization. The released implementation currently uses the verified MOG2-based foreground-correction path in the one-stream evaluator. The residual-guided branch described in the paper is not presented here as a fully verified release.

Released configuration:

- Base tracker: STCMTrack-compatible DINOv2 model
- Config entry: `STCMTrack dinov2`
- Mixins: `disable_torch_compile`, `ltcp`, `ltcp_memory_2`, `ctr`, `ctr_foreground_correction`, `ctr_threshold_040`, `evaluation`
- Full test dataset: `ANTIUAV410-test`
- LTCP memory size: `2`
- CTR confidence threshold: `0.40`
- Evaluation aggregation: per-sequence `AUC`, `P@20`, and `PNorm@0.5`, followed by an unweighted average over sequences
- Paper-reported Anti-UAV410 test result: `AUC=69.2`, `P@20=88.9`, `PNorm=88.4`

The model builder `type` remains `STCMTrack` so existing weights load without remapping parameter names.
