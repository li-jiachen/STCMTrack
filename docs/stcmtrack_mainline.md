# STCMTrack Mainline

The current STCMTrack mainline follows the paper `STCMTrack: Confidence-Guided Spatio-Temporal Context Modeling for Robust Anti-UAV Tracking`.

Module mapping:

- Paper `LTCP`: local-enhanced temporal context propagation. In this project the checkpoint-compatible internal config key remains `model.ltcp`; public mixin name is `ltcp`.
- Paper `CTR`: confidence-triggered re-localization. It is implemented by the one-stream evaluator's `motion_compensation.py`.

Mainline configuration:

- Base tracker: STCMTrack-compatible DINOv2 model
- Config entry: `STCMTrack dinov2`
- Mixins: `disable_torch_compile`, `ltcp`, `ltcp_memory_2`, `ctr`, `ctr_foreground_correction`, `ctr_threshold_040`, `evaluation`
- Full test dataset: `ANTIUAV410-test`
- LTCP memory size: `2`
- CTR confidence threshold: `0.40`
- New evaluation metric: per-sequence `AUC`, `P@20`, `NP@0.5`, then average over sequences
- Final full-test result with the external post-processing script: `AUC=0.6839`, `P@20=0.8773`, `NP@0.5=0.8732`

The model builder `type` remains `STCMTrack` so existing weights load without remapping parameter names.
