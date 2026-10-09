# SPMTrack comparison

This repository contains an independent SPMTrack path for the SPMTrack† comparison in Table 1. It uses the same data splits and evaluation metrics as STCMTrack, while retaining the SPMTrack model and inference conventions:

```bash
DEVICE_IDS=0 ./train_spmtrack.sh                     # add DATASET=antiuav300 for Anti-UAV
VARIANT=spmtrack DEVICE_IDS=0 ./test_stcmtrack.sh
```

`VARIANT=baseline` (alias `stcm_base`) instead selects STCMTrack with LTCP, MCC and RGTC disabled. All eight component ablations use that same STCMTrack foundation; see [ABLATION.md](ABLATION.md) for the mapping and the conflict with the paper's identification of Table 2 row 1 as SPMTrack.

## 1. Source

- Repository: <https://github.com/WenRuiCai/SPMTrack>, commit `c581fe27231f3e16c38578e47daddadfaf6ffd7d`, Apache-2.0.
- `trackit/models/methods/SPMTrack/UPSTREAM.json` records, for every file of the baseline path, the upstream path, the SHA-256 of the upstream file and its status (`verbatim`, `modified` or `ported`). `tests/test_spmtrack_static.py` checks these hashes.

The model files are under `trackit/models/methods/SPMTrack/`; the pipeline and post-processing files are under `trackit/runner/evaluation/distributed/tracker_evaluator/`.

**Verbatim (7 files, byte-identical).** `sample_data_generator.py`, `modules/__init__.py`, `modules/patch_embed.py`, `modules/head/mlp.py`, `modules/tmoe/__init__.py`, `modules/tmoe/merge.py` and `default/pipelines/spmtrack_one_stream/visualization.py` (upstream: `default/pipelines/one_stream/visualization.py`).

**Modified (1 file).** `modules/tmoe/apply.py`: one line, `sorted(...)` instead of `list(set(...))`.

**Ported (9 files, rewritten with the deviations listed in `UPSTREAM.json`).**

- `SPMTrack.py` (upstream: same name): forward pass unchanged; checkpoint loading rewritten.
- `SPMTrack_inference.py` (upstream: `SPMTrack_full_finetune.py`): inherits the training model.
- `builder.py` (upstream: same name): model type `dinov2` only.
- `default/pipelines/spmtrack_one_stream/__init__.py` and `builder.py` (upstream: `default/pipelines/one_stream/`): main pipeline and its builder.
- `default/pipelines/spmtrack_one_stream/template_mask.py` (upstream: `default/pipelines/_common/template_foreground_indicating_mask_generation.py`): template mask plugin.
- `components/post_process/spmtrack_box_with_score_map.py` (upstream: `components/post_process/box_with_score_map.py`): separate post-processing class.
- `config/SPMTrack/run.yaml` and `config/SPMTrack/dinov2/config.yaml` (upstream: same paths): structure and inference settings kept; data and training budget of Sec. 3.1.

Not ported: the `dinov2_full_finetune` model type, the other backbones and heads of the upstream repository, its other dataset configurations and the LaSOT-specific reference rule.

## 2. Structure

The network and the inference logic follow the pinned commit.

- **Backbone and adapters.** DINOv2 ViT-B/14 with frozen weights and TMoE adapters (r = 64, α = 64, 4 routed experts); template 196 × 196, search region 378 × 378; two MLP prediction heads.
- **Templates.** Three templates per frame: the first-frame template and two reference templates. After every tracked frame, a template is cropped around the predicted box (area factor 2.0) and appended to the sequence memory. With n templates in memory, the three slots use the indices `[0, 0, 0]` (n = 1), `[0, 1, 1]` (n = 2), `[0, 1, 2]` (n = 3) and `[0, d // 2, d + d // 2]` with `d = n // 2` (n ≥ 4).
- **Query state.** The query of a frame is the query output of the previous frame plus the learned query; it is zero-initialized on the first tracked frame.
- **Head input.** The encoded search tokens are re-weighted by their dot product with the query output before they enter the heads.
- **Post-processing.** The response map is blended with a Hann window (weight 0.45) to select the peak; the confidence is the response at that peak. The search region uses area factor 5.0 and a minimum object size of 10 pixels.
- **Training.** Three templates and two search frames per sample (`interval` sampling), per-image flip and color augmentation, IoU-aware classification targets, the losses of the two search frames added, weight decay 0 for one-dimensional parameters and embeddings, 2 warm-up epochs and a minimum learning rate of 1e-6.

The independent SPMTrack comparison contains no LTCP, MCC or RGTC. Its model package, pipeline and post-processing class do not import the STCMTrack model, `ltcp`, `ctr` or the STCMTrack pipeline; `tests/test_spmtrack_static.py` and `tools/check_variant_mapping.py` verify this from the import graph.

## 3. Training and evaluation setup

These settings are shared with STCMTrack and replace the upstream training setup:

- **Data.** The Anti-UAV410 and Anti-UAV splits of `config/_dataset/*.yaml`.
- **Training budget.** 80 epochs, global batch size 4, 2048 training and 4096 validation samples per epoch, i.e. the stage-1 budget of Sec. 3.1; `torch.compile` and the efficiency assessment are switched off.
- **Evaluation.** The `one_pass_evaluation_compatible` handler, the PyTracking export and the metrics of `trackit/core/evaluation/antiuav.py` (Sec. 3.2).

## 4. Engineering changes with respect to upstream

The forward computation is unchanged.

| Upstream behaviour | This repository |
|---|---|
| `track_query` and `query_embed` are created with `torch.empty` | initialized with a truncated normal distribution (std 0.02); a loaded checkpoint overrides them |
| inference `load_state_dict` deletes `expert_alpha` unconditionally and ignores the stored value | the checkpoint is validated before any tensor is copied; a TMoE scaling that differs from the configuration is an error |
| the inference model applies TMoE inside `load_state_dict` | the inference class inherits the training class and builds TMoE once in the constructor |
| missing `z_1` / `z_2` inputs are replaced by copies of other templates | the three templates and their masks are required |
| TMoE target layers are collected with `list(set(...))` | `sorted(...)`, so the initialization order does not depend on `PYTHONHASHSEED` |
| the pipeline keeps the templates and masks of finished sequences, hard-codes `.cuda()` and indexes tasks by batch position | sequence memory is released at the end of a sequence and reset on initialization; the device is a constructor argument; the reference selection is computed once and shared with the mask plugin |
| a predicted box that is invalid after clipping fails in the mask step | no template is created for that frame; such frames are counted and reported |
| non-finite model outputs | raise an error, as in the shared evaluation core |

## 5. Checkpoints

- A checkpoint written by `train_spmtrack.sh` (`checkpoint/epoch_79/model.bin`, Safetensors) contains the trainable parameters, the TMoE scaling (`expert_alpha`, `use_rsexpert`) and the marker `_spmtrack_port_version`. Copy it to `weights/spmtrack_baseline.safetensors` (`weights/spmtrack_antiuav300_baseline.safetensors` for Anti-UAV). The frozen backbone always comes from the DINOv2 weights.
- `tools/check_spmtrack_weights.py --weights <file>` checks a file without building the model; `test_stcmtrack.sh` runs it before the evaluation.
- STCMTrack checkpoints are always refused by the baseline, and SPMTrack checkpoints are not accepted by STCMTrack.
- A file without the marker, such as a checkpoint trained with the official SPMTrack code, is accepted only with `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1` (configuration key `model.allow_unmarked_weights`). All trainable parameters must be present and the TMoE scaling must match the configuration.
