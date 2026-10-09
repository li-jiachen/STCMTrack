# SPMTrack baseline

The STCMTrack implementation follows the paper's method and explicit settings; unspecified settings use documented fixed defaults. `VARIANT=baseline` selects the independent SPMTrack model for Table 1 and Table 2 row 1. It contains no LTCP, MCC or RGTC and requires its own trained checkpoint. The published baseline scores have not been reproduced with this configuration.

```bash
DEVICE_IDS=0 ./train_spmtrack.sh
VARIANT=baseline DEVICE_IDS=0 ./test_stcmtrack.sh
```

Add `DATASET=antiuav300` to both commands for Anti-UAV. Only the two Anti-UAV410 STCMTrack weights are published in this repository's release.

## Source and model

The model follows [WenRuiCai/SPMTrack](https://github.com/WenRuiCai/SPMTrack), commit `c581fe27231f3e16c38578e47daddadfaf6ffd7d`, Apache-2.0. `trackit/models/methods/SPMTrack/UPSTREAM.json` records the source files and porting changes.

- DINOv2 ViT-B/14 with frozen pretrained parameters and TMoE adapters; template 196 × 196, search 378 × 378; two MLP prediction heads.
- Three templates: the first-frame template and two historical references. Templates are updated from predicted boxes after tracking each frame.
- The query output of the previous frame is propagated into the next joint encoding. Search tokens are reweighted by their dot product with the current query output before entering the heads.
- Training samples contain three templates and two search frames, using interval sampling and per-image flip/color augmentation.
- Evaluation keeps the original Hann-window weight 0.45, search area factor 5 and minimum target size 10 pixels.

The forward model retains the original computation. The port initializes learned queries explicitly, builds TMoE once, releases per-sequence memory and validates checkpoints before loading.

## Shared experiment settings

The paper specifies the data splits, input sizes, 80-epoch base training budget, AdamW with learning rate 1e-4 and weight decay 0.1, cosine scheduling, equally weighted BCE/GIoU losses and shared evaluation metrics. The implementation additionally uses global batch size 4, 2048 training and 4096 validation samples per epoch, cosine decay to zero without warm-up, binary center targets and averaged search-frame losses. These are fixed public defaults, rather than claims about settings omitted from the paper.

These common settings replace the upstream training configuration. The SPMTrack template, query, sampling and inference conventions remain distinct from STCMTrack. See [ABLATION.md](ABLATION.md) for the resulting limitation of the paper's statement that all remaining settings are shared. Retrain and evaluate this configuration to obtain new baseline results.

## Checkpoints

Copy the `model.bin` produced by `train_spmtrack.sh` to `weights/spmtrack_baseline.safetensors`, or set `BASE_WEIGHT` to its path. For Anti-UAV, the default is `weights/spmtrack_antiuav300_baseline.safetensors`. These files use Safetensors; the extension does not change the format.

The checkpoint contains trainable parameters, TMoE scaling (`expert_alpha`, `use_rsexpert`) and `_spmtrack_port_version`. The matching pretrained DINOv2 weights supply the frozen backbone. `test_stcmtrack.sh` validates the file before evaluation:

```bash
python tools/check_spmtrack_weights.py --weights weights/spmtrack_baseline.safetensors
```

An original SPMTrack checkpoint without the port marker can be loaded with `ALLOW_UNMARKED_SPMTRACK_WEIGHTS=1`. All trainable parameters must be present and the scaling must match the configuration. STCMTrack checkpoints must not be used as SPMTrack weights.
