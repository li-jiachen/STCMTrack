# Implementation details

The STCMTrack implementation follows the method and explicit settings in the paper. This document records those settings together with the fixed defaults needed to train and evaluate the public implementation. Section and equation numbers refer to the paper. Values not explicitly reported in the paper are implementation defaults under `config/STCMTrack/`; they are not claims about the original experiments. The published scores have not been reproduced with this implementation.

## 1. Tracking loop

For every tracked frame, `OneStreamTracker_Evaluation_MainPipeline` and the model run the following steps.

1. **Homography.** When CTR is enabled, one homography $`\mathcal{T}_t`$ between the previous and the current frame is estimated (Sec. 2.3).
2. **Search crop.** The search region is a 378 × 378 crop with area factor 5 around the reference box. With MCC, the reference box is $`B_t^{\mathrm{geo}}`$ (mapped center, previous size); otherwise it is the previous final box $`B_{t-1}`$.
3. **Joint encoding.** The learned query, the tokens of the first-frame template (196 × 196, area factor 2, 14 × 14 tokens) and the search tokens (27 × 27, N = 729) form one sequence of 926 tokens that passes through the 12 blocks of the ViT. The output at the query position is the target state token $`\mathbf{q}_t`$; the outputs at the search positions are $`\mathbf{X}_t`$. The template is cropped once from the first frame and is never updated.
4. **LTCP.** $`\mathbf{X}_t`$ is fused with the context built from the memory (Eqs. 1-3), and $`\mathbf{X}_t`$ itself, before fusion, is written to the memory of the sequence. The memory keeps the m = 2 most recent frames, so LTCP sees 0, 1 and 2 memory frames on the first three tracked frames; with an empty memory the tokens pass through unchanged.
5. **Prediction.** The fused tokens enter the two MLP heads. The box $`B_t^{\mathrm{tracker}}`$ is read at the peak of the sigmoid center response map, and $`s_t`$ is the sigmoid response sampled bilinearly at the center of that box.
6. **RGTC.** If $`s_t < \tau`$, RGTC may replace the box by $`B_t^{\mathrm{corr}}`$ (Eqs. 4-6).
7. **State update.** The final box $`B_t`$ becomes the reference of the next search crop and of the next CTR step. The memory and all per-sequence states are released when the sequence ends.

Training uses the same encoder, LTCP module and heads: `STCMTrack_DINOv2.forward` processes one template and three chronological search frames, and `STCMTrackInference_DINOv2.forward_tracking` processes one frame at a time with the per-sequence memory. Both give identical outputs for the same inputs.

## 2. Settings

### Model

| Item | Value | Configuration key |
|---|---|---|
| Backbone | DINOv2-pretrained ViT-B/14, 12 blocks, D = 768, kept frozen | `model.backbone` |
| Template / search input | 196 × 196 / 378 × 378 | `common.template_size`, `common.search_region_size` |
| Template / search crop | area factor 2 / area factor 5 | `template_area_factor`, `search_region_cropping.area_factor` |
| TMoE adapters | r = 64, α = 64, 4 routed experts, on the `q`, `k`, `v`, `proj`, `fc1` and `fc2` linear layers of every block | `model.tmoe` |
| Token-type embeddings | three embeddings: template background, template foreground (tokens inside the initial box), search | – |
| Target state token | one learned query (`track_query` + `query_embed`) encoded jointly with the template and the search tokens | – |
| Prediction heads | two 3-layer MLPs on every search token: center response (1 value) and box regression (distances from the token center to the left, top, right and bottom sides of the box, normalized by the crop size) | – |
| Parameters | 115.3 M in total; 29.2 M trained in stage 1 (26.9 M TMoE, 2.4 M heads, 3,840 query and type embeddings); 770 trained in stage 2 (LTCP gate) | – |

### LTCP (Sec. 2.2)

| Item | Value | Configuration key (`model.ltcp`) |
|---|---|---|
| Memory size m | 2 frames of search tokens, always stored before fusion and detached | `memory_size`, `detach_memory` |
| Frame weights | softmax over the cosine similarities $`r_k`$ of the mean-pooled tokens, temperature 1 | `frame_softmax_temperature` |
| Global confidence | $`c^{g}_t = (\sum_k w_k r_k + 1) / 2`$ with the frame weights $`w_k`$ | – |
| Local confidence | $`c^{l}_{t,i} = (p_{t,i} + 1) / 2`$ | – |
| Gate bound | $`g_{\max} = 0.05`$ | `max_gate` |
| Gate parameters | $`\mathbf{W}_g \in \mathbb{R}^{1 \times (D+1)}`$, $`\mathbf{b}_g \in \mathbb{R}`$ (770 parameters); initialized with $`b_g = -4`$, weight 2 for $`c^{\mathrm{temp}}`$ and zeros for $`\mathbf{q}_t`$ | `gate_bias_init`, `confidence_weight_init` |
| Memory storage | CPU, float32 | `memory_device`, `memory_dtype` |

### CTR (Sec. 2.3)

| Item | Value | Configuration key (`...pipeline.ctr`) |
|---|---|---|
| Correction threshold τ | 0.40; predictions with $`s_t \ge \tau`$ are accepted directly | `confidence_threshold` |
| Keypoints and matching | 800 ORB keypoints per frame, cross-checked Hamming matching, the 80 best matches | `orb_features`, `max_matches` |
| RANSAC | reprojection inlier threshold 2.0 pixels | `ransac_reproj_threshold` |
| Homography acceptance | at least 8 matches, at least 8 inliers and an inlier ratio of at least 0.25 | `min_matches`, `min_inlier_ratio` |
| MCC | the previous center is mapped with $`\mathcal{T}_t`$ and the previous width and height are kept; the mapped center must lie inside the image and move by at most one image diagonal | `max_center_shift_ratio` |
| Residual threshold (Eq. 5) | α = 4.4478 (3 × 1.4826) | `residual_mad_scale` |
| Valid aligned region | pixels whose bilinear interpolation support lies completely inside the warped previous frame | – |
| MOG2 | history 80, variance threshold 24, shadow pixels excluded; updated with every frame | `mog2_history`, `mog2_var_threshold`, `mog2_detect_shadows` |
| Foreground mask | $`M_t = M_t^{\mathrm{mog2}} \cup M_t^{\mathrm{res}}`$, always used | – |
| Area constraint | bounding rectangles with an area in $`[\max(4,\ 0.05 A_{t-1}),\ 25 A_{t-1}]`$ pixels, where $`A_{t-1}`$ is the area of the previous target box | `min_candidate_area`, `min_candidate_area_ratio`, `max_candidate_area_ratio` |
| Candidate selection | the remaining rectangle whose center is nearest to the motion-corrected center | – |

Fallbacks:

- No accepted homography: the search crop is not re-centered. If RGTC is triggered in that frame, the foreground mask is the MOG2 mask alone and the reference center is the center of the tracker box.
- Mapped center rejected: the search crop is not re-centered and the reference center is the center of the tracker box; the homography is still used for the residual of that frame.
- No candidate after the area constraint: the tracker box is kept.

The ablation mixins `ctr_no_mcc` and `ctr_no_rgtc` switch off one branch. Without MCC the search crop is not re-centered, while RGTC still uses the homography for the frame alignment and the motion-corrected reference center. Without RGTC low-confidence predictions are kept.

The eight variants follow Table 2: `VARIANT=baseline` selects independent SPMTrack for row 1; rows 2–8 use STCMTrack and share its base weights and settings. Data, optimizer and loss settings are aligned between the models. The backbone and MLP head architectures are the same, but SPMTrack retains its template and query computations, sampling and post-processing. These differences limit the paper's statement that all other settings are shared; see [ABLATION.md](ABLATION.md).

## 3. Training (Sec. 3.1)

The paper specifies 80 + 20 epochs, LTCP-only training in stage 2, BCE/GIoU weights 1 : 1, AdamW learning rate 1e-4, weight decay 0.1 and cosine scheduling. Sampling, augmentation, binary center labels, frame-loss averaging, no warm-up, batch/sample counts, mixed precision and clipping below are additional fixed defaults. They make the paper-defined implementation runnable without claiming to recover unspecified historical settings.

| Item | Value |
|---|---|
| Stage 1 | 80 epochs; trains the TMoE adapters, the target-state query, the token-type embeddings and the prediction heads; the DINOv2 weights stay frozen |
| Stage 2 | 20 epochs; trains only the LTCP gate; initialized from the stage-1 checkpoint (`ltcp` and `ltcp_stage2` mixins) |
| Training sample | template from the first frame of a sequence; three consecutive search frames in which the target is present, sampled anywhere after the first frame (`first_frame_causal`) |
| Search crops | area factor 5 around the ground-truth box of each frame, scale jitter 0.25 and translation jitter 3; the three frames of a sample share the jitter |
| Augmentation | horizontal flip (p = 0.5), color jitter (0.4) and one of grayscale / solarization / Gaussian blur, each applied jointly to the template and the three search frames |
| Center target | binary: the token that contains the ground-truth box center is the positive, all other tokens are negatives |
| Loss | binary cross-entropy on the center response plus GIoU on the box predicted at the positive token, weights 1 : 1, averaged over the three search frames |
| Optimizer | AdamW, initial learning rate 1e-4, weight decay 0.1 on all trained parameters |
| Schedule | cosine decay to zero, stepped per iteration, no warm-up |
| Batch | global batch size 4; 2048 training samples and 4096 validation samples per epoch |
| Other | mixed precision (float16), gradient-norm clipping at 1.0 |

MCC and RGTC are not used during training. Checkpoints are written every 20 epochs and at the last epoch (`checkpoint/epoch_<n>/model.bin`).

## 4. Evaluation protocol (Sec. 3.2)

| Item | Convention |
|---|---|
| AUC | area under the success curve over IoU thresholds in [0, 1], computed as the exact integral of the empirical curve (equal to the mean IoU); the 101 sampled thresholds are used for plots only |
| P@20 | fraction of frames with a center error below 20 pixels (strict) |
| P<sub>n</sub> | fraction of frames with a center error divided by $`\sqrt{w_{gt}^2 + h_{gt}^2}`$ below 0.5 (strict) |
| Absent targets | frames with `exist = 0` are excluded from all metrics |
| Initialization frame | included |
| Aggregation | unweighted mean over the sequences |
| Invalid predictions | a box with non-positive size on a frame with a present target counts as a failure; non-finite predictions are errors |
| Coverage | a full evaluation requires every sequence of the split with matching frame counts |

These conventions are written into every JSON report (field `protocol`). The two internal one-pass-evaluation handlers, `tools/evaluate_antiuav_iou_p20.py` and the SPMTrack baseline all call `trackit/core/evaluation/antiuav.py`. Predictions are exported to `results.zip` with one `<sequence>.txt` file per sequence; it has one `x y w h` line per frame (17 significant digits), and the first line is the initialization box.

## 5. Box formats

| Place | Format |
|---|---|
| Paper, $`B_t = (x_t, y_t, w_t, h_t)`$ | center and size |
| `IR_label.json` (`gt_rect`) and exported results | top-left corner, width, height |
| Model, tracking pipeline and CTR | corner coordinates (x1, y1, x2, y2) |

Areas, centers, scales and P<sub>n</sub> are always computed after the corresponding conversion.

## 6. Checkpoints

- Training snapshots written by the current code (`model.bin`, Safetensors) contain the full model in both stages, including the frozen backbone, and two buffers with the TMoE scaling (`_expert_alpha` and `_use_rsexpert`). The `.bin` extension does not change the serialization format.
- `tools/export_stcmtrack_weights.py` splits a snapshot into a base file (everything except LTCP) and an LTCP file (the gate and the two buffers). The default evaluation filenames are `stcmtrack_base.bin` and `stcmtrack_ltcp.bin` for Anti-UAV410, and `stcmtrack_antiuav300_base.bin` and `stcmtrack_antiuav300_ltcp.bin` for Anti-UAV. Both files record the SHA-256 of the source snapshot in their metadata, and a manifest with the SHA-256 of the outputs is written next to the base file. Existing outputs are never overwritten.
- `tools/check_stcmtrack_weights.py` (called by `test_stcmtrack.sh`) validates the base and LTCP files. Newly exported pairs must identify the same source snapshot; the older v1.0.0 release pair is recognized by its file SHA-256 values.
- The loader recognizes the incremental legacy format, translates its scaling fields and can supply frozen parameters from pretrained DINOv2. However, the released base has learned query values that overflow FP32 LayerNorm, so the numerical preflight rejects it before copying parameters. Format compatibility alone does not make this checkpoint usable.
- File SHA-256 checks establish identity of the released pair, not numerical validity or reproduction of the paper's scores. Train new checkpoints with the settings above and run a complete evaluation to measure this implementation's performance; see [weights/README.md](../weights/README.md).
