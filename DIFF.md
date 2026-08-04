# Code vs. Paper Differences

This document flags implementation details that are present in the codebase but not mentioned in the paper, or that differ from the paper's description.

## Reinforcement Learning

- **Track density reward shape** (`rl/reward/rule_based_reward.py`, `rl/config.py:RULES_FOR_REWARD`): the paper describes the Round 2 composite reward as `CLaMP 3 + 0.2 × tanh(TrackDensity / 4)`; the code implements the track term as a per-bar plateau reward (1.0 inside [8, 20] tracks, linear falloff over 4 below / 12 above) followed by `tanh(score / (0.5 / artanh(0.9)))`, averaged over bars — see `compute_linear_plateau_reward` (`rl/reward/utils.py`). The two released variants are selected with `--track_reward 0/1` in `arch/symph/2_reinforce.py`.
- **Symmetric KL regularization** (`rl/grpo/loss.py:full_symmetric_kl_loss`): the KL penalty against the frozen reference model is the mean of forward and reverse full-distribution KL, not the k3 estimator commonly used in GRPO.
- **Per-head loss weighting** (`rl/config.py:GRPO_LOSS_CONFIG`): GRPO optimizes three heads — events, track IDs, instruments — with `loss_scales = (1.0, 5.0, 5.0)` and `kl_betas = (0.03, 0.01, 0.01)`.
- **Composite reward weights** (`rl/reward/composite_reward.py`): the total reward is always `1.0 × CLaMP 3 + 0.2 × track` (`0.0 × track` for the pure-CLaMP variant); the objective metrics computed during evaluation (dissonance, precision/recall, moving, ornament) are reported alongside but carry weight 0 and never contribute to the reward or to best-song picking.

## Architecture

- **Cross-attention in the 2D harmony event decoder** (`arch/symph/model.py:harmony_stage`, `arch/symph/modules/cross_attn_provider.py:HarmCrossAttnProvider`): the paper describes cross-attention only for the 3D music event decoder; in the code the 2D harmony event decoder also uses cross-attention — each layer attends to the same layer's hidden states of the previous bar's harmony events (zeros for the first bar), giving the harmony decoder event-level access to the preceding bar's skeleton in addition to the bar-level context.
- **Alternative cross-attention modes** (`arch/symph/config.py:CROSS_ATTN_STREAM`, `arch/symph/modules/cross_attn_provider.py`): besides the paper's 2-stream design, the code supports a 3-stream variant (additionally attending to the previous track in the same bar) and single-stream "harmony"/"melody" ablation modes.
- **`HARMO_LAST_LAYER = -2`** (`arch/config.py`): the harmony event context used for cross-attention is taken from the second-to-last harmony-decoder layer, for historical reasons.
- **Segment embedding in the shared event encoder** (`arch/symph/modules/hier_encoder.py`): token-type embeddings distinguish harmony from music events.

## Sampling and Inference

- **Metadata-head sampling parameters** (`arch/symph/generate.py`): track-ID and instrument heads are sampled with temperature 0.8 and top-p 0.95; the paper's stated temperature 1.0 / top-p 0.99 applies only to music event tokens.
- **`FORBID_PIANO`** (`Results/config.py`, `arch/symph/generate.py`): the piano instrument family (programs 0–4) is masked out during final-evaluation generation.
- **Sampling constraints** (`arch/symph/sampling.py`, `arch/harmo/sampling.py`): sampling constraints masking invalid positions outside the bar, harmony-generation constraints (bar lengths restricted to time signature of 2/4, 3/4, 4/4; no empty bars). The beat length is recommended to be quarter-note, because data for other beat lengths (e.g., three eighths in 6/8) is scarce, the model is found to have a tendency to copy.
- **Dissonance-averse sampling visualization** (`arch/symph/generator.py:--vis_dissonance`, `arch/symph/sampling.py:DissonanceConstrainer.vis_diss_logit_shift`): the code can visualize the logit adjustment of dissonance-averse sampling — plots of the original vs. adjusted pitch logits with the active harmonic tones, active non-harmonic tones, and remaining harmony-condition pitches highlighted, saved to `<save_dir>/dissonance_vis/`. Plots are produced when at least two non-harmonic tones are active, up to a cap of 16 (`vis_max_plots`) per run.

## Harmony Skeleton

- **Harmonic Analysis** (`data_prep/harmo_analysis.py`): the extension-identification DP allows deleting notes matched by the template with a penalty λ = 3.0.
- **Composite skeleton filter** (`rl/reward/harmo_filter.py`, `rl/config.py:RULES_FOR_HARMO_FILTER`): the paper's density/repetition filters are implemented as a weighted score over four signals (density, bar-level loop repetition with periods 1/2/4, beat-level repetition, cadence count) with an acceptance threshold of 3.8.
- **Two-sided log-probability filter** (`arch/harmo/generator.py:filter_by_log_prob`): skeletons are kept when the per-token loss lies in [0.25, 0.5] — both unusually predictable and unusually surprising skeletons are rejected.
- **Dataset skeletons are also pruned** (`Results/batch_filter_gen.py:get_dataset_harmo`): the paper notes that pruning is unnecessary for analyzed skeletons, but in the released re-orchestration setting (Round 2) the validation-set skeletons are likewise pruned into template chords (with strict triad matching) before conditioning.

## Data Preprocessing

- **Track merging** (`data_prep/merge_tracks.py`): MIDIs with more than 123 tracks are reduced by merging same-program (or drum) tracks, prioritizing drums and high note counts.
- **Long-note slicing** (`data_prep/main.py:extract_and_slice_notes`): notes longer than 8 beats are split.
- **Percussion exclusion in harmony analysis** (`data_prep/main.py:analyze_harmo_outline`): drum tracks and timpani are excluded from the pitch-class statistics. (In the preprocessing used for the paper's experiments, drum tracks of the source MIDIs were serialized with program 0 and thus escaped the exclusion; the released code maps them to the drum program. The dataset contains very few such concert-drum tracks, so the effect is negligible.)
- **Two preprocessing variants** (`arch/config.py:SIMPLIFY_HARMONY_DUR`): the released pipeline stretches skeleton durations to beat boundaries ("simplified"); a "rhythmic" variant stretches to the min and max of skeletal notes' onsets and offsets, but is unused.

