#!/usr/bin/env bash
set -euo pipefail

# Edit this file, then run: bash scripts/run_millerecog2019.sh
# One model/input pairing across the Miller ECoG library (2019) cells.
# Configure checkpoints and model settings in YAML; see README.md.
#
# In this dataset each task set has its own tasks, and not every subject did
# every task set. So every launch below passes --cells: the launcher builds the
# usual task x target grid, then keeps only the task/target pairs listed in that
# cell list (imindbench/cell_manifests/millerecog2019/<name>.json).

# ── Model and input settings ───────────────────────────────────────────────────
# Valid combinations (names refer to YAML configs under imindbench/conf/).
# The miller_* presets do no filtering or re-referencing: the stored signal
# already had both.
# MODEL              PREPROCESSOR                         EXPERIMENT
# logistic/mlp/cnn   miller_multi_stft_1000Hz             default
# logistic/mlp/cnn   miller_wav_hpf_robust_1000to500Hz    default
# htnet_500Hz        miller_wav_hpf_robust_1000to500Hz    default
# popt (PopT-v2)     miller_multi_stft_1000Hz             default
# linear_baseline    miller_stft_brainbert_1000Hz         default
# barista            miller_wav_barista_1000to2048Hz      default
# diver              miller_wav_diver_1000to500Hz         default
MODEL=logistic
PREPROCESSOR=miller_multi_stft_1000Hz
EXPERIMENT=default                  # default runs every listed cell.

# ── Paths, tasks and subject/session pairs ─────────────────────────────────────
# sub<S>_sess<T>: S is the subject number and T the task-set number; both are
# listed in imindbench/conf/dataset/millerecog2019.yaml.
DATASET=millerecog2019
CONFIG_DIR="/path/to/config"          # Contains paths/local.yaml.
OUTPUT_ROOT="/path/to/runs/millerecog2019"

# Main table (task sets 1-8): 153 binary + 25 multiclass = 178 cells.
TASKS=(
  cue_vs_isi fb_target_A_vs_B flex_vs_rest gesture_type
  gesture_vs_rest hand_vs_tongue im_hand_vs_tongue im_move_vs_rest
  mot_hand_vs_tongue mot_kind mot_move_vs_rest move_vs_rest
  nouns_read_vs_isi real_vs_imag_hand real_vs_imag_move real_vs_imag_tongue
  runid_from_isi runid_from_rest stim_vs_isi verbs_speak_vs_isi
  verbs_vs_nouns_cue
)
MULTICLASS_TASKS=(
  direction_4way fingers_5way gesture_type runid_from_rest which_finger
)
TARGETS=(
  sub1_sess3 sub3_sess1 sub3_sess2 sub3_sess4 sub3_sess5 sub3_sess6
  sub4_sess1 sub4_sess5 sub5_sess1 sub5_sess4 sub5_sess5 sub6_sess1
  sub6_sess5 sub7_sess1 sub7_sess2 sub7_sess3 sub8_sess1 sub9_sess1
  sub11_sess1 sub11_sess2 sub11_sess3 sub12_sess1 sub12_sess6 sub13_sess4
  sub16_sess1 sub16_sess2 sub16_sess3 sub16_sess4 sub16_sess6 sub16_sess7
  sub17_sess1 sub18_sess1 sub18_sess2 sub18_sess8 sub19_sess1 sub19_sess4
  sub20_sess1 sub20_sess8 sub22_sess1 sub22_sess2 sub23_sess8 sub24_sess1
  sub24_sess2 sub24_sess8 sub25_sess1 sub26_sess1 sub26_sess4 sub26_sess6
  sub26_sess7 sub26_sess8 sub27_sess4 sub27_sess5 sub28_sess6 sub28_sess7
  sub29_sess1 sub29_sess4 sub29_sess6
)

# Newer task sets 10 (faces_noise) and 11 (memory_nback), and their controls.
NEW_TASKS=(
  noise_high_vs_low noisy_face_vs_house noisy_face_vs_house_le50
  pic_vs_blank target_vs_nontarget
)
CONTROL_TASKS=(fixation_vs_task)
MULTICLASS_CONTROL_TASKS=(nback_level_3way)
NEW_TARGETS=(
  sub1_sess11 sub2_sess10 sub4_sess10 sub4_sess11 sub5_sess11 sub10_sess10
  sub15_sess10 sub21_sess10 sub25_sess11 sub26_sess10 sub29_sess10
)

# Regression (continuous targets): task sets 13, 16, 19 tiled, 22-24 sliding,
# 25 BCI Competition IV split.
REGRESSION_TASKS=(
  cursor_vx cursor_vy flex_index flex_little flex_middle
  flex_ring flex_thumb target_x target_y
)
REGRESSION_TARGETS=(
  sub3_sess13 sub5_sess13 sub7_sess16 sub7_sess19 sub9_sess16 sub9_sess19
  sub13_sess13 sub16_sess13 sub19_sess13 sub22_sess16 sub22_sess19 sub24_sess16
  sub24_sess19 sub26_sess13 sub27_sess13 sub29_sess13
)
SLIDING_TARGETS=(
  sub3_sess22 sub5_sess22 sub7_sess23 sub7_sess24 sub9_sess23 sub9_sess24
  sub13_sess22 sub16_sess22 sub19_sess22 sub22_sess23 sub22_sess24 sub24_sess23
  sub24_sess24 sub26_sess22 sub27_sess22 sub29_sess22
)
BCI4_TASKS=(flex_index flex_little flex_middle flex_ring flex_thumb)
BCI4_TARGETS=(
  sub3_sess25 sub5_sess25 sub13_sess25 sub16_sess25 sub19_sess25 sub26_sess25
  sub27_sess25 sub29_sess25
)

# Class pairs <a>v<b>: score classes a and b of a multiclass task as a binary task.
CLASS_PAIRS=(0v1 0v2 0v3 0v4 1v2 1v3 1v4 2v3 2v4 3v4)

# Uncomment optional commands below; comment out within-session to run only those.
# Each optional block writes to its own --output-group.

# ── Within-session ────────────────────────────────────────────────────────────
# The main Miller table: 153 binary cells, then 25 multiclass cells.
python -m imindbench.launch \
  --dataset "$DATASET" \
  --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
  --task "${TASKS[@]}" --target "${TARGETS[@]}" \
  --model "$MODEL" --preprocessor "$PREPROCESSOR" \
  --experiment "$EXPERIMENT" \
  --cells binary \
  --output-group within_session_binary
python -m imindbench.launch \
  --dataset "$DATASET" \
  --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
  --task "${MULTICLASS_TASKS[@]}" --target "${TARGETS[@]}" \
  --model "$MODEL" --preprocessor "$PREPROCESSOR" \
  --experiment "$EXPERIMENT" \
  --set dataset.label_mode=multiclass \
  --cells multiclass \
  --output-group within_session_multiclass

# ── New task sets ──────────────────────────────────────────────────────────────
# Optional: binary cells of task sets 10-11. Not part of the main table.
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${NEW_TASKS[@]}" --target "${NEW_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --cells new_sets_binary \
#   --output-group new_sets_binary

# ── Controls ───────────────────────────────────────────────────────────────────
# Optional: control tasks whose labels follow time in the session, so a good
# score does not show decoding. Report them apart from every other table.
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${CONTROL_TASKS[@]}" --target "${NEW_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --cells controls_binary \
#   --output-group controls_binary
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${MULTICLASS_CONTROL_TASKS[@]}" --target "${NEW_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.label_mode=multiclass \
#   --cells controls_multiclass \
#   --output-group controls_multiclass

# ── Class pairs ────────────────────────────────────────────────────────────────
# Optional: each multiclass cell scored one pair of classes at a time. Class a
# becomes 0 and class b becomes 1. Each pair writes to its own output group.
# for PAIR in "${CLASS_PAIRS[@]}"; do
#   python -m imindbench.launch \
#     --dataset "$DATASET" \
#     --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#     --task "${MULTICLASS_TASKS[@]}" --target "${TARGETS[@]}" \
#     --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#     --experiment "$EXPERIMENT" \
#     --set dataset.label_mode=multiclass \
#     --set "dataset.class_pair=[${PAIR%v*},${PAIR#*v}]" \
#     --cells "class_pair_${PAIR}" \
#     --output-group "class_pair_${PAIR}"
# done

# ── Regression ─────────────────────────────────────────────────────────────────
# Optional: predict each 1.0 s window's 40 Hz target trajectory. Scored with
# Pearson r (traj_r) instead of AUROC.
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${REGRESSION_TASKS[@]}" --target "${REGRESSION_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.label_mode=regression \
#   --cells regression \
#   --output-group regression

# ── Regression sliding ─────────────────────────────────────────────────────────
# Optional: 1.0 s windows sliding in 50 ms steps; each window is scored on its
# last 2 trajectory samples (its newest 50 ms).
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${REGRESSION_TASKS[@]}" --target "${SLIDING_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.label_mode=regression \
#   --set dataset.regression_target_last_samples=2 \
#   --cells regression_sliding \
#   --output-group regression_sliding

# ── Regression BCI-IV ──────────────────────────────────────────────────────────
# Optional: finger flexion with the BCI Competition IV split (train on the first
# 400 s, test on the rest). This task set has a single fold, fold 0.
# python -m imindbench.launch \
#   --dataset "$DATASET" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${BCI4_TASKS[@]}" --target "${BCI4_TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.label_mode=regression \
#   --set dataset.regression_target_last_samples=1 \
#   --set "dataset.fold_subset=[0]" \
#   --cells regression_bci4 \
#   --output-group regression_bci4

# ── Positions ──────────────────────────────────────────────────────────────────
# Optional: the main table with real MNI152 electrode positions. Needs the Miller
# build with positions (see imindbench/conf/dataset/millerecog2019_pos.yaml) and
# runs only the cells whose recording has trustworthy positions. DIVER's model
# config sets its own coordinate profile, so DIVER needs the extra --set.
# POSITION_SETTINGS=()
# if [[ "$MODEL" == diver ]]; then
#   POSITION_SETTINGS=(--set dataset.coordinate_profile=diver_mni_miller)
# fi
# python -m imindbench.launch \
#   --dataset "${DATASET}_pos" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${TASKS[@]}" --target "${TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   "${POSITION_SETTINGS[@]}" \
#   --cells positions_binary \
#   --output-group positions_binary
# python -m imindbench.launch \
#   --dataset "${DATASET}_pos" \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task "${MULTICLASS_TASKS[@]}" --target "${TARGETS[@]}" \
#   --model "$MODEL" --preprocessor "$PREPROCESSOR" \
#   --experiment "$EXPERIMENT" \
#   "${POSITION_SETTINGS[@]}" \
#   --set dataset.label_mode=multiclass \
#   --cells positions_multiclass \
#   --output-group positions_multiclass
