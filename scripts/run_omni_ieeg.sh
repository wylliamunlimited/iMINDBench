#!/usr/bin/env bash
set -euo pipefail

# Edit this file, then run: bash scripts/run_omni_ieeg.sh
# One model/input pairing across the five Omni-iEEG tasks, with the official
# single-channel protocol: every (recording, channel, window) is one sample,
# and each task is one pooled cell over the upstream patient-disjoint
# train/test split (target `all`). After the launches, every finished cell is
# scored with imindbench.omni_ieeg_scoring, a literal port of the upstream
# evaluation scripts; see README.md ("Omni-iEEG").
#
# Each task has its own window length, split file and label mode, so every
# task is launched on its own with the settings of omni_task_settings below.

# ── Model and input settings ───────────────────────────────────────────────────
# Valid combinations (names refer to YAML configs under imindbench/conf/).
# PREPROCESSOR names the family of Omni-iEEG presets; the script appends the
# window-length variant each task needs (_w60s, _w300s) and uses
# HFO_PREPROCESSOR for the 2 s HFO events. All omni_* presets apply the
# DIVER-style filters and resample 1000 Hz to 500 Hz.
# MODEL              PREPROCESSOR                               EXPERIMENT
# logistic/mlp/cnn   omni_diverstyle_HPF_noSTD_1000Hzto500Hz    default
# diver              omni_diverstyle_HPF_noSTD_1000Hzto500Hz    default
MODEL=logistic
PREPROCESSOR=omni_diverstyle_HPF_noSTD_1000Hzto500Hz
HFO_PREPROCESSOR=omni_hfo_diverstyle_HPF_noSTD_1000Hzto500Hz
EXPERIMENT=default

# ── Paths and tasks ────────────────────────────────────────────────────────────
# paths/local.yaml must set dataset_root (holding omni_ieeg/ and omni_ieeg_hfo/),
# omni_ictal_split_csv, omni_sleep_split_csv, omni_final_split_csv and
# omni_anatomical_mapping_csv (plus omni_raw_root for scoring soz_outcome);
# see paths/example.yaml.
DATASET=omni_ieeg_single_channel
CONFIG_DIR="/path/to/config"          # Contains paths/local.yaml.
OUTPUT_ROOT="/path/to/runs/omni_ieeg"

# The five learnable upstream tasks, in the order they run.
#   ictal_interictal  60 s windows, binary, ictal_split_cross.csv
#   sleep_awake       60 s windows, binary, sleep_split_cross.csv
#   soz_outcome       60 s windows, binary, unlabelled test channels kept (-1)
#   anatomical        300 s windows, 5 classes, selection by macro-F1
#   hfo               2 s events, 3 classes, needs `brainsets prepare omni_ieeg_hfo`
TASKS=(ictal_interictal sleep_awake soz_outcome anatomical hfo)
# Omni-iEEG has one pooled cell per task: train on the upstream train
# patients, score every test patient.
TARGETS=(all)

# Optional: for models that define model.sub_window_sec (DIVER-style backbones),
# run the backbone on consecutive sub-windows of this length and average the
# features. Leave empty to keep each model's default. The HFO event is one
# unit upstream and is never cut.
SUB_WINDOW_SEC=""

# ── Per-task settings ──────────────────────────────────────────────────────────
omni_window_sec() {  # $1 task
  case "$1" in
    anatomical) echo 300 ;;
    hfo) echo 2 ;;
    *) echo 60 ;;
  esac
}

omni_preprocessor() {  # $1 task
  case "$1" in
    hfo) echo "$HFO_PREPROCESSOR" ;;
    *) echo "${PREPROCESSOR}_w$(omni_window_sec "$1")s" ;;
  esac
}

# The Hydra settings of one task, as --set arguments. Window lengths, channel
# selection and labels follow the upstream extractors (see the OmniIEEG dataset
# class); model selection uses val AUROC for the binary tasks and val macro-F1
# for the multiclass ones, their headline metrics.
omni_task_settings() {  # $1 task
  local window
  window=$(omni_window_sec "$1")
  echo --set "dataset.provider_options.window_sec=${window}"
  case "$1" in
    ictal_interictal)
      echo --set 'dataset.provider_options.split_csv=${paths.omni_ictal_split_csv}'
      echo --set dataset.label_mode=binary --set ++model.selection_metric=roc_auc ;;
    sleep_awake)
      echo --set 'dataset.provider_options.split_csv=${paths.omni_sleep_split_csv}'
      echo --set dataset.label_mode=binary --set ++model.selection_metric=roc_auc ;;
    soz_outcome)
      echo --set dataset.label_mode=binary --set ++model.selection_metric=roc_auc
      echo --set runner.ignore_label=-1 ;;
    anatomical)
      echo --set dataset.label_mode=multiclass --set ++model.selection_metric=f1 ;;
    hfo)
      echo --set dataset.provider_options.edge_trim_sec=0
      echo --set dataset.label_mode=multiclass --set ++model.selection_metric=f1 ;;
    *) echo "unknown Omni-iEEG task '$1' (expected: ictal_interictal sleep_awake soz_outcome anatomical hfo)" >&2; return 1 ;;
  esac
  if [[ -n "$SUB_WINDOW_SEC" && "$1" != hfo ]]; then
    echo --set "++model.sub_window_sec=${SUB_WINDOW_SEC}"
  fi
}

# Fail before any preprocessing when SUB_WINDOW_SEC does not tile a task's
# window exactly (the backbone would otherwise fail inside its first forward).
# Sub-windows are counted in 0.1 s patches at 500 Hz, DIVER's patch length.
check_sub_window() {  # $1.. tasks
  [[ -n "$SUB_WINDOW_SEC" ]] || return 0
  local task window patches per_chunk
  per_chunk=$(python -c "import sys; v=float(sys.argv[1]); n=round(v*10)
sys.exit('SUB_WINDOW_SEC must be a positive multiple of 0.1 s, got %s' % v) if n <= 0 or abs(v*10-n) > 1e-6 else print(n)" "$SUB_WINDOW_SEC")
  for task in "$@"; do
    [[ "$task" == hfo ]] && continue
    window=$(omni_window_sec "$task")
    patches=$((window * 10))
    if (( patches > per_chunk && patches % per_chunk != 0 )); then
      echo "SUB_WINDOW_SEC=${SUB_WINDOW_SEC}s does not tile the ${window}s window of task '${task}'" >&2
      return 1
    fi
  done
}
check_sub_window "${TASKS[@]}"

# ── Single-channel (default) ───────────────────────────────────────────────────
# The official protocol: five pooled cells, one per task. Each run writes its
# per-channel-window test predictions (with channel names and logits) next to
# its result JSON, which the scorer below needs. The train sampler draws the
# classes equally often, as the upstream training code does.
for TASK in "${TASKS[@]}"; do
  # shellcheck disable=SC2046
  python -m imindbench.launch \
    --dataset "$DATASET" \
    --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
    --task "$TASK" --target "${TARGETS[@]}" \
    --regime hold-out-subject \
    --model "$MODEL" --preprocessor "$(omni_preprocessor "$TASK")" \
    --experiment "$EXPERIMENT" \
    $(omni_task_settings "$TASK") \
    --set runtime.save_test_predictions=true \
    --set runner.train_sampler=balanced \
    --output-group single_channel
done

# ── Scoring ────────────────────────────────────────────────────────────────────
# Omni-iEEG metrics of every finished single-channel cell (omni_sc_metrics.json
# next to its result JSON). The per-channel-window AUROC in the result JSON is
# not an Omni-iEEG metric; report the scorer's numbers. soz_outcome also needs
# paths.omni_final_split_csv and paths.omni_raw_root (the raw BIDS tree).
for TASK in "${TASKS[@]}"; do
  for TARGET in "${TARGETS[@]}"; do
    CELL_DIR="${OUTPUT_ROOT}/single_channel/${MODEL}_$(omni_preprocessor "$TASK")/hold-out-subject/${TASK}/${TARGET}"
    if [[ -f "${CELL_DIR}/test_predictions.csv" && ! -f "${CELL_DIR}/omni_sc_metrics.json" ]]; then
      python -m imindbench.omni_ieeg_scoring "$CELL_DIR" --task "$TASK"
    fi
  done
done

# ── Multi-channel benchmark ────────────────────────────────────────────────────
# Optional: ictal vs interictal with every channel of a 1 s window as one
# sample (dataset=omni_ieeg), scored by the usual per-window test AUROC. Not
# an Omni-iEEG protocol; comparable with the other datasets' within-session
# numbers.
# python -m imindbench.launch \
#   --dataset omni_ieeg \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task ictal_interictal --target "${TARGETS[@]}" \
#   --regime hold-out-subject \
#   --model "$MODEL" --preprocessor "${PREPROCESSOR}_w1s" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.provider_options.window_sec=1 \
#   --output-group multi_channel_benchmark

# ── Multi-channel paper ────────────────────────────────────────────────────────
# Optional: ictal vs interictal on whole 60 s windows (the Omni-iEEG decision
# unit), then imindbench.omni_ieeg_paper_scoring for Omni-iEEG-comparable AUC,
# macro-F1 and balanced accuracy at the Youden threshold.
# python -m imindbench.launch \
#   --dataset omni_ieeg \
#   --config-dir "$CONFIG_DIR" --paths local --output-root "$OUTPUT_ROOT" \
#   --task ictal_interictal --target "${TARGETS[@]}" \
#   --regime hold-out-subject \
#   --model "$MODEL" --preprocessor "${PREPROCESSOR}_w60s" \
#   --experiment "$EXPERIMENT" \
#   --set dataset.provider_options.window_sec=60 \
#   --set runtime.save_test_predictions=true \
#   --output-group multi_channel_paper
# CELL_DIR="${OUTPUT_ROOT}/multi_channel_paper/${MODEL}_${PREPROCESSOR}_w60s/hold-out-subject/ictal_interictal/all"
# if [[ -f "${CELL_DIR}/test_predictions.csv" ]]; then
#   python -m imindbench.omni_ieeg_paper_scoring "$CELL_DIR"
# fi
