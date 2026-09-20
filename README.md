# iMINDBench

iEEG Multi-Insitution Neural Decoding Benchmark codebase. Includes preprocessing, models, and evaluation code. Obtain datasets w/ splits from [torch_brain](https://github.com/neuro-galaxy/torch_brain/tree/gc/add-seeg-movie-watching-datasets) (see [Prepare data](#2-prepare-data)).

[![Paper](https://img.shields.io/badge/arXiv-2609.18104-red)](http://arxiv.org/abs/2609.18104)
[![Website](https://img.shields.io/badge/Website-blue)](https://imindbench.github.io/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-orange)](https://imindbench.github.io/leaderboard/)
[![Dataset](https://img.shields.io/badge/Dataset-teal)](https://github.com/neuro-galaxy/torch_brain/tree/gc/add-seeg-movie-watching-datasets)

[Getting started](#getting-started) | [Prepare data](#2-prepare-data) | [Full benchmark](#evaluate-the-complete-benchmark) | [Pretrained weights](#pretrained-weights) | [Customize](#customize) | [Outputs](#outputs) | [Changelog](CHANGELOG.md) | [Citation](#citation)

## Getting started

### 1. Install

CPU evaluation is tested on Python 3.10–3.13; GPU models on newer Python versions
are not yet validated. The default environment uses Python 3.10. From this project directory:

```bash
conda env create -f environment.yml
conda activate imindbench
python -m pip install "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@e39f48ce0ec8c8f59be2507dca8ae172cce79d28"
python -m pip install -e '.[models]'
python -m pip check
```

- Includes the software dependencies for all bundled models.
- The editable installation lets you change code or configs without reinstalling.
- Requires internet access and Git; GPU models also need a compatible CUDA runtime.
- Obtain pretrained weights separately; see [Pretrained weights](#pretrained-weights).

### 2. Prepare data

The first pipeline step is **`brainsets prepare <dataset>`**:

```bash
brainsets prepare neuroprobe_2025 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

- Replace `/path/to/...` with absolute paths outside the checkout.
- Preparation downloads recordings and creates an environment for each pipeline.
- Allow sufficient storage and follow the dataset's terms.

| Dataset | `brainsets prepare` name | Evaluation name | Benchmark subject/session pairs |
| --- | --- | --- | ---: |
| NeuroprobeV2 | `neuroprobe_2025` | `neuroprobev2` | 5 |
| Bang! You're Dead | `keles_byd_2024` | `kelesbyd2024` | 29 |
| PIPPI | `berezutskaya_pippi_2022` | `berezutskayapippi2022` | 5 |

Prepare each dataset into the same processed root to run the complete benchmark.
Each pipeline creates its own named subdirectory. Neuroprobe2025 and NeuroprobeV2
use the same prepared artifacts through different dataset views.

### 3. Set paths

```bash
mkdir -p /path/to/config/paths
cp imindbench/conf/paths/example.yaml /path/to/config/paths/local.yaml
```

Set `dataset_root: /path/to/processed` and `dataset_dirname: neuroprobe_2025`.
BYD and PIPPI configs select their own subdirectories under that root. These two
fields are enough for the first baseline. Add checkpoint/cache paths only when
needed; omitted optional resources inherit packaged `null` defaults.

### 4. Run a first model

Run one CPU Logistic evaluation:

```bash
python -m imindbench.launch --dataset neuroprobev2 --model logistic --experiment default \
  --preprocessor multi_stft_2048Hz --task onset --target sub1_sess1 \
  --device cpu --paths local --config-dir /path/to/config --output-root /path/to/runs/logistic
```

This baseline needs no pretrained weights. Existing output JSONs are skipped.
Remote logging is disabled by default. For full experiments, see below. 

## Evaluate the complete benchmark

We provide scripts to run for each dataset: 
| Script | What it does |
| --- | --- |
| [run_neuroprobev2.sh](scripts/run_neuroprobev2.sh) | NeuroprobeV2: within-session, optional transfer and sample efficiency |
| [run_kelesbyd2024.sh](scripts/run_kelesbyd2024.sh) | BYD: within-session and optional transfer |
| [run_berezutskayapippi2022.sh](scripts/run_berezutskayapippi2022.sh) | PIPPI: within-session and optional transfer |

Open the script for your dataset. Each file lists all benchmark subject/session
pairs and all 15 tasks; the default model is Logistic with multi-STFT inputs.

| Setting | Where to edit it |
| --- | --- |
| Model and inputs | `MODEL` and `PREPROCESSOR` in the script; use its compatibility table |
| Cohort and training-sample caps | `EXPERIMENT=default` or `EXPERIMENT=decodable` in the script |
| Paths and evaluation selection | `CONFIG_DIR`, `OUTPUT_ROOT`, `TASKS` and `TARGETS` in the script |
| Pretrained checkpoints | `popt_checkpoint`, `brainbert_checkpoint`, `barista_checkpoint` or `diver_checkpoint` in `paths/local.yaml` |
| Model architecture and training hyperparameters | `imindbench/conf/model/<MODEL>.yaml` |

After configuring the selected model, run the dataset script:

```bash
bash scripts/run_neuroprobev2.sh
```

**Within-session runs the one pairing you selected** across the listed tasks
and targets. For example, NeuroprobeV2 defaults to:

```bash
MODEL=logistic
PREPROCESSOR=multi_stft_2048Hz
EXPERIMENT=default
```

`EXPERIMENT` selects the cohort and training-sample caps independently of the
model/input pairing:

- `default`: unfiltered cohort, no training-sample cap.
- `decodable`: decodable cohort with automatic training-sample caps.

Training hyperparameters live in the model YAML. Supported inputs are:

| Models | Input preprocessing |
| --- | --- |
| Logistic, MLP, CNN | Multi-STFT or 500 Hz waveforms |
| HTNet | 500 Hz waveforms |
| PopT-v2 (`popt` config) | Multi-STFT matching its checkpoint |
| BrainBERT + linear readout | BrainBERT-specific STFT |
| BaRISTA | Session waveforms with its normalization |
| DIVER-1 | Filtered 500 Hz waveforms with 15-second context |

- **Within-session is enabled by default.** The other experiment commands are commented out.
- Uncomment a whole command to include it; comment out within-session to run only another family.
- Every block uses `MODEL` and `PREPROCESSOR` directly.
- For transfer, select `MODEL=popt` and its matching multi-STFT input.
  Transfer blocks use `TRANSFER_EXPERIMENT=decodable`; within-session and sample
  efficiency use `EXPERIMENT`.

| Family block | Scope |
| --- | --- |
| `within_session` | All selected tasks/targets for the chosen model/input pairing |
| `within_dataset` | PopT-v2; training across sessions within the dataset, Main cohort |
| `multi_dataset` | PopT-v2; training across all three datasets, Main cohort |
| `sample_efficiency` | NeuroprobeV2 only; selected Logistic, MLP, CNN or PopT-v2 on multi-STFT, training fractions 1, 1/2, 1/4, 1/8, 1/16 |

**Complete within-session coverage: 15 tasks × 39 subject/session pairs = 585
evaluations per model/input pairing**, before folds.

- Covers the benchmark selection, not every recording in the original datasets.
- Applies no decodable-target filter.
- Scripts own the task/target selections; coverage tests check that launch previews
  match those explicit lists.

<details>
<summary>Smaller runs and cohort details</summary>

**Smaller runs and settings**

- For a smoke test, keep only the desired entries in `TASKS` and `TARGETS`.
  Keep at least one task and subject/session pair.
- Set training options such as `max_iter` in the model YAML.
  PyTorch `training_mode` accepts `epoch_based` or `steps_based` (default:
  `epoch_based`); `optimizer` names are case-sensitive PyTorch optimizer classes
  such as `Adam`, `AdamW` or `SGD` (default: `Adam`). Invalid names fail before
  evaluation instead of silently selecting a different training setup.
- Logistic, MLP, CNN and all HTNet model configs use `tol: 1e-4` across inputs.
  Logistic uses it for optimizer convergence; epoch-based Torch training uses
  it as the minimum validation-score improvement (ROC AUC by default) to save
  a checkpoint and reset early-stopping patience. Model YAMLs own this setting;
  the bundled experiment presets inherit it.
  This standardizes the former Logistic `1e-3` and MLP `1e-8` defaults, so new
  runs can differ in iteration count, stopping epoch or selected checkpoint.
- `runtime.deterministic: true` requests deterministic Torch execution for all
  inputs. Set it to `false` to opt out; `model.deterministic` has been removed.
  The former `waveform500` experiment is now `default`.
- Store checkpoint paths in `paths/local.yaml`. DIVER also needs a writable
  `diver_shape_cache_dir` in that same file.
- Scripts run each enabled block serially and skip valid existing output JSONs.
  If a block reports failures, the script stops before the next block.
- Use a new output root after changing settings.

The shared runtime config defaults to 4 data-loader workers, pinned memory,
persistent workers, and 6 preprocessing Torch threads, matching the former
Multi-STFT execution preset. These now apply to every model/input pairing.
Adjust `runner.num_workers`, `runner.pin_memory`, `runner.persistent_workers`,
and `runtime.preprocess_torch_num_threads` for your machine. Use
`experiment=default` in place of the former `multi_stft/*` experiments.

`runtime.sklearn_num_threads` caps each loaded BLAS/OpenMP pool during fitting
(default: 4). Smaller existing limits, including those set by the environment,
are preserved; previous limits are restored afterward. This avoids enlarging
small OpenBLAS pools, which can crash SciPy's L-BFGS solver on some builds.

**Preprocessing**

The bundled collection contains 18 presets: the 10 main input presets listed in
the dataset scripts, plus 8 variants selected for the paper notebooks' final
plots. Each additional family below has both `1000` and `2048` Hz versions; replace `{rate}`
with the dataset's native rate (BYD: 1000; NeuroprobeV2 and PIPPI: 2048).

All bundled presets use Laplacian referencing, so filenames omit the
`laplacian_` prefix. Spectral preset names are `stft_{rate}Hz`,
`stft_brainbert_{rate}Hz`, `multi_stft_{rate}Hz`, and
`multi_stft_zscore_{rate}Hz`. Update existing commands by dropping the prefix and
moving the Multi-STFT `zscore` suffix before the rate (for example,
`multi_stft_2048Hz_zscore` becomes `multi_stft_zscore_2048Hz`). Processing settings
are unchanged. Historical result folders retain their old names; new runs use
the shorter names. Old preset aliases are not bundled.

Chain configs contain an ordered `chain:` list; each stage has its own `name:`.
Remove the old top-level `name:` from external chain configs. Single-stage
configs still require `name:`. Logs and result descriptions show the ordered
stage names, and result metadata retains the full preprocessing configuration.
Removing the old field changes cache identities; existing preprocessing caches
will be rebuilt (or must be refreshed before using `read_only` cache mode).

The old combined `name: laplacian_stft` stage has been removed. External chains
should use `laplacian_rereference` followed by `stft`, with each stage's settings
on its own entry. The rereferencing implementation now lives in
`imindbench.preprocessors.laplacian_rereference_preprocessor`.

| Paper variant | Preprocessor config (without `.yaml`) |
| --- | --- |
| Single-STFT | `stft_{rate}Hz` |
| Multi-STFT with per-sample, per-channel normalization | `multi_stft_zscore_{rate}Hz` |
| 500 Hz waveform with high-pass filtering and per-sample, per-channel normalization | `wav_hpf_zscore_{rate}to500Hz` |
| 500 Hz waveform without high-pass filtering, with robust scaling | `wav_nohpf_robust_{rate}to500Hz` |

Waveform names use `wav_<recipe>_<source>[to<target>]Hz`. A single rate
means no resampling. Every bundled waveform recipe uses Laplacian referencing;
filter and normalization details are explicit in its YAML.

| Recipe | Filtering/context | Normalization |
| --- | --- | --- |
| `hpf_robust` | Notch + high-pass, 15-second context, cropped to target window | Global robust scaling fitted on training data |
| `hpf_zscore` | Same filtering/context as `hpf_robust` | Per-sample, per-channel z-score |
| `nohpf_robust` | Notch without high-pass, 15-second context, cropped to target window | Global robust scaling fitted on training data |
| `diver` | DIVER filter, 15-second context, cropped to target window | No standardization stage; DIVER applies its input scaling |
| `barista` | Session-wise notch + high-pass filtering, 2048 Hz output | Global robust scaling followed by per-sample, per-channel z-score |

Selection follows the notebooks' active config choices and final model filters,
excluding commented alternatives and hidden series:

| Notebook selection | Retained inputs |
| --- | --- |
| Figure 4 preprocessing baselines, after `HIDE_LAST_PLOT_MODELS` | Single-STFT, standard/z-scored Multi-STFT, and the three 500 Hz baseline waveform recipes |
| Figure 4 STFT sweeps/selection and Appendix 1 challenge-unit comparison | Single-STFT sweeps and the main 500 Hz waveform recipe |
| Appendix 3 coverage, with `MODEL_TO_PLOT = 'Logistic (multi-STFT)'` | Standard Multi-STFT |
| Other scorecards, task breakouts, scaling, sample-efficiency, and input visualization | Main presets and single-STFT |

The native-rate waveform presets (`wav_nohpf_pooled_{rate}Hz`, previously
`laplacian_wav_{rate}Hz`) and 400 Hz Multi-STFT presets
(`laplacian_multi_stft_high_{rate}Hz`) have been removed: their entries are
hidden or unselected in the final plots. The retained single-STFT sweeps can
still vary their frequency limit independently.

HTNet uses `model=htnet_500Hz` with the 500 Hz waveform presets. The native-rate
`htnet_1000Hz` and `htnet_2048Hz` model presets have also been removed: all final
paper notebook selections use the 500 Hz model.

<details>
<summary>Previous waveform names and migration</summary>

These are filename-only renames: preprocessing chains are unchanged. New run
folders use the new names; existing paper result folders and notebook paths
retain their historical names. Update external commands/config references using
this mapping. Old preset aliases are not bundled.

| Previous name (without `.yaml`) | New name |
| --- | --- |
| `laplacian_wav_HPF_global_robust_scalar_long_context_15s_1000Hzto500Hz` | `wav_hpf_robust_1000to500Hz` |
| `laplacian_wav_HPF_global_robust_scalar_long_context_15s_2048Hzto500Hz` | `wav_hpf_robust_2048to500Hz` |
| `laplacian_wav_HPF_sample_per_channel_time_long_context_15s_1000Hzto500Hz` | `wav_hpf_zscore_1000to500Hz` |
| `laplacian_wav_HPF_sample_per_channel_time_long_context_15s_2048Hzto500Hz` | `wav_hpf_zscore_2048to500Hz` |
| `laplacian_wav_diverstyle_HPF_noSTD_long_context_15s_1000Hzto500Hz` | `wav_diver_1000to500Hz` |
| `laplacian_wav_diverstyle_HPF_noSTD_long_context_15s_2048Hzto500Hz` | `wav_diver_2048to500Hz` |
| `laplacian_wav_global_robust_scalar_long_context_15s_1000Hzto500Hz` | `wav_nohpf_robust_1000to500Hz` |
| `laplacian_wav_global_robust_scalar_long_context_15s_2048Hzto500Hz` | `wav_nohpf_robust_2048to500Hz` |
| `laplacian_wav_session_HPF_global_robust_scalar_1000Hz_2048Hz_zscore` | `wav_barista_1000to2048Hz` |
| `laplacian_wav_session_HPF_global_robust_scalar_2048Hz_zscore` | `wav_barista_2048Hz` |

</details>

The audited notebooks live in
`torch_brain/examples/neuroprobe_eval/notebooks/paper_figs`. The restored z-scored
Multi-STFT variants match the preprocessing settings in the corresponding saved
`09_neurips/{dataset}/logistic_laplacian_multi_stft*/.../.hydra/config.yaml` runs,
with Torch padding made explicit. The long-context waveform variants come from
the original evaluation configs, migrated to `resample`. Historical DIVER output
folders ending in `1000to500` or `2048to500` correspond to the main DIVER presets
named `wav_diver_1000to500Hz` or `wav_diver_2048to500Hz`.

Other bundled variations have been removed; their YAMLs remain in Git history.
For an additional ablation, copy a retained config into your external config
directory and override its settings. Keeping a paper recipe available does not
establish numerical parity with historical runs made with older implementations.

| Input | Processing |
| --- | --- |
| Multi-STFT | Uses the dataset's native sampling rate |
| 500 Hz waveform baselines | Filter with 15-second context, crop to the target window, apply Laplacian referencing, downsample, and fit robust scaling on training data |

For multi-dataset training, `dataset.train_sources[].preprocessor` selects a
bundled preprocessor preset by filename without `.yaml`. Omit it to inherit the
top-level pipeline. Individual stage names such as `raw` are not preset names.

**Window slicing and historical reproduction**

`dataset.window_slicing_policy` applies to evaluation windows and context-window
reads across all splits and training sources:

- `ceil` (default): the current TorchBrain behavior, snapping near-grid timestamps
  before rounding both boundaries up.
- `legacy_floor`: floor both boundaries without snapping, relative to the
  recording's time origin, matching the former historical diagnostic launcher.

Historical mode preserves the existing context placement and crop calculations;
it changes waveform reads, as the former diagnostic did. It does not establish
parity for unrelated preprocessing or training changes.

Select `dataset.window_slicing_policy=legacy_floor` through the normal evaluator
or launcher. Logs and result JSON (`config.window_slicing_policy`) record the
selection. Both preprocessing caches include the policy, and their versions have
changed to exclude old entries that did not record it. Rebuild caches before
using `read_only` mode. Caching can be enabled for either policy after this update.
Use a new output root when changing policy, since completed result JSONs are
still skipped independently of cache identity.

STFT and multi-STFT use Torch with centered windows, `pad_mode: reflect` and
`padded: false` in the bundled presets. SciPy STFT is no longer supported; remove
legacy `use_scipy` and `boundary` keys from external configs. SciPy is still used
for filtering and resampling.

Waveform rate conversion uses one `resample` stage with required positive integer
`source_rate` and `target_rate` values in Hz. It accepts NumPy arrays or Torch
tensors shaped `(channels, time)` and returns float32 NumPy arrays, preserving
channel metadata and setting `sampling_rate` to the target rate. Output length is
`ceil(input_length * target_rate / source_rate)`, matching SciPy's polyphase
resampler. Crop context windows before resampling.

For external preprocessor YAMLs, replace `name: downsample` or `name: upsampler`
with `name: resample` and specify both rates. The old stages have been removed.
Upsampling no longer rounds fractional output lengths to the nearest integer;
it can retain one additional sample. The bundled one-second waveform windows
still produce 500 samples for the waveform baselines and 2048 for BaRISTA.
Use a new output root after migrating; existing result JSONs are still skipped.

**Dataset subsets**

| Dataset | Default subset |
| --- | --- |
| NeuroprobeV2 | `lite` |
| BYD | `full` |
| PIPPI, including DIVER | `high-cov` |

Subset tiers select eligible recordings and prepared splits; `TASKS` and `TARGETS`
select the evaluation grid. For the listed PIPPI within-session targets,
`high-cov` and `full` use identical splits and channels.

BaRISTA's model config selects `dataset.destrieux_brain_area_key` as the active
brain-area field and owns the benchmark learning rates (`upstream_lr: 1e-3`,
`head_lr: 1e-3`) and fixed scheduler (500 warmup updates, decay every 95 updates).
Use `experiment=default` or `experiment=decodable`; the separate `barista`
experiment has been removed.

Selecting `model=diver` selects `dataset.coordinate_profile=diver_mni`.
Its training settings stay in the model YAML; use `default` or `decodable`
in place of the former `diver` experiment.

**Transfer cohorts and sample caps**

The `default` and `decodable` experiment presets expose the same fields:

| Field | `default` | `decodable` |
| --- | --- | --- |
| `paths.decodable_subject_sessions_dir` | `null` | Packaged Main-cohort manifest directory |
| `dataset.label_mode` | `binary` | `binary` |
| `dataset.train_same_subject_only` | `false` | `false` |
| `dataset.train_sample_fraction` | `1.0` | `1.0` |
| `dataset.max_train_samples_per_subject` | `null` | `auto` |
| `dataset.decodable_subject_sessions_only` | `false` | `true` |

`default` is selected when no experiment is specified; scripts can explicitly
select `--experiment default`. It replaces the former `baseline` and
`within_session` presets. Only these two experiment presets are bundled. Model
settings live in model configs; runtime settings live in
`imindbench/conf/runtime/default.yaml`.

| Setting | Within-session / sample efficiency | Within-dataset / multi-dataset |
| --- | --- | --- |
| Training cohort filter | Disabled | Validation-selected Main cohort |
| Evaluation targets | Listed targets | Listed targets filtered per task by the standard manifest |
| Per-subject/session training cap | Disabled | `auto`: capped at the target session's training-sample count |

The `decodable` transfer preset supplies the manifest, sets
`decodable_subject_sessions_only=true`, and enables the sample cap.
The cap is an experiment setting, not a PopT-v2 requirement.
`dataset.decodable_subject_sessions_only` replaces the former
`dataset.train_decodable_subject_sessions_only` field. The launcher applies the
manifest to evaluation targets; the data adapter applies it to training
recordings. Direct `imindbench.run_eval` calls retain their explicit evaluation
target and apply the training filter. Old external YAMLs must rename the field;
validation rejects the retired spelling. Use a fresh output root when migrating.

- To use a custom cohort, add `--decodable-rule NAME` or
  `--decodable-dir /path/to/manifests` to the transfer command.
- The loader calls within-dataset training `hold-in-session`.

</details>

## Pretrained weights

Store model weights outside the checkout and configure their absolute paths.
Model dependencies are included in the installation above.

| Model | Weights | Configuration |
| --- | --- | --- |
| PopT-v2 | Available upon request. | `paths.popt_checkpoint` |
| BrainBERT | [Official weights ZIP](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing), linked by the [upstream project](https://github.com/czlwang/BrainBERT#using-brainbert-embeddings) | `paths.brainbert_checkpoint` |
| BaRISTA | Available upon request. | `paths.barista_checkpoint` |
| DIVER-1 | [Official iEEG checkpoint](https://drive.google.com/file/d/1svTMyxABZ-9kvk-BiiZ6-2sNyZ5io8mg/view), linked by the [upstream project](https://github.com/DIVER-Project/DIVER-1#weights) | `paths.diver_checkpoint` and writable `paths.diver_shape_cache_dir` |

<details>
<summary>Checkpoint compatibility and DIVER configuration</summary>

| Model | Compatibility notes |
| --- | --- |
| PopT-v2 | Use the multi-STFT weights shared upon request. Accepts `model_cfg`/`model` or `config`/`model_state` checkpoint dictionaries. |
| BrainBERT | Expects the upstream `model_cfg`/`model` format. |
| BaRISTA | Use the weights shared upon request and prepared Destrieux metadata. The `barista` preset selects `localization_Destrieux` for NeuroprobeV2 or `label_destrieux` for BYD/PIPPI. |

For DIVER, select its table entry and add these fields to `paths/local.yaml`:

```yaml
diver_checkpoint: /path/to/ieeg_checkpoint.pt
diver_shape_cache_dir: /path/to/diver_shapes
```

- **Default architecture:** width 256, depth 12, patch size 50; DeepSpeed `module` format.
- **Other checkpoints:** match their architecture. Set `model.deepspeed_pth_format=false`
  for `model_state_dict` format.
- **Validation:** the linked checkpoint passed strict weight loading and a synthetic
  CPU forward pass with fresh and existing shape caches.
- Record the checkpoint hash with your results.

</details>

## Customize

```text
brainsets prepare → prepared recordings and labels → dataset task/split
  → preprocessor chain → model training/evaluation → result JSON and logs
```

| What to change | Location |
| --- | --- |
| Dataset and split settings | `imindbench/conf/dataset/` |
| Model settings / implementation | `imindbench/conf/model/` / `imindbench/models/` |
| Preprocessing settings / implementation | `imindbench/conf/preprocessor/` / `imindbench/preprocessors/` |
| Cohort / runtime presets | `imindbench/conf/experiment/` / `imindbench/conf/runtime/` |
| Task and recording selections | `TASKS` and `TARGETS` in each dataset script |
| Launching / single evaluation | `imindbench/launch.py` / `imindbench/run_eval.py` |

Preparation pipelines and dataset loader implementations live in TorchBrain,
under `torch_brain/pipeline/brainsets-pipelines/` and `torch_brain/datasets/`.

<details>
<summary>Change settings with a small YAML file</summary>

Create `/path/to/config/experiment/my_trial.yaml`:

```yaml
# @package _global_
model:
  max_iter: 20
  learning_rate: 0.001
```

Select it through the generic launcher:

```bash
imindbench-grid --dataset neuroprobev2 --model mlp --preprocessor multi_stft_2048Hz --experiment my_trial --task onset --target sub1_sess1 --device cuda:0 --config-dir /path/to/config --paths local --output-root /path/to/runs/my_trial
```

Keep model settings in the YAML file so the configuration is easy to review and
reuse. Dataset scripts select the model, preprocessor and experiment; use
`imindbench-grid` for new combinations.

</details>

<details>
<summary>Compare preprocessors using a baseline model</summary>

Select a different preprocessor config while keeping the model, task and unit
selection fixed. This runs Logistic on one NeuroprobeV2 task/recording:

```bash
imindbench-grid --dataset neuroprobev2 --model logistic --preprocessor stft_2048Hz --experiment default --task onset --target sub1_sess1 --device cpu --config-dir /path/to/config --paths local --output-root /path/to/runs/single_stft
```

- Repeat with `--preprocessor multi_stft_2048Hz` and a different output root.
- Use the 1000 Hz config for BYD and the 2048 Hz config for NeuroprobeV2/PIPPI.
- For a custom chain, copy a compatible YAML to
  `/path/to/config/preprocessor/my_chain.yaml`, edit `chain`, and select
  `--preprocessor my_chain`.

</details>

<details>
<summary>Add your own model</summary>

1. Add `imindbench/models/my_model.py` and register it with `@register_model("my_model")`.
   Modules in this directory are discovered automatically.
2. Implement the interface for your model type:

| Type | Starting point | Interface |
| --- | --- | --- |
| PyTorch | `mlp_model.py` | Subclass `TorchBaseModel`; implement `_create_network(input_shape, n_classes)` and `build_model(input_shape, n_classes, device=None)`. The runner owns training. |
| sklearn | `logistic_model.py` | Implement `fit` and `predict_proba`, set `classes_`, and use `prepare_batch` if input adaptation is needed. Probabilities must follow `classes_` order. |

3. Add `imindbench/conf/model/my_model.yaml` with `name: my_model`, required
   `backend: torch` or `backend: sklearn`, input requirements and training settings.
   The backend selects the runner and batch handling independently of the model name;
   the launcher applies `--device` only to Torch models. Existing external model
   configs must also declare their backend.
4. Select `--model my_model` with the dataset, preprocessor and path arguments
   from the custom experiment example.

Start with one task/target and check input shapes and class probabilities.
Then expand `TASKS` and `TARGETS` in the dataset script for a larger run.
The CLI requires both selections explicitly; automatic all-target expansion and
`--unit-set` have been removed.
Repeat for all three datasets with matching preprocessors.

</details>

<details>
<summary>Add your own preprocessor</summary>

Add a module under `imindbench/preprocessors/` and register its class. For example:

```python
from imindbench.preprocessors import register_preprocessor
from imindbench.preprocessors.base_preprocessor import BasePreprocessor

@register_preprocessor("gain")
class GainPreprocessor(BasePreprocessor):
    def transform_samples(self, samples):
        return [{**sample, "x": sample["x"] * self.cfg.factor} for sample in samples]
```

- Add `- {name: gain, factor: 2.0}` to a copied preprocessor chain.
- Preserve sample metadata; keep shapes, channel labels and coordinates aligned.
- For transforms that learn statistics, use `execution_type = "fold_fit_transform"`
  and implement `fit_split`, `get_state`, `set_state` and `reset_state`.
  Fit on training data only.
- See `standardization_preprocessor.py` for a stateful example.

</details>

## Outputs

Each run writes `population_*.json`, resolved Hydra config, `launch.json` and
`launcher.log`.

| Situation | Behavior |
| --- | --- |
| Valid output JSON exists | Skip the evaluation |
| Output JSON is missing or malformed | Restart the evaluation; no checkpoint resume |
| Settings change | Use a new output root; existing results are reused by filename |
| Concurrent launchers | Use separate output roots |

The launcher accepts `--dry-run` to preview commands or `--count` to count evaluations.

<details>
<summary>Development checks</summary>

```bash
python -m pip install -e '.[models,dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests use synthetic data; optional model checks skip when their dependencies are unavailable.

</details>

See [LICENSE.txt](LICENSE.txt) and [third-party notices](THIRD_PARTY.md).
BaRISTA retains its [upstream license](LICENSES/BaRISTA-LICENSE.md).
PopT retains its [upstream MIT license](LICENSES/PopT-LICENSE.txt), with separate
terms for tutorial, PyTorch, and SciPy portions described in the third-party notices.
DIVER-1 retains its [upstream MIT license](LICENSES/DIVER-1-LICENSE.txt), with
separate Apache-2.0 terms for uni2ts portions described in the third-party notices.

## Citation

```bibtex
@misc{chau2026imindbench,
  title={{iMINDBench}: {iEEG} Multi-Institution Neural Decoding Benchmark},
  author={Geeling Chau and Saba Hashemi and Yonghyeon Gwon and Eshani Patel and Jan DeWitt and Christopher Wang and Andrii Zahorodnii and Sabera J Talukder and Danny Dongyeop Han and Chun Kee Chung and Maryam M Shanechi and Yisong Yue},
  year={2026},
  eprint={2609.18104},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={http://arxiv.org/abs/2609.18104},
}
```
