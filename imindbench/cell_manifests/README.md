# Cell lists

A cell is one task on one recording, written as `sub<S>_sess<T>`. A cell list
says which cells a run covers.

## How the launcher uses a cell list

`--cells NAME|PATH` is an option of `python -m imindbench.launch`.

1. `NAME` (for example `binary`) means the file
   `imindbench/cell_manifests/<dataset.provider>/NAME.json`. Anything ending in
   `.json` or containing `/` is read as a path instead.
2. The launcher first builds every `--task` × `--target` pair, as usual.
3. It then keeps only the pairs listed in the file and drops the rest.
4. Every `--task` must appear in the file. A task with an empty list adds no
   evaluations.
5. When `--decodable-rule` / `--decodable-dir` (or a preset that sets
   `dataset.decodable_subject_sessions_only`) is also used, a pair must be in
   both files.

The file layout is the same as the decodable lists:

```json
{
  "tasks": {
    "move_vs_rest": {"subject_sessions": ["sub3_sess1", "sub7_sess1"]}
  }
}
```

Other top-level keys (`dataset`, `list`, `n_cells`) and per-task keys
(`n_subject_sessions`) are for people reading the file; the launcher ignores
them.

## Why Miller needs cell lists

In the other datasets every task exists on every recording, so the run scripts
run the full task × target grid. In the Miller ECoG library each task set has
its own tasks, not every subject did every task set, and a task only becomes a
cell when the brainsets pipeline found enough windows and trials for each
class. A plain grid would include combinations that do not exist, so the Miller
run script passes `--cells` to every launch.

## Miller lists (`millerecog2019/`)

The packaged lists were converted from the lists Danny Han used for the
original Miller runs (`millerecog2019_pr19*.json`); each file names its source.
Task sets are numbered as in `conf/dataset/millerecog2019.yaml`.

| List | Cells | Run with |
|---|---|---|
| `binary.json` | 153 binary cells of task sets 1-8 | `dataset.label_mode=binary` |
| `multiclass.json` | 25 multiclass cells of task sets 1-8 | `dataset.label_mode=multiclass` |
| `positions_binary.json`, `positions_multiclass.json` | cells of the two lists above whose recording has trustworthy (quality A or B) MNI152 positions | `dataset=millerecog2019_pos` and the matching label mode |
| `class_pair_<a>v<b>.json` | cells of `multiclass.json` with more than `b` classes | `dataset.label_mode=multiclass`, `dataset.class_pair=[a,b]` |
| `new_sets_binary.json` | binary cells of task sets 10 (faces_noise) and 11 (memory_nback), without the control tasks | `dataset.label_mode=binary` |
| `controls_binary.json`, `controls_multiclass.json` | control tasks: their labels follow time in the session, so a good score does not show decoding | the matching label mode |
| `regression.json` | regression targets of task sets 13, 16, 19 (1.0 s windows) | `dataset.label_mode=regression` |
| `regression_w500.json`, `regression_w250.json` | the same targets with 0.5 s and 0.25 s windows | `dataset.label_mode=regression` |
| `regression_sliding.json` | task sets 22-24 (1.0 s windows sliding in 50 ms steps) | `dataset.label_mode=regression`, `dataset.regression_target_last_samples=2` |
| `regression_bci4.json` | task set 25 (BCI Competition IV split, one fold) | `dataset.label_mode=regression`, `dataset.regression_target_last_samples=1`, `dataset.fold_subset=[0]` |

`binary.json` plus `multiclass.json` is the main Miller table: 178 cells.

The face sets with 0.8 s windows (task sets 9 and 12) are not packaged: they
need 0.8 s preprocessing presets, which iMINDBench does not bundle yet. The
builder below writes them as `faces08_binary.json`.

## Rebuilding the Miller lists

The lists come from the prepared H5 files:

```bash
python -m imindbench.cell_manifests.build_millerecog2019 \
    --data-dir <dataset_root>/miller_ecog_library_2019
```

The script reads the subject and task-set tables from the `MillerECoG2019`
dataset module in `torch_brain.datasets`. If the installed torch_brain does
not have that class yet, install the brainsets version (for example the
brainsets fork branch `miller-ecog-modularize`) and add `--from-brainsets`.

The script opens every H5 file, reads the task lists the brainsets pipeline
stored in it (`tasks_json`, `control_tasks_json`, `regression_tasks_json`),
turns each recording id into `sub<S>_sess<T>`, and writes the lists above into
`imindbench/cell_manifests/millerecog2019/`. It prints the number of cells in
each list. The `positions_*` lists are written only when the files carry MNI152 positions;
`--min-position-channels` (default 1) sets how many good channels a recording
must keep. The packaged positions lists come from Danny's tier A/B lists, whose
rule may be stricter, so compare before replacing them.
