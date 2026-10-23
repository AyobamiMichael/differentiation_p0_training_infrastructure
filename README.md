# Differentiation World — P0 Training Infrastructure (v2)

Input `[OP_i, OP_j, STATE_x, QUERY]`, target `f_j(f_i(x))`. State-token IDs are randomised per task seed.
Encoder is **bidirectional (unmasked)**; only the QUERY position is read out.

## Install and verify (no GPU, no torch needed for the checks)
```bash
pip install -r requirements.txt
python tests/check_world.py          # split, ID holdout, sampler, baselines, utils  (numpy only)
python tests/simulate_result.py      # result assembly + strict JSON + summarize.py   (numpy only)
```

## Evaluation sets (logged every `--eval-every` epochs)
| set   | meaning                                                           | use it for        |
|-------|-------------------------------------------------------------------|-------------------|
| train | trained (pair, state) cells                                       | fit               |
| id    | held-out STATES of trained composite functions (per-function)     | state generalization |
| val   | held-out composite functions (disjoint signatures)                | ALL selection/tuning |
| test  | held-out composite functions (disjoint signatures)                | final report only |

Splits are by complete composite-function signature; train/val/test signatures are pairwise disjoint and every
operator appears as first AND second operator in >= `--min-slot-coverage` (default 8) training pairs. With the
default `--test-fraction 0.2` and `--split-seed 0` the test set is identical to the v1 test set (50 pairs);
validation pairs are carved out of the v1 training pairs.
The held-out set depends on `--split-seed` ONLY (not on the task seed): vary `--split-seed` to vary which
functions are OOD.

## Sampler and conditions (budgets matched)
* `--condition fixed`: full table of allowed cells every epoch. This is also the **no-shift control**.
* `--condition changing --q-early 0.9 --q-late 0.1 --shift-epoch 300 [--ramp-epochs R]`: pairs with D1/D2 as
  first or second operator are resampled so shared-region exposure of the D operand follows q(epoch)
  (operand is x for first position, z = f_i(x) for second). Every pair keeps exactly the same example count,
  so operator marginals are identical to `fixed`.
* `q = 0.5` is the unmodified table: `changing --q-early 0.5 --q-late 0.5` == `fixed` exactly.
* Constant worlds: `--q-early 0.9 --q-late 0.9` (high overlap), `--q-early 0.1 --q-late 0.1` (low overlap).
* Realised exposure and unique-cell coverage are logged in every history row (`exposure`).

## Running
```bash
python run_p0.py --mode smoke                                    # 3 epochs
python run_p0.py --mode sweep --estimate-only                    # prints run count / hours before you commit
python run_p0.py --mode sweep --models m1 --weight-decay 0.1 0.3 1.0 --train-seed 0 1 2 --epochs 600 --save-checkpoints --skip-existing --steps-per-sec 50

python run_p0.py --mode train --epochs 60 --id-holdout 0 --warmup-steps 0 --min-lr-ratio 1.0 --weight-decay 0 --eval-every 1 --train-seed 0 1 2 3 4 --outdir runs_diag_a
python run_p0.py --mode train --epochs 60 --id-holdout 0 --warmup-steps 0 --min-lr-ratio 1.0 --weight-decay 0 --eval-every 1 --val-fraction 0.01 --train-seed 0 1 2 3 4 --outdir runs_diag_b

python run_p0.py --mode train --model m1 --weight-decay 0.3 --train-seed 3 4 5 6 7 8 9 --epochs 600 --checkpoint-every 10 --save-checkpoints --skip-existing --steps-per-sec 50


```
 
Weight Decay: 0.3 best learning perfomance accross training seed 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 at 600 epochs

Defaults: 600 epochs (~23k optimizer steps at batch 128), AdamW lr 1e-3, 200 warmup steps, cosine to 0.1x, grad clip 1.0,
weight decay on matrices/embeddings only. `--skip-existing` makes sweeps resumable.

## What each run writes
`metrics.json` (config + hash + data fingerprint + versions, baselines for val and test, per-epoch history for
train/id/val/test incl. calibration and D-region accuracy, per-pair / per-family / per-unique-function breakdown
for val and test, selection by validation), `final_model.pt`, and dense log-spaced checkpoints with `--save-checkpoints`.

## How to read the baselines
All baselines use training tuples only. Compare the network with: chance (1/32); the **shortcut floor**
(`max` of the P(y|x), P(y|i,x), P(y|j,x) probes; it can reach ~6% in some worlds, so the null to beat is
max(chance, shortcut floor)); `rectangle_solver_matched_cells` (information-matched compositional reference);
`rectangle_solver_full_tables` (identifiability ceiling; more information than the network sees).
`composition_lookup_by_signature.coverage` must be 0.

## Status
Data, split, sampler, baselines, metrics and aggregation are unit-tested (numpy only). The torch training loop in
`run_p0.py` has been statically checked (pyflakes) but NOT executed by the author of v2: run `--mode smoke` first.
