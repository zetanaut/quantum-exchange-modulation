# QIP submission revision: frozen numerical record

This directory contains a new, separate numerical study for the QIP revision. It does not replace the historical `results/` records. The starting repository commit is recorded in `run_manifest.json`; the fresh-data design and independent root seeds are in `protocol.json`. All five `confirmation_*` datasets were generated after the development choices were fixed.

## Reproduce the record

Run from the repository root with Python 3.12, NumPy, Torch, and PennyLane versions recorded in `run_manifest.json`. Use `PYTHONPATH=.` and set `OPENBLAS_NUM_THREADS=1` for the matrix fits. The scripts below read the frozen files here and write their outputs back here:

1. `scripts/submission_dataset.py` generates the development and confirmation archives.
2. `scripts/submission_validate_datasets.py` checks density matrices, response identities, state and physical-axis isolation, and exact duplicates.
3. `scripts/submission_product_baselines.py` and `scripts/submission_analytic_head.py` fit linear controls.
4. `scripts/submission_classical_search.py` records the development search. `classical_selection.json` fixes the configurations used by `scripts/submission_classical_confirmation.py`.
5. `scripts/submission_quantum_confirmation.py` fits the compact, matched-QNN, no-commutator, and neutral-initialization controls. `scripts/submission_generic_checks.py` verifies the generic matrix implementation against an independent PennyLane circuit. For the nonlinear conditioner, run `scripts/submission_generic_fit.py --phase development`, then `--phase confirmation` after fixing `generic_selection.json`. Run `scripts/submission_generic_fit.py --phase confirmation --blind --output results/submission_revision/context_blind_fits` for its separate context-blind control.
6. `scripts/submission_validate_checkpoints.py` reloads all classical checkpoints. `scripts/submission_statistics.py`, `scripts/submission_copy_risk.py`, and `scripts/submission_amortization.py` generate paired and source-copy analyses. The ablation copy-risk outputs use `scripts/submission_copy_risk.py --models no_commutator,joint_exchange_neutral --output results/submission_revision/copy_risk_ablations`. `scripts/submission_summarize.py` writes `all_scores.csv` and `summary.json`; `scripts/submission_release_gate.py` checks the complete record.

Each learned run has its own JSON record, including the validation-selected checkpoint and held-out split metrics. Classical Torch checkpoints are saved as `.pt` files. The summary averages optimizer starts within a dataset before averaging the five dataset values. Physical states, rather than state-context rows, are the sampling unit for the paired uncertainty analysis.

## Scope

The target is an analytically known directional-dephasing response. Training and the reported test metrics use exact expectation values. The finite-copy study concerns **inference only** under specified ideal measurement laws. It excludes finite-shot training, hardware noise, source-preparation overhead, labels, optimizer queries, and gradient-estimation costs. The 36-feature Gram product fit and fixed analytic circuit are strong target-informed controls and prevent a general quantum learning-advantage claim. The direct collective observable is an alternative implementation of the same ideal distribution as the analytic exchange estimator.

The revised manuscript and figure-generation sources accompany this directory in the submission bundle. Authors must check all scientific claims and fill the title-page and declaration fields before submission.
