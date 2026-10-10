# Reproduce the results

The repository separates the saved scientific record from new experiments. Recorded data, model parameters, histories, and predictions are already included. Most scripts accept `--output`; choose a new directory for a rerun so the published evidence remains easy to compare. Use the pinned [installation](install.md) and execute commands from the repository root.

## Verification without retraining

```bash
python scripts/run_all.py
python examples/layer_forward.py
python examples/pretrained_inference.py
```

The first command runs 13 groups of independent matrix tests, seven groups of actual PennyLane checks, and the state-valued layer/gradient checks. The reports include signs, physicality, copy degree, global exchange, equivalent gate constructions, parameter derivatives, and two-layer differentiability. [Theory checks](assets/verification/theory_results.json) · [PennyLane checks](assets/verification/pennylane_results.json) · [layer checks](assets/results/verification/layer_checks.json).

The verification workflow in `.github/workflows/verify.yml` also runs the user-facing examples and the independent physics checks. It performs a two-update streaming smoke run; the supplied illustrative training curve contains the separately executed 40-update run. The workflow verifies the implementation and examples rather than retraining the entire comparison suite on every commit.

## Figures and their source commands

| Figure family | Scientific record | Reproduction command |
|---|---|---|
| Main learning results, generalization, measurement cost | `results/summary.*`, quantum/classical/shot records | `python scripts/summarize_results.py` |
| Data-efficiency learning curve | `results/learning_curve/` | `python scripts/learning_curve.py --plot-only` |
| Bell/GHZ response and copy cost | `results/physics_examples/correlation*` | `python examples/physics_examples.py --output results/physics_reproduction` |
| Finite-time purity loss | `results/physics_examples/dephasing_transfer*` | Same physics command |
| QFI/sub-QFI sensitivity | `results/physics_examples/qfi*` | Same physics command |
| Three-spin Gibbs response | `results/physics_examples/spin_gibbs*` | Same physics command |
| One-state directional response map and spectator-purity check | `results/physics_applications/` | `python scripts/physics_application_diagnostics.py --output results/physics_applications_reproduction` |
| Streaming training | `results/streaming_demo/` | `python examples/train_streaming.py --output results/streaming_reproduction` |
| Resource scaling | `results/resources/` | `python scripts/resource_estimate.py --plot --output results/resources_reproduction` |
| Network and exchange schematics | `docs/assets/figures/network_architecture.svg`, `exchange_block.svg` | Editable vector diagrams supplied directly |

Numerical plot scripts write PNG, PDF, and SVG. The two architecture diagrams are static vector sources; they do not represent sampled experimental data. The asset-preparation script uses the explicitly recorded result locations, so a reproduction directory does not silently replace the published figures.

The application diagnostics make no new fits. `directional_response_map.csv` evaluates one fixed `test_joint` state and the frozen saved checkpoint on an exploratory 11-by-21 angular grid, with five actual exchange-QNode spot checks. `spectator_purity.csv` verifies the product-state factorization and spectator-invariant QFI for 21 deterministic states, with three actual exchange-QNode checks. Their manifest records input hashes, versions, exact tolerances, and scope. The published `results/physics_applications/` is the reviewed record; use a separate output directory for a rerun.

## Learning-curve study

```bash
python scripts/learning_curve.py --output results/learning_curve_reproduction
```

This executes the smaller training budgets and verifies/imports the pinned 64-state results. It preserves the full protocol: nested 8/16/32/64-state subsets, all five contexts per included state, the same optimizer seeds and splits, and the complete FiLM width/regularization sweep at each budget. The imported full-size runs are checked by content hashes and checkpoint prediction agreement. The saved execution source and postprocessing provenance accompany the published run.

To refit all sizes, including the full-size points:

```bash
python scripts/learning_curve.py --fresh-full-size --output results/learning_curve_all_fresh
```

Use `--resume` with the same output directory after an interruption. Use `--plot-only --output ...` to regenerate its tables and figures. The script validates the stored protocol and indices before resuming. The original run contains **96 complete model/configuration/seed records**, of which 72 were freshly fitted for the extension and 24 were verified historical fits. [Protocol](assets/results/learning_curve/protocol.json) · [indices](assets/results/learning_curve/indices.json) · [metrics](assets/results/learning_curve/metrics.json) · [independent audit](assets/results/learning_curve/audit.json).

The three seeds vary optimization. The fixed train, validation, and test states do not provide repeated-dataset confidence intervals. Validation uses 96 additional pairs at every size. The primary figure uses the joint holdout and new-state split because the original context-only split changes its seen-state composition under subsampling.

## Refit the primary benchmark

The complete original training sequence is:

```bash
python scripts/run_all.py --train
```

This regenerates the fixed-seed dataset and overwrites the canonical benchmark results. Run it in a separate checkout or a copy of the repository when preserving the supplied record. Add `--reference` to rerun the slower independent NumPy quantum fits as well. The Torch/PennyLane fits are the primary learning results; the independent implementation is a cross-check.

For selected comparisons in new directories:

```bash
python scripts/train_torch.py --output results/quantum_reproduction
python scripts/train_torch_classical.py --output results/film_reproduction
python scripts/run_classical_benchmark.py --output results/classical_reproduction
python scripts/finite_shot_benchmark.py --quantum results/quantum_reproduction --classical results/film_reproduction --output results/shots_reproduction
```

`summarize_results.py` reads the canonical recorded directories. Comparing an alternative run should use its own metrics and predictions or an explicitly updated plotting source; changing a directory name must not implicitly change the scientific claim. If you refit the canonical checkpoints, the learning-curve importer will intentionally reject historical hashes; use `--fresh-full-size` for a newly generated curve.

The finite-shot study holds the trained weights fixed. It combines exact variance formulas with simulated measurement repetitions at matched raw-state-copy budgets. It does not include finite-shot training, device noise, label generation, or a complete hardware-time comparison. The [evidence page](evidence.md) explains the strongest analytic and quantum comparators.

## Regenerate the physics suite

```bash
python examples/physics_examples.py --output results/physics_reproduction
```

The suite includes exact matrix identities, actual exchange QNodes, a Liouvillian exponential, a separate SLD-based QFI calculation, and a phase-kickback global-exchange check. The supplied run passed all 21 declared checks. Every result directory contains its arrays/tables and a manifest that identifies the inputs, normalization conventions, checks, and checkpoint where relevant. [Physics manifest](assets/results/physics_examples/manifest.json) · [check results](assets/results/physics_examples/checks.json).

## Rebuild the documentation

```bash
python -m pip install -r requirements-docs.txt
python scripts/prepare_site_assets.py
python -m mkdocs build --strict
python scripts/check_site.py
```

The first script mirrors recorded source files, numerical evidence, and figures into download assets and writes a SHA-256 inventory. The checker validates local HTML links/anchors, required figure families, and the asset hashes. Website builds do not rerun training. New scientific results should be reviewed, given a precise scope label, and intentionally promoted into the recorded sources before publishing.

Wall-clock times in the JSON reports describe the machine used for that execution. They are useful for estimating a local run, but are not speed comparisons between classical and quantum hardware. Historical metadata can retain the original run's input path; data hashes and repository-relative reproduction commands identify the portable inputs.
