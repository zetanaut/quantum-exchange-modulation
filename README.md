# Context-controlled exchange modulation

A reproducible **PyTorch + PennyLane** research prototype for conditioning quantum-state features on several classical context variables. A quantum memory collides with a freshly prepared, context-rotated source state. A nonlinear Torch conditioner chooses the rotation and the physically allowed exchange coefficients.

![Streaming quantum-network architecture](docs/assets/figures/network_architecture.svg)

The retained memory obeys

$$
\omega' = a\omega+(1-a)\tau+i b[\omega,\tau],\qquad
\tau=V(c)\rho V(c)^\dagger,\qquad b^2\leq a(1-a).
$$

The repository contains a reusable state-valued network, a measured-observable regression head, a compact two-copy dephasing specialization, trained checkpoints, comparisons, and five physics-example plots. The website explains the mathematics, code, learning protocol, hardware accounting, and deployment in detail.

## Start here

Use Python 3.12. The recorded environment uses CPU Torch 2.14.1 and PennyLane 0.45.1.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/run_all.py
python examples/layer_forward.py
python examples/pretrained_inference.py
python examples/train_streaming.py
```

[Installation](docs/install.md), [quickstart](docs/quickstart.md), [custom-data training](docs/training.md), and [reproduction commands](docs/reproduction.md) explain the inputs, outputs, and alternatives.

## What the evidence supports

The directional-dephasing study tests new quantum states and held-out context directions. With 8 training states / 40 training pairs, the structured exchange model reaches a mean test MSE of **9.31e-6**, the matched two-copy QNN **1.16e-6**, and the validation-selected FiLM baseline **0.0839** across three optimizer seeds. Every size also uses 96 validation pairs. At larger recorded sizes both quantum architectures reach very small errors. The known Gram formula solves the same target exactly when its Pauli features are available.

These results support the value of the chosen physical structure for this benchmark. The [results page](docs/evidence.md) keeps the matched quantum model, analytic comparator, commutator ablation, seed variability, measurement costs, and experimental limits visible. The [learning-curve record](results/learning_curve/metrics.json) contains every run.

![Recorded data-efficiency study](results/learning_curve/learning_curve.svg)

The examples cover Bell/GHZ correlations hidden from single-qubit marginals, finite-time dephasing purity loss, the established sub-QFI sensitivity bound, and a three-spin Gibbs model. These examples have explicit scope labels: trained prediction, analytic transfer, exact diagnostic, or software integration. See [physics examples](docs/physics.md) and [references](docs/references.md).

## Build the website

```bash
python -m pip install -r requirements-docs.txt
python scripts/prepare_site_assets.py
python -m mkdocs build --strict
python scripts/check_site.py
python -m mkdocs serve
```

Open the address printed by MkDocs. The static site contains 11 documentation pages, equations, 13 figure families, downloadable code and numerical evidence, and an interactive exchange-control and resource explorer. Python runs locally or in CI; the browser serves documentation and analytic controls. [Deployment](docs/deploy.md) explains the supplied GitHub Actions workflow.

## Repository map

| Directory | Contents |
|---|---|
| `qem/` | Quantum layers, predictor API, benchmark models, data generator, classical controls |
| `examples/` | Forward execution, frozen inference, supervised streaming training, physics demonstrations |
| `scripts/` | Training, learning curves, finite-shot comparisons, figures, resource and website tools |
| `data/` | The fixed benchmark and split metadata |
| `results/` | Recorded fits, checkpoints, predictions, CSV/JSON data, PNG/PDF/SVG plots |
| `verification/` | Independent matrix identities, actual PennyLane checks, resource certificate |
| `docs/` | Complete MkDocs source and mirrored download assets |
| `.github/workflows/` | Scientific verification and GitHub Pages build/deployment |

All reported experiments use exact simulator expectations for training. Finite-shot results concern inference after that training. Device noise, large-system scaling, and a broad quantum advantage remain research questions. Raw quantum preparations, measurement settings, and gate costs are documented separately from labeled-example counts.
