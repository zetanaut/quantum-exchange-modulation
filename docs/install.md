# Installation

The recorded experiments use **Python 3.12, CPU Torch 2.14.1, PennyLane 0.45.1, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8**. These exact scientific dependencies are pinned in `requirements.txt`; the documentation dependencies are separate. [Recorded scientific environment](assets/results/environment.json).

## Scientific environment

Run these commands from the repository root on Linux or a compatible CPU environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. On macOS, use the standard PyPI Torch wheel rather than the Linux CPU-wheel command:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The local validation reported here ran on Linux CPU. Other operating systems can use the same source, but their numerical runtime and packaging should be checked with the verification commands. A GPU is not required. The reusable state-valued layer uses PennyLane's dense `default.mixed` simulator and was exercised on CPU.

Verify the installation:

```bash
python scripts/run_all.py
python examples/layer_forward.py
python examples/pretrained_inference.py
```

`run_all.py` verifies the matrix identities, actual PennyLane circuits and gradients, and two-layer state-valued composition. It does not retrain the benchmark unless you add `--train`. The inference example loads the supplied checkpoint and evaluates 128 held-out pairs. See [quickstart](quickstart.md) for the outputs.

## Precision and input conventions

Call this before constructing and executing the reusable state-valued layers:

```python
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
```

Use complex128 density matrices and real float64 contexts. The pinned PennyLane `default.mixed` implementation constructs internal Kraus tensors using the Torch default precision. The layer therefore checks the global default and raises a clear error if it is not float64. The explicit per-example QNode loop also avoids a batched density-matrix initialization issue in this pinned version. These choices are recorded implementation constraints, not requirements of the underlying exchange physics. The official [PennyLane Torch interface guide](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html) explains how QNode outputs participate in autograd.

Start with one or two qubits per register. A dense state on two $n$-qubit registers contains $16^n$ complex entries, before temporary arrays and autograd storage. The [resource explorer](control_explorer.md) separates simulator memory from logical hardware costs.

## Documentation environment

The website can be built without Torch, PennyLane, or a quantum device:

```bash
python -m pip install -r requirements-docs.txt
python scripts/prepare_site_assets.py
python -m mkdocs build --strict
python scripts/check_site.py
python -m mkdocs serve
```

The public site is a static MkDocs Material site. Search, navigation, equations, downloadable files, and the analytic control explorer run in the browser. MathJax is loaded from the pinned CDN URL in `mkdocs.yml`; equation rendering therefore needs network access to that CDN. The scientific scripts execute in your local Python environment or CI. [MkDocs documentation](https://www.mkdocs.org/) and [GitHub Pages setup](deploy.md).

## Troubleshooting

| Symptom | Concrete action |
|---|---|
| `ModuleNotFoundError` for `qem` | Activate the intended environment, run commands from the repository root, and install with `python -m pip install -e . --no-deps`. |
| Float64 runtime error | Set `torch.set_default_dtype(torch.float64)` before executing the layer. |
| Density-matrix/context shape error | Match `[batch, 2**n, 2**n]` with `[batch, context_dim]`; see the [training schema](training.md). |
| Memory use grows rapidly | Reduce register size or batch size; the generic simulator explicitly represents the joint density matrix. |
| Existing learning-curve output | Choose a new `--output` directory, `--resume`, or `--plot-only`; the script protects its recorded runs from an accidental restart. |
| Site build reports a missing asset | Run `python scripts/prepare_site_assets.py` before rebuilding. |
| Pages workflow cannot configure deployment | Set Pages' build source to GitHub Actions; follow the [deployment instructions](deploy.md). |
