# Quickstart: from a layer to a prediction

Run the following examples after [installing the scientific environment](install.md). Each command works from the repository root, writes a distinct result directory, and prints a short JSON report. The generic streaming network and the compact trained dephasing model share the exchange mechanism but have different parameterizations and readouts.

## 1. Execute a physical state-valued layer

```bash
python examples/layer_forward.py
```

This creates four valid two-qubit mixed states and contexts, executes the nonlinear context-conditioned layer, and checks the trace, eigenvalues, and coefficient constraint. The recorded run has trace error below $2.3\times10^{-16}$ and positive output eigenvalues. It writes `results/layer_forward/states_and_controls.npz` and `metrics.json`. [Recorded report](assets/results/layer_forward/metrics.json) · [script](assets/code/examples/layer_forward.py).

The minimal API is:

```python
import torch
from qem.layers import ContextExchangeLayer, StreamingExchangeNetwork

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

rho = torch.eye(4, dtype=torch.complex128)[None] / 4
context = torch.tensor([[0.2, -0.1, 0.8]])

layer = ContextExchangeLayer(n_qubits=2, context_dim=3, hidden=16)
state = layer.self_modulate(rho, context)  # [1, 4, 4]
controls = layer.gate_parameters(context)

network = StreamingExchangeNetwork(
    n_qubits=2, context_dim=3, hidden=8, depth=2
)
final_state = network(rho, context)
print(network.copies_per_execution)  # 3 independent preparations
```

The maximally mixed state is a simple shape/trace check: it commutes with every rotation and remains unchanged. Use the random-state script to see nontrivial modulation. For a separate memory and source, call `layer(memory, source, context)`. Passing the same tensor twice represents independently prepared copies; no physical state is cloned.

## 2. Evaluate the trained dephasing specialization

```bash
python examples/pretrained_inference.py --split test_joint
```

The script loads the **historical validation-selected seed-31 checkpoint** from `results/torch_quantum/weights.npz`, evaluates 128 new-state/oriented-cap context pairs, and writes predictions and metrics to `results/pretrained_inference/`. The recorded MSE is $3.86735\times10^{-12}$ in exact simulation. Its physical axes overlap training support through antipodes. The [current five-dataset QIP study](qip_revision.md) uses a corrected physical-axis split and retains all starts and new-state outcomes.

For your own two-qubit states, create an NPZ containing:

| Array | Shape and meaning |
|---|---|
| `rho` | `[B,4,4]`, Hermitian, positive, unit-trace density matrices |
| `angles` | `[B,2]`, real polar/azimuthal angles $[\theta,\phi]$ in radians |
| `target` | Optional `[B]` or `[B,1]` real labels for scoring |

The corresponding direction is $\mathbf n=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$, acting on the first qubit. For example, export a supplied split as a custom-input file:

```python
import numpy as np

data = np.load("data/benchmark.npz", allow_pickle=False)
np.savez(
    "my_inference.npz",
    rho=data["test_joint_rho"],
    angles=data["test_joint_angles"],
    target=data["test_joint_y"],
)
```

```bash
python examples/pretrained_inference.py --input my_inference.npz --output results/my_inference
```

This checkpoint predicts the specified dephasing target. A different observable or physical target requires a new model/readout and appropriate labels. [Inference script](assets/code/examples/pretrained_inference.py).

## 3. Train and reload a streaming DNN

```bash
python examples/train_streaming.py
```

The default example trains a two-layer, one-qubit streaming predictor with a nonlinear context conditioner and an affine head on X, Y, and Z expectations. It uses a synthetic teacher fixture with disjoint physical-state splits: 16 training, 8 validation, and 8 test pairs. It runs 40 Adam updates, selects by validation MSE, and then evaluates the test set. It is a software integration example; the teacher and learner belong to the same model family.

The output directory contains `dataset.npz`, `weights.pt`, `history.csv`, `predictions.npz`, `metrics.json`, and the training plot in SVG/PDF/PNG. Checkpoint reload reproduces the saved predictions exactly in the recorded run. [Metrics](assets/results/streaming_demo/metrics.json) · [training script](assets/code/examples/train_streaming.py).

```python
import numpy as np
import torch
from qem.predictors import load_predictor

torch.set_default_dtype(torch.float64)
model = load_predictor("results/streaming_demo/weights.pt")
data = np.load("results/streaming_demo/dataset.npz", allow_pickle=False)
with torch.no_grad():
    prediction = model(data["test_rho"], data["test_context"])
print(prediction.shape)  # [8, 1]
```

The [training guide](training.md) specifies the custom NPZ schema, a training loop, state-based splitting, physical readout choices, and the limits of exact simulator labels.

## 4. Run the physics and resource examples

```bash
python examples/physics_examples.py
python scripts/resource_estimate.py --n-qubits 2 --depth 2 --shots 1000 --settings 3 --plot
```

The physics script executes Bell/GHZ, dephasing-transfer, QFI, and three-spin examples with independent numerical checks. The resource script performs explicit arithmetic for a logical exchange construction. Read [physics examples](physics.md) alongside the figures to understand what was fitted, what was derived, and which comparisons are stronger.

The static [control explorer](control_explorer.md) changes $\eta$, $\chi$, register size, depth, shots, and readout settings without a Python runtime. For full training comparisons and figure regeneration, continue to the [reproduction guide](reproduction.md).
