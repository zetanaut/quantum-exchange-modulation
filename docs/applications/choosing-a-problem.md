# Choose a physics problem and an access model

The exchange network is most natural when an apparatus can repeatedly prepare a quantum state and a classical vector specifies **which response** is requested. A density matrix in the code is a simulator representation of such a preparation. A classical array from an experiment or simulation does not become native quantum data merely because it is encoded into a circuit; encoding digitized NMR traces or scattering cross sections needs its own cost and classical-baseline study.

## Decision guide

| Question | What to establish before using exchange |
|---|---|
| Can the same physical state be prepared independently? | Two-copy response estimation needs at least two source preparations per shot. A depth-$L$ stream initialized from $\rho$ consumes $L+1$. |
| Is a quantum memory and two-copy interaction available? | Full-register exchange couples two equal-size registers; parallel local partial swaps are a different circuit. Include compilation, routing, preparation, and readout. |
| Is the target linear, quadratic, or more complex in $\rho$? | For a fixed channel, ordinary observable expectations are linear. Purity and $D_P$ are quadratic; further targets need a degree/error analysis. |
| Does a known simple estimator already solve it? | Check global Pauli parity, analytic Gram features, direct $K_P$, grouped measurements, and classical shadows before fitting a network. |
| How many contexts are queried for each state? | Reusing measured Pauli or shadow records may amortize preparation cost across directions; a fresh exchange query usually consumes new copies. |
| Which budget is binding? | Count labels, raw preparations, measurement settings, shot variance, logical gates/depth, and hardware noise separately. |

The recorded directional-dephasing benchmark is a worked case: two-spin source states, context $\mathbf n$, target $D_{P_{\mathbf n}}$, exact labels for training, and state-disjoint plus direction-held-out evaluation. The [learning curve](../evidence.md#learning-curves-count-complete-source-states) measures fitted error against a matched two-copy QNN and a FiLM sweep. The [finite-copy table](../evidence.md#finite-copy-inference) evaluates frozen models under specified shot allocations. Neither axis alone is a total experimental-resource comparison.

## Adaptation recipe

1. Define a repeatable state source $\rho$ and its preparation cost. Decide whether simulation-only density matrices or native hardware preparations are the intended input.
2. Choose a physical observable or functional, the context components supplied at inference, and an independent label procedure. Do not feed hidden state-generation parameters into the model unless they would truly be measured.
3. Split by state identity and reserve a context region to test transfer. Keep label generation, validation selection, and test evaluation separate.
4. Start with the simplest observable and direct estimator. Then choose the [compact benchmark](../training.md#load-and-use-the-trained-specialization) for this dephasing target or the [state-valued layer and readout](../architecture.md#6-use-the-state-valued-api) for a new target.
5. Train through Torch, save a validation-selected checkpoint, and reload it before measuring test errors. The [custom trainer](../training.md#train-and-checkpoint-a-custom-streaming-predictor) accepts `train/validation/test` density matrices, contexts, labels, and optional state IDs.
6. Report accuracy with the strongest task-informed and access-matched alternatives. Include raw-copy, settings, shots, gate depth, state preparation, and label costs.

Run the working examples after [installation](../install.md):

```bash
python examples/pretrained_inference.py --output /tmp/qem-frozen-review
python examples/train_streaming.py --steps 2 --output /tmp/qem-custom-smoke
```

The second command uses a synthetic teacher fixture unless `--dataset` supplies your labeled states. The [reproduction guide](../reproduction.md) describes the saved scientific records and safe output directories. The [application hub](../physics.md) explains which use cases are trained, analytic, diagnostic, and proposed.
