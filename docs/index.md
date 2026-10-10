# Context-controlled exchange modulation

**A quantum-state analogue of feature modulation, implemented in Torch and PennyLane.** A classical context vector controls gates that mix a quantum memory with a newly prepared source state. The output remains a physical quantum state and can enter the next layer. After the final layer, selected observables feed a Torch prediction head.

![Network architecture and the streaming exchange module](assets/figures/network_architecture.svg)

## The central idea

Classical FiLM uses context-dependent feature scales and offsets. Here, the context chooses an operator rotation and the strengths of a physical exchange interaction. For memory $\omega$, input state $\rho$, and context $c$,

$$
\tau(c)=V(c)\rho V(c)^\dagger,\qquad
\omega'=a(c)\omega+[1-a(c)]\tau(c)+i b(c)[\omega,\tau(c)].
$$

The commutator couples noncommuting operator features. A bounded gate parameterization enforces $b^2\leq a(1-a)$, so the construction stays physical. A multilayer Torch network can choose the gate parameters nonlinearly in several context variables. Reusing the numerical description of $\rho$ in a simulator represents fresh physical preparations of the state. For a depth-$L$ streaming network, each execution consumes $L+1$ such preparations.

The [architecture guide](architecture.md) derives the map, explains global versus pairwise exchange, and distinguishes nonlinear dependence on a context from nonlinear dependence on an unknown quantum input. The [control explorer](control_explorer.md) makes the coefficient region and resource arithmetic interactive.

## What has been demonstrated

The strongest fitted example estimates **directional dephasing susceptibility**,

$$
D_P(\rho)=\operatorname{Tr}(\rho^2)-\operatorname{Tr}(\rho P\rho P),\qquad P^2=I.
$$

For two-qubit states, the benchmark conditions on a direction for $P=(\mathbf n\cdot\boldsymbol\sigma)\otimes I$. It tests generalization to new states and a held-out cap of context directions. The learned exchange model, a matched two-copy QNN, a commutator-removal ablation, classical FiLM, generic kernels, and a physics-informed Gram model are all included.

| Training budget | Structured exchange: mean joint-test MSE | Matched two-copy QNN | Selected FiLM |
|---|---:|---:|---:|
| 8 states / 40 pairs | $9.31\times10^{-6}$ | $1.16\times10^{-6}$ | $8.39\times10^{-2}$ |
| 16 states / 80 pairs | $1.49\times10^{-8}$ | $8.54\times10^{-8}$ | $1.11\times10^{-1}$ |
| 32 states / 160 pairs | $3.46\times10^{-9}$ | $1.24\times10^{-7}$ | $8.82\times10^{-2}$ |
| 64 states / 320 pairs | $6.05\times10^{-9}$ | $2.05\times10^{-8}$ | $5.37\times10^{-2}$ |

Each entry averages seeds 17, 31, and 73. The state subsets are nested, all models share them, and every size additionally uses 96 validation pairs. These optimizer seeds do not measure uncertainty from resampling the dataset. The matched QNN is better at the smallest budget; both quantum models perform well. The untrained exact Gram formula reaches numerical precision when its required Pauli features are given. [Full protocol, comparisons, and data](evidence.md).

![Learning curve with all seeds and strong comparators](assets/figures/learning_curve.svg)

The demonstrated benefit is a useful physical structure for this target relative to the tested generic feature models and ablations. The study does not establish a universal learning advantage, an asymptotic speedup, or superior performance on noisy hardware.

## Physics examples you can run

| Problem | What the example computes | Evidence type |
|---|---|---|
| Directional dephasing | A context-dependent quadratic response from two-copy exchange | Trained models and held-out comparisons |
| Bell/GHZ correlations | Varying global response despite identical local marginals | Exact QNodes, matrices, and a stronger known parity estimator |
| Finite-time decoherence | Purity loss across rates and times from the frozen learned response | Analytic transfer of a trained checkpoint |
| Phase sensitivity | Sub-QFI and its relation to quantum Fisher information | Independent spectral, SLD, and exchange checks |
| Three-spin thermal states | Local perturbation sensitivity across field and temperature | Gibbs-state generation and global-exchange diagnostics |
| A streaming DNN | End-to-end training and checkpoint reload through two quantum layers | Explicitly synthetic software integration |

The [physics guide](physics.md) gives equations, assumptions, figures, data, and extension directions for each problem. Some examples are direct physics computations, while the two-layer teacher fixture demonstrates the API. The scope label accompanies every plot.

## Choose a starting point

1. **Run a model:** follow [installation](install.md) and the [quickstart](quickstart.md). A frozen checkpoint evaluates immediately.
2. **Build your own DNN:** read [architecture](architecture.md) and [training](training.md), then supply density matrices, contexts, labels, and disjoint splits to the training script.
3. **Assess the evidence:** read [results and comparisons](evidence.md), then use the [reproduction guide](reproduction.md) to regenerate a selected result or the full benchmark.
4. **Publish or extend this site:** use the [deployment guide](deploy.md). The supplied workflow builds the static site and keeps numerical experiments in a separate verification job.

All figures have portable vector versions; the numerical downloads include saved predictions, selected parameters, run histories, split indices, and verification reports. [The asset inventory](assets/manifest.json) records file sizes and SHA-256 hashes.
