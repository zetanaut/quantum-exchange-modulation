# Context-controlled exchange modulation

**Which perturbation direction makes a repeatedly prepared quantum state lose purity, and can a conditional circuit learn that response from limited labeled states?** This project answers that question for a two-spin dephasing benchmark. A classical direction context controls gates acting on fresh quantum-state preparations; measured output ports feed a Torch prediction head. The architecture also supplies a state-valued streaming layer for other targets.

Start with the [current five-dataset QIP revision](qip_revision.md) and [directional decoherence](applications/decoherence.md), or use the [problem-selection guide](applications/choosing-a-problem.md) to decide whether your state source and measurement access fit. The [application hub](physics.md) covers correlated registers, sensing, and interacting thermal spins. Each guide identifies a prepared input, context, measured or computed output, runnable command, data, strongest comparator, and evidence status.

![Network architecture and the streaming exchange module](assets/figures/network_architecture.svg)

## The physical input and conditional network

Classical FiLM uses context-dependent feature scales and offsets. Here, the context chooses an operator rotation and the strengths of a physical exchange interaction. For memory $\omega$, input state $\rho$, and context $c$,

$$
\tau(c)=V(c)\rho V(c)^\dagger,\qquad
\omega'=a(c)\omega+[1-a(c)]\tau(c)+i b(c)[\omega,\tau(c)].
$$

The commutator couples noncommuting operator features. A bounded gate parameterization enforces $b^2\leq a(1-a)$, so the construction stays physical. A multilayer Torch network can choose the gate parameters nonlinearly in several context variables. Reusing the numerical description of $\rho$ in a simulator represents fresh physical preparations of the state. For a depth-$L$ streaming network, each execution consumes $L+1$ such preparations.

The [architecture guide](architecture.md) derives the map, explains global versus pairwise exchange, and distinguishes nonlinear dependence on a context from nonlinear dependence on an unknown quantum input. The [control explorer](control_explorer.md) makes the coefficient region and resource arithmetic interactive.

**Run now:** after [installation](install.md), execute `python examples/pretrained_inference.py --output /tmp/qem-frozen-review` to reload the historical two-spin checkpoint as an API example. Its older axis split is documented in the [historical evidence](evidence.md#historical-single-dataset-study); the current paper uses the [five-dataset revision](qip_revision.md). For a custom target, provide valid density matrices, context vectors, independent labels, and disjoint state splits to the [streaming trainer](training.md#train-and-checkpoint-a-custom-streaming-predictor). The reusable network retains quantum memory between collisions; its two-layer training example is a synthetic software integration fixture.

## What has been demonstrated

The strongest fitted example estimates **directional dephasing susceptibility**,

$$
D_P(\rho)=\operatorname{Tr}(\rho^2)-\operatorname{Tr}(\rho P\rho P),\qquad P^2=I.
$$

For two-qubit states, the benchmark conditions on a direction for $P=(\mathbf n\cdot\boldsymbol\sigma)\otimes I$. The current study trains and validates outside both caps $|n_x|>0.55$, then tests on those held-out physical axes and new states across five independent datasets. The learned exchange model, a matched two-copy QNN, a commutator-removal ablation, classical FiLM, product-feature regressions, and a physics-informed Gram model are all included.

| Training budget | Compact exchange MSE | Matched exchange QNN | Product FiLM | Fitted Gram-36 |
|---|---:|---:|---:|---:|
| 8 states / 40 pairs | $1.87\times10^{-6}$ | $1.53\times10^{-6}$ | $8.18\times10^{-2}$ | $1.82\times10^{-27}$ |
| 16 states / 80 pairs | $1.39\times10^{-8}$ | $1.51\times10^{-8}$ | $4.67\times10^{-2}$ | $1.95\times10^{-30}$ |
| 32 states / 160 pairs | $6.07\times10^{-9}$ | $1.04\times10^{-8}$ | $3.19\times10^{-2}$ | $2.41\times10^{-30}$ |
| 64 states / 320 pairs | $1.56\times10^{-9}$ | $1.75\times10^{-8}$ | $1.31\times10^{-2}$ | $1.19\times10^{-30}$ |

Each entry averages five dataset outcomes; learned models also average starts 17, 31, and 73. The state subsets are nested within each dataset, and every size additionally uses 96 validation pairs. The matched QNN is slightly better at the smallest budget; both quantum models fit the task from their prescribed initializations. The fitted Gram-36 control already reaches numerical precision with eight states. [Full protocol, comparisons, and data](qip_revision.md).

![Five-dataset physical-axis learning curves](assets/figures/qip_Fig2.png)

The demonstrated result is a physical layer with verified coefficient constraints, differentiable composition, and transparent source-copy accounting. Learning the analytic target is sensitive to initialization. The study does not establish a universal learning advantage, an asymptotic speedup, or superior performance on noisy hardware.

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

The revised paper includes a [fresh-protocol directional response map](qip_revision.md#transfer-checks-and-source) for one new test state. The historical [spectator-purity diagnostic](applications/metrology.md#an-uncorrelated-spectator-changes-the-hilbertschmidt-response) illustrates a limit of interpreting sub-QFI as QFI. Their scripts and CSV files accompany the figures; neither adds a new training result.

## Choose a starting point

1. **Run a model:** follow [installation](install.md) and the [quickstart](quickstart.md). A frozen checkpoint evaluates immediately.
2. **Build your own DNN:** read [architecture](architecture.md) and [training](training.md), then supply density matrices, contexts, labels, and disjoint splits to the training script.
3. **Assess the evidence:** read [results and comparisons](evidence.md), then use the [reproduction guide](reproduction.md) to regenerate a selected result or the full benchmark.
4. **Publish or extend this site:** use the [deployment guide](deploy.md). The supplied workflow builds the static site and keeps numerical experiments in a separate verification job.

All figures have portable vector versions; the numerical downloads include saved predictions, selected parameters, run histories, split indices, and verification reports. [The asset inventory](assets/manifest.json) records file sizes and SHA-256 hashes.
