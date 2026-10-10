# Conditional open-system characterization and control

**Evidence status: proposed research extension, grounded in the trained dephasing response and exact examples.** A practical context might contain field direction, coupling strength, dissipation rate, or elapsed time. Only variables actually passed to the model are conditioning inputs. Parameters used by a simulator to *prepare* $\rho$ or generate labels are not automatically available at inference. The current trained benchmark conditions on direction angles; the finite-time example adds rate and time through a known analytic formula after prediction, not through a newly trained dynamics network.

## Match the target to its state-copy degree

| Target at fixed context $c$ | Dependence on unknown input $\rho$ | First comparator |
|---|---|---|
| $\operatorname{Tr}[O\mathcal E_c(\rho)]$ for a fixed channel | Linear: $\operatorname{Tr}[\mathcal E_c^\dagger(O)\rho]$ | Measure the effective observable or fit a linear state-feature model |
| Purity $\operatorname{Tr}[\mathcal E_c(\rho)^2]$, $D_P$, or a specified Hilbert–Schmidt overlap | Quadratic when the channel is linear in $\rho$ | Two-copy observable, known Pauli/Gram formula, measured-feature estimator |
| Entropy, optimized witnesses, general nonlinear dynamics targets | May require higher polynomial degree, approximation, or optimization | Problem-specific estimator and an explicit error/resource analysis |

The first row follows from linearity of quantum channels and the adjoint identity. An ordinary finite-time expectation value does not intrinsically need a nonlinear quantum-input layer. A circuit using $k$ independent preparations evaluates an expectation of the form $\operatorname{Tr}[M_c\rho^{\otimes k}]$, a polynomial of degree at most $k$ in the input state. More operations on the **same fixed number of copies** do not raise that degree. A depth-$L$ streaming network uses $L+1$ copies when initialized with $\omega_0=\rho$; its depth increases both processing and copy count. [Direct nonlinear-state estimation](https://arxiv.org/abs/quant-ph/0203016) and [classical shadows](https://arxiv.org/abs/2002.08953) are useful prior comparators.

## A concrete proposed loop

1. Prepare a state under the current classical control setting and record which preparation protocol produced it.
2. Query selected directions or rates using repeated preparations. Use a direct known estimator whenever it meets the task and budget.
3. Estimate a response with uncertainty and choose the next classical control setting.
4. Prepare a **new** state at that setting and evaluate it on independent shots.

The [directional map](decoherence.md#a-directional-map-for-one-held-out-state) illustrates step 2 on one frozen model; the [finite-time transfer](decoherence.md#transfer-through-a-known-time-law) supplies a known-law example. No closed-loop control experiment, noise-recovery channel, or error-correction protocol has been executed here. Channel noncommutativity, nonunitality, state drift, preparation cost, and measurement latency must be tested for a real apparatus.

For a new dataset, specify the context vector and labels independently, reserve disjoint preparation states for testing, and compare the [general state-valued API](../architecture.md#6-use-the-state-valued-api) with a direct estimator and the [custom trainer](../training.md#train-and-checkpoint-a-custom-streaming-predictor). The current two-layer training curve is a synthetic software fixture, not a control result.
