# Correlations hidden from individual-spin marginals

**Evidence status: verified circuit identity and a stronger target-specific estimator.** A coherent two- or three-spin register can have a direction-dependent response even when every one-spin state is unchanged. This matters when a diagnostic uses a local perturbation but the prepared input contains multipartite coherence.

The executed family is $\rho_p=p|\mathrm{GHZ}^+\rangle\langle\mathrm{GHZ}^+|+(1-p)|\mathrm{GHZ}^-\rangle\langle\mathrm{GHZ}^-|$, with $|\mathrm{GHZ}^{\pm}\rangle=(|0\cdots0\rangle\pm|1\cdots1\rangle)/\sqrt2$. For $m=2$ these are Bell states. The context fixes $P=Z_1$; the state variable $p$ is used to **prepare** the family, not passed to a learned conditioner. Each one-spin marginal equals $I/2$ for all $p$, while

$$
D_{Z_1}(\rho_p)=(2p-1)^2,\qquad
\langle X^{\otimes m}\rangle_{\rho_p}=2p-1.
$$

A predictor limited to the individual-spin marginals sees identical inputs for all $p$. A fixed one-copy expectation is affine in $p$ and cannot trace the full parabola. Neither statement excludes nonlinear classical processing of repeated measurements. The [physics page](../physics.md#2-bell-and-ghz-mixtures-expose-global-correlations) derives these restricted comparisons.

![Bell and GHZ response across the mixture weight.](../assets/figures/correlation_sensitivity.svg)

The script constructs 41 mixture weights for each of two and three qubits, then compares an actual prescribed exchange QNode with the independent matrix polynomial. All 82 cases pass the declared checks. The horizontal variable in this figure is mixture weight $p$, not time or dephasing rate. [Cases CSV](../assets/results/physics_examples/correlation.csv) · [checks](../assets/results/physics_examples/checks.json) · [source](../assets/code/examples/physics_examples.py).

## The comparator that matters

For this known family, measure $X$ locally on **every** qubit, multiply the signs to obtain a global parity outcome, and multiply outcomes from two independent copies. This unbiased estimator recovers $(2p-1)^2$ without an entangling readout. At $C$ raw source copies, the recorded ideal variances are $(2-2D^2)/C$ for parity and $(4-2D^2)/C$ for exchange. The parity estimator is better throughout the plotted family.

![Copy-matched exchange and global-parity variances.](../assets/figures/correlation_copy_cost.svg)

[Copy-cost CSV](../assets/results/physics_examples/correlation_copy_cost.csv) gives the numerical comparison. The exchange circuit remains a useful demonstration of a collective quadratic feature, but this target already has a simpler known estimator. If many context observables are needed, a reusable measured record or [classical shadows](https://arxiv.org/abs/2002.08953) should also be assessed against the cost of fresh preparations for each exchange query.

This example diagnoses a coherence and correlation-sensitive response. It does not by itself certify entanglement: an entanglement claim needs an appropriate witness or separability bound, with preparation and measurement assumptions stated. To reproduce, run `python examples/physics_examples.py --output /tmp/qem-correlations-review`; the `correlation*` files in that output contain the curves and circuit checks.
