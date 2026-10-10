# Three interacting spins at finite temperature

**Evidence status: verified global-exchange diagnostic, with no three-qubit training.** A small Gibbs register asks how a local $X_1$ or $Z_1$ perturbation acts on a correlated thermal preparation. The current script uses an open three-site chain (bonds $1$–$2$ and $2$–$3$, with no $3$–$1$ bond):

$$
H=-J(Z_1Z_2+Z_2Z_3)-h(X_1+X_2+X_3)-\epsilon(Z_1-Z_3),
\qquad \rho_\beta=\frac{e^{-\beta H}}{\operatorname{Tr}(e^{-\beta H})}.
$$

Here $J=1$ fixes the energy unit, $\epsilon/J=0.13$ breaks the end-spin symmetry, the plotted horizontal axis is $h/J\in[0,1.8]$, and the vertical axis is $\beta J\in\{0,0.25,0.5,1,2,4\}$. $\beta$ is inverse temperature in reciprocal energy units; lower temperature means larger $\beta J$. The query axis, $X_1$ or $Z_1$, is the context of the diagnostic. The Hamiltonian parameters generate the state; the supplied three-spin example does not train a conditioner on them.

![Global-exchange responses for the three-spin Gibbs register.](../assets/figures/spin_gibbs.svg)

The first panel maps $D_{Z_1}$ across field and inverse temperature. The second compares $X_1$ and $Z_1$ at $\beta J=2$, plus the output of a **different** circuit made from parallel local partial swaps. A different response to $X_1$ and $Z_1$ means basis-dependent coherence under those chosen perturbations. The saved connected $Z_1Z_2$ value supplies a separate correlation diagnostic; a smaller $D$ at high temperature cannot alone be read as the disappearance of a specific correlation.

At $\beta=0$, $\rho=I/8$ commutes with every $P$, so both responses vanish. At $h=0$, $H$ is diagonal in the $Z$ basis and $D_{Z_1}=0$; $D_{X_1}$ can remain nonzero at finite $\beta$ because $X_1$ changes the diagonal sectors. These are checks on interpretation, not claims about a thermodynamic phase transition from three spins.

The actual full-register exchange QNode agrees with independent matrix responses to within $7.78\times10^{-16}$ over the recorded scan. Three selected cases also verify the six-Fredkin phase-kickback construction. Parallel local pair swaps differ from full-register exchange by up to $0.0970$ in the recorded $\beta J=2$ comparison. Full-register SWAP is the operation $S_{A:B}$ on two complete three-qubit registers; a product of three local *partial* swaps does not implement $e^{-i\eta S_{A:B}}$.

[Full scan CSV](../assets/results/physics_examples/spin_gibbs.csv) · [local-circuit comparison](../assets/results/physics_examples/spin_gibbs_local_comparison.csv) · [state arrays](../assets/results/physics_examples/spin_gibbs.npz) · [manifest](../assets/results/physics_examples/manifest.json) · [script](../assets/code/examples/physics_examples.py). Reproduce with `python examples/physics_examples.py --output /tmp/qem-thermal-review`.

A quantum simulator with repeatedly preparable thermal states is a proposed deployment setting. It would need a defined Gibbs preparation procedure, two equal-size registers, a compiled global exchange with connectivity/routing, chosen measurement settings, and raw-copy accounting. The current figure verifies an identity on three spins; it does not show a trained many-body advantage or a large-system scaling result.
