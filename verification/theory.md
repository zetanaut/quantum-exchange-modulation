# Mathematical verification notes: quantum exchange feature modulation

This file records exact statements used by `verify.py`. It distinguishes algebraic identities and restricted representation results from learning performance. The numerical checks provide independent finite-dimensional matrix verification; the proofs below establish the stated general identities.

## 1. Conventions and the physical operation

The two registers have the same dimension `d`. Their full SWAP is

\[
S\,|i\rangle_A|j\rangle_B=|j\rangle_A|i\rangle_B,
\qquad S^\dagger=S,\quad S^2=I.
\]

Use the exchange convention

\[
U_\eta=e^{-i\eta S}=\cos\eta\,I-i\sin\eta\,S.
\]

For independently supplied input states `rho` and `tau`, tracing register B gives

\[
\operatorname{Tr}_B[U_\eta(\rho\otimes\tau)U_\eta^\dagger]
=\cos^2\eta\,\rho+\sin^2\eta\,\tau
+i\cos\eta\sin\eta[\rho,\tau].
\]

The sign follows from
\(\operatorname{Tr}_B[(A\otimes B)S]=AB\) and
\(\operatorname{Tr}_B[S(A\otimes B)]=BA\).

Mixing the positive and negative exchange angles with probabilities
\((1+\kappa)/2\) and \((1-\kappa)/2\) yields

\[
F_{a,b}(\rho,\tau)
=a\rho+(1-a)\tau+ib[\rho,\tau],
\qquad
a=\cos^2\eta,\quad
b=\kappa\sqrt{a(1-a)},\quad -1\leq\kappa\leq1.
\]

Here choose \(0\leq\eta\leq\pi/2\) for the square-root parametrization. A coherent sign-control ancilla, followed by tracing that ancilla, gives the same mixture. This is a CPTP channel on the *joint input*. When \(\tau=V(c)\rho V(c)^\dagger\), the resulting state-valued function of \(\rho\) is quadratic and consumes two independently prepared copies. It is not a linear quantum channel on one unknown copy of \(\rho\).

## 2. Sharp necessary and sufficient parameter domain

**Proposition.** For real `a,b` and any Hilbert-space dimension at least two, the expression \(F_{a,b}(\rho,\tau)\) is a state for every pair of density matrices if and only if

\[
0\leq a\leq1,\qquad b^2\leq a(1-a).
\]

The same necessity holds if the permitted inputs are only pairs \(\tau=V\rho V^\dagger\) with arbitrary \(\rho,V\). Equivalently, the allowed coefficient domain is the closed disk

\[
(a-1/2)^2+b^2\leq1/4.
\]

**Sufficiency.** For `0<a<1`, set \(\eta=\arccos\sqrt a\) and \(\kappa=b/\sqrt{a(1-a)}\). The signed-exchange mixture above is a physical CPTP realization. At the endpoints, the inequality forces `b=0`, leaving either \(\rho\) or \(\tau\).

**Necessity.** Orthogonal pure states give eigenvalues `a` and `1-a`, so `a` must lie in `[0,1]`. For the remaining condition use two pure qubit states with Bloch vectors \(\mathbf r,\mathbf t\), both of unit length, and dot product \(u=\mathbf r\cdot\mathbf t\). The output Bloch vector is

\[
\mathbf r'=a\mathbf r+(1-a)\mathbf t-b\,\mathbf r\times\mathbf t,
\]

and therefore

\[
\|\mathbf r'\|^2-1
=(1-u)\{-2a(1-a)+b^2(1+u)\}.
\]

If `b^2>a(1-a)`, choose `u<1` sufficiently close to one. The right side is positive and the smallest density-matrix eigenvalue \((1-\|\mathbf r'\|)/2\) is negative. Every pair of pure states is unitarily related, proving the restricted-pair assertion as well. Embedding this two-dimensional counterexample proves necessity in every larger dimension.

This domain is a physical constraint on the proposed formula, not a free choice of independent classical feature gains.

## 3. Conditional multivariable features

For qubits, let

\[
\rho=\tfrac12(I+\mathbf r\cdot\boldsymbol\sigma),\qquad
V(c)\rho V(c)^\dagger=\tfrac12(I+R(c)\mathbf r\cdot\boldsymbol\sigma).
\]

Then the exact feature map is

\[
\mathbf r'
=a\mathbf r+(1-a)R(c)\mathbf r
-b\,\mathbf r\times R(c)\mathbf r.
\]

For a general dimension, choose traceless Hermitian generators with
\(\operatorname{Tr}(T_aT_b)=2\delta_{ab}\) and
\([T_a,T_b]=2i\sum_c f_{abc}T_c\). Write
\(\rho=I/d+\frac12\sum_a r_aT_a\), and define the adjoint matrix
\(R_{ab}=\frac12\operatorname{Tr}(T_aVT_bV^\dagger)\). Then

\[
r'_c=ar_c+(1-a)\sum_dR_{cd}r_d
-b\sum_{a,b}f_{abc}\,r_a(R\mathbf r)_b.
\]

The last term is explicitly quadratic in the native-state coordinates. Its coefficients are determined by the operator algebra and the conditional rotations. Nonlinear dependence of gate angles on classical context is a separate source of nonlinearity and should not be confused with this polynomial dependence on \(\rho\).

The `b=0` ablation retains the exact linear mixture and removes only the commutator contribution. If `V=I`, the entire module returns \(\rho\). More generally the quadratic contribution vanishes whenever \([\rho,V\rho V^\dagger]=0\).

### Exact qubit feature bank

At `a=b=1/2`, writing \(\mathbf r=(x,y,z)\),

| Rotation of the second copy | Output Bloch vector |
|---|---|
| \(R_z(\pi)\) | \((-yz,xz,z)\) |
| \(R_y(\pi)\) | \((yz,y,-xy)\) |

With \(V=R_j(\pi/2)\), the component along the rotation axis is

\[
r'_j=r_j-\tfrac12(r_k^2+r_l^2),\qquad\{j,k,l\}=\{x,y,z\}.
\]

Thus five settings, Pauli readouts, the linear input features, and a constant span the ten scalar monomials of total degree at most two. For example, with \(s_x=y^2+z^2\), \(s_y=z^2+x^2\), \(s_z=x^2+y^2\), one obtains \(x^2=(s_y+s_z-s_x)/2\). This is a *bank of settings with a linear readout*, not a claim that one output qubit stores ten independently adjustable coordinates. Measurement settings and additional copies must be counted in any benchmark.

## 4. The dephasing response and its normalization

Let \(P=P^\dagger\), \(P^2=I\), and define

\[
D_P(\rho)=\operatorname{Tr}(\rho^2)-\operatorname{Tr}(\rho P\rho P)
=\tfrac12\|[P,\rho]\|_F^2.
\]

For the dephasing generator
\(\dot\rho=\gamma(P\rho P-\rho)\),

\[
\frac{d}{dt}\operatorname{Tr}(\rho^2)=-2\gamma D_P(\rho).
\]

This fixes the benchmark normalization. If instead one defines the complete two-sector dephasing map \(\Delta_P(\rho)=(\rho+P\rho P)/2\), then

\[
\operatorname{Tr}(\rho^2)-\operatorname{Tr}(\Delta_P(\rho)^2)=D_P(\rho)/2.
\]

These two quantities must not be given the same numerical label.

Take \(V_+=e^{-i\pi P/4}\), prepare \(\rho\otimes V_+\rho V_+^\dagger\), and apply \(U_{\pi/4}\). With \(m=\operatorname{Tr}(P\rho)\), the two output ports satisfy

\[
z_A=\langle P_A\rangle=m-D_P(\rho)/2,\qquad
z_B=\langle P_B\rangle=m+D_P(\rho)/2.
\]

Changing the quarter-rotation sign reverses the commutator contribution. Independent positive/negative rotation runs also recover `D` by a branch difference, but **one circuit retaining both output ports already suffices** for this terminal readout.

### Exact equivalence to a direct collective observable

Define

\[
W=e^{-i\pi S/4}(I\otimes e^{-i\pi P/4}),\qquad
K=(I-P\otimes P)S.
\]

There is an operator identity on the entire joint Hilbert space:

\[
W^\dagger(P_B-P_A)W=K.
\]

To see it, set \(\Delta=P_B-P_A\). Since \(S\Delta=-\Delta S\), conjugating by the half exchange gives \(iS\Delta\). Conjugating by the second-register quarter rotation then gives \((I-P\otimes P)S\). Consequently the port-difference circuit is a physical implementation of the known collective observable `K`, with the same combined outcome distribution, not a sample advantage over measuring `K` directly.

The measurements of `P_A` and `P_B` commute. If a joint shot produces signs `s,t`, the estimator uses `t-s`, with values in `{-2,0,2}`. The correlation is

\[
\langle P_AP_B\rangle=m^2,
\]

because the exchange commutes with \(P\otimes P\) and the quarter rotation commutes with `P`. Therefore `N` joint shots, using `2N` independent raw-state copies, have exact variance

\[
\operatorname{Var}(\widehat D)
=\frac{2(1-m^2)-D_P(\rho)^2}{N}.
\]

The direct `K` estimator has exactly the same variance: \(K^2=2(I-P\otimes P)\). The verification script checks the operator identity, matrix POVM probabilities, first and second moments, and deterministic Monte Carlo means and variances.

### Bell-mixture fixture

Let

\[
\rho_t=\tfrac{1+t}{2}|\Phi^+\rangle\langle\Phi^+|
+\tfrac{1-t}{2}|\Phi^-\rangle\langle\Phi^-|,
\quad -1\leq t\leq1,
\quad P=Z\otimes I.
\]

Both single-qubit marginals are maximally mixed, but

\[
D_P(\rho_t)=t^2,\qquad z_A=-t^2/2,\qquad z_B=t^2/2.
\]

Any deterministic one-copy circuit with a linear expectation readout is affine in `t`; it cannot equal `t^2` throughout this family. On uniform `t` in `[-1,1]`, the best affine approximation is `1/3`, with mean-squared error `4/45`. This restricted representation separation disappears if the comparison model is allowed enough repeated preparations and nonlinear classical processing of an informative measured feature. Such a comparison remains necessary for performance claims.

## 5. Copy order, depth, and feedback

For fixed context, any deterministic physical circuit supplied with exactly `k` copies has a scalar linear readout of the form

\[
f(\rho,c)=\operatorname{Tr}[O(c)\rho^{\otimes k}],
\]

where fixed ancillas and CPTP maps can be absorbed into the effective observable. Hence its degree in the entries of \(\rho\) is at most `k`. Extra gate depth and fresh *fixed* ancillas do not increase that degree. A CNOT parity circuit on \(\rho_z^{\otimes k}\), with \(\rho_z=(I+zZ)/2\), attains the degree exactly by returning `z^k` in a single local expectation.

This statement excludes conditioning on postselection success and nonlinear classical processing of estimated expectations. Such operations change the mathematical input-output expression and must be accounted for explicitly.

A streaming construction uses a memory state \(\sigma_l\) and a fresh transformed *raw input* \(\tau_l=V_l(c)\rho V_l(c)^\dagger\):

\[
\sigma_{l+1}=F_{a_l,b_l}(\sigma_l,\tau_l).
\]

If \(\sigma_0=\rho\), `L` collisions consume `L+1` independent raw copies and yield degree at most `L+1`. They do not require exponentially many hidden-state copies. In contrast, recursively preparing two independent copies of the *entire previous nonlinear hidden state* multiplies its preparation cost at each stage.

## 6. Purity contraction and the fixed maximally mixed state

For \(\tau=V\rho V^\dagger\), the spectra and purities of the two inputs agree. Expanding the output square gives

\[
\operatorname{Tr}[F_{a,b}(\rho,\tau)^2]-\operatorname{Tr}(\rho^2)
=-a(1-a)\|\rho-\tau\|_F^2+b^2\|[\rho,\tau]\|_F^2.
\]

If \(\delta=\lambda_{\max}(\rho)-\lambda_{\min}(\rho)\leq1\), diagonalizing \(\rho\) proves

\[
\|[\rho,\tau]\|_F
=\|[\rho,\tau-\rho]\|_F
\leq\delta\|\tau-\rho\|_F.
\]

Combined with \(b^2\leq a(1-a)\), this proves that the self-rotated block cannot increase input purity. This is a structural limitation, not an empirical noise result. An independently prepared reservoir of a different purity is a different architecture and is not subject to this particular comparison.

Also \(F_{a,b}(I/d,V(I/d)V^\dagger)=I/d\). The self-rotated block therefore supplies no freely adjustable constant feature shift. A separate collision with an independently prepared context state can give a physically constrained affine shift.

## 7. Symmetry and charge accounting

Because \(S(W\otimes W)=(W\otimes W)S\),

\[
F_{a,b}(W\rho W^\dagger,W\tau W^\dagger)
=W F_{a,b}(\rho,\tau)W^\dagger.
\]

For a conditional rotation family, equivariance follows when
\(V(g\cdot c)=W_gV(c)W_g^\dagger\), with compatible scalar conditioner transformations.

Covariance does not mean the expectation of each register's conserved charge remains fixed. Exchange preserves a total charge of the form `Q_A+Q_B`, but may transfer that charge between ports. For example, a qubit initially along `+X` and its `RZ(pi/2)` rotated copy both have zero `Z` expectation; the half-exchange outputs have `Z` expectations `-1/2` and `+1/2`. Fixed-sector preservation is available when both inputs are supported in the same charge eigenspace and the conditional rotations preserve that eigenspace.

## 8. Gate conventions and register size

For an expectation `f(eta)` in which only the exchange gate varies,

\[
\frac{df}{d\eta}=f(\eta+\pi/4)-f(\eta-\pi/4),
\qquad U_\eta=e^{-i\eta S}.
\]

There is no `1/2` prefactor in this convention. For \(e^{-i\theta S/2}\), the usual formula has shifts `+/-pi/2` and prefactor `1/2`. Shared parameters appearing in other gates require the appropriate chain rule and separate gate contributions.

For two multi-qubit registers,

\[
S_{AB}=\prod_j S_{A_jB_j},\qquad
e^{-i\eta S_{AB}}\ne\prod_j e^{-i\eta S_{A_jB_j}}
\]

in general. At `eta=pi/4`, inputs `|00>` and `|11>` yield retained state `diag(1/2,0,0,1/2)` for global exchange, but `I/4` for independent local exchanges. Their trace norm difference is `1`, corresponding to trace distance `1/2`.

A genuine global exchange can be implemented using a control ancilla. With `C_S` the controlled global SWAP and `B=(H tensor I) C_S (H tensor I)`, the sequence

\[
B\,[R_z(2\eta)\otimes I],B
\]

acts on `|0> tensor |psi>` as `|0> tensor exp(-i eta S)|psi>`. This uses two controlled global SWAP operations, each decomposable into `n` controlled individual SWAPs for `n`-qubit registers. Connectivity and native gates determine actual depth and noise. Replacing this with local pairwise exchange changes the global model and its verification requirements.

### A concrete CNOT construction for two-qubit registers

For the specific logical gate basis used in the resource certificate, a controlled SWAP has the chronological decomposition

\[
\operatorname{Fredkin}(c,a,b):
\quad \operatorname{CX}(b,a),\quad
\operatorname{CCX}(c,a,b),\quad
\operatorname{CX}(b,a).
\]

The verification script compiles each Toffoli into a standard sequence of `H`, `T`, `T-dagger`, and six CNOT gates, and compares the complete complex unitary with the literal Toffoli permutation matrix. The comparison retains all phases. Consequently a Fredkin uses eight CNOTs in this construction.

For `n=2`, the global exchange implementation contains two controlled global SWAPs, each containing two Fredkins. The resulting construction uses four Fredkins, or **32 CNOTs**, and one control ancilla. The script verifies the complete five-qubit unitary and the control ancilla's return to zero.

Any weight-two Pauli rotation \(e^{-i\theta(P\otimes Q)/2}\), with each factor in `{X,Y,Z}`, has the familiar basis-change, CNOT, `Rz`, CNOT, inverse-basis-change realization. All nine factor combinations are checked against matrix exponentials. Four such rotations add eight CNOTs. Thus a module consisting of precisely one global exchange plus those four Pauli rotations has a **40-CNOT upper construction** in this basis.

This is not an optimal compilation claim or a lower bound. It assumes all-to-all logical CNOT availability and does not include state preparation, routing, measurement, error correction, other layers, or optimization through cancellation between blocks. It therefore cannot justify calling this module and a generic eight-CNOT QNN an equal-gate-budget comparison. A resource comparison must report its actual gate basis and circuit construction explicitly.

## 9. Verification scope and reproduction

Run:

```bash
python verification/verify.py
```

The default outputs are `verification/theory_results.json` and `verification/resource_certificate.json`. Every check records its cases, numerical error or statistical discrepancy, tolerance, and status. The process exits nonzero on any failed metric or exception. The deterministic seed is `20261009`; `--seed` permits an explicit alternative and `--output` selects the main report path, with the resource certificate written beside it.

Dependencies are Python, NumPy, and SciPy. No automatic differentiation, quantum framework, symbolic algebra package, or external service is required. The script contains matrix tests of the core operation, sharp-domain counterexamples, covariance, dephasing identities, joint-port statistics, feature span, purity contraction, shift conventions, global/local distinctions, and copy degree.

Neither the partial-SWAP identity nor multicopy expressivity is asserted to be novel. These checks establish the consistency of this proposed implementation and the exact scope of its mathematical claims. They do not establish better trainability, lower sample cost than optimal collective measurements, or an unconditional quantum advantage.
