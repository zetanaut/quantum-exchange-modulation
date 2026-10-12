# References and scientific scope

This project combines established quantum operations with a particular context-controlled architecture and evaluation protocol. The following primary sources locate the proposal within that literature. Links lead to the papers themselves or their primary preprint records.

## Exchange operations and repeated quantum inputs

### Partial SWAP and state addition

**Ziman et al., “Quantum homogenization” (2001 preprint); published as “Diluting quantum information: An analysis of information transfer in system-reservoir interactions,” Physical Review A 65, 042105 (2002).** [Paper](https://arxiv.org/abs/quant-ph/0110164) · [DOI](https://doi.org/10.1103/PhysRevA.65.042105)

Studies partial-SWAP collisions and information transfer between a system and reservoir states. The collision mechanism underlying the layer is established physics.

**Audenaert, Datta, and Ozols, “Entropy power inequalities for qudits,” Journal of Mathematical Physics 57, 052202 (2016).** [Paper](https://arxiv.org/abs/1503.04213) · [DOI](https://doi.org/10.1063/1.4950785)

Defines a partial-SWAP state-addition operation with the mixture-plus-commutator structure used here, up to exchange sign convention. The basic reduced-state formula is not a new primitive of this project.

### Quantum data used repeatedly

**Ekert, Alves, Oi, Horodecki, Horodecki, and Kwek, “Direct estimations of linear and non-linear functionals of a quantum state,” Physical Review Letters 88, 217901 (2002).** [Paper](https://arxiv.org/abs/quant-ph/0203016) · [DOI](https://doi.org/10.1103/PhysRevLett.88.217901)

An early controlled-SWAP network for direct estimation of purity and other nonlinear density-operator functionals. It is a target-informed alternative to fitting a conditional network when the desired functional is known.

**Cha, Park, and Lee, “Re-uploading quantum data: a universal function approximator for quantum inputs,” arXiv:2509.18530, version 6 (2026).** [Version examined](https://arxiv.org/abs/2509.18530v6)

A close precedent for nonlinear functions obtained from repeated native quantum inputs, with an approximation framework. The streaming architecture should be compared against this work rather than presented as the first nonlinear quantum-input model.

**Bromley and Rebentrost, “Batched quantum state exponentiation and quantum Hebbian learning,” Quantum Machine Intelligence 1, 31–40 (2019).** [Paper](https://arxiv.org/abs/1803.07039) · [DOI](https://doi.org/10.1007/s42484-019-00002-9)

Connects copy-consuming quantum processing with learning. It is relevant to interpretations of state-dependent interactions and their preparation costs.

**Connerty and Evans, “Controllable Quantum Memory Capacity in Quantum Reservoir Networks with Tunable partial-SWAPs,” arXiv:2605.12713v3 (2026).** [Paper](https://arxiv.org/abs/2605.12713v3)

Examines tunable partial-SWAP interactions in quantum reservoir networks. Trainable exchange strengths and quantum memory already have related learning applications.

## Conditioning and quantum representations

**Perez et al., “FiLM: Visual Reasoning with a General Conditioning Layer,” AAAI (2018).** [Paper](https://arxiv.org/abs/1709.07871)

The classical analogy: context-dependent affine modulation of numerical features. The exchange proposal translates conditioning into a physical operation on quantum representations.

**Haboury et al., “A supervised hybrid quantum machine learning solution to the emergency escape routing problem” (2023).** [Paper](https://arxiv.org/abs/2307.15682)

Includes a Quantum FiLM construction. The name and broad idea of quantum conditioning therefore precede this repository.

**Lee, Chen, and Tseng, “Learning to Program Adaptive Non-Local Observables for Machine Learning,” arXiv:2609.18655v1 (2026).** [Paper](https://arxiv.org/abs/2609.18655v1)

Programs circuit parameters and nonlocal observables from classical inputs. Its classical-input forecasting and reinforcement-learning setting differs from the repeated native quantum-state input and retained memory used here.

**Pérez-Salinas et al., “Data re-uploading for a universal quantum classifier,” Quantum 4, 226 (2020).** [Paper](https://arxiv.org/abs/1907.02085) · [DOI](https://doi.org/10.22331/q-2020-02-06-226)

Repeated classical data encodings into a circuit. Its access model differs from preparing independent copies of an unknown quantum state.

**Korkmaz and Türkpençe, “Information-driven Nonlinear Quantum Neuron” (2023).** [Paper](https://arxiv.org/abs/2307.09017)

A related nonlinear quantum-neuron proposal. Broad claims to the first physically motivated quantum activation are inappropriate.

**Nguyen et al., “Theory for Equivariant Quantum Neural Networks,” PRX Quantum 5, 020328 (2024).** [Paper](https://arxiv.org/abs/2210.08566) · [DOI](https://doi.org/10.1103/PRXQuantum.5.020328)

Provides a framework for symmetry-compatible quantum models. Exchange covariance becomes an equivariant architecture only when conditioners and readouts transform consistently.

**Ivashkov et al., “QKAN: quantum Kolmogorov–Arnold networks with applications in machine learning and multivariate state preparation,” npj Quantum Information 12, 73 (2026).** [Paper](https://arxiv.org/abs/2410.04435) · [DOI](https://doi.org/10.1038/s41534-026-01202-5)

A neighboring approach to multivariable quantum function representation. Compare input encoding, oracles, and preparation assumptions before treating different architectures as interchangeable.

## Physical sensitivity and measurement

**Paris, “Quantum estimation for quantum technology,” International Journal of Quantum Information 7, 125 (2009).** [Paper](https://arxiv.org/abs/0804.2981) · [DOI](https://doi.org/10.1142/S0219749909004839)

Reviews quantum estimation and the spectral form of quantum Fisher information. It supplies the normalization used for the parameter family generated by \(P/2\).

**Cerezo, Sone, Beckey, and Coles, “Sub-Quantum Fisher Information,” Quantum Science and Technology 6, 035008 (2021).** [Paper](https://arxiv.org/abs/2101.10144) · [DOI](https://doi.org/10.1088/2058-9565/abfbef)

Studies a measurable lower bound on QFI and its connections to coherence, asymmetry, and purity loss. For generator \(G=P/2\), \(P^2=I\), the response \(D_P\) is this established sub-QFI. The sensing connection is an application of prior theory, not a new bound.

**Huang, Kueng, and Preskill, “Predicting Many Properties of a Quantum System from Very Few Measurements,” Nature Physics 16, 1050–1057 (2020).** [Paper](https://arxiv.org/abs/2002.08953) · [DOI](https://doi.org/10.1038/s41567-020-0932-7)

Classical shadows are relevant measurement-based competitors, especially when many observables or contexts are queried from a reusable classical record.

**Huang et al., “Quantum advantage in learning from experiments,” Science 376, 1182–1186 (2022).** [Paper](https://arxiv.org/abs/2112.00778) · [DOI](https://doi.org/10.1126/science.abn7293)

Demonstrates the importance of quantum-data access models for learning advantages. Its results motivate the question; they do not automatically establish an advantage for this small dephasing benchmark.

## Trainability and classical alternatives

**Cerezo et al., “Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits,” Nature Communications 12, 1791 (2021).** [Paper](https://arxiv.org/abs/2001.00550) · [DOI](https://doi.org/10.1038/s41467-021-21728-w)

Gradient scaling depends on the cost, depth, locality, and initialization assumptions. A small parameter count is useful engineering information, not a trainability theorem.

**Bermejo et al., “Quantum Convolutional Neural Networks are Effectively Classically Simulable,” PRX Quantum 7, 020304 (2026).** [Paper](https://arxiv.org/abs/2408.12739v2) · [DOI](https://doi.org/10.1103/8qt9-72ts)

Shows why classical simulation and measurement assumptions deserve explicit analysis. Its scope is architectural; it should not be read as a statement about every quantum learning model.

## How to describe this project

A supported description is: **a context-controlled exchange module for native quantum inputs, with a differentiable implementation, explicit preparation accounting, and physical benchmarks that include strong collective and classical comparisons.**

The project does not claim a new partial-SWAP identity, the first quantum FiLM layer, the first nonlinear quantum-input function, or a new QFI lower bound. Its [results](evidence.md) also do not establish general quantum superiority. The research question is whether its particular conditional orientation and parameter structure become useful under a well-specified task and access model.

The papers establish prior scientific context. Numerical claims on this site refer separately to the downloadable experiment records. Absence of a matching search result is never treated as proof of originality.
