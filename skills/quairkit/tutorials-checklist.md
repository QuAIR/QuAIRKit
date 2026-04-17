# Task Pattern Checklist

## How To Use This File

This file is a capability checklist for common QuAIRKit workloads.

For each task pattern:

- identify the target capability
- identify the core APIs involved
- combine the relevant APIs from the other reference files
- reproduce the workflow without copying external tutorial code

## State Preparation And Inspection

- Goal: prepare named or random states, inspect their tensor forms, and explain pure vs mixed semantics
- Core APIs: `to_state`, `zero_state`, `bell_state`, `isotropic_state`, `random_state`, `haar_state_vector`, `random_density_matrix`, `State.ket`, `State.bra`, `State.density_matrix`, `State.numpy`, `State.trace`, `State.rank`, `State.vec`, `State.evolve`, `State.transform`, `State.expec_val`, `State.measure`
- Success standard: produce valid states, inspect at least one representation, and correctly explain when the workflow is pure-state versus density-matrix based

## Circuit Construction And Custom Blocks

- Goal: build a circuit with standard gates, custom unitary/channel blocks, and inspect its structure
- Core APIs: `Circuit`, `oracle`, `param_oracle`, `depolarizing`, `kraus_channel`, `choi_channel`, `stinespring_channel`, `linear_entangled_layer`, `real_block_layer`, `complex_block_layer`, `operator_history`, `matrix`
- Success standard: construct a runnable circuit, add at least one custom block, and explain how the inserted operators appear in the circuit history

## Hamiltonian Modeling And Expectation Values

- Goal: define standard or custom Hamiltonians and evaluate expectation values on states or circuit outputs
- Core APIs: `Hamiltonian`, `ising_hamiltonian`, `xy_hamiltonian`, `heisenberg_hamiltonian`, `random_hamiltonian_generator`, `ExpecVal`, `State.expec_val`
- Success standard: create the Hamiltonian, inspect or use `ham.matrix`, and compute at least one expectation value correctly

## Matrix And Channel Utilities

- Goal: generate gate matrices and channel representations, then relate them to state evolution or channel action
- Core APIs: `h`, `rx`, `ry`, `rz`, `cnot`, `swap`, `random_unitary`, `random_channel`, `channel_repr_convert`, `bit_flip_kraus`, `depolarizing_kraus`, `State.evolve`, `State.transform`
- Success standard: distinguish clearly between matrix generators and circuit methods, and demonstrate one unitary example plus one channel example

## Qinfo Analysis

- Goal: run linear-algebra checks and quantum-information metrics on states or matrices
- Core APIs: `trace`, `direct_sum`, `NKron`, `dagger`, `decomp_1qubit`, `von_neumann_entropy`, `trace_distance`, `state_fidelity`, `purity`, `relative_entropy`, `p_norm`, `is_positive`, `is_ppt`, `is_unitary`
- Success standard: compute at least one validation quantity and one information-theoretic quantity, while handling input types correctly

## Measurement And POVM Workflows

- Goal: perform computational-basis, custom-basis, or POVM measurement and interpret the resulting probabilities or post-selected states
- Core APIs: `Measure`, `pauli_str_povm`, `State.measure`, `prob_sample`, `Circuit.measure`
- Success standard: show at least one partial-system or custom-basis measurement and correctly distinguish `Measure` from `Circuit.measure`

## VQE And Standard Training Loops

- Goal: train a variational circuit with the standard optimization template
- Core APIs: `Circuit`, `complex_entangled_layer`, `ExpecVal`, `ising_hamiltonian`, `torch.optim.Adam`, `ReduceLROnPlateau`
- Success standard: use the standard training template, optimize a scalar loss, and report a meaningful validation or comparison metric

## Batch Execution

- Goal: run many parameterized circuits or matrix operations in parallel with explicit batch semantics
- Core APIs: batched parameterized gates, `oracle` with batched unitaries, `kraus_channel`, `choi_channel`, `ExpecVal`, `Measure`, `State.measure`
- Success standard: explain what the leading dimensions mean and produce output whose batch dimensions are interpreted correctly

## Third-Party Backend Integration

- Goal: define or explain a `StateOperator`-style backend with shot-based execution
- Core APIs: `StateOperator`, `SimpleStateOperator`, `set_backend`, `Measure`, `ExpecVal`
- Success standard: show the required backend skeleton and accurately state that execution backends support shot-oriented workflows rather than direct state inspection

## Hybrid Classical-Quantum Models

- Goal: combine `Circuit` with a larger `torch.nn.Module` and manage parameters cleanly
- Core APIs: `Circuit.update_param`, `randomize_param`, indexing into `Circuit`, `param_oracle`, `torch.nn.Module`, `torch.nn.Parameter`, `ExpecVal`
- Success standard: explain when to use QuAIRKit-managed parameters versus outer `torch.nn.Module` parameters, and provide a runnable hybrid pattern

## Circuit Visualization

- Goal: generate publication-ready circuit figures or Quantikz code
- Core APIs: `plot`, `to_latex`, `oracle`, `param_oracle`, noise channels, layers, `swap`, `permute`, `measure`, `reset`
- Success standard: produce either a plotted circuit or Quantikz output and mention the required TeX dependency and paper/arXiv caveats

## Qudit And Mixed-Dimension Workflows

- Goal: build circuits and states whose local dimensions are not all qubits
- Core APIs: `Circuit(..., system_dim=...)`, `random_state(..., system_dim=...)`, `universal_qudits`, `oracle`, `Measure`, `State.measure`
- Success standard: specify `system_dim` explicitly, use at least one non-qubit subsystem, and explain how indexing and dimensions change

## Barren Plateau And Gradient Diagnostics

- Goal: study how gradient magnitude behaves for random variational circuits
- Core APIs: parameterized `Circuit`, expectation-value style losses, gradient-based analysis workflows
- Success standard: build a parameterized model, extract gradient information, and summarize the optimization difficulty meaningfully

## Process Learning Without PQCombNet

- Goal: learn or approximate a process transformation using ordinary QuAIRKit building blocks
- Core APIs: `Circuit`, `universal_qudits`, `oracle`, `random_unitary`, `trace_distance`, `state_fidelity`, standard training loop
- Success standard: formulate the task without `PQCombNet`, train with a stable loss, and evaluate with a task-relevant metric

## Teleportation And Low-Level LOCC

- Goal: implement teleportation-style conditional correction using `Circuit.locc`
- Core APIs: `Circuit.locc`, `bell_state`, `random_state`, `state_fidelity`, `trace`, `nkron`, `eye`, `x`, `z`
- Success standard: use the standard two-step LOCC pattern and verify correct transmission with a fidelity-style check

## OneWayLOCCNet Protocols

- Goal: build party-structured discrimination, communication, or distillation workflows
- Core APIs: `OneWayLOCCNet`, party circuit access, `param_locc`, `set_init_state`, `physical_circuit`, `Measure`, `partial_trace`, `state_fidelity`
- Success standard: use party-aware indices, initialize states correctly, and explain the distinction between logical protocol structure and `physical_circuit`

## Simon-Type Oracle Workflows

- Goal: build an oracle-driven algorithmic circuit or a PQC rediscovery workflow
- Core APIs: `Circuit`, `extend`, `oracle`, `Uf`, `Measure`, `universal_two_qubits`
- Success standard: show how the oracle enters the circuit and explain the role of the measurement or training objective

## Hamiltonian Simulation And Trotterization

- Goal: approximate time evolution under a Hamiltonian with Trotter methods
- Core APIs: `Circuit.trotter`, `TrotterLayer`, `Hamiltonian`, `gate_fidelity`, `torch.linalg.matrix_exp`
- Success standard: build a Trotterized circuit, compare against the exact unitary, and discuss step-count or order trade-offs

## Quasi-Probability Error Mitigation

- Goal: express a noisy or mitigated process using quasi-probability branches
- Core APIs: `Circuit.quasi`, `eye`, `x`, `y`, `z`, `random_state`, `trace`
- Success standard: explain how quasi-probability branches work and show a small but valid mitigation-style example

## Entanglement And Communication Primitives

- Goal: build Bell-pair-based communication or entanglement demos such as superdense coding
- Core APIs: `bell_state`, `Circuit`, `h`, `cnot`, Pauli operations, measurement, standard state factories
- Success standard: prepare the entangled resource, apply local encoding/decoding or verification steps, and interpret the output correctly
- For the standard `|Phi+>` superdense-coding convention, a common message mapping is first bit -> `z` and second bit -> `x` on Alice's qubit; keep the computational-basis bit ordering consistent with how you read measurement outcomes.

## QPE With Mixed Dimensions

- Goal: build quantum phase estimation with controlled powers of a unitary and possibly a non-qubit target system
- Core APIs: mixed `system_dim`, `oracle` with controlled powers, `qft_matrix`, `dagger`, `measure`, `probability`, `torch.linalg.matrix_power`
- Success standard: construct the controlled-power pipeline, measure the phase register, and explain the resulting probability distribution

## Grover-Style Search

- Goal: build an unstructured-search example with one marked computational basis state
- Core APIs: `Of`, `grover_matrix`, `h`, `measure`, `probability`
- Success standard: identify the marked state explicitly, report its final probability, and explain whether you used a manual diffusion-plus-`Of` construction or QuAIRKit's `grover_matrix`
- `Of(f, n)` is the `n`-qubit phase oracle for search, while `Uf(f, n)` is Simon's `2n`-qubit XOR oracle and is not the usual Grover phase oracle

## Reproduction Guardrails

- Prefer reproducing capabilities, not copying notebook prose or code blocks.
- Keep the same scientific meaning even if constants, seeds, or formatting are simplified.
- If a task depends on unsupported infrastructure, use a clearly labeled fallback.
- If an older example relies on weaker or deprecated APIs, prefer current stable APIs unless exact historical reproduction is requested.
