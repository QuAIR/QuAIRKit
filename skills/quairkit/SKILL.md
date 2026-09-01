---
name: quairkit
description: Use for QuAIRKit 0.5.1 only when writing, explaining, reviewing, validating, or reproducing code involving circuits, states, ansatz layers, database factories, qinfo, losses, LOCC, Quantikz plotting, QASM, training loops, PyTorch integration, import paths, or StateOperator backends.
license: Apache-2.0
---

# QuAIRKit 0.5.1

## Version Gate

1. Resolve every referenced path relative to this skill directory.
2. Before reading API references, recommending imports, or producing QuAIRKit code, run:

   ```bash
   python scripts/check_version.py
   ```

3. If QuAIRKit is missing or its installed version is not exactly `0.5.1`, stop. Ask the user to install an official QuAIRKit 0.5.1 package; do not adapt these instructions with APIs from another version.

## Source Of Truth

1. Treat the installed QuAIRKit 0.5.1 package and its public release source as authoritative.
2. Use `Circuit` as the default user-facing circuit interface.
3. Use `database` to construct matrices, channels, states, bases, and random data.
4. Use `qinfo` for analysis and `loss` for reusable training-oriented wrappers.
5. Treat `StateOperator` as an execution-backend interface, not a numeric state constructor.
6. Verify ambiguous signatures, shapes, or insertion behavior with a minimal runnable snippet in the gated 0.5.1 environment.

## Setup Defaults

- Use an official QuAIRKit 0.5.1 wheel that matches the target Python and operating system. Do not install from a repository tree.
- Use Python 3.10 and PyTorch 2.11 for the verified baseline.
- Use `python -m pip install "quairkit==0.5.1"` when installing from the official package index.
- Use `qkit.set_dtype("complex128")` for numerical stability, `qkit.set_device(...)` for device choice, and `qkit.set_seed(...)` for reproducibility.
- `Circuit.plot()` requires a TeX toolchain with `pdflatex`.

Read [references/environment-and-style.md](references/environment-and-style.md) for installation, imports, plotting dependencies, and deliverable style.

## Reference Routing

Read only the references needed for the task:

- states, Hamiltonians, backend switching: [references/api-core.md](references/api-core.md)
- circuits, gates, channels, measurement, QASM2, plotting: [references/api-circuit.md](references/api-circuit.md)
- template layers, encodings, subcircuits: [references/api-ansatz.md](references/api-ansatz.md)
- database factories and quantum-information utilities: [references/api-database-qinfo.md](references/api-database-qinfo.md)
- loss wrappers and `OneWayLOCCNet`: [references/api-loss-application.md](references/api-loss-application.md)
- PyTorch integration and training loops: [references/api-torch.md](references/api-torch.md)
- tutorial reconstruction and common workloads: [references/tutorials-checklist.md](references/tutorials-checklist.md)
- canonical QuAIRKit 0.5.1 import routes: [references/imports.json](references/imports.json)

## Workflow

### Write Or Review Code

1. Confirm the version gate passes.
2. Identify the API family: state preparation, circuit construction, analysis, training, plotting, or backend integration.
3. Query an uncertain function, class, or method before choosing its import:

   ```bash
   python scripts/recommend_import.py Circuit
   python scripts/recommend_import.py Circuit.rx
   python scripts/recommend_import.py state_fidelity --json
   ```

4. Prefer `Circuit` plus public `database`, `qinfo`, and `loss` routes over low-level operator classes.
5. Keep batch shape, probability axes, `system_dim`, dtype, device, and randomness explicit.
6. Run a minimal snippet when API behavior matters.

### Reproduce A Tutorial

1. Confirm the version gate passes.
2. Use [references/tutorials-checklist.md](references/tutorials-checklist.md) to identify the capability and relevant APIs.
3. Rebuild from QuAIRKit 0.5.1 APIs instead of copying tutorial cells.
4. Preserve scientific meaning; simplify constants only when exact reproduction is unnecessary.
5. Seed stochastic steps and provide a simulator fallback for optional external backends when possible.

### Write Training Code

1. Confirm the version gate passes.
2. Separate the optimization objective from the validation metric.
3. Prefer a stable differentiable loss even when the final metric differs.
4. Follow [references/api-torch.md](references/api-torch.md), including a nonzero logging interval.

## API Conventions

- `Circuit.append(layer)` preserves a child-module boundary; `Circuit.extend(layer)` flattens it.
- Built-in trainable layers use parameters shaped `[batch_size, total_param_num]`.
- Ordinary parameterized gates flatten accepted parameter input to one batch axis. They support the same batch size or size-one broadcasting, not arbitrary multidimensional NumPy broadcasting.
- `int`, `List[int]`, and `List[List[int]]` insertion targets are API- and arity-dependent; consult the circuit reference.
- `database.rx(...)` returns a matrix, not a callable gate object.
- Do not use low-level operator classes such as `RX`, `CNOT`, `Oracle`, `Collapse`, or `OneWayLOCC` in ordinary user code when a `Circuit` method exists.
- `Circuit.measure` changes circuit structure; `loss.Measure` consumes an existing state.
- Only qubit circuits export to OpenQASM 2.0.
- QuAIRKit 0.5.1 has no QuAIRKit-specific circuit compiler, public IR, binding table, executable plan, or generic circuit snapshot API. Do not invent `Circuit.save` or `Circuit.load`.
- An inherited `Circuit.compile` attribute is PyTorch `torch.nn.Module` behavior, not a QuAIRKit quantum-circuit compiler.
- `Circuit.from_operators(...)` only reconstructs a qubit circuit from a non-daggered `operator_history`; it is not general persistence or a cross-version interchange format.
- Do not recommend `PQCombNet`.

## Backend Guardrails

- `StateOperator` backends support execution, shots, and operator-history workflows.
- Only promise `measure` and `expec_val` unless a provider explicitly supports more.
- Prefer `_multi_execute()` returning batched CSR counts for shot-based backends.
- Shot-based measurement may return sparse probabilities while preserving dense simulator outcome indexing.
- Use `SimpleStateOperator` as a local stand-in when demonstrating interface design.

## Validation

- Run `python scripts/check_version.py` before any other validation.
- Run `python scripts/verify_examples.py` after changing examples or references.
- Run `python scripts/recommend_import.py` to validate query behavior; version mismatch exits before recommendations are produced.
- Inspect the reported installation path and exact version when freshness matters.
- State missing TeX or external-backend dependencies explicitly; do not imply unexecuted behavior was verified.

## Gotchas

- This skill is not cross-version guidance. Do not infer APIs from a different QuAIRKit release.
- A local package tree that shadows the official wheel fails the version gate; remove it from the import path rather than bypassing the check.
- `Circuit.rx`, `database.rx`, and `quairkit.operator.RX` are different abstraction levels.
- Leading state dimensions can include probability branches introduced by measurement, LOCC, or quasi operations; they are not always independent training batches.
- A plain `torch.Tensor` passed as `param` does not become a registered module parameter.
- Sparse forward outputs do not guarantee sparse backward memory for every PyTorch operation.
- Do not silently mix qubit-only and qudit-aware assumptions.
- Match the user's language in conversation, but keep code, comments, and repository-facing artifacts in English unless the task explicitly requires another language.
