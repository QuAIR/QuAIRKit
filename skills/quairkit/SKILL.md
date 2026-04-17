---
name: quairkit-guide
description: Write, explain, validate, and reproduce QuAIRKit code using only the reference files in this directory. Use when working with QuAIRKit circuits, states, ansatzes, LOCC, Quantikz plotting, training loops, common workload patterns, or third-party execution backends.
---

# QuAIRKit Guide

## When To Use

Use this skill when a task involves:

- writing or reviewing QuAIRKit code
- reproducing or adapting examples from the official QuAIRKit tutorials
- explaining QuAIRKit APIs for states, circuits, training, plotting, or LOCC
- creating tutorial-like demos for the paper appendix
- integrating a `StateOperator` backend

## First Principles

1. Treat the QuAIRKit installed package source as the source of truth.
2. Prefer `Circuit` as the default user-facing interface.
3. Use `database` for construction and `qinfo` for analysis.
4. Use `loss` wrappers for training-oriented code.
5. When an insertion rule is ambiguous, verify it in code instead of guessing.

## Installation

### Runtime Install

- Recommended environment: Python 3.10 and PyTorch 2.9.x.
- On supported wheel platforms, use `pip install quairkit`.
- Release wheels are acceptable if their PyTorch compatibility matches the environment.

### Source Install

- Use source install when modifying QuAIRKit itself or when no compatible wheel exists.
- Requirements: active Python environment, PyTorch >= 2.4, and a C++17 toolchain.
- Standard editable install:

```bash
pip install -e . --no-build-isolation
```

- Better IDE compatibility:

```bash
pip install -e . --config-settings editable_mode=strict --no-build-isolation
```

- For VSCode/Pylance, `python.analysis.extraPaths` may still be useful after editable install.

### Optional Plotting Dependencies

- `Circuit.plot()` requires `pdflatex`.
- Recommended TeX distribution: TeX Live or MacTeX.
- On macOS, install `poppler` if PDF page-count errors appear.

## Default Imports

```python
import time
import numpy as np
import torch
import quairkit as qkit

from quairkit import Circuit, State, Hamiltonian, to_state
from quairkit.database import *
from quairkit.loss import *
from quairkit.qinfo import *
```

Use narrower imports in final user-facing code when that improves readability.

## Default Global Setup

- Use `qkit.set_dtype("complex128")` when numerical stability matters.
- Use `qkit.set_device("cpu")` or `qkit.set_device("cuda")` with PyTorch-style device strings.
- Use `qkit.set_seed(seed)` when reproducibility matters.
- Backend choice, dtype, device, and seed are global settings.

## Core Mental Model

- `Circuit` is the main user entry point.
- `Layer` is a reusable subcircuit template and is an `OperatorList`.
- `State` is the top-level alias users normally see.
- `StateSimulator` is for tensor-level simulation and state inspection.
- `StateOperator` is for execution-style backends such as cloud or shot-based providers.
- `database` builds matrices, channels, states, bases, and random data.
- `qinfo` is the analysis toolbox.
- `loss` provides training-friendly wrappers.
- `OneWayLOCCNet` is the only application-level wrapper covered by this skill.

## Routing

### States, Hamiltonians, `to_state`, backend switching, or cloud backends

Read [api-core.md](api-core.md).

### Circuit creation, gates, channels, oracles, measurement, plotting, or QASM2

Read [api-circuit.md](api-circuit.md).

### Template layers, encodings, or custom subcircuits

Read [api-ansatz.md](api-ansatz.md).

### Matrix/state generators, random data, or quantum-information utilities

Read [api-database-qinfo.md](api-database-qinfo.md).

### Loss wrappers or `OneWayLOCCNet`

Read [api-loss-application.md](api-loss-application.md).

### Training loops, PyTorch integration, hybrid models, or NumPy/Torch interop

Read [api-torch.md](api-torch.md).

### Common workload patterns and tutorial-like reconstruction

Read [tutorials-checklist.md](tutorials-checklist.md).

## Default Working Patterns

### Writing A New Example

1. Decide whether the task is about state preparation, circuit construction, analysis, training, plotting, or backend integration.
2. Pick the right API family instead of mixing abstractions randomly.
3. Prefer `Circuit` plus `database` factories for concise examples.
4. Keep batch shape and `system_dim` semantics explicit.
5. If plotting or cloud execution is involved, mention external dependencies.

### Reproducing A Tutorial

1. Use [tutorials-checklist.md](tutorials-checklist.md) to identify the target capability and APIs.
2. Rebuild the workflow from APIs and patterns, not by copying tutorial cells.
3. Preserve the same conceptual pipeline, but simplify constants or logging if the user does not need an exact replica.
4. If the tutorial depends on randomness, seed it or state clearly that output is stochastic.
5. If the tutorial depends on third-party infrastructure, provide a simulator fallback when possible.

### Writing Training Code

1. Separate the training objective from the validation metric.
2. Prefer numerically stable losses even if the final success metric is different.
3. Use the standard loop in [api-torch.md](api-torch.md).
4. For simple VQE-like tasks, the validation metric can be omitted; otherwise keep it.

## Non-Negotiable API Conventions

- For trainable built-in layers in `quairkit.ansatz.layer`, batched parameters use `[batch_size, total_param_num]`.
- `Circuit.append(layer)` keeps the layer as a child module; `Circuit.extend(layer)` flattens the layer into its internal operators.
- For operator insertion, `int`, `List[int]`, and `List[List[int]]` can mean different things depending on arity. Check [api-circuit.md](api-circuit.md) before writing examples.
- Build circuits with `Circuit.*`; use `database.*` when you need matrices, channels, or named states outside a circuit.
- Do not import from `quairkit.operator` in ordinary user code. Names such as `RX`, `CNOT`, `Oracle`, `Collapse`, and `OneWayLOCC` are low-level operator classes, not the default user interface.
- `database.rx(...)` returns a matrix (`torch.Tensor` or `numpy.ndarray`), not a callable gate object.
- Conceptually, state tensors follow `(batch, prob_1, ..., prob_K, state_rows, state_cols)`. See [api-core.md](api-core.md) before reshaping or indexing leading dimensions.
- `qinfo` is more analysis-oriented; `loss` is more training-oriented.
- `Circuit.measure` is structurally different from `loss.Measure`.
- Only qubit circuits can be exported to OpenQASM 2.0.
- Do not document or recommend `PQCombNet` in this skill.
- Do not create a separate backend-integration skill; backend integration belongs in the core/backend notes.

## Plotting And Paper Integration

- `Circuit.to_latex()` returns Quantikz code.
- `Circuit.plot()` depends on `pdflatex`.
- For arXiv, include the Quantikz support file if the archive does not provide it.
- If users only need a publication figure, exporting code and compiling in Overleaf is a valid fallback.

## Backend Integration Guardrails

- `StateOperator` backends are for shots, execution, and operator-history workflows.
- They do not support direct numeric state construction from matrices or state vectors.
- For backend examples, only promise `measure` and `expec_val` unless the provider explicitly supports more.
- Use `SimpleStateOperator` as a lightweight local stand-in when demonstrating interface design.

## What To Avoid

- Do not invent undocumented batch rules.
- Do not assume tutorial prose is newer than the source.
- Do not paste large tutorial code blocks when a smaller runnable example is enough.
- Do not mix qubit-only and qudit-aware assumptions silently.
- Do not use Windows-style paths inside the skill files.

## Common Pitfalls

- `Circuit.rx(...)`, `database.rx(...)`, and `quairkit.operator.RX(...)` live at three different abstraction levels. Use the first for circuit building, the second for matrices, and avoid the third in ordinary examples.
- Not every leading state dimension is an independent training batch. Later leading dimensions can be probability branches created by `measure`, `locc`, or `quasi`.
- Passing a plain `torch.Tensor` as `param` does not register a module parameter. Use `param=None` or an explicit `torch.nn.Parameter` when the parameter must appear in `model.parameters()`.
- `Circuit.measure` changes circuit structure; `loss.Measure` consumes an already prepared state.
- `StateOperator` backends cannot be initialized from numeric matrices or state vectors.

## Deliverable Style

- Keep examples short and runnable.
- Prefer current APIs over deprecated wrappers.
- Use English only.
- If something is uncertain, say it needs verification instead of guessing.

## Additional Resources

- State and backend details: [api-core.md](api-core.md)
- Circuit and plotting details: [api-circuit.md](api-circuit.md)
- Layers and encodings: [api-ansatz.md](api-ansatz.md)
- Data generation and analysis tools: [api-database-qinfo.md](api-database-qinfo.md)
- Training and PyTorch usage: [api-torch.md](api-torch.md)
- Tutorial coverage targets: [tutorials-checklist.md](tutorials-checklist.md)
