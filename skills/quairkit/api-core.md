# Core APIs

## Scope

Use this file for:

- `State`, `StateSimulator`, and `StateOperator`
- `Hamiltonian`
- `to_state`
- global backend/device/dtype/seed control
- third-party execution backends

## Top-Level Exports

The top-level `quairkit` package exposes these commonly used names:

- `Circuit`
- `State` (`StateSimulator` alias)
- `Hamiltonian`
- `Operator`
- `to_state`
- `StateOperator`
- `set_backend`, `set_device`, `set_dtype`, `set_seed`
- `get_backend`, `get_device`, `get_dtype`, `get_seed`, `get_float_dtype`
- subpackages: `ansatz`, `database`, `operator`, `loss`, `qinfo`, `application`

## Global Runtime Settings

### Backend

- `qkit.set_backend("default")` selects the default simulator.
- `qkit.set_backend(SomeBackendClass)` registers and selects a custom backend class.
- Backends are global, just like dtype and device.

### Device

- `qkit.set_device("cpu")`
- `qkit.set_device("cuda")`
- Use PyTorch-style device strings.

### Dtype

- Supported complex defaults are `complex64` and `complex128`.
- `qkit.get_float_dtype()` returns the matching real dtype for parameter tensors and probabilities.

### Seed

- `qkit.set_seed(seed)` sets common random sources used by QuAIRKit, including PyTorch, NumPy, and Python `random`.

## `Hamiltonian`

### Construction

Typical construction patterns:

```python
from quairkit import Hamiltonian

ham = Hamiltonian([
    [-1.0, "Z0,Z1"],
    [0.5, "X0"],
])
```

### What To Access

- `ham.matrix` for matrix form
- Pauli-string structure and coefficients for decomposition-based workflows

Use `database.ising_hamiltonian`, `database.xy_hamiltonian`, or `database.heisenberg_hamiltonian` when a standard model is enough.

## `to_state`

```python
state = to_state(data, system_dim=2, eps=1e-4, backend=None, prob=None)
```

### Arguments

- `data`: `torch.Tensor`, `numpy.ndarray`, an existing `State`, or execution-backend operator history
- `system_dim`: an integer for uniform systems or a list for heterogeneous systems
- `eps`: tolerance for state validation; use `None` to disable checking
- `backend`: optional backend override
- `prob`: optional probability annotations for probabilistic workflows

### Behavior

- Numeric tensor/array input goes through a simulator backend.
- Existing `State` input is cloned; it is not numerically re-parsed.
- `StateOperator` backends cannot be created from numeric state data.
- If the current backend is a `StateOperator` backend and you try to pass numeric data, QuAIRKit raises an error and asks you to switch back to a simulator.

### As Circuit Input

Prepared states are commonly fed into a circuit explicitly:

```python
output_state = cir(input_state)
```

This is the standard way to run a circuit on a chosen initial state instead of the default zero state.

### Default-Backend Semantics

- Pure-state numerical data usually stays in state-vector semantics under unitary evolution.
- Density matrices, mixed-state operations such as partial transpose or trace, and noisy channels move the state into density-matrix semantics.
- For users, the important distinction is usually pure vs mixed vs product structure, not the internal class name.

## `State` And `StateSimulator`

At the top level, users normally see `State`. Conceptually:

- `StateSimulator` is the tensor-execution path
- `StateOperator` is the execution-backend path

## State Tensor Shape Convention

For user-facing reasoning, treat state tensors as having this conceptual layout:

```text
(batch, prob_1, prob_2, ..., prob_K, state_rows, state_cols)
```

- `batch` is the outer parallel-workload axis
- each `prob_i` axis is introduced by one probabilistic operation such as `measure`, `locc`, or `quasi`
- the trailing state block is `(dim, 1)` for a pure-state `ket` view or `(dim, dim)` for a density-matrix view

Verified runtime examples:

- a batch of two 1-qubit pure states has `ket.shape == (2, 2, 1)`
- after a branching measurement with two outcomes, the post-measurement state can have `ket.shape == (2, 2, 2, 1)` and `batch_dim == [2, 2]`
- after two sequential `locc` calls on a batch of two 3-qubit inputs, QuAIRKit reports `batch_dim == [2, 2, 2]`, `probability.shape == (2, 2, 2)`, `ket.shape == (2, 2, 2, 8, 1)`, and `density_matrix.shape == (2, 2, 2, 8, 8)`

Practical interpretation:

- the first leading dimension is the usual workload batch
- later leading dimensions can be probability branches rather than independent training samples
- each new probabilistic operation inserts one new axis before the trailing state dimensions
- broadcasting still follows the usual rule: `N` operators with `N` states pair elementwise; one side can broadcast to the other

Important caveats:

- do not assume that `batch_dim` always means only a training batch; probability axes are folded into the same leading-dimension view
- for a single pure state, `ket` may be shown as `(dim, 1)` rather than `(1, dim, 1)`
- mixed states do not expose `ket`; use `density_matrix` or `vec`

### Common Simulator Operations

- `evolve(unitary, sys_idx=None)`
- `transform(op, sys_idx=None, repr_type="kraus")`
- `permute(target_seq)`
- `reset(reset_idx, replace_state)`
- `trace(trace_idx=None)`
- `expec_val(hamiltonian, shots=None, decompose=False)`
- `measure(system_idx=None, shots=None, desired_result=None, keep_state=False)`
- `expec_state(prob_idx=None)`
- `sqrt()`
- `log()`
- `transpose(transpose_idx=None)`

### Common State Information

- `ket`
- `bra`
- `density_matrix`
- `vec`
- `rank`
- `trace()`
- `batch_dim`
- `probability`
- `num_systems`
- `system_dim`

These interfaces preserve batch/probability information where applicable.

### Notes

- `State.measure` supports POVM computation in addition to PVM workflows.
- `trace()` without an index returns the trace value; with a subsystem index it performs a partial trace.
- `expec_state()` is useful in probabilistic workflows when a branch-weighted expected state is needed.
- When a probabilistic output must be compared against one reference state per workload sample, the usual recipe is `output_state.expec_state().trace(...)` rather than tracing the raw branched state directly.

## `tensor_state`

`tensor_state(state1, state2, ...)` is the state-level tensor/Kronecker-product helper. In the current runtime, `tensor_state` is not exported from the top-level `quairkit` package. Treat it as a core/state helper rather than a default top-level import.

If the inputs are possibly arrays/tensors instead of `State` objects, prefer `qinfo.nkron`. In general, one should use `qinfo.nkron` unless you are pretty sure the inputs are definitely `State` objects.

For a workload batch of product states, a practical recipe is: build batched kets with shape `(batch, dim, 1)`, combine factors with `qinfo.nkron` along that leading dimension, and then pass the final `(batch, total_dim, 1)` tensor to `to_state(...)`.

## `StateOperator` Versus `StateSimulator`

### `StateSimulator`

- numerical state construction is allowed
- tensor inspection is allowed
- suitable for algorithm prototyping and analysis

### `StateOperator`

- intended for execution-like backends, hardware submissions, or compatible shot-based emulators
- numeric state construction from matrices or state vectors is not supported
- capabilities such as measurement, expectation values, gradient support, and batch submission depend on the concrete implementation

`SimpleStateOperator` is a lightweight local stand-in for interface testing.

## Third-Party Backend Integration

This section is for backend integrators, not ordinary end users.

### Required Pattern

Subclass `StateOperator` and define:

- a non-empty class attribute `backend`
- `clone()`
- `_execute(qasm: str, shots: int)` or `_multi_execute(list_qasm, list_shots)`

### Minimal Template

```python
from quairkit import StateOperator


class MyBackend(StateOperator):
    backend = "my-backend"

    def clone(self):
        return MyBackend(self._data, self._sys_dim)

    def _execute(self, qasm: str, shots: int):
        ...
```

If the provider supports batched submissions, prefer `_multi_execute`.

### Setup

```python
qkit.set_backend(MyBackend)
```

Reset to the simulator:

```python
qkit.set_backend("default")
```

### What Users Can Expect

- `Measure` and `ExpecVal` can be reused on `StateOperator` backends
- the supported user-facing operations are typically shot-based `measure` and `expec_val`
- tensor inspection belongs to simulator backends, not execution backends

### Gradients

- QuAIRKit can differentiate through `StateOperator` backends using generalized parameter-shift rules or finite differences, depending on how the parameterized operators are represented
- do not promise exact training cost or precision; these depend on the backend and shot budget

### Example Integration Flow

1. Build a `Circuit`.
2. Set the backend to a `StateOperator` subclass.
3. Run the circuit to produce an execution-style state.
4. Call `Measure` or `ExpecVal` with a `shots` argument.

## Notes For Writing Examples

- Use `State` in user-facing prose unless the distinction between simulator and operator backends matters.
- Mention `StateSimulator` and `StateOperator` explicitly when discussing semantics, backends, or extensibility.
- For beginner examples, mention `eps=None` only in prose; do not encourage disabling state validation by default.

## Common Pitfalls

- Do not reshape away leading dimensions unless you know whether they are workload batch axes or probability axes.
- `State.probability` stores the joint probability grid of all probabilistic branches seen so far; it is not a separate object unrelated to `batch_dim`.
- `StateOperator` backends are execution-style interfaces and cannot be initialized from numeric matrices or state vectors.
- Mixed states do not provide `ket`; use `density_matrix` or `vec` instead.
- For probabilistic outputs, tracing the raw branched state can keep probability axes in the batch view and break later comparisons. Use `expec_state()` first when you want the branch-averaged state.
