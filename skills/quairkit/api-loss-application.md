# Loss And Application APIs

## Scope

Use this file for:

- `ExpecVal`
- `Measure`
- `TraceDistance`
- `StateFidelity`
- responsibility boundaries between `loss`, `State`, `Circuit`, and `qinfo`
- `OneWayLOCCNet`

## Public Exports

### Loss

- `Measure`
- `ExpecVal`
- `TraceDistance`
- `StateFidelity`

### Application

- `OneWayLOCCNet`

`PQCombNet` exists in the source tree but is intentionally excluded from this skill.

## Recommended Responsibility Split

| Interface | Typical use case | Notes |
| --- | --- | --- |
| `ExpecVal` | training or modular forward pass | `Operator`-style wrapper; natural after a circuit forward |
| `State.expec_val` | one-off calculation on an existing `State` | shorter code for ad hoc analysis |
| `Measure` | training or modular measurement post-processing | good for reusable measurement nodes |
| `State.measure` | one-off measurement on an existing `State` | direct and convenient; also supports POVM |
| `Circuit.measure` | represent measurement or post-selection as part of circuit structure | useful for plotting, LaTeX, and structural workflows |
| `loss.StateFidelity` / `loss.TraceDistance` | compare to a fixed target during training | optimized wrapper form for iterative training |
| `qinfo.state_fidelity` / `qinfo.trace_distance` | analysis, scripting, post-processing, NumPy interop | more function-style and analysis-oriented |

## `ExpecVal`

Use `ExpecVal(hamiltonian)` when the expectation value is part of a reusable forward pipeline.

```python
loss_op = ExpecVal(ham)
value = loss_op(cir())
```

### Why Use It

- natural in training loops
- works cleanly after `Circuit` or any module that returns a state
- supports batching
- on execution backends, availability depends on backend support

If the code only computes one expectation once and already holds a `State`, `state.expec_val(...)` is often enough.

## `Measure`

Use `Measure(...)` when measurement should behave like a reusable operator node.

Possible measurement basis forms:

- `None` for the computational basis
- a Pauli string such as `"xy"`
- a custom PVM tensor

Useful capabilities:

- partial-system measurement
- `desired_result`
- `keep_state`
- modular composition in training or post-processing code

### Simulator Versus Execution Backend

- on simulators, measurement bases are flexible
- on execution backends, behavior is usually closer to Pauli-basis and shot-based semantics
- `keep_state` is simulator-oriented and usually not meaningful for execution backends

## `TraceDistance` And `StateFidelity`

Use these when a fixed target state should be packaged as a reusable loss object.

```python
target = bell_state(2)
loss_op = StateFidelity(target)
score = loss_op(state)
```

These wrappers are preferable to direct `qinfo` calls in repeated training code because they choose more suitable implementations for:

- pure vs pure
- pure vs mixed
- mixed vs mixed

They also handle batch alignment for target and input states.

## Practical Training Guidance

- Use `ExpecVal`, `Measure`, `TraceDistance`, and `StateFidelity` when the object should live as part of a model.
- Use `State.expec_val` or `State.measure` for one-off calculations.
- Use `qinfo` functions for post-processing, analysis, and NumPy-mixed workflows.

## `Circuit.measure` Versus `Measure`

These two interfaces overlap conceptually but are not interchangeable.

### The `Measure` Wrapper

- acts on an already available state
- modular and reusable
- oriented toward computation

### The `Circuit.measure` API

- inserts a measurement/post-selection step into the circuit structure itself
- better for structural diagrams and Quantikz output
- better when the circuit description itself must show the measurement
- not supported by OpenQASM 2.0 export

Use `Circuit.measure` when the measurement is part of the circuit narrative.
Use `Measure` when the measurement is part of the computation pipeline.

## `OneWayLOCCNet`

`OneWayLOCCNet` is a high-level wrapper for one-way LOCC protocols organized by party.

It is the only application-level wrapper covered by this skill.

### Initialization

Each party can be declared in one of these forms:

```python
{"Alice": 2}
{"Alice": {"num_systems": 2, "system_dim": 3}}
{"Alice": {"num_systems": 2, "system_dim": [2, 3]}}
```

This means:

- default systems are qubits when only a count is provided
- uniform qudit parties are allowed
- mixed local dimensions are allowed

### Access Per-Party Circuits

```python
net["Alice"].u3()
net["Bob"].cnot([0, 1])
```

The per-party circuit access pattern is central to how `OneWayLOCCNet` is used.

Per-party circuits are ordinary `Circuit` objects, so their regular methods are still available.
That includes patterns such as:

```python
net["Alice"].u3([0])
```

### Party Metadata

Useful interfaces:

- `net.keys()`
- `net.items()`
- `net.party_info`
- `net.physical_circuit`

`physical_circuit` returns the combined physical circuit plus appended LOCC operators.

### Initial State Preparation

Use:

```python
net.set_init_state(system_idx, state=None)
```

Logical indices are written as `(party_name, local_index)` tuples, for example:

```python
net.set_init_state([("Alice", 0), ("Bob", 0)])
```

Important behavior:

- if `state` is omitted and exactly two subsystems are specified, QuAIRKit uses a Bell state (or generalized Bell state for qudits)
- subsystems not covered by `set_init_state` default to zero state
- a pre-existing initial state on the same physical indices cannot be set twice

### `locc` And `param_locc`

These are the high-level LOCC insertion APIs.

For both interfaces:

- the first element of `system_idx` is the measured party subsystem or subsystems
- the remaining entries specify the subsystems that receive local operations

The party-aware form uses the same `(party_name, local_index)` tuples, for example:

```python
net.param_locc(u3, 3, [("Alice", 0), ("Bob", 0)], support_batch=False)
```

`locc` takes a non-parameterized local unitary.

`param_locc` takes:

- a generator function
- `num_acted_param`
- optional `param`
- optional `support_batch`

Important parameterization detail:

- the local unitary/generator must support batch size equal to the product of the measured subsystem dimensions
- `param_locc` itself does not represent an outer batch over repeated LOCC blocks

### Measurement Reuse Restriction

A subsystem that has already been used as the measured subsystem in an earlier LOCC step cannot be reused as the measured subsystem later.

This is a real structural restriction enforced by the implementation.

## Forward Execution

`OneWayLOCCNet` is callable:

```python
output_state = net()
```

This prepares the registered initial state configuration, applies all per-party circuits, then applies the LOCC blocks in sequence.

## Example Skeleton

```python
from quairkit.application import OneWayLOCCNet

net = OneWayLOCCNet({
    "Alice": 2,
    "Bob": 1,
})

net["Alice"].h(0)
net["Alice"].cnot([0, 1])
```

This is only a local-circuit sketch. For real LOCC protocols, also add `set_init_state(...)`, then one or more `locc` / `param_locc` steps, and finally call `net()`.

## Guidance For Writing Examples

- Use `OneWayLOCCNet` only when the party abstraction is the point.
- For low-level LOCC operations inside a single circuit, regular `Circuit.locc` and `Circuit.param_locc` may be sufficient.
- When discussing targets, distinguish between protocol structure and the final physical circuit via `physical_circuit`.

## Common Pitfalls

- `Measure` and `Circuit.measure` are not interchangeable. The first consumes an existing state; the second inserts a structural measurement step into the circuit.
- `OneWayLOCCNet` uses party-aware indices such as `("Alice", 0)`, not only plain integer system indices.
- `param_locc` needs a generator that already matches the required measurement-branch batch semantics. It is not an outer batch wrapper for many independent LOCC blocks.
- Once a subsystem has been used as a measured subsystem in one LOCC step, it cannot be reused later as another measured subsystem.
