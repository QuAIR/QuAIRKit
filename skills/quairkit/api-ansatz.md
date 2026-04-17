# Ansatz APIs

## Scope

Use this file for:

- `Layer` and `OperatorList`
- built-in trainable layers
- encoding layers
- `TrotterLayer`
- custom reusable subcircuits

## `Layer` Versus `Circuit`

- `Circuit` is the main end-user entry point.
- `Layer` is a reusable subcircuit template.
- `Layer` is an `OperatorList`, so it behaves like a quantum submodule rather than a separate execution model.

Use `Circuit` when writing an end-to-end workflow.
Use `Layer` when you want a named, reusable subcircuit that can be inserted into larger circuits.

## Public Exports

### Container

- `Layer`
- `OperatorList`

### Trainable Layers

- `LinearEntangledLayer`
- `RealEntangledLayer`
- `ComplexEntangledLayer`
- `ComplexBlockLayer`
- `RealBlockLayer`
- `Universal2`
- `Universal3`

### Encodings

- `BasisEncoding`
- `AmplitudeEncoding`
- `AngleEncoding`
- `IQPEncoding`

### Trotter

- `TrotterLayer`

## How To Insert A Layer

There are two important ways to attach a custom layer to a circuit.

### Preserve The Layer As A Child

```python
cir.append(layer)
```

This keeps the layer object as a child module. It is the right default when:

- you want the layer to remain visible in `operator_history`
- you want to index or plot the layer as a single component
- you want to preserve the submodule boundary

### Flatten The Layer Into Operators

```python
cir.extend(layer)
```

This inserts the layer's internal operators one by one.

Use it only when you explicitly want flattening behavior.

## Batch Rules For Built-In Trainable Layers

For common trainable layers in `quairkit.ansatz.layer`, the batched parameter rule is:

```text
[batch_size, total_param_num]
```

Important details:

- old class-specific batched shapes are no longer the default interface
- a single sample is still accepted when `numel == total_param_num`
- the layer reshapes its internal parameters on its own
- the user-facing input stays flat even when the layer internally reshapes it into per-block or per-gate tiles
- users should not manually rebuild the old per-gate layouts

This rule applies to the trainable layer family, not to all encodings or templates.

## Encoding Layers

Encoding layers follow their own API-specific input conventions.

Do not assume the trainable-layer batch rule applies to:

- `BasisEncoding`
- `AmplitudeEncoding`
- `AngleEncoding`
- `IQPEncoding`

When writing examples, make the encoding input shape explicit.

## Built-In Layer Families

### `LinearEntangledLayer`

Use when you want a structured entangling ansatz with a linear entanglement pattern.

### `RealEntangledLayer`

Use when a real-valued ansatz is enough and you want a lighter parameterization.

### `ComplexEntangledLayer`

Use as a strong default for variational quantum circuits when no problem-specific template is required.

Common construction pattern:

```python
layer = ComplexEntangledLayer(qubits_idx=[0, 1, 2, 3], depth=2, param=None)
```

For batched initialization, pass `param` with shape `[batch_size, total_param_num]`, not an old per-gate tiled layout.
If you need `total_param_num` for a specific layer instance, a practical runtime-safe pattern is to instantiate the layer once with `param=None` and read `layer.param.numel()`.

### `RealBlockLayer` And `ComplexBlockLayer`

Use when the circuit is easier to describe in repeated block structure rather than a line-entangled structure.

### `Universal2` And `Universal3`

Use when you want built-in trainable universal templates at the layer level.

### `TrotterLayer`

Use when building Hamiltonian-simulation circuits or when a tutorial/task explicitly follows Trotter-Suzuki structure.

## Locating A Layer Inside A Circuit

Useful interfaces:

- `cir[i]` to access a child layer/operator
- `cir.operator_history` to inspect the circuit structure
- `cir.param` to read all registered trainable parameters as a flattened tensor
- `cir.grad` to read flattened gradients of registered trainable parameters
- `cir.matrix` if the circuit is unitary-only
- `layer.matrix` if the layer is unitary-only
- `cir.plot()` or `cir[i].plot()` for visualization

## Custom Layer Pattern

```python
import torch
from quairkit import Circuit
from quairkit.ansatz import Layer


class MyLayer(Layer):
    def __init__(self, system_idx):
        super().__init__(num_systems=len(system_idx), system_dim=2, physical_idx=system_idx)
        self.ry(system_idx, param=None)
        self.cnot(system_idx)


cir = Circuit(2)
layer = MyLayer([0, 1])
cir.append(layer)
```

When writing custom layers, prefer:

- stable system indexing
- explicit parameter semantics
- reuse of existing `Circuit`/`OperatorList` insertion methods

## Plotting And Matrix Access

- `Layer` participates in the same plotting pipeline as `Circuit`
- `cir[i].plot()` is useful when a layer should be explained separately
- for unitary-only layers, `matrix` is the modern interface

## Guidance For Tutorial Reproduction

- For VQE-like and ansatz-heavy tutorials, start with `ComplexEntangledLayer` unless the tutorial needs a specific structure.
- For data-loading examples, choose the encoding API explicitly instead of forcing everything into trainable-layer semantics.
- For Hamiltonian-simulation workflows, look at `TrotterLayer` first.
