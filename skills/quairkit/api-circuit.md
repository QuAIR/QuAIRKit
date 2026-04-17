# Circuit APIs

## Scope

Use this file for:

- `Circuit(...)`
- gate/channel/layer/oracle insertion
- measurement, LOCC, quasi-probability operations
- parameter handling
- plotting and Quantikz export
- QASM2 import/export

## Circuit Construction

### Standard Qubit Circuit

```python
cir = Circuit(3)
```

Creates a 3-qubit circuit.

### Uniform-Qudit Circuit

```python
cir = Circuit(3, system_dim=5)
```

Creates 3 systems, each of dimension 5.

### Mixed-Dimension Circuit

```python
cir = Circuit(system_dim=[2, 3, 2])
```

Creates a mixed qubit-qutrit-qubit circuit.

### With Physical Indices

```python
cir = Circuit(2, physical_idx=[3, 8])
```

This separates:

- logical indices inside the circuit
- physical indices carried by the state/circuit mapping

Use this when composing systems across larger workflows.

In the current runtime, circuit insertion methods are in-place APIs and are normally used as statements. Do not assume calls such as `Circuit(2).h(0)` or `Circuit(2).depolarizing(0.1, 0)` return a new circuit object for fluent chaining.

## Operator Insertion Rules

This is the most error-prone part of QuAIRKit.

Do not improvise these rules.

## Index Formats

### Multiple Operators Of The Same Type
Default pattern:

```python
cir.cnot([[0, 1], [1, 2]])
```

This means two operators of the same type are inserted in sequence.

### One Multi-System Operator

```python
cir.cnot([0, 1])
```

This means one two-system operator.

### One Single-System Operator

```python
cir.h(0)
cir.h([0])
```

Both are accepted for single-system operators.

### One Call Over Several Single-System Targets

```python
cir.x([0, 1])
```

This is still one inserted child module. Internally, its `system_idx` records multiple one-system applications.

### Important Warning

`List[int]` does not mean the same thing for all APIs.

- for a one-system operator, it may mean one gate on one system or many gates on many one-system targets depending on that API
- for a multi-system operator, it often means one operator acting on multiple systems

When writing examples for unfamiliar methods, verify with code.

## Parameter Rules For Parameterized Gates

### Accepted Input Types

- `torch.Tensor`
- `torch.nn.Parameter`
- `numpy.ndarray`
- `float`
- `None`

### Shared Parameters

```python
cir.ry([0, 1, 2], param=theta, param_sharing=True)
```

`param_sharing=True` means all gates in that inserted family share one parameter set.

### Separate Parameters
Without `param_sharing=True`, each inserted operator needs its own parameter set.

### Common Shape Model
The common conceptual layout is:

```text
[1 or num_operators, batch_size, num_params_per_operator]
```

The actual input does not always need to be written in exactly that shape, as long as it can be reshaped to the expected internal format.

### `param=None`
If `param=None`, QuAIRKit creates trainable parameters automatically.

### `torch.nn.Parameter`
If `param` is a `torch.nn.Parameter`, it is registered by the operator.

### Plain Tensor / ndarray / float
These behave like fixed parameters from the module-management perspective.

Important nuance:
- a plain tensor can still participate in autograd
- but it is not treated the same way as a registered module parameter
- a fixed tensor/ndarray/scalar parameter does not appear in `list(circuit.parameters())`

## `universal_two_qubits`, `universal_three_qubits`, `universal_qudits`

### Two- and Three-Qubit Universal Gates

- `universal_two_qubits(...)`
- `universal_three_qubits(...)`

Use these when a built-in universal parameterization is enough.

### `universal_qudits`

Key rules:

- `system_idx` must be `List[int]`
- parameter rules match the general parameterized-gate family
- supports `param_sharing`
- supports `manifold=True`, use this to save memory and time in the price of lossing trainability; could be useful when the dimension is large, ex., at least 64
- supports `identity_init=True`, for verification

`universal_qudits` is a gate family, not a `Layer`.

## Noise Channels
Noise-channel insertion follows the same broad indexing pattern as operators.

Examples:

```python
cir.depolarizing(0.1, 0)
cir.amplitude_damping(0.2, 1)
cir.bit_flip(0.3, 0)
cir.bit_phase_flip(0.3, 2)
```

These insertion methods mutate the existing `Circuit` in place and return `None`. Use:

```python
cir = Circuit(2)
cir.depolarizing(0.1, 0)
out = cir(input_state)
```

Do not assume `Circuit(...).depolarizing(...)(state)` works as a fluent chain.

Important differences from parameterized gates:

- channel parameters are not managed as trainable gate parameters in the same special way
- `param=None` is not allowed for the usual parameterized channel path

## Layer Methods On `Circuit`

Built-in layer methods include:

- `linear_entangled_layer`
- `real_entangled_layer`
- `complex_entangled_layer`
- `real_block_layer`
- `complex_block_layer`
- `basis_encoding`
- `amplitude_encoding`
- `angle_encoding`
- `iqp_encoding`
- `trotter`

For trainable layers in `quairkit.ansatz.layer`, batched parameters follow the `[batch_size, total_param_num]` rule.

Runtime-verified `Circuit.trotter` signature:

```python
cir.trotter(hamiltonian, time, qubits_idx=None, num_steps=1, order=1, name="H")
```

Like other insertion methods on `Circuit`, `trotter(...)` mutates the circuit in place and returns `None`.
Use `time` for the total simulated evolution time and `num_steps` for the Trotter step count. When comparing approximation quality, vary `num_steps` while keeping `time` fixed.

## `append` Versus `extend`

### `append(layer)`

Preserves the layer as a child module.

### `extend(layer)`

Flattens the layer and inserts its internal operators.

This distinction matters for:

- indexing
- `operator_history`
- plotting a layer as a single component
- preserving submodule boundaries

Verified runtime behavior:
- `append(layer)` keeps one child whose type is the appended subcircuit/layer
- `extend(layer)` turns the layer into its internal operator children

## `oracle`

```python
cir.oracle(unitary, system_idx, control_idx=None, gate_name=None, latex_name=None)
```

### Uncontrolled Oracle

```python
cir.oracle(unitary, [0, 2], latex_name=r"U_1")
```

### Controlled Oracle

```python
cir.oracle(unitary, [[0, 1], 2, 3], control_idx=2, latex_name=r"V_2")
```

Interpretation:

- the first element can represent the control subsystem group
- the rest are target subsystems

For example:

- `[0, 2]` and `[[0], 2]` both describe a single control and one target in the controlled case
- `[[0, 1], 2]` means controls `[0, 1]` and target `2`

### `control_idx`

`control_idx` selects which computational-basis label on the control space triggers the oracle.

For qubits with two control qubits:

| `control_idx` | binary condition |
| --- | --- |
| `0` | `00` |
| `1` | `01` |
| `2` | `10` |
| `3` | `11` |

For qudits, the control space dimension is the product of the control subsystem dimensions.

For mixed-dimension controls, `control_idx` is interpreted in the corresponding computational-basis index of that product space.

### Labels

- `gate_name` affects the logical name
- `latex_name` affects the rendered Quantikz label

## `param_oracle`

```python
cir.param_oracle(generator, num_acted_param, system_idx, control_idx=None, param=None, gate_name=None, latex_name=None, support_batch=True)
```

Key rules:

- `generator` is `Callable[[torch.Tensor], torch.Tensor]`
- parameter rules are like parameterized gates
- there is no `param_sharing`
- `num_acted_param` is required
- `support_batch=False` is required if the generator cannot handle batched parameters
- `param=None` creates trainable parameters automatically

## `measure`

```python
cir.measure(system_idx=None, post_selection=None, if_print=False, measure_basis=None)
```

### Index Forms

- `None` means measure all systems
- `int`, `List[int]`, and some string alias paths are accepted

### Post-Selection

- `post_selection` supports one desired result
- if `if_print=True`, the average post-selection probability is printed

### Measurement Basis

- default is the computational basis
- `measure_basis` allows custom basis/PVM specification

Remember that `Circuit.measure` is a structural insertion API, not the same thing as `loss.Measure`.

Use `Circuit.measure(...)` when the measured-outcome distribution should remain part of the circuit output and appear in `State.probability`.
Use `State.measure(...)` or `loss.Measure(...)` when you already have a state object and want a post-processing measurement step instead of a structural circuit branch.

## `locc` And `param_locc`

### `locc`

```python
cir.locc(local_unitary, system_idx, label="M", latex_name="O")
```

### `param_locc`

```python
cir.param_locc(generator, num_acted_param, system_idx, param=None, label="M", latex_name="U", support_batch=True)
```

Key rules:

- the first part of `system_idx` is the measurement subsystem group
- later entries are the subsystems receiving local operations
- the supplied local unitary/generator must already carry the measurement-branch batch
- that batch size must equal the product of the measured subsystem dimensions
- the LOCC wrapper itself is not an outer batch of repeated LOCC blocks
- `param=None` creates a trainable parameter tensor with branch batch size

Recommended teleportation pattern (sequential single-qubit measurements):

```python
M1 = torch.stack([eye(), x()])   # corrections for qubit 1 outcome
M2 = torch.stack([eye(), z()])   # corrections for qubit 0 outcome
cir = Circuit(3)
cir.cnot([0, 1])
cir.h(0)
cir.locc(M1, [1, 2])   # measure qubit 1, correct qubit 2
cir.locc(M2, [0, 2])   # measure qubit 0, correct qubit 2
```

This two-step pattern is the standard LOCC idiom in QuAIRKit tutorials.

## `quasi` And `param_quasi`

### `quasi`

```python
cir.quasi(list_unitary, probability, system_idx, latex_name=r"\mathcal{E}")
```

### `param_quasi`

```python
cir.param_quasi(generator, num_acted_param, probability, system_idx, probability_param=False, param=None, latex_name=r"\mathcal{E}", support_batch=True)
```

Key rules:

- local unitary or generator must be batch-aware
- if `probability_param=False`, expected batch size matches the number of probabilities
- if `probability_param=True`, expected batch size is `len(probability) + 1`
- when `probability_param=True`, the probability vector itself becomes trainable
- `param=None` creates the trainable gate parameters automatically

Probabilities must be real-valued, not complex.

## Circuit Information Interfaces

- `system_idx`
- `system_dim`
- `num_systems`
- `num_qubits`
- `param`
- `grad`
- `operator_history`
- `matrix`
- `unitary_matrix()` as the old compatibility form
- `to_latex()`

Use `matrix` only when the circuit is unitary-only.

## Plotting And Quantikz

### `to_latex`

`cir.to_latex(style="standard", decimal=2)` returns Quantikz code.

### `plot`

Current public signature:

```python
cir.plot(style="standard", decimal=2, dpi=300, print_code=False, show_plot=True, include_empty=False)
```

Important implementation note:
- plotting depends on `pdflatex`
- if `pdflatex` is missing, QuAIRKit warns and skips rendering
- `print_code=True` still helps because it prints the LaTeX code

Tutorial note:
- some tutorial prose discusses a `latex` switch for Quantikz versus matplotlib fallback
- in the current source, the stable documented knobs are the explicit signature above plus the Quantikz-based default rendering path

### Plot Styles

- `standard`
- `compact`
- `detailed`

### What To Mention In Plotting Examples

- custom oracles and parametric oracles
- controlled oracles and `control_idx`
- noise channels
- parameterized layers
- `swap`, `cswap`, `permute`
- `measure`, post-selection, and `reset`
- `cir[i].plot()` for plotting a single child layer/operator list

If you need controlled permutations, mention the tutorial pattern explicitly:

```python
cir.permute(perm, [control] + targets, control_idx=some_basis_label)
```

### TeX Installation

- use TeX Live or MacTeX
- verify with `latex --version`
- on macOS, install `poppler` if PDF page-count errors occur

### Paper / arXiv Integration

To paste exported Quantikz code into LaTeX:

```latex
\usepackage{tikz}
\usetikzlibrary{quantikz}
```

For arXiv:
- Quantikz may not be installed on the archive side
- include `tikzlibraryquantikz2.code.tex` manually if needed
- or compile in Overleaf and export the figure PDF

## QASM2 Support

### Export

Use:

- `cir.qasm2`

Restrictions:

- only qubit systems are supported
- only a restricted OpenQASM 2.0 subset is supported
- classical instructions are not supported
- measurement is not supported in QASM2 export

### Import

Use:

```python
round_trip = Circuit.from_qasm2(qasm2_source)
```

## QPE-Friendly Pattern

When building controlled powers of a unitary for QPE-style code, a robust pattern is:

```python
power = torch.linalg.matrix_power(U, 2**k)
cir.oracle(power, [control, target], control_idx=1)
```

Prefer this over relying on an undocumented `.matrix_power(...)` method on the returned object.

For controlled parameterized gates such as `crz`, `cp`, and `cu`, the current calling convention is:

```python
cir.crz([control, target], param=theta)
cir.cp([control, target], param=theta)
cir.cu([control, target], param=params)
```

Pass the acted-on qubit indices first and the parameter as `param=...`.

## Working Advice

- prefer `Circuit` methods over low-level operator classes for end-user examples
- use `latex_name` and `label` to keep executable code and rendered circuit diagrams synchronized
- when documenting a method whose shape semantics are subtle, verify with code in the target environment before making a claim

## Common Pitfalls

- `List[int]` does not have one universal meaning. For one-system operators it can mean several one-system targets, while for multi-system operators it often means one operator acting on several systems.
- `param=None` creates and registers trainable parameters. Passing a plain tensor does not register a module parameter even if autograd can still see it.
- `Circuit` insertion methods are in-place APIs. Do not write fluent chains such as `Circuit(2).depolarizing(...)(state)` or assume `Circuit(...).trotter(...).matrix` works.
- `locc` expects the local unitary batch to match the measurement-branch count. That batch is the probability-branch axis, not an outer training batch of repeated LOCC blocks.
- Prefer the two-step teleportation idiom shown above instead of compressing teleportation into one larger multi-outcome correction unless you intentionally want that representation.
- `Circuit.measure` modifies circuit structure and later state shapes. It is not interchangeable with `loss.Measure`.
- `Circuit.trotter(...)` mutates the circuit in place and requires explicit `time` and `num_steps` choices; vary `num_steps` at fixed `time` when comparing Trotter accuracy.
- For `crz`, `cp`, and related controlled parameterized gates, do not pass the angle as the first positional argument. Use `cir.crz([control, target], param=theta)`.
