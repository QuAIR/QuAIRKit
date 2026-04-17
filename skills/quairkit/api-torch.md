# PyTorch And NumPy Integration

## Scope

Use this file for:

- `Circuit` as a `torch.nn.Module`
- parameter registration and optimization
- standard training loops
- hybrid classical-quantum models
- dtype/device handling
- autograd behavior
- NumPy/Torch interop rules

## Module Hierarchy

- `Circuit -> OperatorList -> torch.nn.Sequential -> torch.nn.Module`
- individual quantum operators ultimately derive from `Operator`, which is also a `torch.nn.Module`

This is why QuAIRKit works naturally with:

- `parameters()`
- optimizers
- schedulers
- custom `torch.nn.Module` wrappers

## Parameter Management

### `param=None`

For parameterized gates/layers/oracles, `param=None` usually means:

- QuAIRKit creates a trainable parameter
- the parameter is registered as a `torch.nn.Parameter`
- initialization is random unless the API says otherwise

### Passing `torch.nn.Parameter`

If you pass an explicit `torch.nn.Parameter`, it is registered and managed as part of the module.

### Passing Tensor / ndarray / float

If you pass a non-`Parameter` tensor, NumPy array, or scalar:

- the operator behaves like a fixed-parameter operator
- a plain tensor can still participate in autograd if it is part of a larger graph
- but it is not automatically registered as a trainable module parameter

Verified runtime behavior:

- `param=None` registers trainable parameters
- explicit `torch.nn.Parameter` stays registered
- a plain tensor does not appear in `list(module.parameters())`

## Standard Training Template

This is the default training-loop template for QuAIRKit work in this project.

- `loss_fcn(model)` is the training objective
- `fidelity_fcn(model)` is the validation metric
- the validation metric may be omitted for simple no-batch tasks such as basic VQE, but keeping it is generally preferred
- the same template works for a bare `Circuit` or for a larger `torch.nn.Module`

```python
import time
import numpy as np
import torch

from quairkit import Circuit

PRINT_TIMES: int = ...
LR: float = ...
NUM_ITR: int = ...

def loss_fcn(circuit: Circuit) -> torch.Tensor:
    r"""Training objective.
    Use a numerically stable surrogate that is easy to optimize, even if it is
    not identical to the final success metric.
    """
    ...
    pass

def fidelity_fcn(circuit: Circuit) -> torch.Tensor:
    r"""Validation metric.
    Use the metric that best reflects the real task goal.
    """
    ...
    pass

loss_list, time_list = [], []

opt = torch.optim.Adam(lr=LR, params=cir.parameters()) # cir is a Circuit type
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5) # activate scheduler

for itr in range(NUM_ITR):
    start_time = time.time()
    opt.zero_grad()

    loss = loss_fcn(cir) # compute loss

    loss.backward()
    opt.step()

    loss = loss.item()
    scheduler.step(loss)

    loss_list.append(loss)
    time_list.append(time.time() - start_time)

    if itr % (NUM_ITR // PRINT_TIMES) == 0 or itr == NUM_ITR - 1:
        fidelity = fidelity_fcn(cir).item() # compute fidelity
        print(f"iter: {str(itr).zfill(len(str(NUM_ITR)))}, " +
              f"loss: {loss:.8f}, fidelity: {fidelity:.8f}, " +
              f"lr: {scheduler.get_last_lr()[0]:.2E}, avg_time: {np.mean(time_list):.4f}s")
        time_list = []
```

## `loss_fcn` Versus `fidelity_fcn`

### `loss_fcn`

Use `loss_fcn` for optimization.

It does not need to be identical to the final task metric.
It should be:

- differentiable enough for the task
- numerically stable
- reasonably easy to optimize

Typical choices:

- MSE-style objectives
- averaged expectation values
- averaged trace distance

### `fidelity_fcn`

Use `fidelity_fcn` for validation.

It should reflect the real task goal, even if it is not the easiest quantity to optimize directly.

Typical choices:

- minimum state fidelity across a batch
- task-specific accuracy-like figures
- a post-selected success probability

### Why Separate Them

Some metrics are ideal for evaluation but awkward or unstable for direct training.
For example:

- a fidelity-like metric may be the most meaningful final target
- a surrogate such as trace distance or MSE may train more smoothly

## Hybrid Classical-Quantum Models

The same template applies to hybrid models from `torch.nn.Module`.

Typical pattern:

1. store trainable classical parameters or submodules on `self`
2. build or reuse one or more `Circuit` objects
3. turn final quantum outputs into classical tensors using `ExpecVal`, `Measure`, or explicit state processing
4. feed those tensors into standard PyTorch layers

Use `net.parameters()` instead of `cir.parameters()` when optimizing the full hybrid model.

## Dtype And Device Rules

### Global Control

- `qkit.set_dtype("complex64")`
- `qkit.set_dtype("complex128")`
- `qkit.set_device("cpu")`
- `qkit.set_device("cuda")`

### Practical Rule

When you construct tensors manually, keep them compatible with the active QuAIRKit dtype/device.

Typical examples:

```python
qkit.set_dtype("complex128")
qkit.set_device("cpu")

theta = torch.tensor([0.1, 0.2], dtype=qkit.get_float_dtype())
```

Use `get_float_dtype()` for real-valued trainable parameters and probability vectors.

## Autograd Notes

- simulator workflows are PyTorch-native and support autograd naturally
- execution-style backends use custom autograd logic where needed
- converting tensors to NumPy via `.detach().cpu().numpy()` breaks gradient flow
- some destructive state operations may interrupt gradients; do not assume every state-manipulation path is gradient-preserving

One explicit caution:

- resetting the whole state can break gradient flow in simulator code paths

## NumPy Interop Rules

The exact per-family rules come from QuAIRKit's `_type_fetch` / `_type_transform` mechanism plus individual implementations. For detailed NumPy/Tensor behavior, see `api-database-qinfo.md`, especially `Database Signature Patterns`, `Qinfo Signature Patterns`, and `High-Frequency Runtime Facts`.

## `to_state`

- accepts NumPy arrays
- converts them into simulator-state data using the current QuAIRKit dtype/device

## `State.numpy()`

Use `state.numpy()` when a NumPy array is explicitly required for external analysis or plotting.

## Frequently Used Torch APIs In QuAIRKit Workflows

### Construction

- `torch.tensor`
- `torch.eye`
- `torch.stack`
- `torch.cat`
- `torch.kron`

### Linear Algebra

- `torch.linalg.matrix_power`
- `torch.linalg.matrix_exp`
- `torch.linalg.eigvalsh`

### Logging / Aggregation

- `.mean()`
- `.item()`

## Recommended Defaults

- Use `Circuit` directly when the model is purely quantum and parameter management is straightforward.
- Use a custom `torch.nn.Module` when combining quantum computation with classical preprocessing or postprocessing.
- Keep `loss_fcn` and `fidelity_fcn` separate unless the task is simple enough not to need the distinction.

## Common Pitfalls

- `loss.backward()` must happen before converting `loss` to a Python scalar with `.item()`. After `loss = loss.item()`, the computational graph is gone.
- `.detach().cpu().numpy()` breaks gradient flow. Only use it for logging, plotting, or non-differentiable post-processing.
- Passing a plain tensor as `param` does not make it appear in `model.parameters()`. Use `param=None` or `torch.nn.Parameter` for trainable module parameters.
- Do not assume all `database` helpers preserve NumPy round-tripping. Fixed constructors are often tensor-only, while parameterized helpers mirror the input type more often.
