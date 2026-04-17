# !/usr/bin/env python3
# Copyright (c) 2023 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
The library of intrinsic functions of the QuAIRKit. 
NOT ALLOWED to be exposed to users.
"""

import copy
import functools
import gc
import io
import math
import warnings
from typing import (Callable, Dict, Iterable, List, Optional, Tuple, TypeVar,
                    Union)

import IPython
import numpy as np
import PIL
import torch
from torch.nn.parameter import Parameter

from . import base as _base
from .base import get_dtype, get_float_dtype, get_seed
from .state import StateOperator, StateSimulator, to_state


def _format_total_dim(num_systems: int, system_dim: Union[List[int], int]) -> int:
    r"""Given number of systems with dimension, check the dimension of the total system
    
    Args:
        num_systems: total number of systems
        system_dim: dimension of each system
        
    Returns:
        dimension of the total system
    
    """
    if isinstance(system_dim, int):
        return system_dim ** num_systems 
    
    assert len(system_dim) == num_systems, \
        f"Dimension of each system should be equal to the number of systems: received {system_dim} and {num_systems}"
    return math.prod(system_dim)


def _format_system_dim(total_dim: int, system_dim: Union[List[int], int]) -> List[int]:
    r"""Given total dimension with dimension of each system, check the dimension of each system
    
    Args:
        total_dim: dimension of the total system
        system_dim: dimension of each system (could be an int)
        
    Returns:
        dimension of each 
    
    """
    if isinstance(system_dim, int):
        num_system = int(math.log(total_dim, system_dim))
        system_dim = [system_dim] * num_system
        assert math.prod(system_dim) == total_dim, \
            f"Total system dimension {total_dim} does not match the dimension of each system {len(system_dim)}"
    else:
        assert math.prod(system_dim) == total_dim, \
            f"Dimension of each system should be equal to the total dimension: received {system_dim} and {total_dim}"
    return system_dim


def __norm(list_idx: Union[Iterable[int], int], n: int) -> Union[Iterable[int], int]:
    return [__norm(item, n) for item in list_idx] if (isinstance(list_idx, list) or isinstance(list_idx, tuple)) else int(list_idx) % n

def __find_max_abs(list_idx: Iterable[int]) -> int:
    stack = [iter(list_idx)]
    found_any = False
    best = 0

    while stack:
        try:
            x = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue

        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray)):
            stack.append(iter(x))
            continue

        if not isinstance(x, int):
            x = int(x)

        ax = -x if x < 0 else x
        if not found_any or ax > best:
            best = ax
            found_any = True

    if not found_any:
        raise ValueError("No integers found (empty iterable).")

    return best

def _format_sequential_idx(system_idx: Union[Iterable[int], int, str],
                           num_systems: int, num_acted_system: int) -> Iterable[int]:
    r"""Formatting the system indices in a sequential way, and translate negative indices to positive ones

    Args:
        system_idx: input system indices. Supports negative indices (counting from the end).
        num_systems: total number of systems
        num_acted_system: the number of systems that one operation acts on

    Note:
        The shape of output system indices are formatted as [# of vertical gates, num_acted_system].
    """
    if not isinstance(system_idx, str):
        if isinstance(system_idx, int):
            system_idx = [system_idx]
        
        assert (
            (max_idx := __find_max_abs(system_idx)) < num_systems
        ), (f"Invalid input system idx: {max_idx} cannot match with {num_systems} systems." + 
            " One may change the number of systems by calling property `num_systems` or method `add_systems`.")
            
        return [__norm(item, num_systems) for item in system_idx]
    
    if num_acted_system == 1:
        if system_idx == 'full':
            system_idx = list(range(num_systems))
        elif system_idx == 'even':
            system_idx = list(range(num_systems, 2))
        elif system_idx == 'odd':
            system_idx = list(range(1, num_systems, 2))
        else:
            raise ValueError(
                f"Unsupported system_idx string for single-system operator: received {system_idx}")

    elif system_idx == 'cycle':
        assert num_systems >= num_acted_system, \
            f"# of qubits should be >= # of acted qubits: received {num_systems} and {num_acted_system}"
        system_idx = [
            list(range(idx, idx + num_acted_system))
            for idx in range(num_systems - num_acted_system)
        ]
        system_idx.extend(
            list(range(idx, num_systems))
            + list(range(idx + num_acted_system - num_systems))
            for idx in range(num_systems - num_acted_system, num_systems)
        )

    elif system_idx == 'linear':
        assert num_systems >= num_acted_system, \
            f"# of system should be >= # of acted systems: received {num_systems} and {num_acted_system}"

        system_idx = [
            list(range(idx, idx + num_acted_system))
            for idx in range(num_systems - num_acted_system + 1)
        ]
        
    else:
        raise ValueError(
            f"Unsupported system_idx string for multi-system operator: received {system_idx}")

    return system_idx


def _format_operator_idx(
        system_idx: Union[List[List[int]], List[int], int], num_acted_system: int
) -> List[List[int]]:
    r"""Formatting the system indices that operations acts on

    Args:
        system_idx: input system indices
        num_acted_system: the number of systems that one operation acts on

    Note:
        The shape of output system indices are formatted as [# of vertical gates, num_acted_system].
    """
    
    system_idx = np.array(system_idx).reshape([-1, num_acted_system])
    assert system_idx.shape == np.unique(system_idx, axis=1).shape, \
        f"Operators do not allow repeated indices: received {system_idx.tolist()}"
    return system_idx.tolist()


def _infer_num_acted_system(
        system_idx: Union[List[Union[List[int], int]], int]
) -> int:
    r"""Infer number of acted systems from raw operator indices.

    Args:
        system_idx: raw operator indices before formatting.

    Returns:
        Number of systems that one operation acts on.
    """
    if isinstance(system_idx, int):
        return 1

    if not isinstance(system_idx, (list, tuple)) or len(system_idx) == 0:
        raise ValueError(f"Invalid system_idx: received {system_idx}")

    first = system_idx[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        if len(first) == 0:
            raise ValueError("Invalid empty system index group.")
        return len(first)
    return len(system_idx)


def _format_param_shape(system_idx: List[List[int]], num_acted_param: int, 
                        param_sharing: bool, batch_size: int = 1) -> List[int]:
    r"""Formatting the shape of parameters of param gates

    Args:
        system_idx: list of input system indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set
        batch_size: size of gate batch, -1 for variable batch size

    Note:
        The input ``system_idx`` must be formatted by ``_format_system_idx`` first.
        The shape of parameters are formatted as [len(system_idx), batch_size, num_acted_param].

    """
    return [1 if param_sharing else len(system_idx), batch_size, num_acted_param]


def _theta_generation(net: torch.nn.Module, param: Optional[Union[torch.Tensor, float, List[float]]], 
                      system_idx: List[List[int]], num_acted_param: int, param_sharing: bool) -> None:
    r""" determine net.theta, and create parameter if necessary

    Args:
        net: neural network instance
        param: input theta
        system_idx: list of input system indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set

    Note:
        ``param`` will be transformed to a parameter:
        - if ``param`` is ``None``
        
        or ``param`` will be added to the parameter list:
        - if ``param`` is a Parameter instance (reshaped to expect shape if needed)
        
        or ``param`` will keep unchanged:
        - if ``param`` is a Tensor but not a ParamBase
        - if ``param`` is an array of floats
        
        ``param`` will be shared by all operations:
        - if ``param_sharing`` is True
        - if ``param`` is a float scalar
    """
    float_dtype = _get_float_dtype(net.dtype)
    if param is None:
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, batch_size=1)
        theta = torch.rand(expect_shape, dtype=float_dtype) * 2 * math.pi
        net.register_parameter('theta', Parameter(theta))
        return
    
    if isinstance(param, (int, float)):
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, batch_size=1)
        net.theta = torch.ones(expect_shape, dtype=float_dtype) * param
        return
    
    if isinstance(param, Parameter):
        batch_size = int(param.numel() // (1 if param_sharing else len(system_idx)) // num_acted_param)
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, batch_size=batch_size)
        
        assert param.numel() == math.prod(expect_shape), \
            f"Numel mismatch for input parameter: receive {param.numel()}, expect {math.prod(expect_shape)}"
        assert param.dtype == (torch.float32 if float_dtype == torch.float32 else torch.float64), \
            f"Dtype assertion failed for input parameter: receive {param.dtype}, expect {float_dtype}"
        if list(param.shape) != expect_shape:
            param = Parameter(param.data.reshape(expect_shape), requires_grad=param.requires_grad)
        net.register_parameter('theta', param)
    
    elif isinstance(param, torch.Tensor):
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, -1)
        net.theta = param.to(net.device, dtype=float_dtype).reshape(expect_shape)
    
    else:
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, -1)
        net.theta = torch.tensor(param, dtype=float_dtype).view(expect_shape)


def _get_float_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex128:
        float_dtype = torch.float64
    else:
        raise ValueError(
            f"The dtype should be torch.complex64 or torch.complex128: received {complex_dtype}")
    return float_dtype


def _get_complex_dtype(float_dtype: torch.dtype) -> torch.dtype:
    if float_dtype == torch.float32:
        complex_dtype = torch.complex64
    elif float_dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(
            f"The dtype should be torch.float32 or torch.float64: received {float_dtype}")
    return complex_dtype


_ArrayLike = Union[np.ndarray, torch.Tensor]
_State = Union[StateSimulator, StateOperator]
_StateLike = Union[np.ndarray, torch.Tensor, _State]
_ParamLike = Union[np.ndarray, torch.Tensor, Iterable[float]]
_SingleParamLike = Union[_ParamLike, float]
_General = Union[_ArrayLike, _StateLike, _SingleParamLike]


def _type_fetch(data: _General, 
                ndim: Optional[int]= None) -> Union[str, Tuple[str, List[int]]]:
    r"""Fetch the type of ``data``

    Args:
        data: the input data, and datatype of which should be either ``numpy.ndarray``,
            ``torch.Tensor``, ``quairkit.State``, ``Iterable[float]`` or ``float``
            where the last two types will be considered as "tensor".
        ndim: the number of dimensions to be removed, used for batched data

    Returns:
        When ndim is not none, the returned tuple contains the following variables:
        - a string of datatype of ``data``, can be either ``"numpy"``, ``"tensor"``, ``"state"``, or ``"other"``
        - the batch dimension
        When ndim is none, the returned variable is just the string.

    """
    if isinstance(data, np.ndarray):
        return "numpy" if ndim is None else ("numpy", list(data.shape[:-ndim]))

    if isinstance(data, torch.Tensor):
        return "tensor" if ndim is None else list(data.shape[:-ndim])
    
    if isinstance(data, StateSimulator):
        return "state" if ndim is None else ("state", list(data.batch_dim))

    assert ndim is None, \
        ("Cannot obtain batch dimension for data type other than " +
         f"np.ndarray, torch.Tensor or quairkit.State: received {type(data)}")
    return "other"


def _type_transform(data: _General, output_type: str, 
                    system_dim: Optional[Union[List[int], int]] = None) -> _StateLike:
    r""" transform the datatype of ``input`` to ``output_type``

    Args:
        data: data to be transformed, can be and datatype of which should be either ``numpy.ndarray``,
            ``quairkit.State``, ``float``, ``Iterable[float]`` or ``torch.Tensor``
        output_type: datatype of the output data, type is either ``"numpy"``, ``"state"``, ``"tensor"``
        system_dim: dimension of the system, used for transforming to state. Defaults to qubit case.

    Returns:
        the output data with expected type

    Raises:
        ValueError: does not support transformation to type.

    """
    current_type = _type_fetch(data)

    if current_type == "other":
        current_type = "tensor"
        data = torch.tensor(data)

    if output_type == "other":
        output_type = "tensor"
    else:
        assert output_type in {"numpy", "tensor", "state"}, \
            f"does not support transformation from {current_type} to type {output_type}"

    if current_type == output_type:
        return data


    if current_type == "numpy":
        if output_type == "tensor":
            return torch.from_numpy(data)

        if system_dim is None:
            shape = data.shape
            system_dim = shape[-2] if (len(shape) >= 2 and shape[-1] == 1) else shape[-1]
        else:
            system_dim = _format_system_dim(data.shape[-1], system_dim)

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "tensor":
        if output_type == "numpy":
            return data.detach().cpu().resolve_conj().numpy()

        if system_dim is None:
            shape = data.shape
            system_dim = shape[-2] if (len(shape) >= 2 and shape[-1] == 1) else shape[-1]
        else:
            system_dim = _format_system_dim(data.shape[-1], system_dim)

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "state":
        if output_type == "numpy":
            return data.numpy()
        if hasattr(data, "_data_tensor"):
            return data._data_tensor.clone()
        return data._data.clone()
    
    raise ValueError(
        f"does not support transformation from {current_type} to type {output_type}: received {data}")


def _flatten_batch(x: torch.Tensor, tail_ndim: int) -> Tuple[torch.Tensor, List[int]]:
    r"""Flatten all leading batch dimensions into a single batch dim.

    This helper is intended to be used by *external* interfaces before calling
    strict internal implementations.

    Examples:
        - x.shape == [d1, d2, n], tail_ndim=1  -> ([B, n], batch_shape=[d1, d2])
        - x.shape == [d1, d2, m, n], tail_ndim=2 -> ([B, m, n], batch_shape=[d1, d2])
    """
    if tail_ndim < 0:
        raise ValueError(f"tail_ndim must be >= 0, got {tail_ndim}")
    if x.ndim < tail_ndim:
        raise ValueError(f"x.ndim must be >= tail_ndim, got x.ndim={x.ndim}, tail_ndim={tail_ndim}")

    batch_shape: List[int] = list(x.shape[:-tail_ndim]) if tail_ndim > 0 else list(x.shape)
    tail_shape: List[int] = list(x.shape[-tail_ndim:]) if tail_ndim > 0 else []
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    return x.reshape([batch_size] + tail_shape), batch_shape


def _unflatten_batch(y: torch.Tensor, batch_shape: List[int]) -> torch.Tensor:
    r"""Inverse of `_flatten_batch` for outputs shaped as [B, ...]."""
    if not isinstance(batch_shape, list):
        batch_shape = list(batch_shape)
    if len(batch_shape) == 0:
        return y.squeeze(0)
    return y.reshape(batch_shape + list(y.shape[1:]))


def _ensure_param_2d(theta: torch.Tensor, n: int) -> Tuple[torch.Tensor, List[int]]:
    r"""Normalize parameter-like input to shape [B, n] and return original batch_shape.

    Accepted inputs:
        - scalar (only when n == 1)
        - [n] (single, unbatched)
        - [B] (only when n == 1, treated as batched scalars)
        - [..., n] (multi-batch)
    """
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if theta.ndim == 0:
        if n != 1:
            raise ValueError(f"Scalar parameter only supported for n==1, got n={n}")
        return theta.reshape(1, 1), []

    if theta.ndim == 1:
        if theta.shape[0] == n:
            return theta.reshape(1, n), []
        if n == 1:
            return theta.reshape(theta.shape[0], 1), [int(theta.shape[0])]
        raise ValueError(
            f"1D parameter must have shape [n] (n={n}) or [B] (only when n==1), got {list(theta.shape)}"
        )

    if theta.shape[-1] != n:
        raise ValueError(f"Parameter last dim mismatch: expect [...,{n}], got {list(theta.shape)}")
    theta2, batch_shape = _flatten_batch(theta, tail_ndim=1)
    return theta2, batch_shape


def _ensure_mat_3d(mat: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    r"""Normalize matrix-like input to shape [B, d, d] and return original batch_shape."""
    if not isinstance(mat, torch.Tensor):
        raise TypeError(f"mat must be torch.Tensor, got {type(mat)}")
    if mat.ndim < 2:
        raise ValueError(f"mat.ndim must be >= 2, got {mat.ndim}")
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f"mat must be square on last 2 dims, got {list(mat.shape)}")
    mat3, batch_shape = _flatten_batch(mat, tail_ndim=2)
    return mat3, batch_shape


def _ensure_vec_3d(vec: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    r"""Normalize state-vector-like input to shape [B, d, 1] and return original batch_shape."""
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"vec must be torch.Tensor, got {type(vec)}")
    if vec.ndim < 2:
        raise ValueError(f"vec.ndim must be >= 2, got {vec.ndim}")
    if vec.shape[-1] != 1:
        raise ValueError(f"State vector must have last dim 1, got {list(vec.shape)}")
    vec3, batch_shape = _flatten_batch(vec, tail_ndim=2)
    return vec3, batch_shape


def _ensure_set_4d(set_op: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    r"""Normalize operator-set input to shape [B, K, d, d] and return original batch_shape."""
    if not isinstance(set_op, torch.Tensor):
        raise TypeError(f"set_op must be torch.Tensor, got {type(set_op)}")
    if set_op.ndim < 3:
        raise ValueError(f"set_op.ndim must be >= 3, got {set_op.ndim}")
    if set_op.shape[-1] != set_op.shape[-2]:
        raise ValueError(f"set_op must be square on last 2 dims, got {list(set_op.shape)}")
    set4, batch_shape = _flatten_batch(set_op, tail_ndim=3)
    return set4, batch_shape


def _assert_gradient(func: Union[Callable[[torch.Tensor], torch.Tensor], str], param: torch.Tensor, 
                     eps: float = 1e-2, atol: float = 1e-2, rtol: float = 1e-2, nondet_tol: float = 0.0,
                     feedback: bool = False, message: str = None) -> None:
    r"""
    Check the correctness of the gradient computation using finite difference method.

    Args:
        function_name: function to be tested.
        param: variables that perform finite differences.
        eps: perturbation for finite differences. Defaults to 1e-2.
        atol: absolute tolerance. Defaults to 1e-3.
        rtol: relative tolerance. Defaults to 1e-2.
        nondet_tol: tolerance for nondeterministic operations, for randomized functions only. Defaults to 0.
        feedback: whether to print the result for confirmation. Defaults to False.
        message: optional message to be printed if the gradient check fails. Defaults to None.
    
    """
    seed, dtype = get_seed(), get_float_dtype()
    assert dtype == torch.float64, \
        "The gradient check only supports double precision, which can be set by `quairkit.set_dtype('complex128')`."
    param = param.to(dtype).clone().requires_grad_(True)
    
    result = False
    last_exc: Optional[Exception] = None
    for trial_eps in (eps, eps * 0.1, eps * 0.01):
        if trial_eps <= 0:
            continue
        try:
            result = torch.autograd.gradcheck(
                func,
                param,
                eps=trial_eps,
                atol=atol,
                rtol=rtol,
                raise_exception=True,
                nondet_tol=nondet_tol,
            )
            if result:
                break
        except Exception as e:
            last_exc = e
            result = False
            continue
    if not result and last_exc is not None:
        warnings.warn(
            "The following error occurred when trying to compute the gradient\n" + str(last_exc),
            RuntimeWarning,
        )
    
    assert result, (
        "Gradient test failed \n"
        f"Setting: seed {seed}, precision {atol}, message {message} "
    )
        
    if feedback:
        print(f"Gradient test with seed {seed} passed",
              f"under abs tolerance {atol} and rel tolerance {rtol}.")


def _is_sample_linear(
    func: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
    info: Union[List[int], Callable[[], torch.Tensor], Callable[[], np.ndarray]],
    input_dtype: torch.dtype = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[], torch.Tensor], torch.dtype, str]:
    r"""Check whether the inputs in `is_linear`, `create_matrix`, `create_choi_repr` are legal

    Args:
        func: A callable function to be tested.
        info: Information of the sample function, either the shape of sample data or the function itself.
        input_dtype: The data type of the input tensor. Defaults to None.

    Returns:
        A tuple containing:
        - A function to be tested, which ensures the input and output are torch.Tensors.
        - A function that generates input samples as torch.Tensors.
        - The data type of the input tensor.
        - A string representing the original type of the input function's output.

    Raises:
        TypeError: The input arguments are not of expected types.
    """
    assert isinstance(
        func, Callable
    ), f"The input 'func' must be callable, received {type(func)}"

    generator = info

    if input_dtype is None:
        input_dtype = get_dtype()
    elif not isinstance(input_dtype, torch.dtype):
        raise TypeError(f"Input must be of type torch.dtype or None. Received type: {type(input_dtype)}")

    if isinstance(info, list) and all((isinstance(x, int) and x > 0) for x in info):
        generator = lambda: torch.randn(info, dtype=input_dtype)
        sample_tensor = generator()
        func_in = func

    elif isinstance(info, Callable):
        sample = info()

        if isinstance(sample, np.ndarray):
            generator = lambda: torch.from_numpy(info())
            func_in = func
            sample_tensor = torch.from_numpy(sample)
        elif isinstance(sample, torch.Tensor):
            generator = lambda: info()
            func_in = func
            sample_tensor = sample
        else:
            raise TypeError(
                f"The output of info must be a torch.Tensor or numpy.ndarray: received {type(sample)}"
            )

    else:
        raise TypeError(
            "the info entry should either the shape of input data, or a Callable functions: "
            + f"received info {info} with type {type(info)}"
        )

    is_output_numpy = False
    is_output_tensor = False
    
    if isinstance(sample_tensor, torch.Tensor):
        try:
            sample_numpy = sample_tensor.detach().cpu().numpy()
            output_numpy = func_in(sample_numpy)
            is_output_numpy = isinstance(output_numpy, np.ndarray)
            is_output_tensor = isinstance(output_numpy, torch.Tensor)
        except (TypeError, AttributeError, RuntimeError):
            output = func_in(sample_tensor)
            is_output_numpy = isinstance(output, np.ndarray)
            is_output_tensor = isinstance(output, torch.Tensor)
    else:
        output = func_in(sample_tensor)
        is_output_numpy = isinstance(output, np.ndarray)
        is_output_tensor = isinstance(output, torch.Tensor)
    
    if not (is_output_numpy or is_output_tensor):
        raise TypeError(
            f"The input function either return torch.Tensor or np.ndarray: received type {type(output)}"
        )

    output_type_str = "numpy" if is_output_numpy else "tensor"
    
    func_tensor = func_in
    if is_output_numpy:
        def wrapped_func_tensor(mat):
            if isinstance(mat, torch.Tensor):
                mat_np = mat.detach().cpu().numpy()
                result = func_in(mat_np)
            else:
                result = func_in(mat)
            if isinstance(result, np.ndarray):
                return torch.from_numpy(result)
            return result
        func_tensor = wrapped_func_tensor

    return func_tensor, generator, input_dtype, output_type_str


def _alias(aliases: Dict[str, str]) -> Callable:
    r"""alias decorator for function arguments
    
    Args:
        aliases: a dictionary of alias names. The key is the original name, and the value is the alias name.
    
    Returns:
        A decorator that replaces the alias name with the original name.
    
    Note:
        See stack overflow #29374425
    
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for name, alias in aliases.items():
                if name not in kwargs and alias in kwargs:
                    kwargs[name] = copy.deepcopy(kwargs[alias])
                    del kwargs[alias]
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _digit_to_int(digits: str, base: Union[int, List[int]]) -> int:
    r"""Convert a string of digits in a given base to an integer
    
    Args:
        digits: a string of digits to be converted
        base: the base of the digits, or a list of bases for each digit
    """
    if isinstance(base, int):
        return int(digits, base=base)
    
    number = 0
    multiplier = 1
    for digit, base in zip(reversed(digits), reversed(base)):
        if not digit.isdigit() or int(digit) >= base:
            raise ValueError(
                f"Digit '{digit}' is invalid for base {base}.")
        
        number += int(digit) * multiplier
        multiplier *= base   
    return number


def _int_to_digit(number: int, base: Union[int, List[int]]) -> str:
    r"""Convert an integer to a string of digits in a given base
    
    Args:
        number: the integer to be converted
        base: the base to convert the integer, or a list of bases for each digit
    """
    if number < 0:
        return f'-{_int_to_digit(-number, base)}'

    if isinstance(base, int):
        return np.base_repr(number, base=base)

    digits = []
    for base in reversed(base):
        digits.append(str(number % base))
        number //= base
    return ''.join(reversed(digits))


def _replace_indices(l: Iterable[int], 
                     m: Union[List[int], Dict[int, int]]) -> Iterable[int]:
    r"""Replace the indices in l with the corresponding elements in m
    
    Args:
        l: a nested list to be replaced, where each element is an index
        m: a list or a dictionary of replacement
        
    Returns:
        a mapped list
    """
    def replace_recursive(element):
        if isinstance(element, Iterable):
            return [replace_recursive(sub_element) for sub_element in element]
        else:
            return m[element]

    return replace_recursive(l)


def __is_jupyter() -> bool:
    r"""Check whether the current environment is Jupyter notebook
    
    Returns:
        True if the current environment is Jupyter notebook
    """
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False

def _display_png(img: IPython.display.Image) -> None:
    r"""Display input picture that is stored using IPython
    
    Args:
        img: the image stored in png to be displayed
    """
    if __is_jupyter():
        IPython.display.display(img)
    else:
        pil_img = PIL.Image.open(io.BytesIO(img.data))
        pil_img.show()


def _merge_qasm(qasm1: Union[str, List[str]], qasm2: Union[str, List[str]]) -> Union[str, List[str]]:
    r"""Merge two qasm which can be Union[str, List[str]]

        Args:
            qasm1: first qasm
            qasm2: second qasm

        Returns:
            A string if both inputs are strings else a list of strings.
    """
    if isinstance(qasm1, str):
        if isinstance(qasm2, str):
            qasm1 += qasm2
        else:
            qasm1 = [qasm1 + item for item in qasm2]
    elif isinstance(qasm2, str):
        qasm1 = [item + qasm2 for item in qasm1]
    else:
        if len(qasm1) != len(qasm2):
            raise ValueError("The circuit has different batch sizes for different gates!")
        for ii in range(len(qasm1)):
            qasm1[ii] += qasm2[ii]
    return qasm1


def _reset_all(
    *vars: object,
    force_gc: bool = True,
    clear_cuda_cache: bool = True,
    reset_globals: bool = False,
) -> Tuple[None, ...]:
    r"""Reset QuAIRKit global defaults and help release memory held by passed variables.

    Args:
        *vars: Any number of variables to be reset. The function returns the same
            number of ``None`` so users can write ``x, y = _reset_all(x, y)``.
        force_gc: Whether to force Python garbage collection. Defaults to True.
        clear_cuda_cache: Whether to clear CUDA cache when available. Defaults to True.
        reset_globals: Whether to reset QuAIRKit global defaults (device/dtype/seed)
            and sync PyTorch defaults. Defaults to False.

    Returns:
        A tuple of ``None`` with the same length as the input ``vars``.

    Note:
        This function cannot forcibly free objects that are still referenced elsewhere.
        The intended usage is to drop references by assigning the returned ``None``.

        If any input is a ``torch.Tensor`` (including ``torch.nn.Parameter``) with
        ``requires_grad=True``, a warning will be emitted to avoid accidental misuse
        in training code.
    """
    for v in vars:
        if isinstance(v, torch.Tensor) and getattr(v, "requires_grad", False):
            warnings.warn(
                "_reset_all received a Tensor/Parameter with requires_grad=True. "
                "This may break training if called on model parameters or graph intermediates. "
                "Consider detaching or avoid resetting during training.",
                UserWarning,
            )
            break

    if reset_globals:
        try:
            _base.set_device("cpu")
        except Exception:
            pass
        try:
            _base.set_dtype("complex64")
        except Exception:
            pass
        try:
            _base.SEED = None
        except Exception:
            pass

    if clear_cuda_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    if force_gc:
        try:
            gc.collect()
        except Exception:
            pass

    return (None,) * len(vars)
