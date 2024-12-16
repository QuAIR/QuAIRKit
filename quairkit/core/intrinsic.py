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
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

from .base import Operator, get_dtype, get_float_dtype, get_seed
from .state import State, to_state


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


def _format_circuit_idx(system_idx: Union[List[List[int]], List[int], int, str],
                        num_systems: Union[int, None], num_acted_system: int) -> List[List[int]]:
    r"""Formatting the system indices in a circuit

    Args:
        system_idx: input system indices
        num_systems: total number of systems
        num_acted_system: the number of systems that one operation acts on

    Note:
        The shape of output system indices are formatted as [# of vertical gates, num_acted_system].
    """
    if not isinstance(system_idx, str):
        return system_idx
    
    assert (
        not isinstance(system_idx, str) or num_systems is not None
    ), f"Cannot specify the system indices when num_systems is None: received system_idx {system_idx} and num_systems {num_systems}"

    if num_acted_system == 1:
        if system_idx == 'full':
            system_idx = list(range(num_systems))
        elif system_idx == 'even':
            system_idx = list(range(num_systems, 2))
        elif system_idx == 'odd':
            system_idx = list(range(1, num_systems, 2))

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
    return system_idx


def _format_layer_idx(qubits_idx: Union[List[int], str], num_qubits: int) -> List[int]:
    r"""Check the validity of ``qubits_idx`` and ``num_qubits``.

    Args:
        qubits_idx: Indices of qubits.
        num_qubits: Total number of qubits.

    Raises:
        RuntimeError: You must specify ``qubits_idx`` or ``num_qubits`` to instantiate the class.
        ValueError: The ``qubits_idx`` must be ``Iterable`` or ``None``.

    Returns:
        Checked indices of qubits.
    """
    if qubits_idx is None or qubits_idx == 'full':
        if num_qubits is None:
            raise RuntimeError(
                "You must specify qubits_idx or num_qubits to instantiate the class.")
        return list(range(num_qubits))
    elif isinstance(qubits_idx, Iterable):
        assert len(np.array(qubits_idx).shape) == 1, \
            "The input qubit index must be a list of int for layers."
        qubits_idx = list(qubits_idx)
        assert len(qubits_idx) > 1, \
            f"Requires more than 1 qubit for a layer to act on: received length {len(qubits_idx)}"
        assert len(qubits_idx) == len(set(qubits_idx)), \
            f"Layers do not allow repeated indices: received {qubits_idx}"
        return qubits_idx

    raise ValueError(f"The qubits_idx must be a list of int or None: received {type(qubits_idx)}")


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


def _format_param_shape(system_idx: List[List[int]], num_acted_param: int, 
                        param_sharing: bool, batch_size: int = 1) -> List[int]:
    r"""Formatting the shape of parameters of param gates

    Args:
        system_idx: list of input system indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set
        batch_size: size of gate batch

    Note:
        The input ``system_idx`` must be formatted by ``_format_system_idx`` first.
        The shape of parameters are formatted as [len(system_idx), batch_size, num_acted_param].

    """
    return [1 if param_sharing else len(system_idx), batch_size, num_acted_param]


def _theta_generation(net: Operator, param: Union[torch.Tensor, float, List[float]], 
                      system_idx: List[List[int]], num_acted_param: int, param_sharing: bool) -> None:
    r""" determine net.theta, and create parameter if necessary

    Args:
        net: neural network instance
        param: input theta
        system_idx: list of input system indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set

    Note:
        In the following cases ``param`` will be transformed to a parameter:
            - ``param`` is ``None``
        or ``param`` will be added to the parameter list:
            - ``param`` is a ParamBase instance
        or ``param`` will keep unchanged:
            - ``param`` is a Tensor but not a ParamBase
            - ``param`` is an array of floats
        In the following cases ``param`` will be shared by all operations:
            - ``param_sharing`` is True, or
            - ``param`` is a float scalar
    """
    float_dtype = _get_float_dtype(net.dtype)
    expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing)
    
    # TODO unify support for batch and non-batch case
    
    if param is None:
        theta = torch.rand(expect_shape, dtype=float_dtype) * 2 * math.pi
        net.register_parameter('theta', Parameter(theta))
    
    elif isinstance(param, Parameter):
        assert list(param.shape) == expect_shape, \
            f"Shape assertion failed for input parameter: receive {list(param.shape)}, expect {expect_shape}"
        assert param.dtype == (torch.float32 if float_dtype == torch.float32 else torch.float64), \
            f"Dtype assertion failed for input parameter: receive {param.dtype}, expect {float_dtype}"
        net.register_parameter('theta', param)
    
    elif isinstance(param, (int, float)):
        net.theta = torch.ones(expect_shape, dtype=float_dtype) * param
    
    elif isinstance(param, torch.Tensor):
        expect_shape[1] = -1
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, -1)
        net.theta = param.to(net.device, dtype=float_dtype).reshape(expect_shape)
    
    else:  # when param is an Iterable
        expect_shape[1] = -1
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
_StateLike = Union[np.ndarray, torch.Tensor, State]
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
    
    if isinstance(data, State):
        return "state" if ndim is None else ("state", list(data.batch_dim))

    assert ndim is None, \
        ("Cannot obtain batch dimension for data type other than " +
         f"np.ndarray, torch.Tensor or quairkit.State: received {type(data)}")
    return "other"


def _density_to_vector(rho: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r""" transform a density matrix to a state vector

    Args:
        rho: a density matrix (pure state)

    Returns:
        a state vector

    Raises:
        ValueError: the output state may not be a pure state

    """
    type_str = _type_fetch(rho)
    rho = _type_transform(rho, "tensor")
    eigval, eigvec = torch.linalg.eigh(rho)

    max_eigval = torch.max(eigval).item()
    err = np.abs(max_eigval - 1)
    if err > 1e-6:
        raise ValueError(
            f"the output state may not be a pure state, maximum distance: {err}")

    state = eigvec[:, torch.argmax(eigval)]

    return state.detach().numpy() if type_str == "numpy" else state


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

    #TODO remove when all state manipulation functions are implemented
    if system_dim is None:
        shape = data.shape
        system_dim = shape[-2] if (len(shape) >= 2 and shape[-1] == 1) else shape[-1]
    else:
        system_dim = _format_system_dim(data.shape[-1], system_dim)

    if current_type == "numpy":
        if output_type == "tensor":
            return torch.from_numpy(data)

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "tensor":
        if output_type == "numpy":
            return data.detach().cpu().resolve_conj().numpy()

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "state":
        if output_type == "numpy":
            return data.numpy()
        return data._data.clone()


def _cnot_idx_fetch(num_qubits: int, qubits_idx: List[Tuple[int, int]]) -> List[int]:
    r"""
    Compute the CNOT index obtained by applying the CNOT gate without using matrix multiplication.

    Args:
        num_qubits: The total number of qubits in the system.
        qubits_idx: A list of tuples, where each tuple contains the indices of the two qubits
                    involved in the CNOT gate.

    Returns:
        List: A list of integers representing the decimal values of all binary strings
                obtained by applying the CNOT gate.
    """
    assert len(np.shape(qubits_idx)) == 2, \
        "The CNOT system_idx should be list of tuple of integers, e.g., [[0, 1], [1, 2]]."
    binary_list = [bin(i)[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]
    system_idx_length = len(qubits_idx)
    for item in range(len(binary_list)):
        for bin_idx in range(system_idx_length):
            id1 = qubits_idx[system_idx_length - bin_idx - 1][0]
            id2 = qubits_idx[system_idx_length - bin_idx - 1][1]
            if binary_list[item][id1] == "1":
                if binary_list[item][id2] == '0':
                    binary_list[item] = binary_list[item][:id2] + '1' + binary_list[item][id2 + 1:]
                else:
                    binary_list[item] = binary_list[item][:id2] + '0' + binary_list[item][id2 + 1:]

    decimal_list = [int(binary, 2) for binary in binary_list]
    return decimal_list


def _assert_gradient(func: Union[Callable[[torch.Tensor], torch.Tensor], str], param: torch.Tensor, 
                     atol: float = 1e-3, rtol: float = 1e-2, eps: float = 1e-2,
                     feedback: bool = False, exception: bool = False, message: str = None ) -> None:
    r"""
    Check the correctness of the gradient computation using finite difference method.

    Args:
        function_name: function to be tested.
        param: variables that perform finite differences.
        atol: absolute tolerance. Defaults to 1e-3.
        rtol: relative tolerance. Defaults to 1e-2.
        eps: perturbation for finite differences. Defaults to 1e-2.
        feedback: whether to print the result for confirmation. Defaults to False.
        exception: whether to raise an exception if the gradient check fails. Defaults to False.
        message: optional message to be printed if the gradient check fails. Defaults to None.
    
    """
    seed = get_seed()
    assert get_float_dtype() == torch.float64, \
        "The gradient check only supports double precision, which can be set by `quairkit.set_dtype('complex128')`."
    
    param = param.clone().requires_grad_(True)
    result = torch.autograd.gradcheck(func, param, eps=eps, atol=atol, 
                                        rtol=rtol, raise_exception = exception)
    assert result, (
        "The function is not accurate enough for precision \n"
        f"Function name: {func}, seed: {seed}, precision: {atol}, message: {message} "
    )
        
    
    # Print the result for confirmation
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

    # Process the 'info' argument to create a tensor generator
    generator = info

    if input_dtype is None:
        input_dtype = get_dtype()
    elif not isinstance(input_dtype, torch.dtype):
        raise TypeError(f"Input must be of type torch.dtype or None. Received type: {type(input_dtype)}")

    if isinstance(info, list) and all((isinstance(x, int) and x > 0) for x in info):
        # If 'info' is a list of integers, create a torch.tensor generator based on the shape of input data and the input dtype
        generator = lambda: torch.randn(info, dtype=input_dtype)
        sample = generator()
        func_in = func

    elif isinstance(info, Callable):
        sample = info()
        # Handling different types of generated inputs

        if isinstance(sample, np.ndarray):
            generator = lambda: torch.from_numpy(info())
            func_in = lambda input_data: func(torch.from_numpy(input_data))
        elif isinstance(sample, torch.Tensor):
            generator = lambda: info()
            func_in = func
        else:
            raise TypeError(
                f"The output of info must be a torch.Tensor or numpy.ndarray: received {type(sample)}"
            )

    else:
        raise TypeError(
            "the info entry should either the shape of input data, or a Callable functions: "
            + f"received info {info} with type {type(info)}"
        )

    # guarantee the output type is torch.Tensor
    output = func(sample)

    func_tensor = func_in
    if isinstance(output, np.ndarray):
        func_tensor = lambda mat: torch.from_numpy(func_in(mat))
    elif not isinstance(output, torch.Tensor):
        raise TypeError(
            f"The input function either return torch.Tensor or np.ndarray: received type {type(output)}"
        )

    return func_tensor, generator, input_dtype, _type_fetch(output)


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

