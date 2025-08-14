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

from .base import get_dtype, get_float_dtype, get_seed
from .state import State, StateOperator, StateSimulator, to_state


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


def _format_sequential_idx(system_idx: Union[List[List[int]], List[int], int, str],
                           num_systems: int, num_acted_system: int) -> List[List[int]]:
    r"""Formatting the system indices in a sequential way

    Args:
        system_idx: input system indices
        num_systems: total number of systems
        num_acted_system: the number of systems that one operation acts on

    Note:
        The shape of output system indices are formatted as [# of vertical gates, num_acted_system].
    """
    if not isinstance(system_idx, str):
        return system_idx

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
        batch_size: size of gate batch, -1 for variable batch size

    Note:
        The input ``system_idx`` must be formatted by ``_format_system_idx`` first.
        The shape of parameters are formatted as [len(system_idx), batch_size, num_acted_param].

    """
    return [1 if param_sharing else len(system_idx), batch_size, num_acted_param]


def _theta_generation(net: torch.nn.Module, param: Union[torch.Tensor, float, List[float]], 
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
        - if ``param`` is a ParamBase instance
        
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
        
        assert list(param.shape) == expect_shape, \
            f"Shape assertion failed for input parameter: receive {list(param.shape)}, expect {expect_shape}"
        assert param.dtype == (torch.float32 if float_dtype == torch.float32 else torch.float64), \
            f"Dtype assertion failed for input parameter: receive {param.dtype}, expect {float_dtype}"
        net.register_parameter('theta', param)
    
    elif isinstance(param, torch.Tensor):
        expect_shape = _format_param_shape(system_idx, num_acted_param, param_sharing, -1)
        net.theta = param.to(net.device, dtype=float_dtype).reshape(expect_shape)
    
    else:  # when param is an Iterable
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

        #TODO remove when all state manipulation functions are implemented
        if system_dim is None:
            shape = data.shape
            system_dim = shape[-2] if (len(shape) >= 2 and shape[-1] == 1) else shape[-1]
        else:
            system_dim = _format_system_dim(data.shape[-1], system_dim)

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "tensor":
        if output_type == "numpy":
            return data.detach().cpu().resolve_conj().numpy()

        #TODO remove when all state manipulation functions are implemented
        if system_dim is None:
            shape = data.shape
            system_dim = shape[-2] if (len(shape) >= 2 and shape[-1] == 1) else shape[-1]
        else:
            system_dim = _format_system_dim(data.shape[-1], system_dim)

        return to_state(data, system_dim=system_dim, eps=None)

    if current_type == "state":
        return data.numpy() if output_type == "numpy" else data._data.clone()
    
    raise ValueError(
        f"does not support transformation from {current_type} to type {output_type}: received {data}")


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
    
    try:
        result = torch.autograd.gradcheck(func, param, eps=eps, atol=atol, 
                                          rtol=rtol, raise_exception=True, nondet_tol=nondet_tol)
    except Exception as e:
        result = False
        warnings.warn("The following error occurred when trying to compute the gradient\n" 
                      + str(e), RuntimeWarning)
    
    assert result, (
        "Gradient test failed \n"
        f"Setting: seed {seed}, precision {atol}, message {message} "
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
