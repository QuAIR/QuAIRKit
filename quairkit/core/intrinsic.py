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

import math
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

from . import base, utils
from .state import State, to_state


def _format_qubits_idx(
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int, str],
        num_qubits: int, num_acted_qubits: int
) -> List[List[int]]:
    r"""Formatting the qubit indices that operations acts on

    Args:
        qubits_idx: input qubit indices, could be a string
        num_qubits: total number of qubits
        num_acted_qubits: the number of qubits that one operation acts on

    Note:
        The shape of output qubit indices are formatted as follows:
        - If num_acted_qubits is 1, the output shape is [# of qubits that one operation acts on];
        - otherwise, the output shape is [# of vertical gates, num_acted_qubits].

    """
    assert not (isinstance(qubits_idx, str) and num_qubits is None), \
        f"Cannot specify the qubit indices when num_qubits is None: received qubit_idx {qubits_idx} and num_qubits {num_qubits}"
    if num_acted_qubits == 1:
        if qubits_idx == 'full':
            qubits_idx = list(range(num_qubits))
        elif qubits_idx == 'even':
            qubits_idx = list(range(num_qubits, 2))
        elif qubits_idx == 'odd':
            qubits_idx = list(range(1, num_qubits, 2))
        elif isinstance(qubits_idx, Iterable):
            assert len(qubits_idx) == len(set(qubits_idx)), \
                f"Single-qubit operators do not allow repeated indices: received {qubits_idx}"
        else:
            qubits_idx = [qubits_idx]
        qubits_idx = [[idx] for idx in qubits_idx]
    else:
        if qubits_idx == 'cycle':
            assert num_qubits >= num_acted_qubits, \
                f"# of qubits should be >= # of acted qubits: received {num_qubits} and {num_acted_qubits}"
            qubits_idx = []
            for idx in range(num_qubits - num_acted_qubits):
                qubits_idx.append(
                    [i for i in range(idx, idx + num_acted_qubits)])
            for idx in range(num_qubits - num_acted_qubits, num_qubits):
                qubits_idx.append([i for i in range(idx, num_qubits)] +
                                  [i for i in range(idx + num_acted_qubits - num_qubits)])
        elif qubits_idx == 'linear':
            assert num_qubits >= num_acted_qubits, \
                f"# of qubits should be >= # of acted qubits: received {num_qubits} and {num_acted_qubits}"
                
            qubits_idx = []
            for idx in range(num_qubits - num_acted_qubits + 1):
                qubits_idx.append(
                    [i for i in range(idx, idx + num_acted_qubits)])
        elif len(np.shape(qubits_idx)) == 1 and len(qubits_idx) == num_acted_qubits:
            qubits_idx = [list(qubits_idx)]
        elif len(np.shape(qubits_idx)) == 2 and all((len(indices) == num_acted_qubits for indices in qubits_idx)):
            qubits_idx = [list(indices) for indices in qubits_idx]
        else:
            raise TypeError(
                "The qubits_idx should be iterable such as list, tuple, and so on whose elements are all integers."
                "And the length of acted_qubits should be consistent with the corresponding gate."
                f"\n    Received qubits_idx type {type(qubits_idx)}, qubits # {len(qubits_idx)}, gate dimension {num_acted_qubits}"
            )
    return qubits_idx


def _format_param_shape(qubits_idx: List[List[int]], num_acted_param: int, 
                        param_sharing: bool, batch_size: int = 1) -> List[int]:
    r"""Formatting the shape of parameters of param gates

    Args:
        qubits_idx: list of input qubit indices
        num_acted_param: the number of parameters required for a single operation
        param_sharing: whether all operations are shared by the same parameter set
        batch_size: size of gate batch

    Note:
        The input ``qubits_idx`` must be formatted by ``_format_qubits_idx`` first.
        The shape of parameters are formatted as [len(qubits_idx), batch_size, num_acted_param].

    """
    return [1 if param_sharing else len(qubits_idx), batch_size, num_acted_param]


def _theta_generation(net: torch.nn.Module, param: Union[torch.Tensor, float, List[float]], 
                      qubits_idx: List[List[int]], num_acted_param: int, param_sharing: bool) -> None:
    r""" determine net.theta, and create parameter if necessary

    Args:
        net: neural network instance
        param: input theta
        qubits_idx: list of input qubit indices
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
    expect_shape = _format_param_shape(qubits_idx, num_acted_param, param_sharing)
    
    # TODO unify support for batch and non-batch case
    
    if param is None:
        theta = torch.rand(expect_shape, dtype=float_dtype) * 2 * math.pi
        net.register_parameter('theta', Parameter(theta))
    
    elif isinstance(param, Parameter):
        assert param.shape == expect_shape, \
            f"Shape assertion failed for input parameter: receive {list(param.shape)}, expect {expect_shape}"
        assert param.dtype == (torch.float32 if float_dtype == torch.float32 else torch.float64), \
            f"Dtype assertion failed for input parameter: receive {param.dtype}, expect {float_dtype}"
        net.register_parameter('theta', param)
    
    elif isinstance(param, (int, float)):
        net.theta = torch.ones(expect_shape, dtype=float_dtype) * param
    
    elif isinstance(param, torch.Tensor):
        expect_shape[1] = -1
        expect_shape = _format_param_shape(qubits_idx, num_acted_param, param_sharing, -1)
        net.theta = param.to(float_dtype).reshape(expect_shape)
    
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


def _type_fetch(data: Union[np.ndarray, torch.Tensor, State]) -> str:
    r""" fetch the type of ``data``

    Args:
        data: the input data, and datatype of which should be either ``numpy.ndarray``,
    ''torch.Tensor'' or ``quairkit.State``

    Returns:
        string of datatype of ``data``, can be either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``

    Raises:
        ValueError: does not support the current backend of input state.
        TypeError: cannot recognize the current type of input data.

    """
    if isinstance(data, np.ndarray):
        return "numpy"

    if isinstance(data, torch.Tensor):
        return "tensor"

    if isinstance(data, State):
        return data.backend

    raise TypeError(
        f"cannot recognize the current type {type(data)} of input data.")


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


def _type_transform(data: Union[np.ndarray, torch.Tensor, State],
                    output_type: str) -> Union[np.ndarray, torch.Tensor, State]:
    r""" transform the datatype of ``input`` to ``output_type``

    Args:
        data: data to be transformed
        output_type: datatype of the output data, type is either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``

    Returns:
        the output data with expected type

    Raises:
        ValueError: does not support transformation to type.

    """
    current_type = _type_fetch(data)

    support_type = {"numpy", "tensor", "state_vector", "density_matrix"}
    if output_type not in support_type:
        raise ValueError(
            f"does not support transformation to type {output_type}")

    if current_type == output_type:
        return data

    if current_type == "numpy":
        if output_type == "tensor":
            return torch.tensor(data)

        data = np.squeeze(data)
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = _density_to_vector(data)
        
        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ np.conj(data.T)
        return to_state(data)

    if current_type == "tensor":
        if output_type == "numpy":
            return data.detach().cpu().resolve_conj().numpy()

        data = torch.squeeze(data)
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = utils.linalg._density_to_vector(data)

        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ torch.conj(data.T)
        return to_state(data)

    if current_type == "state_vector":
        if output_type == "density_matrix":
            return to_state(data, state_backend='density_matrix')
        return data.ket.detach().numpy() if output_type == "numpy" else data.ket

    # density_matrix data
    if output_type == "state_vector":
        raise NotImplementedError(
            "The transformation from density matrix to state vector is not supported.")
    return data.detach().numpy() if output_type == "numpy" else data.density_matrix


def _perm_to_swaps(perm: List[int]) -> List[Tuple[int]]:
    r"""This function takes a permutation as a list of integers and returns its
        decomposition into a list of tuples representing the two-permutation (two conjugated 2-cycles).

    Args:
        perm: the target permutation

    Returns:
        the decomposition of the permutation.
    """
    n = len(perm)
    swapped = [False] * n
    swap_ops = []

    for idx in range(n):
        if not swapped[idx]:
            next_idx = idx
            swapped[next_idx] = True
            while not swapped[perm[next_idx]]:
                swapped[perm[next_idx]] = True
                if next_idx < perm[next_idx]:
                    swap_ops.append((next_idx, perm[next_idx]))
                else:
                    swap_ops.append((perm[next_idx], next_idx))
                next_idx = perm[next_idx]

    return swap_ops


def _trans_ops(state: torch.Tensor, swap_ops: List[Tuple[int]], batch_dims: List[int], num_qubits: int,
               extra_dims: int = 1) -> torch.Tensor:
    r"""Transpose the state tensor given a list of swap operations.

    Args:
        swap_ops: given list of swap operations
        batch_dims: intrinsic dimension of the state tensor
        num_qubits: the number of qubits in the system
        extra_dims: labeling the dimension of state, 1 for statevector; 2 for density operator

    Returns:
        torch.Tensor: transposed state tensor given the swap list
    """
    num_batch_dims = len(batch_dims)
    for swap_op in swap_ops:
        shape = batch_dims.copy()
        shape.extend([2**(swap_op[0]), 2, 2**(swap_op[1] - swap_op[0] - 1),
                     2, 2**(extra_dims * num_qubits - swap_op[1] - 1)])
        state = torch.reshape(state, shape)
        state = torch.permute(
            state, tuple(range(num_batch_dims)) +
            tuple(item + num_batch_dims for item in [0, 3, 2, 1, 4])
        )
    return state


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
        "The CNOT qubits_idx should be list of tuple of integers, e.g., [[0, 1], [1, 2]]."
    binary_list = [bin(i)[2:].zfill(num_qubits) for i in range(2 ** num_qubits)]
    qubits_idx_length = len(qubits_idx)
    for item in range(len(binary_list)):
        for bin_idx in range(qubits_idx_length):
            id1 = qubits_idx[qubits_idx_length - bin_idx - 1][0]
            id2 = qubits_idx[qubits_idx_length - bin_idx - 1][1]
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
    seed = base.get_seed()
    func, param = lambda x: func(x).double(), param.clone().detach().double().requires_grad_(True)
    if not param.requires_grad:
        # Perform gradient check
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

        