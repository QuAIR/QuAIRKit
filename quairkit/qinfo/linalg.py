# !/usr/bin/env python3
# Copyright (c) 2024 QuAIR team. All Rights Reserved.
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

import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from quairkit.core import utils

from ..core import get_dtype, tensor_state, to_state, utils
from ..core.intrinsic import (_ArrayLike, _format_system_dim,
                              _is_sample_linear, _SingleParamLike, _StateLike,
                              _type_fetch, _type_transform)
from ..database import pauli_basis

__all__ = [
    "abs_norm",
    "block_enc_herm",
    "create_matrix",
    "dagger",
    "direct_sum",
    "gradient",
    "hessian",
    "herm_transform",
    "kron_power",
    "logm",
    "nkron",
    "NKron",
    "p_norm",
    "partial_trace",
    "partial_trace_discontiguous",
    "partial_transpose",
    "pauli_decomposition",
    "permute_systems",
    "prob_sample",
    "schmidt_decompose",
    "sqrtm",
    "trace",
    "trace_norm",
]


def abs_norm(mat: _StateLike) -> float:
    r"""tool for calculation of matrix norm

    Args:
        mat: matrix

    Returns:
        norm of input matrix

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.to(get_dtype())
    return utils.linalg._abs_norm(mat)


def block_enc_herm(
    mat: _ArrayLike, num_block_qubits: int = 1
) -> _ArrayLike:
    r"""generate a (qubitized) block encoding of hermitian ``mat``

    Args:
        mat: matrix to be block encoded
        num_block_qubits: ancilla qubits used in block encoding

    Returns:
        a unitary that is a block encoding of ``mat``

    """
    type_mat = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")

    assert utils.check._is_hermitian(mat), "the input matrix is not a hermitian"
    assert mat.shape[0] == mat.shape[1], "the input matrix is not a square matrix"

    block_enc = utils.linalg._block_enc_herm(mat, num_block_qubits)

    return _type_transform(block_enc, type_mat)


def create_matrix(
    linear_map: Callable[[_ArrayLike], _ArrayLike],
    input_dim: int,
    input_dtype: Optional[torch.dtype] = None,
) -> _ArrayLike:
    r"""Create a matrix representation of a linear map without needing to specify the output dimension.

    This function constructs a matrix representation for a given linear map and input dimension.

    Args:
        linear_map: A function representing the linear map, which takes an input_dim-dimensional vector and returns a vector.
        input_dim: The dimension of the input space.
        input_dtype: The dtype of the input. Defaults to None.

    Returns:
        A matrix representing the linear map.

    Raises:
        RuntimeWarning: the input`linear_map` may not be linear.
    """
    linear_map, generator, input_dtype, type_str = _is_sample_linear(
        func=linear_map, info=[input_dim, 1], input_dtype=input_dtype
    )

    if not utils.check._is_linear(linear_map, generator, input_dtype):
        warnings.warn("the input linear_map may not be linear", RuntimeWarning)

    return _type_transform(
        utils.linalg._create_matrix(
            linear_map=linear_map, input_dim=input_dim, input_dtype=input_dtype
        ),
        type_str,
    )


def dagger(mat: _ArrayLike) -> _ArrayLike:
    r"""tool for calculation of matrix dagger

    Args:
        mat: matrix

    Returns:
        The dagger of matrix

    """
    type_mat = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")

    dag = utils.linalg._dagger(mat)

    return _type_transform(dag, type_mat)


def direct_sum(
    A: _ArrayLike, B: _ArrayLike
) -> _ArrayLike:
    r"""calculate the direct sum of A and B

    Args:
        A: :math:`m \times n` matrix
        B: :math:`p \times q` matrix

    Returns:
        a direct sum of A and B, with shape :math:`(m + p) \times (n + q)`


    """
    type_A, type_B = _type_fetch(A), _type_fetch(B)
    A, B = _type_transform(A, "tensor"), _type_transform(B, "tensor")
    B = B.to(A.dtype)

    mat = utils.linalg._direct_sum(A, B)

    return _type_transform(mat, type_A) if type_A == type_B else mat


def gradient(loss_function: Callable[[torch.Tensor], torch.Tensor], var: _ArrayLike, n: int) -> _ArrayLike:
    r"""
    Computes the gradient of a given loss function with respect to its input variable.

    Args:
        loss_function : A loss function to compute the gradient.
        var : A vector of shape (m,1) as the input variables for the loss function.
        n : The number of iterations for gradient computation.

    Returns:
        torch.Tensor: The gradient vector of shape (m,1).
    """
    type_str = _type_fetch(var)
    var = _type_transform(var, "tensor")
    # Check if the input is a vector
    var_ = torch.squeeze(var)
    shape = var_.shape
    if len(shape) != 1:
        warnings.warn(
            f"The input of var is not a (m,1) vector: received {var.shape}", RuntimeWarning
        )
    # Compute and return the gradient
    return _type_transform(utils.linalg._gradient(loss_function, var, n), type_str)


def hessian(loss_function: Callable[[torch.Tensor], torch.Tensor], var: _ArrayLike) -> _ArrayLike:
    r"""
    Computes the Hessian matrix of a given loss function with respect to its input variables.

    Args:
        loss_function : The loss function to compute the Hessian.
        var : A matrix of shape (n, m) as input variables for the loss function.

    Returns:
        torch.Tensor: Hessian matrix of shape (m, n, n).
    """
    shape = var.shape
    # Check if the input is a square matrix
    if len(shape) != 2:
        warnings.warn(
            f"The input of var is not a (n,m) matrix: received {var.shape}", RuntimeWarning
        )
    type_str = _type_fetch(var)
    var = _type_transform(var, "tensor")
    # Compute and return the hessian
    return _type_transform(utils.linalg._hessian(loss_function, var), type_str)


def herm_transform(
    fcn: Callable[[float], float],
    mat: _StateLike,
    ignore_zero: Optional[bool] = False,
) -> _ArrayLike:
    r"""function transformation for Hermitian matrix

    Args:
        fcn: function :math:`f` that can be expanded by Taylor series
        mat: hermitian matrix :math:`H`
        ignore_zero: whether ignore eigenspaces with zero eigenvalue, defaults to be ``False``

    Returns
        :math:`f(H)`

    """
    type_str = _type_fetch(mat)
    mat = (
        _type_transform(mat, "tensor")
        if type_str != "state"
        else mat.density_matrix
    )

    assert utils.check._is_hermitian(
        mat
    ), "the input matrix is not Hermitian: check your input"

    mat = utils.linalg._herm_transform(fcn, mat, ignore_zero)

    return mat.detach().numpy() if type_str == "numpy" else mat


def kron_power(matrix: _StateLike, n: int) -> _ArrayLike:
    r"""Calculate Kronecker product of identical matirces
    
    Args:
        matrix: the matrix to be powered
        n: the number of identical matrices
    
    Returns:
        Kronecker product of n identical matrices
    """
    if n == 0:
        return np.array([[1.0]]) if isinstance(matrix, np.ndarray) else torch.tensor([[1.0]])
    return nkron(matrix, *[matrix for _ in range(n - 1)])


def logm(mat: _StateLike) -> _ArrayLike:
    r"""Calculate log of a matrix

    Args:
        mat: Input matrix.

    Returns:

        The matrix of natural base logarithms

    """
    type_str = _type_fetch(mat)
    if type_str == "state":
        return mat.log()

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._logm(mat), type_str)


def nkron(matrix_1st: _StateLike, *args: _StateLike) -> _StateLike:
    r"""calculate Kronecker product of matirces

    Args:
        matrix_1: the first matrix 
        args: other matrices

    Returns:
        Kronecker product of matrices

    .. code-block:: python

        from quairkit.state import density_op_random
        from quairkit.linalg import NKron
        A = density_op_random(2)
        B = density_op_random(2)
        C = density_op_random(2)
        result = NKron(A, B, C)

    Note:
        ``result`` from above code block should be A \otimes B \otimes C
    """
    if not args:
        return matrix_1st
    
    type_1st = _type_fetch(matrix_1st)
    type_list = [type_1st] + [_type_fetch(arg) for arg in args]

    if all(type_arg == "state" for type_arg in type_list):
        return tensor_state(matrix_1st, *args)

    if all(type_arg == "numpy" for type_arg in type_list):
        return_type = "numpy"
    else:
        return_type = "tensor"

    matrix_1st = _type_transform(matrix_1st, "tensor")
    args = [_type_transform(mat, "tensor") for mat in args]
    result = utils.linalg._nkron(matrix_1st, *args)

    return _type_transform(result, return_type)

NKron = nkron


def p_norm(mat: _StateLike, p: _SingleParamLike) -> _ArrayLike:
    r"""tool for calculation of Schatten p-norm

    Args:
        mat: matrix
        p: p-norm parameter

    Returns:
        p-norm of input matrix

    """
    type_mat = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")
    mat = mat.to(get_dtype())
    p = p if isinstance(p, torch.Tensor) else torch.tensor(p)
    norm = utils.linalg._p_norm_herm(mat, p) if utils.check._is_hermitian(mat) \
        else utils.linalg._p_norm(mat, p)
    return norm.detach().numpy() if type_mat == "numpy" else norm


def partial_trace(
    state: _StateLike, trace_idx: Union[List[int], int], 
    system_dim: Union[List[int], int] = 2
) -> _StateLike:
    r"""Calculate the partial trace of the quantum state

    Args:
        state: Input quantum state.
        trace_idx: The system indices to be traced out.
        system_dim: The dimension of all systems. Defaults to be the qubit case.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.

    """
    type_str = _type_fetch(state)
    if type_str == "state":
        new_state = to_state(state, system_dim)
        return new_state.trace(trace_idx)
    
    trace_idx = [trace_idx] if isinstance(trace_idx, int) else trace_idx
    system_dim = _format_system_dim(state.shape[-1], system_dim)
    assert max(trace_idx) < len(system_dim), \
        f"The trace index {trace_idx} should be smaller than number of systems {len(system_dim)}"
    
    state = _type_transform(state, "state").density_matrix
    state = utils.linalg._partial_trace(state, trace_idx, system_dim)

    return _type_transform(state, type_str)


def partial_trace_discontiguous(
    state: _StateLike, preserve_qubits: List[int] = None
) -> _StateLike:
    r"""Calculate the partial trace of the quantum state with arbitrarily selected subsystem

    Args:
        state: Input quantum state.
        preserve_qubits: Remaining qubits, default is None, indicate all qubits remain.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.

    Note:
        suitable only when the systems are qubits.
    
    """
    type_str = _type_fetch(state)
    if type_str == "state":
        new_state = to_state(state, 2)
        return new_state.trace([x for x in range(state.num_qubits) if x not in preserve_qubits])
    
    rho = _type_transform(state, "state", system_dim=2).density_matrix
    
    rho = utils.linalg._partial_trace_discontiguous(rho, preserve_qubits)

    return _type_transform(rho, type_str)


def partial_transpose(
    state: _StateLike, transpose_idx: Union[List[int], int], 
    system_dim: Union[List[int], int] = 2
) -> _ArrayLike:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        state: input quantum state.
        transpose_idx: The system indices to be transposed.
        system_dim: The dimension of all systems. Defaults to be the qubit case.

    Returns:
        The partial transpose of the input quantum state.
    """
    type_str = _type_fetch(state)
    if type_str == "state":
        new_state = to_state(state, system_dim)
        return new_state.transpose(transpose_idx)
    
    transpose_idx = [transpose_idx] if isinstance(transpose_idx, int) else transpose_idx
    system_dim = _format_system_dim(state.shape[-1], system_dim)
    assert max(transpose_idx) < len(system_dim), \
        f"The transpose index {transpose_idx} should be smaller than number of systems {len(system_dim)}"
    
    type_str = _type_fetch(state)
    state = _type_transform(state, "state").density_matrix
    
    state = utils.linalg._partial_transpose(state, transpose_idx, system_dim)

    return state.detach().numpy() if type_str == "numpy" else state


def pauli_decomposition(mat: _ArrayLike) -> _ArrayLike:
    r"""Decompose the matrix by the Pauli basis.

    Args:
        mat: the matrix to be decomposed

    Returns:
        The list of coefficients corresponding to Pauli basis.

    """
    type_str = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")

    dimension = mat.shape[0]
    num_qubits = int(math.log2(dimension))
    assert (
        2**num_qubits == dimension
    ), f"Input matrix is not a valid quantum data: received shape {mat.shape}"

    basis = pauli_basis(num_qubits)
    coef = torch.cat([torch.trace(mat @ basis[i]).view([1]) for i in range(dimension**2)])
    return _type_transform(coef, type_str)


def permute_systems(
    state: _StateLike, perm_list: List[int], system_dim: Union[List[int], int] = 2,
) -> _StateLike:
    r"""Permute quantum system based on a permute list

    Args:
        mat: A given matrix representation which is usually a quantum state.
        perm_list: The permute list. e.g. input ``[0,2,1,3]`` will permute the 2nd and 3rd subsystems.
        system_dim: A list of dimension sizes of each subsystem.

    Returns:
        The permuted matrix
    """
    type_str = _type_fetch(state)
    if type_str == "state":
        new_state = to_state(state, system_dim)
        return new_state.permute(perm_list)
    
    system_dim = _format_system_dim(state.shape[-1], system_dim)
    state = _type_transform(state, "tensor")
    perm_mat = utils.linalg._permute_systems(state, perm_list, system_dim)
    return _type_transform(perm_mat, type_str)


def prob_sample(distribution: _ArrayLike, shots: int = 1024,
                binary: bool = True, proportional: bool = False) -> Dict[str, Union[int, float]]:
    r"""Sample from a probability distribution.

    Args:
        distribution: The probability distribution.
        shots: The number of shots. Defaults to 1024.
        binary: Whether the sampled result is recorded as binary. Defaults to True.
        proportional: Whether the counts are shown in proportion

    Returns:
        A dictionary containing the ordered sampled results and their counts.

    """
    assert shots > 0, \
        f"The number of shots must be a positive integer, received {shots}"
    distribution = _type_transform(distribution, "tensor")

    prob_sum = distribution.sum(dim=-1)
    if torch.any(torch.abs(prob_sum - 1) > 1e-3):
        warnings.warn(
            "The sum of the probability distribution is not 1" +
            f": error {prob_sum}. Automatically normalized.", RuntimeWarning
        )
    return utils.linalg._prob_sample(distribution / distribution.sum(dim=1, keepdim=True), shots, binary, proportional)


def schmidt_decompose(
    psi: _StateLike, sys_A: List[int] = None
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    r"""Calculate the Schmidt decomposition of a quantum state :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`.

    Args:
        psi: State vector form of the quantum state, with shape (2**n)
        sys_A: Qubit indices to be included in subsystem A (other qubits are included in subsystem B), default are the first half qubits of :math:`\lvert \psi\rangle`

    Returns:
        contains elements

        * A one dimensional array composed of Schmidt coefficients, with shape ``(k)``
        * A high dimensional array composed of bases for subsystem A :math:`\lvert i_A\rangle`, with shape ``(k, 2**m, 1)``
        * A high dimensional array composed of bases for subsystem B :math:`\lvert i_B\rangle` , with shape ``(k, 2**m, 1)``
    """
    type_psi = _type_fetch(psi)
    # TODO
    psi = _type_transform(psi, "state_vector").ket

    assert math.log2(
        psi.numel()
    ).is_integer(), "The dimensional of input state must be an integral power of 2."

    c, u, v = utils.linalg._schmidt_decompose(psi, sys_A)
    return (
        (
            c.detach().resolve_conj().numpy(),
            u.detach().resolve_conj().numpy(),
            v.detach().resolve_conj().numpy(),
        )
        if type_psi == "numpy"
        else (c, u, v)
    )


def sqrtm(mat: _StateLike) -> _ArrayLike:
    r"""Calculate square root of a matrix

    Args:
        mat: Input matrix.

    Returns:
        The square root of the matrix
    """
    type_str = _type_fetch(mat)
    if type_str == "state":
        return mat.sqrt()

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._sqrtm(mat), type_str)


def trace(mat: _StateLike, axis1: int = -2, axis2: int = -1) -> _ArrayLike:
    r"""Return the sum along diagonals of the tensor.

    If :math:`mat` is 2-D tensor, the sum along its diagonal is returned.

    If :math:`mat` has more than two dimensions, then the axes specified by ``axis1`` and ``axis2`` are used to determine the 2-D sub-tensors whose traces are returned. The shape of the resulting tensor is the same as the shape of :math:`mat` with ``axis1`` and ``axis2`` removed.

    Args:
        mat: Input tensor, from which the diagonals are taken.
        axis1: The first axis of the 2-D sub-tensors along which the diagonals should be taken. Defaults to -2.
        axis2: The second axis of the 2-D sub-tensors along which the diagonals should be taken. Defaults to -1.

    Returns:
        The sum along the diagonals. If :math:`mat` is 2-D tensor, the sum along its diagonal is returned. If :math:`mat` has larger dimensions, then a tensor of sums along diagonals is returned.

    Raises:
        ValueError: The 2-D tensor from which the diagonals should be taken is not square.

    Note:
        The 2-D tensor/array from which the diagonals is taken should be square.
    """

    if mat.shape[axis1] != mat.shape[axis2]:
        raise ValueError(
            f"The 2-D tensor from which the diagonals should be taken is not square, as {mat.shape[axis1]} != {mat.shape[axis2]}."
        )

    type_str = _type_fetch(mat)
    if type_str == "state":
        assert {axis1, axis2} == {-1, -2}, \
            f"Only support tracing out the state matrix: received {axis1} and {axis2}"
        return mat.trace()
    
    mat_tensor = _type_transform(mat, "tensor")
    trace_mat = utils.linalg._trace(mat_tensor, axis1, axis2)

    return _type_transform(trace_mat, type_str)


def trace_norm(mat: _StateLike) -> _ArrayLike:
    r"""tool for calculation of trace norm

    Args:
        mat: matrix

    Returns:
        trace norm of input matrix

    """
    return p_norm(mat, 1)
