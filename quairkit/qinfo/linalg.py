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
    r"""Tool for calculation of matrix norm.

    Args:
        mat: matrix

    Returns:
        norm of input matrix

    Examples:
        .. code-block:: python

            abs_nor = abs_norm(eye(2) / 2)
            print(f'The abs norm is:\n{abs_nor}')

        ::

            The abs norm is:
            0.7071067690849304
    """
    mat = _type_transform(mat, "tensor")
    mat = mat.to(get_dtype())
    return utils.linalg._abs_norm(mat)


def block_enc_herm(
    mat: _ArrayLike, num_block_qubits: int = 1
) -> _ArrayLike:
    r"""Generate a (qubitized) block encoding of Hermitian ``mat``.

    Args:
        mat: matrix to be block encoded
        num_block_qubits: ancilla qubits used in block encoding

    Returns:
        a unitary that is a block encoding of ``mat``.

    Examples:
        .. code-block:: python

            block_enc = block_enc_herm(x())
            print(f'The block encoding of X is:\n{block_enc}')

        ::

            The block encoding of X is:
            tensor([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
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
        RuntimeWarning: the input `linear_map` may not be linear.

    Examples:
        .. code-block:: python

            def f(X):
                return X[0] + X[1]

            mat_repr = create_matrix(f, input_dim=2)
            print(f'The matrix representation is:\n{mat_repr}')

        ::

            The matrix representation is:
            tensor([1.+0.j, 1.+0.j])
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
    r"""Tool for calculation of matrix dagger.

    Args:
        mat: matrix

    Returns:
        The dagger of matrix

    Examples:
        .. code-block:: python

            dag = dagger(t())
            print(f'The dagger of this matrix is:\n{dag}')

        ::

            The dagger of this matrix is:
            tensor([[1.0000-0.0000j, 0.0000-0.0000j],
                    [0.0000-0.0000j, 0.7071-0.7071j]])
    """
    type_mat = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")

    dag = utils.linalg._dagger(mat)

    return _type_transform(dag, type_mat)


def direct_sum(
    A: _ArrayLike, B: _ArrayLike
) -> _ArrayLike:
    r"""Calculate the direct sum of A and B.

    Args:
        A: :math:`m \times n` matrix
        B: :math:`p \times q` matrix

    Returns:
        A direct sum of A and B, with shape :math:`(m + p) \times (n + q)`

    Examples:
        .. code-block:: python

            dir_sum = direct_sum(x(), y())
            print(f'The direct sum is:\n{dir_sum}')

        ::

            The direct sum is:
            tensor([[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, -0.-1.j],
                    [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j]])
    """
    type_A, type_B = _type_fetch(A), _type_fetch(B)
    A, B = _type_transform(A, "tensor"), _type_transform(B, "tensor")
    B = B.to(A.dtype)

    mat = utils.linalg._direct_sum(A, B)

    return _type_transform(mat, type_A) if type_A == type_B else mat


def gradient(loss_function: Callable[[torch.Tensor], torch.Tensor], var: _ArrayLike, n: int) -> _ArrayLike:
    r"""Compute the gradient of a given loss function with respect to its input variable.

    Args:
        loss_function: A loss function to compute the gradient.
        var: A vector of shape (m, 1) as the input variables for the loss function.
        n: The number of iterations for gradient computation.

    Returns:
        torch.Tensor: The gradient vector of shape (m, 1).

    Examples:
        .. code-block:: python

            def quadratic_loss(x: torch.Tensor) -> torch.Tensor:
                # loss function is: L(x) = x₁² + 2x₂² + 3x₃²
                return x[0]**2 + 2 * x[1]**2 + 3 * x[2]**3

            var = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
            grad = gradient(quadratic_loss, var, n=1)

            print(f"Input variable is:\n{var}")
            print(f"Computed gradient is:\n{grad}")

        ::

            Input variable is:
            tensor([[1.],
                    [2.],
                    [3.]], requires_grad=True)
            Computed gradient is:
            tensor([[ 2.],
                    [ 8.],
                    [81.]], grad_fn=<CopySlices>)
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
    r"""Compute the Hessian matrix of a given loss function with respect to its input variables.

    Args:
        loss_function: The loss function to compute the Hessian.
        var: A matrix of shape (n, m) as input variables for the loss function.

    Returns:
        torch.Tensor: Hessian matrix of shape (m, n, n).

    Examples:
        .. code-block:: python

            def quadratic_loss(x: torch.Tensor) -> torch.Tensor:
                # loss function is: L(x) = x₁² + 2x₂² + 3x₃²
                return x[0]**2 + 2 * x[1]**2 + 3 * x[2]**3

            var = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
            hes = hessian(quadratic_loss, var)

            print(f"Input variable is:\n{var}")
            print(f"Computed Hessian is:\n{hes}")

        ::

            Input variable is:
            tensor([[1.],
                    [2.],
                    [3.]], requires_grad=True)
            Computed Hessian is:
            tensor([[[ 2.,  0.,  0.],
                     [ 0.,  4.,  0.],
                     [ 0.,  0., 54.]]])
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
    r"""Function transformation for a Hermitian matrix.

    Args:
        fcn: A function :math:`f` that can be expanded by Taylor series.
        mat: Hermitian matrix :math:`H`.
        ignore_zero: Whether to ignore eigenspaces with zero eigenvalue. Defaults to False.

    Returns:
        :math:`f(H)`

    Examples:
        .. code-block:: python

            fH = herm_transform(math.exp, eye(2))
            print(f'The result is:\n{fH}')

        ::

            The result is:
            tensor([[2.7183+0.j, 0.0000+0.j],
                    [0.0000+0.j, 2.7183+0.j]])
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
    r"""Calculate the Kronecker product of identical matrices.
    
    Args:
        matrix: The matrix to be powered.
        n: The number of identical matrices.

    Returns:
        Kronecker product of n identical matrices.

    Examples:
        .. code-block:: python

            kp = kron_power(x(), 2)
            print(f'The Kronecker product of 2 X is:\n{kp}')

        ::

            The Kronecker product of 2 X is:
            tensor([[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    if n == 0:
        return np.array([[1.0]]) if isinstance(matrix, np.ndarray) else torch.tensor([[1.0]])
    return nkron(matrix, *[matrix for _ in range(n - 1)])


def logm(mat: _StateLike) -> _ArrayLike:
    r"""Calculate the logarithm of a matrix.

    Args:
        mat: Input matrix.

    Returns:
        The matrix of natural base logarithms.
        
    Examples:
        .. code-block:: python

            lgm = logm(x())
            print(f'The log of X is:\n{lgm}')

        ::

            The log of X is:
            tensor([[-1.8562e-16+1.5708j,  1.8562e-16-1.5708j],
                    [ 1.8562e-16-1.5708j, -1.8562e-16+1.5708j]])
    """
    type_str = _type_fetch(mat)
    if type_str == "state":
        return mat.log()

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._logm(mat), type_str)


def nkron(matrix_1st: _StateLike, *args: _StateLike) -> _StateLike:
    r"""Calculate the Kronecker product of matrices.

    Args:
        matrix_1st: The first matrix.
        args: Other matrices.

    Returns:
        Kronecker product of the given matrices.

    Examples:
        .. code-block:: python

            A = random_state(1)
            B = random_state(1)
            C = random_state(1)
            result = nkron(A, B, C)
            print(f'The result is:\n{result}')

        ::

            The result is:
            -----------------------------------------------------
             Backend: density_matrix
             System dimension: [2, 2, 2]
             System sequence: [0, 1, 2]
            [[ 0.02+0.j    0.01-0.01j -0.01+0.j   -0.-0.01j  0.04+0.04j  0.05-0.01j
              -0.04-0.03j -0.04+0.02j]
             [ 0.01+0.01j  0.01-0.j   -0.01-0.01j -0.01+0.j   -0.01+0.05j  0.04+0.03j
              -0.-0.04j -0.03-0.02j]
             [-0.01-0.j   -0.01+0.01j  0.02+0.j    0.01-0.02j -0.03-0.04j -0.04+0.j
               0.06+0.06j  0.07-0.02j]
             [-0.-0.01j -0.01-0.j    0.01+0.02j  0.02-0.j    0.02-0.04j -0.02-0.03j
              -0.01+0.08j  0.05+0.05j]
             [ 0.04-0.04j -0.01-0.05j -0.03+0.04j  0.02+0.04j  0.21-0.j    0.1-0.16j
              -0.16+0.03j -0.06+0.14j]
             [ 0.05+0.01j  0.04-0.03j -0.04-0.j   -0.02+0.03j  0.1+0.16j  0.18-0.j
              -0.11-0.11j -0.14+0.03j]
             [-0.04+0.03j -0.-0.04j  0.06-0.06j -0.01-0.08j -0.16-0.03j -0.11+0.11j
               0.29-0.j    0.15-0.23j]
             [-0.04-0.02j -0.03+0.02j  0.07+0.02j  0.05-0.05j -0.06-0.14j -0.14-0.03j
               0.15+0.23j  0.25-0.j  ]]
            -----------------------------------------------------
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
    r"""Calculate the Schatten p-norm of a matrix.

    Args:
        mat: matrix
        p: p-norm parameter

    Returns:
        p-norm of input matrix

    Examples:
        .. code-block:: python

            p_nor = p_norm(x(), p=2)
            print(f'The 2-norm of X is:\n{p_nor}')

        ::

            The 2-norm of X is:
            1.4142135381698608
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
    r"""Calculate the partial trace of the quantum state.

    Args:
        state: Input quantum state.
        trace_idx: The system indices to be traced out.
        system_dim: The dimension of all systems. Defaults to the qubit case.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.

    Examples:
        .. code-block:: python

            pt = partial_trace(bell_state(2), 0, [2, 2])
            print(f'The partial trace of Bell state is:\n{pt}')

        ::

            The partial trace of Bell state is:
            -----------------------------------------------------
             Backend: density_matrix
             System dimension: [2]
             System sequence: [0]
            [[0.5+0.j 0. +0.j]
             [0. +0.j 0.5+0.j]]
            -----------------------------------------------------
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
    r"""Calculate the partial trace of the quantum state with arbitrarily selected subsystem.

    Args:
        state: Input quantum state.
        preserve_qubits: Remaining qubits; if None, all qubits are preserved.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.

    Examples:
        .. code-block:: python

            ptdis = partial_trace_discontiguous(bell_state(2), [0])
            print(f'The partial trace of Bell state is:\n{ptdis}')

        ::

            The partial trace of Bell state is:
            -----------------------------------------------------
             Backend: density_matrix
             System dimension: [2]
             System sequence: [0]
            [[0.5+0.j 0. +0.j]
             [0. +0.j 0.5+0.j]]
            -----------------------------------------------------
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
        state: Input quantum state.
        transpose_idx: The system indices to be transposed.
        system_dim: The dimension of all systems. Defaults to the qubit case.

    Returns:
        The partial transpose of the input quantum state.

    Examples:
        .. code-block:: python

            pt = partial_transpose(bell_state(2), [0])
            print(f'The partial transpose of Bell state is:\n{pt}')

        ::

            The partial transpose of Bell state is:
            -----------------------------------------------------
             Backend: density_matrix
             System dimension: [2, 2]
             System sequence: [0, 1]
            [[0.5+0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0.5+0.j 0. +0.j]
             [0. +0.j 0.5+0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0.5+0.j]]
            -----------------------------------------------------
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
        mat: The matrix to be decomposed.

    Returns:
        A list of coefficients corresponding to the Pauli basis.

    Examples:
        .. code-block:: python

            pauli_dec = pauli_decomposition(random_state(1).density_matrix)
            print(f'The decomposition is:\n{pauli_dec}')

        ::

            The decomposition is:
            tensor([ 0.7071+0.j,  0.5076+0.j,  0.4494+0.j, -0.0572+0.j])
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
    r"""Permute quantum systems based on a permutation list.

    Args:
        state: A matrix representation of a quantum state.
        perm_list: The permutation list. For example, [0, 2, 1, 3] will swap the 2nd and 3rd subsystems.
        system_dim: A list of dimension sizes of each subsystem.

    Returns:
        The permuted matrix.

    Examples:
        .. code-block:: python

            result = permute_systems(random_state(3), [2, 1, 0])
            print(f'The permuted matrix is:\n{result}')

        ::

            The permuted matrix is:
            -----------------------------------------------------
             Backend: density_matrix
             System dimension: [2, 2, 2]
             System sequence: [0, 1, 2]
            [[ 0.06+0.j   -0.02+0.j   -0.02-0.07j -0.03-0.j    0.06+0.01j -0.02+0.03j
              -0.01-0.06j  0.01+0.01j]
             [-0.02-0.j    0.13+0.j   -0.05+0.01j -0.02-0.01j  0.06-0.06j -0.01-0.03j
              -0.+0.08j   0.03-0.03j]
             [-0.02+0.07j -0.05-0.01j  0.24+0.j    0.06+0.03j -0.09+0.05j  0.01-0.05j
               0.04-0.05j -0.13+0.04j]
             [-0.03+0.j   -0.02+0.01j  0.06-0.03j  0.1+0.j    -0.07+0.02j  0.02-0.05j
              -0.01+0.01j -0.03+0.j  ]]
            -----------------------------------------------------
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
        proportional: Whether the counts are shown in proportion.

    Returns:
        A dictionary containing the ordered sampled results and their counts.

    Examples:
        .. code-block:: python

            dist = torch.abs(haar_state_vector(3))
            result = prob_sample(dist / torch.sum(dist))
            print(f'The sample result is:\n{result}')

        ::

            The sample result is:
            {'0': tensor([1024, 1024, 1024])}
    """
    assert shots > 0, \
        f"The number of shots must be a positive integer, received {shots}"
    distribution = _type_transform(distribution, "tensor")

    prob_sum = distribution.sum(dim=-1, keepdim=True)
    if torch.any(torch.abs(err := prob_sum - 1) > 1e-5):
        warnings.warn(
            "The sum of (some) probability distribution is not close to 1" +
            f" and automatically normalized: received error\n{err}. ", RuntimeWarning
        )
        distribution = distribution / prob_sum
    return utils.linalg._prob_sample(distribution, shots, binary, proportional)


def schmidt_decompose(
    psi: _StateLike, sys_A: List[int] = None
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    r"""Calculate the Schmidt decomposition of a quantum state.

    For a state :math:`\lvert\psi\rangle=\sum_i c_i \lvert i_A\rangle\otimes\lvert i_B \rangle`.

    Args:
        psi: State vector form of the quantum state, with shape :math:`(2**n)`.
        sys_A: Qubit indices to be included in subsystem A. By default, the first half of the qubits belong to subsystem A.

    Returns:
        A tuple containing:
            - A one-dimensional array of Schmidt coefficients with shape (k).
            - A high-dimensional array of bases for subsystem A with shape (k, 2**m, 1).
            - A high-dimensional array of bases for subsystem B with shape (k, 2**m, 1).

    Examples:
        .. code-block:: python

            # Example usage (assuming a proper state vector 'psi'):
            c, u, v = schmidt_decompose(psi)
            print("Schmidt coefficients:", c)
            print("Subsystem A bases:", u)
            print("Subsystem B bases:", v)

        ::

            Schmidt coefficients: tensor([...])
            Subsystem A bases: tensor([...])
            Subsystem B bases: tensor([...])
    """
    type_psi = _type_fetch(psi)
    # TODO: Provide a complete example when psi is defined.
    psi = _type_transform(psi, "state_vector").ket

    assert math.log2(
        psi.numel()
    ).is_integer(), "The dimension of input state must be an integral power of 2."

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
    r"""Calculate the square root of a matrix.

    Args:
        mat: Input matrix.

    Returns:
        The square root of the matrix.

    Examples:
        .. code-block:: python

            sqrt = sqrtm(x())
            print(f'The square root of X is:\n{sqrt}')

        ::

            The square root of X is:
            tensor([[0.5000+0.5000j, 0.5000-0.5000j],
                    [0.5000-0.5000j, 0.5000+0.5000j]])
    """
    type_str = _type_fetch(mat)
    if type_str == "state":
        return mat.sqrt()

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._sqrtm(mat), type_str)


def trace(mat: _StateLike, axis1: int = -2, axis2: int = -1) -> _ArrayLike:
    r"""Return the sum along the diagonals of the tensor.

    If :math:`mat` is a 2-D tensor, the sum along its diagonal is returned.
    For tensors with more than two dimensions, the axes specified by ``axis1`` and ``axis2``
    determine the 2-D sub-tensors whose traces will be taken.

    Args:
        mat: Input tensor from which the diagonal is taken.
        axis1: The first axis for the 2-D sub-tensor. Defaults to -2.
        axis2: The second axis for the 2-D sub-tensor. Defaults to -1.

    Returns:
        The trace (sum along the diagonal) of the tensor.

    Examples:
        .. code-block:: python

            tr = trace(x())
            print(f'The trace of X is:\n{tr}')

        ::

            The trace of X is:
            0j
    """
    if mat.shape[axis1] != mat.shape[axis2]:
        raise ValueError(
            f"The 2-D tensor from which the diagonal is taken is not square, as {mat.shape[axis1]} != {mat.shape[axis2]}."
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
    r"""Calculate the trace norm of a matrix.

    Args:
        mat: matrix

    Returns:
        Trace norm of the input matrix

    Examples:
        .. code-block:: python

            tr_norm = trace_norm(x())
            print(f'The trace norm of X is:\n{tr_norm}')

        ::

            The trace norm of X is:
            2.0
    """
    return p_norm(mat, 1)
