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
The library of functions in quantum information.
"""

import math
import warnings
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .core import State, get_dtype, utils
from .core.intrinsic import _type_fetch, _type_transform
from .database import pauli_basis
from .operator import Channel


def trace_distance(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the trace distance of two quantum states.

    .. math::
        D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The trace distance between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "density_matrix").density_matrix
    sigma = _type_transform(sigma, "density_matrix").density_matrix
    assert rho.shape == sigma.shape, "The shape of two quantum states are different"

    dist = utils.qinfo._trace_distance(rho, sigma.to(rho.dtype))
    return (
        dist.detach().numpy() if type_rho == "numpy" and type_sigma == "numpy" else dist
    )


def state_fidelity(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the fidelity of two quantum states.

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The fidelity between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "density_matrix").density_matrix
    sigma = _type_transform(sigma, "density_matrix").density_matrix
    assert rho.shape == sigma.shape, "The shape of two quantum states are different"

    fidelity = utils.qinfo._state_fidelity(rho, sigma.to(rho.dtype))

    return (
        fidelity.detach().numpy()
        if type_rho == "numpy" and type_sigma == "numpy"
        else fidelity
    )


def gate_fidelity(
    U: Union[np.ndarray, torch.Tensor], V: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    r"""calculate the fidelity between gates

    .. math::

        F(U, V) = |\text{tr}(UV^\dagger)|/2^n

    :math:`U` is a :math:`2^n\times 2^n` unitary gate

    Args:
        U: quantum gate :math:`U`  in matrix form
        V: quantum gate :math:`V`  in matrix form

    Returns:
        fidelity between gates

    """
    type_u, type_v = _type_fetch(U), _type_fetch(V)
    U, V = _type_transform(U, "tensor"), _type_transform(V, "tensor")
    assert U.shape == V.shape, "The shape of two matrices are different"

    fidelity = utils.qinfo._gate_fidelity(U, V.to(dtype=U.dtype))

    return (
        fidelity.detach().numpy()
        if type_u == "numpy" and type_v == "numpy"
        else fidelity
    )


def purity(
    rho: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the purity of a quantum state.

    .. math::

        P = \text{tr}(\rho^2)

    Args:
        rho: Density matrix form of the quantum state.

    Returns:
        The purity of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "density_matrix").density_matrix

    gamma = utils.qinfo._purity(rho)

    return gamma.detach().numpy() if type_rho == "numpy" else gamma


def von_neumann_entropy(
    rho: Union[np.ndarray, torch.Tensor, State], base: Optional[int] = 2
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the von Neumann entropy of a quantum state.

    .. math::

        S = -\text{tr}(\rho \log(\rho))

    Args:
        rho: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        The von Neumann entropy of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "density_matrix").density_matrix

    entropy = utils.qinfo._von_neumann_entropy(rho, base)

    return entropy.detach().numpy() if type_rho == "numpy" else entropy


def relative_entropy(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
    base: Optional[int] = 2,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the relative entropy of two quantum states.

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

    Args:
        rho: Density matrix form of the quantum state.
        sigma: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        Relative entropy between input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "density_matrix").density_matrix
    sigma = _type_transform(sigma, "density_matrix").density_matrix
    assert rho.shape == sigma.shape, "The shape of two quantum states are different"

    entropy = utils.qinfo._relative_entropy(rho, sigma.to(rho.dtype), base)

    return (
        entropy.detach().numpy()
        if type_rho == "numpy" and type_sigma == "numpy"
        else entropy
    )


# TODO
def pauli_str_convertor(observable: List) -> List:
    r"""Concatenate the input observable with coefficient 1.

    For example, if the input ``observable`` is ``[['z0,x1'], ['z1']]``,
    then this function returns the observable ``[[1, 'z0,x1'], [1, 'z1']]``.

    Args:
        observable: The observable to be concatenated with coefficient 1.

    Returns:
        The observable with coefficient 1
    """

    for i in range(len(observable)):
        assert len(observable[i]) == 1, "Each term should only contain one string"

    return [[1, term] for term in observable]


# TODO
def partial_transpose_2(
    density_op: Union[np.ndarray, torch.Tensor, State], sub_system: int = None
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state. The shape should be (m, 4, 4).
        sub_system: 1 or 2. 1 means to perform partial transpose on system A;
                    2 means to perform partial transpose on system B. Default is 2.

    Returns:
        The partial transpose of the input quantum state.

    :Example:
        .. code-block:: python

            import torch
            from quairkit.qinfo import partial_transpose_2

            rho_test = torch.arange(1, 17).reshape([4, 4])
            partial_transpose_2(rho_test, sub_system=1)

    ::

       [[ 1,  2,  9, 10],
        [ 5,  6, 13, 14],
        [ 3,  4, 11, 12],
        [ 7,  8, 15, 16]]
    """
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix").data

    transposed_density_op = utils.qinfo._partial_transpose_2(density_op, sub_system)

    return (
        transposed_density_op.detach().numpy()
        if type_str == "numpy"
        else transposed_density_op
    )


def partial_transpose(
    density_op: Union[np.ndarray, torch.Tensor, State], n: int
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.
        n: Number of qubits of subsystem A, with qubit indices as [0, 1, ..., n-1]

    Returns:
        The partial transpose of the input quantum state.
    """
    # Copy the density matrix and not corrupt the original one
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix").density_matrix
    density_op = utils.linalg._partial_transpose(density_op, n)

    return density_op.detach().numpy() if type_str == "numpy" else density_op


def permute_systems(
    mat: Union[np.ndarray, torch.Tensor, State],
    perm_list: List[int],
    dim_list: List[int],
) -> Union[np.ndarray, torch.Tensor, State]:
    r"""Permute quantum system based on a permute list

    Args:
        mat: A given matrix representation which is usually a quantum state.
        perm: The permute list. e.g. input ``[0,2,1,3]`` will permute the 2nd and 3rd subsystems.
        dim: A list of dimension sizes of each subsystem.

    Returns:
        The permuted matrix
    """
    mat_type = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")
    perm_mat = utils.linalg._permute_systems(mat, perm_list, dim_list)
    return _type_transform(perm_mat, mat_type)


def negativity( #Todo
    density_op: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Compute the Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Negativity of the input quantum state.
    """
    # Implement the partial transpose
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix").density_matrix

    neg = utils.qinfo._negativity(density_op)

    return neg.detach().numpy() if type_str == "numpy" else neg


def logarithmic_negativity(
    density_op: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Logarithmic Negativity of the input quantum state.
    """
    # Calculate the negativity
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix").density_matrix

    log_neg = utils.qinfo._logarithmic_negativity(density_op)

    return log_neg.detach().numpy() if type_str == "numpy" else log_neg


def is_ppt(density_op: Union[np.ndarray, torch.Tensor, State]) -> Union[bool, List[bool]]:
    r"""Check if the input quantum state is PPT.
        Support batch input.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        Whether the input quantum state is PPT. 
        For batch input, return a boolean array with the same batch dimensions as input.
    """
    density_op = _type_transform(density_op, "density_matrix").density_matrix
    density_op = density_op.squeeze()
    if len(density_op.shape) in {2, 3}:
        return utils.check._is_ppt(density_op)
    return False


def is_choi(op: Union[np.ndarray, torch.Tensor]) -> Union[bool, List[bool]]:
    r"""Check if the input op is a Choi operator of a quantum operation.
        Support batch input.

    Args:
        op: matrix form of the linear operation.

    Returns:
        Whether the input op is a valid quantum operation Choi operator. 
        For batch input, return a boolean array with the same batch dimensions as input.

    Note:
        The operation op is (default) applied to the second system.
    """
    op = _type_transform(op, "tensor").to(torch.complex128)
    op = op.squeeze()
    return utils.check._is_choi(op) if len(op.shape) in (2, 3) else False


def schmidt_decompose(
    psi: Union[np.ndarray, torch.Tensor, State], sys_A: List[int] = None
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


def NKron(
    matrix_A: Union[torch.Tensor, np.ndarray],
    matrix_B: Union[torch.Tensor, np.ndarray],
    *args: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    r"""calculate Kronecker product of at least two matrices

    Args:
        matrix_A: matrix, as torch.Tensor or numpy.ndarray
        matrix_B: matrix, as torch.Tensor or numpy.ndarray
        *args: other matrices, as torch.Tensor or numpy.ndarray

    Returns:
        Kronecker product of matrices, determined by input type of matrix_A

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
    type_A, type_B = _type_fetch(matrix_A), _type_fetch(matrix_B)

    matrix_A = _type_transform(matrix_A, "tensor")
    matrix_B = _type_transform(matrix_B, "tensor")
    args = [_type_transform(mat, "tensor") for mat in args]
    result = utils.linalg._nkron(matrix_A, matrix_B, *args)

    return _type_transform(result, type_A) if type_A == type_B else result


def diamond_norm(
    channel_repr: Union[Channel, torch.Tensor],
    dim_io: Union[int, Tuple[int, int]] = None,
    **kwargs,
) -> float:
    r"""Calculate the diamond norm of input.

    Args:
        channel_repr: A ``Channel`` or a ``torch.Tensor`` instance.
        dim_io: The input and output dimensions.
        **kwargs: Parameters to set cvx.

    Raises:
        RuntimeError: `channel_repr` must be `Channel` or `torch.Tensor`.
        TypeError: "dim_io" should be "int" or "tuple".

    Warning:
        `channel_repr` is not in Choi representation, and is converted into `ChoiRepr`.

    Returns:
        Its diamond norm.

    Reference:
        Khatri, Sumeet, and Mark M. Wilde. "Principles of quantum communication theory: A modern approach."
        arXiv preprint arXiv:2011.04672 (2020).
        Watrous, J. . "Semidefinite Programs for Completely Bounded Norms."
        Theory of Computing 5.1(2009):217-238.
    """
    if isinstance(channel_repr, Channel):
        choi_matrix = channel_repr.choi_repr

    elif isinstance(channel_repr, torch.Tensor):
        choi_matrix = channel_repr

    else:
        raise RuntimeError(
            "`channel_repr` must be `ChoiRepr`or `KrausRepr` or `StinespringRepr` or `torch.Tensor`."
        )

    return utils.qinfo._diamond_norm(choi_matrix, dim_io, **kwargs)


# TODO
def channel_repr_convert(
    representation: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
    source: str,
    target: str,
    tol: float = 1e-6,
) -> Union[torch.Tensor, np.ndarray]:
    r"""convert the given representation of a channel to the target implementation

    Args:
        representation: input representation
        source: input form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``
        target: target form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``
        tol: error tolerance for the conversion from Choi, :math:`10^{-6}` by default

    Raises:
        ValueError: Unsupported channel representation: require Choi, Kraus or Stinespring.

    Returns:
        quantum channel by the target implementation

    Note:
        choi -> kraus currently has the error of order 1e-6 caused by eigh

    Raises:
        NotImplementedError: does not support the conversion of input data type

    """
    source, target = source.lower(), target.lower()
    if target not in ["choi", "kraus", "stinespring"]:
        raise ValueError(
            f"Unsupported channel representation: require 'choi', 'kraus' or 'stinespring', not {target}"
        )
    if source == target:
        return representation
    if source not in ["choi", "kraus", "stinespring"]:
        raise ValueError(
            f"Unsupported channel representation: require 'choi', 'Kraus' or 'stinespring', not {source}"
        )
    if isinstance(representation, List):
        if isinstance(representation[0], torch.Tensor):
            representation = torch.stack(representation)
        else:
            representation = np.stack(representation)
    
    if len(representation.shape) > 2:
        assert (
            source == "kraus"
        ), f"Unsupported data input: expected Kraus representation, received {source}"
    type_str = _type_fetch(representation)
    representation = _type_transform(representation, "tensor")

    oper = Channel(source, representation)
    if source == "choi":
        if target == "kraus":
            representation = oper.kraus_repr

        # stinespring repr
        representation = oper.stinespring_repr

    elif source == "kraus":
        representation = oper.choi_repr if target == "choi" else oper.stinespring_repr

    else:  # if source == 'stinespring'
        if target == "kraus":
            representation = oper.kraus_repr

        # choi repr
        representation = oper.choi_repr

    return _type_transform(representation, type_str)


def abs_norm(mat: Union[np.ndarray, torch.Tensor, State]) -> float:
    r"""tool for calculation of matrix norm

    Args:
        mat: matrix

    Returns:
        norm of input matrix

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.to(get_dtype())
    return utils.linalg._abs_norm(mat)


def trace_norm(mat: Union[np.ndarray, torch.Tensor, State]) -> Union[np.ndarray, torch.Tensor]:
    r"""tool for calculation of trace norm

    Args:
        mat: matrix

    Returns:
        trace norm of input matrix

    """
    return p_norm(mat, 1)


def p_norm(mat: Union[np.ndarray, torch.Tensor, State],
           p: Union[np.ndarray, torch.Tensor, float]) -> Union[np.ndarray, torch.Tensor]:
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
    norm = utils.linalg._p_norm_herm(mat, p) if is_hermitian(mat, sys_dim = mat.shape[0]) \
        else utils.linalg._p_norm(mat, p)
    return norm.detach().numpy() if type_mat == "numpy" else norm


def dagger(mat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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


def is_hermitian(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6, sys_dim: Union[int, List[int]] = 2
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is Hermitian.
        Support batch input.

    Args:
        mat: hermitian candidate :math:`P`
        eps: tolerance of error
        sys_dim: dimension of subsystems, default to be `2` i.e. all subsystems are qubit systems

    Returns:
        determine whether :math:`P - P^\dagger = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.squeeze()
    if len(mat.shape) in {2, 3}:
        return utils.check._is_hermitian(mat, eps, sys_dim)
    return False


def is_positive(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6, sys_dim: Union[int, List[int]] = 2
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a positive semi-definite matrix.
        Support batch input.

    Args:
        mat: positive operator candidate :math:`P`
        eps: tolerance of error
        sys_dim: dimension of subsystems, default to be `2` i.e. all subsystems are qubit systems

    Returns:
        determine whether :math:`P` is Hermitian and eigenvalues are non-negative
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.squeeze()
    if len(mat.shape) in {2, 3}:
        return utils.check._is_positive(mat, eps, sys_dim)
    return False


def is_state_vector(
    vec: Union[np.ndarray, torch.Tensor], eps: Optional[float] = None, sys_dim: Union[int, List[int]] = 2
) -> Union[Tuple[bool, int],Tuple[List[bool], List[int]]]:
    r"""Verify whether ``vec`` is a legal quantum state vector.
        Support batch input.

    Args:
        vec: state vector candidate :math:`x`
        sys_dim: (list of) dimension(s) of the systems, can be a list of integers or an integer
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
        sys_dim: dimension of subsystems, default to be `2` i.e. all subsystems are qubit systems

    Returns:
        determine whether :math:`x^\dagger x = 1`, and return the number of qudits or an error message
        For batch input, return a boolean array and an integer array with the same batch dimensions as input.

    Note:
        error message is:
        * ``-1`` if the above equation does not hold
        * ``-2`` if the dimension of ``vec`` is not a product of dim
        * ``-3`` if ``vec`` is not a vector

    """
    vec = _type_transform(vec, "tensor")
    vec = vec.squeeze()

    is_batch = False
    if len(vec.shape) == 3:
        is_batch = True
    elif len(vec.shape) != 2:
        return False
    
    if eps is None:
        eps = 1e-4 if get_dtype() == torch.complex64 else 1e-6
        
    return utils.check._is_state_vector(vec, eps, sys_dim, is_batch)


def is_density_matrix(
    rho: Union[np.ndarray, torch.Tensor], eps: Optional[float] = None, sys_dim: Union[int, List[int]] = 2
) -> Union[Tuple[bool, int],Tuple[List[bool], List[int]]]:
    r"""Verify whether ``rho`` is a legal quantum density matrix
        Support batch input

    Args:
        rho: density matrix candidate
        sys_dim: (list of) dimension(s) of the systems, can be a list of integers or an integer
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
        sys_dim: dimension of subsystems, default to be `2` i.e. all subsystems are qubit systems

    Returns:
        determine whether ``rho`` is a PSD matrix with trace 1 and return the number of qudits or an error message.
        For batch input, return a boolean array and an integer array with the same batch dimensions as input.

    :Example:
        .. code-block:: python

            import torch
            from quairkit.database.random import random_density_matrix
            from quairkit.qinfo import is_density_matrix

            rho_test_1, rho_test_2 = random_density_matrix(2), random_density_matrix(2)
            rho_test_2 = rho_test_2 + 0.1
            rho_test_3 = torch.tensor([[1, 0, 0, 0], \
                                    [0, 1, 0, 0], \
                                    [0, 0,-2, 0], \
                                    [0, 0, 0, 1]])
            rho_test = torch.stack([rho_test_1, rho_test_2, rho_test_3])
            is_density_matrix(rho_test, is_batch=True)

    ::

       ([True, False, False], [2, -2, -1])

    Note:
        error message is:
        * ``-1`` if ``rho`` is not PSD
        * ``-2`` if the trace of ``rho`` is not 1
        * ``-3`` if the dimension of ``rho`` is not a product of dim
        * ``-4`` if ``rho`` is not a square matrix

    """
    rho = _type_transform(rho, "tensor")
    rho = rho.squeeze()

    is_batch = False
    if len(rho.shape) == 3:
        is_batch = True
    elif len(rho.shape) != 2:
        return False

    if eps is None:
        eps = 1e-4 if get_dtype() == torch.complex64 else 1e-6

    return utils.check._is_density_matrix(rho, eps, sys_dim, is_batch)


def is_projector(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a projector.
        Support batch input.

    Args:
        mat: projector candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP - P = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor").to(torch.complex128)
    mat = mat.squeeze()
    if len(mat.shape) in {2, 3}:
        return utils.check._is_projector(mat, eps)
    return False


def is_unitary(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-4
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a unitary.
        Support batch input.

    Args:
        mat: unitary candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP^\dagger - I = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor").to(torch.complex128)
    mat = mat.squeeze()
    return utils.check._is_unitary(mat, eps) if len(mat.shape) in {2, 3} else False


def block_enc_herm(
    mat: Union[np.ndarray, torch.Tensor], num_block_qubits: int = 1
) -> Union[np.ndarray, torch.Tensor]:
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


def direct_sum(
    A: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
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


def herm_transform(
    fcn: Callable[[float], float],
    mat: Union[torch.Tensor, np.ndarray, State],
    ignore_zero: Optional[bool] = False,
) -> torch.Tensor:
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
        if type_str != "state_vector"
        else mat.ket @ mat.bra
    )

    assert utils.check._is_hermitian(
        mat
    ), "the input matrix is not Hermitian: check your input"

    mat = utils.linalg._herm_transform(fcn, mat, ignore_zero)

    return mat.detach().numpy() if type_str == "numpy" else mat


# TODO
def pauli_decomposition(
    mat: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
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


def partial_trace(
    state: Union[np.ndarray, torch.Tensor, State], dim1: int, dim2: int, A_or_B: int
) -> Union[np.ndarray, torch.Tensor, State]:
    r"""Calculate the partial trace of the quantum state.
    
    Args:
        state: Input quantum state. 
        dim1: The dimension of system A.
        dim2: The dimension of system B.
        A_or_B: 1 or 2. 1 means to calculate partial trace on system A; 2 means to calculate partial trace on system B.
    
    Returns:
        Partial trace of the input quantum state.
    """
    type_str = _type_fetch(state)
    state = (
        _type_transform(state, "density_matrix").density_matrix
        if type_str == "state_vector"
        else _type_transform(state, "tensor")
    )

    new_state = utils.linalg._partial_trace(state, dim1, dim2, A_or_B)

    return _type_transform(new_state, type_str)


def partial_trace_discontiguous(
    state: Union[np.ndarray, torch.Tensor, State], preserve_qubits: list = None
) -> Union[np.ndarray, torch.Tensor, State]:
    r"""Calculate the partial trace of the quantum state with arbitrarily selected subsystem
    
    Args:
        state: Input quantum state.
        preserve_qubits: Remaining qubits, default is None, indicate all qubits remain.
    
    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.
    
    """
    type_str = _type_fetch(state)
    rho = (
        _type_transform(state, "density_matrix").density_matrix
        if type_str == "state_vector"
        else _type_transform(state, "tensor")
    )

    rho = utils.linalg._partial_trace_discontiguous(rho, preserve_qubits)

    return _type_transform(rho, type_str)


def logm(mat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate log of a matrix

    Args:
        mat: Input matrix.

    Returns:

        The matrix of natural base logarithms

    """
    type_str = _type_fetch(mat)

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._logm(mat), type_str)


def sqrtm(mat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate square root of a matrix

    Args:
        A: Input matrix.

    Returns:
        The square root of the matrix
    """
    type_str = _type_fetch(mat)

    mat = _type_transform(mat, "tensor")

    return _type_transform(utils.linalg._sqrtm(mat), type_str)


def __check_sample(
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


def is_linear(
    func: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
    info: Union[List[int], Callable[[], np.ndarray], Callable[[], torch.Tensor]],
    input_dtype: torch.dtype = None,
    eps: Optional[float] = 1e-5,
) -> bool:
    """Check if the provided function 'func' is linear.

    Args:
        func: A callable function to be tested. This function should accept and return either a torch.Tensor or a numpy.ndarray.
        info: A parameter specifying the shape of the input for 'func'. It can be a list of two integers (defining the shape of a tensor),
              a callable that returns a numpy.ndarray, or a callable that returns a torch.Tensor.
        eps: An optional tolerance value used to determine if the function's behavior is close enough to linear. Default value is 1e-6.

    Returns:
        bool: True if 'func' behaves as a linear function within the specified tolerance 'eps'; False otherwise.

    Raises:
        TypeError: If 'func' is not callable, does not accept a torch.Tensor or numpy.ndarray as input, or does not return a torch.Tensor or numpy.ndarray.
                   If 'info' is not a valid type (not a list of integers or a callable returning a torch.Tensor or numpy.ndarray).
    """
    func, generator, input_dtype, _ = __check_sample(func, info, input_dtype)
    return utils.check._is_linear(
        func=func, generator=generator, input_dtype=input_dtype, eps=eps
    )


def create_matrix(
    linear_map: Callable[
        [Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]
    ],
    input_dim: int,
    input_dtype: torch.dtype = None,
) -> Union[torch.Tensor, np.ndarray]:
    r"""Create a matrix representation of a linear map without needing to specify the output dimension.

    This function constructs a matrix representation for a given linear map and input dimension.

    Args:
        linear_map: A function representing the linear map, which takes an input_dim-dimensional vector and returns a vector.
        input_dim: The dimension of the input space.
        is_tensor_input: A boolean indicating whether the input of linear_map is a torch.Tensor. Defaults to True.

    Returns:
        A matrix representing the linear map.

    Raises:
        RuntimeWarning: the input`linear_map` may not be linear.
    """
    linear_map, generator, input_dtype, type_str = __check_sample(
        func=linear_map, info=[input_dim, 1], input_dtype=input_dtype
    )

    if not is_linear(func=linear_map, info=generator, input_dtype=input_dtype):
        warnings.warn("the input linear_map may not be linear", RuntimeWarning)

    return _type_transform(
        utils.linalg._create_matrix(
            linear_map=linear_map, input_dim=input_dim, input_dtype=input_dtype
        ),
        type_str,
    )


def create_choi_repr(
    linear_map: Callable[
        [Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]
    ],
    input_dim: int,
    input_dtype: torch.dtype = None,
) -> torch.Tensor:
    r"""Create the Choi representation of a linear map with input checks.

    This function verifies if the map is linear and if the output is a square matrix.

    Args:
        linear_map: A function representing the linear map, which takes and returns a square matrix.
        input_dim: The dimension of the space in which the linear map operates.
        is_tensor_input: A boolean indicating whether the input of linear_map is a torch.Tensor. Defaults to True.


    Returns:
        torch.Tensor: The Choi matrix of the linear map.

    Raises:
        RuntimeWarning: If `linear_map` is not linear or the output is not a square matrix.
    """
    linear_map, _, input_dtype, type_str = __check_sample(
        linear_map, [input_dim, input_dim], input_dtype
    )

    # Check if the linear_map is linear and issue a warning if not
    if not is_linear(linear_map, [input_dim, input_dim]):
        warnings.warn("linear_map is not linear", RuntimeWarning)

    # Check if the output is a square matrix
    sample = linear_map(torch.randn(input_dim, input_dim, dtype=input_dtype))
    if sample.shape[0] != sample.shape[1]:
        warnings.warn(
            f"The output of this linear map is not a square matrix: received {sample.shape}",
            RuntimeWarning,
        )

    # Compute and return the Choi representation
    return _type_transform(
        utils.qinfo._create_choi_repr(
            linear_map=linear_map, input_dim=input_dim, input_dtype=input_dtype
        ),
        type_str,
    )


def trace(mat: Union[np.ndarray, torch.Tensor], axis1: Optional[int]=-2, axis2: Optional[int]=-1) -> Union[np.ndarray, torch.Tensor]:
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
    mat_tensor = _type_transform(mat, "tensor")
    trace_mat = utils.linalg._trace(mat_tensor, axis1, axis2)

    return _type_transform(trace_mat, type_str)

def hessian(loss_function: Callable[[torch.Tensor], torch.Tensor], var: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]: 
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

def gradient(loss_function: Callable[[torch.Tensor], torch.Tensor], var: Union[torch.Tensor, np.ndarray], n: int) -> Union[torch.Tensor, np.ndarray]:
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


def decomp_1qubit(
    unitary: Union[np.ndarray, torch.Tensor], return_global: bool = False
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a single-qubit unitary operator into Z-Y-Z rotation angles.

    Args:
        unitary: A batch of 2x2 unitary matrices representing single-qubit gates,
                 as either a numpy ndarray or a torch Tensor. The shape should be (m, 2, 2).
        return_global: If set to True, the global phase angle `alpha` is also returned.

    Returns:
        A tuple containing the angles `(beta, gamma, delta)` or `(alpha, beta, gamma, delta)` if `return_global` is True.
        The type of the tuple elements matches the input type.

    Raises:
        ValueError: Raises a ValueError if the input matrix is not a batch of 2x2 unitary matrices.
    """
    type_str = _type_fetch(unitary)

    unitary = _type_transform(unitary, "tensor")

    if unitary.shape[-2:] != (2, 2):
        raise ValueError(
            f"Input must be a batch of 2x2 unitary matrices with shape (m, 2, 2), but got shape {unitary.shape}"
        )
    result = utils.qinfo._decomp_1qubit(unitary, return_global)
    # Transform each tensor in the result tuple individually
    return tuple(_type_transform(tensor, type_str) for tensor in result)


def decomp_ctrl_1qubit(
    unitary: Union[np.ndarray, torch.Tensor]
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a controlled single-qubit unitary operator into its components.

    Args:
        unitary: A 2x2 unitary matrix representing the single-qubit gate, as either a numpy ndarray or a torch Tensor.

    Returns:
        A tuple containing the global phase `alpha` and the matrices `A, B, C`, which are components of the decomposition.
        The type of the tuple elements matches the input type.

    Raises:
        ValueError: Raises a ValueError if the input matrix is not a 2x2 unitary matrix.

    Reference:
        Nielsen, M. A., & Chuang, I. L. (2000). Quantum Computation and Quantum Information. 
        Corollary 4.2 for the decomposition of a single-qubit unitary operation.
    """
    type_str = _type_fetch(unitary)

    unitary = _type_transform(unitary, "tensor")

    if unitary.shape[-2:] != (2, 2) or not utils.check._is_unitary(unitary):
        raise ValueError(f"Input must be a batch of 2x2 unitary matrices with shape (m, 2, 2), but got shape {unitary.shape}")

    result = utils.qinfo._decomp_ctrl_1qubit(unitary)
    # Transform each tensor in the result tuple individually
    return tuple(_type_transform(tensor, type_str) for tensor in result)


def prob_sample(distribution: torch.Tensor, shots: int = 1024, 
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
