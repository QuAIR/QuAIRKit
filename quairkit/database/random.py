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

"""
The library of random data generation functions
"""

import math
from typing import List, Optional, Union

import numpy as np
import scipy
import torch
from scipy.stats import unitary_group

# TODO this is added due to channel_repr_convert, move it to intrinsic
import quairkit as qkit

from ..core import Hamiltonian, State, get_dtype, get_float_dtype, to_state
from ..core.intrinsic import _alias, _format_total_dim
from ..core.utils.linalg import _dagger

__all__ = [
    "random_pauli_str_generator",
    "random_state",
    "random_hamiltonian_generator",
    "random_hermitian",
    "random_orthogonal_projection",
    "random_density_matrix",
    "random_unitary",
    "random_unitary_hermitian",
    "random_unitary_with_hermitian_block",
    "haar_orthogonal",
    "haar_unitary",
    "haar_state_vector",
    "haar_density_operator",
    "random_channel",
]


def random_pauli_str_generator(num_qubits: int, terms: Optional[int] = 3) -> List:
    r"""Generate a random observable in list form.

    An observable :math:`O=0.3X\otimes I\otimes I+0.5Y\otimes I\otimes Z`'s list form is
    ``[[0.3, 'x0'], [0.5, 'y0,z2']]``.  Such an observable is generated by 
    ``random_pauli_str_generator(3, terms=2)`` 

    Args:
        num_qubits: Number of qubits.
        terms: Number of terms in the observable. Defaults to 3.

    Returns:
        The Hamiltonian of randomly generated observable.
    """
    pauli_str = []
    for sublen in np.random.randint(1, high=num_qubits + 1, size=terms):
        # Tips: -1 <= coeff < 1
        coeff = np.random.rand() * 2 - 1
        ops = np.random.choice(['x', 'y', 'z'], size=sublen)
        pos = np.random.choice(range(num_qubits), size=sublen, replace=False)
        op_list = [ops[i] + str(pos[i]) for i in range(sublen)]
        op_list.sort(key=lambda x: int(x[1:]))
        pauli_str.append([coeff, ','.join(op_list)])
    return pauli_str


@_alias({"num_systems": "num_qubits"})
def random_state(num_systems: int, 
                 rank: Optional[int] = None, 
                 is_real: Optional[bool] = False, 
                 size: Optional[Union[int, List[int]]] = 1,
                 system_dim: Union[List[int], int] = 2) -> State:
    r"""Generate a random quantum state.

    Args:
        num_systems: The number of qubits contained in the quantum state.
        rank: The rank of the density matrix. Defaults to ``None`` which means full rank.
        is_real: If the quantum state only contains the real number. Defaults to ``False``.
        size: Batch size. Defaults to 1
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    dim = _format_total_dim(num_systems, system_dim)
    size = [size] if isinstance(size, int) else list(size)
    rank = np.random.randint(1, dim + 1) if rank is None else rank
    total_size = int(np.prod(size))
    
    if rank == 1:
        list_state = torch.stack([haar_state_vector(dim, is_real) 
                                  for _ in range(total_size)]).view(size + [dim, 1])
    else:
        list_state = torch.stack([haar_density_operator(dim, rank, is_real) 
                                  for _ in range(total_size)]).view(size + [dim, dim])
    
    list_state = list_state if total_size > 1 else list_state.squeeze()
    return to_state(list_state, system_dim)


def random_hamiltonian_generator(num_qubits: int, terms: Optional[int] = 3) -> Hamiltonian:
    r"""Generate a random Hamiltonian. 

    Args:
        num_qubits: Number of qubits.
        terms: Number of terms in the Hamiltonian. Defaults to 3.

    Returns:
        The randomly generated Hamiltonian.
    """
    return Hamiltonian(random_pauli_str_generator(num_qubits, terms))


def random_hermitian(num_qubits: int) -> torch.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` hermitian matrix

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` hermitian matrix

    """
    assert num_qubits > 0
    n = 2 ** num_qubits

    mat = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    for i in range(n):
        mat[i, i] = np.abs(mat[i, i])
        for j in range(i):
            mat[i, j] = np.conj(mat[j, i])

    eigval= np.linalg.eigvalsh(mat)
    max_eigval = np.max(np.abs(eigval))
    return torch.tensor(mat / max_eigval, dtype=get_dtype())


def random_orthogonal_projection(num_qubits: int) -> torch.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` rank-1 orthogonal projector

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` orthogonal projector
    """
    assert num_qubits > 0
    n = 2 ** num_qubits
    float_dtype = get_float_dtype()
    vec = torch.randn([n, 1], dtype=float_dtype) + 1j * torch.randn([n, 1], dtype=float_dtype)
    mat = vec @ _dagger(vec)
    return mat / torch.trace(mat)


def random_density_matrix(num_qubits: int) -> torch.Tensor:
    r""" randomly generate an num_qubits-qubit state in density matrix form

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
        a :math:`2^n \times 2^n` density matrix

    """
    dim = 2 ** num_qubits
    return haar_density_operator(dim, rank=np.random.randint(1, dim + 1))


@_alias({"num_systems": "num_qubits"})
def random_unitary(num_systems: int,
                   size: Optional[Union[int, List[int]]] = 1,
                   system_dim: Union[List[int], int] = 2) -> torch.Tensor:
    r"""randomly generate a :math:`d \times d` unitary

    Args:
        num_systems: number of systems in this unitary. Alias of ``num_qubits``.
        size: batch size. Defaults to 1
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
         (a) :math:`d \times d` unitary matrix

    """
    dim = _format_total_dim(num_systems, system_dim)
    size = [size] if isinstance(size, int) else list(size)
    total_size = math.prod(size)
    list_unitary = torch.stack([torch.tensor(unitary_group.rvs(dim), dtype=get_dtype()) 
                                for _ in range(total_size)]).view(size + [dim, dim])
    return list_unitary if total_size > 1 else list_unitary.squeeze()


def random_unitary_hermitian(num_qubits: int) -> torch.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` hermitian unitary

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` hermitian unitary matrix

    """
    proj_mat = random_orthogonal_projection(num_qubits)
    id_mat = torch.eye(2 ** num_qubits)
    return (2 + 0j) * proj_mat - id_mat


def random_unitary_with_hermitian_block(num_qubits: int, is_unitary: bool = False) -> torch.Tensor:
    r"""randomly generate a unitary :math:`2^n \times 2^n` matrix that is a block encoding of a :math:`2^{n/2} \times 2^{n/2}` Hermitian matrix

    Args:
        num_qubits: number of qubits :math:`n`
        is_unitary: whether the hermitian block is a unitary divided by 2 (for tutorial only)

    Returns:
         a :math:`2^n \times 2^n` unitary matrix that its upper-left block is a Hermitian matrix

    """
    assert num_qubits > 0

    if is_unitary:
        mat0 = random_unitary_hermitian(num_qubits - 1).detach().numpy() / 2
    else:
        mat0 = random_hermitian(num_qubits - 1).detach().numpy()
    id_mat = np.eye(2 ** (num_qubits - 1))
    mat1 = 1j * scipy.linalg.sqrtm(id_mat - np.matmul(mat0, mat0))

    mat = np.block([[mat0, mat1], [mat1, mat0]]).astype(complex)

    return torch.tensor(mat, dtype=get_dtype())


def haar_orthogonal(dim: int) -> torch.Tensor:
    r""" randomly generate an orthogonal matrix following Haar random, referenced by arXiv:math-ph/0609050v2

    Args:
        dim: dimension of orthogonal matrix

    Returns:
        a :math:`2^n \times 2^n` orthogonal matrix

    """
    # Step 1: sample from Ginibre ensemble
    ginibre = (np.random.randn(dim, dim))
    # Step 2: perform QR decomposition of G
    mat_q, mat_r = np.linalg.qr(ginibre)
    # Step 3: make the decomposition unique
    mat_lambda = np.diag(mat_r) / abs(np.diag(mat_r))
    mat_u = mat_q @ np.diag(mat_lambda)
    return torch.tensor(mat_u, dtype=get_dtype())


def haar_unitary(dim: int) -> torch.Tensor:
    r""" randomly generate a unitary following Haar random, referenced by arXiv:math-ph/0609050v2

    Args:
        dim: dimension of unitary

    Returns:
        a :math:`d \times d` unitary

    """
    # Step 1: sample from Ginibre ensemble
    ginibre = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    # Step 2: perform QR decomposition of G
    mat_q, mat_r = np.linalg.qr(ginibre)
    # Step 3: make the decomposition unique
    mat_lambda = np.diag(mat_r) / np.abs(np.diag(mat_r))
    mat_u = mat_q @ np.diag(mat_lambda)
    return torch.tensor(mat_u, dtype=get_dtype())


def haar_state_vector(dim: int, is_real: Optional[bool] = False) -> torch.Tensor:
    r""" randomly generate a state vector following Haar random

    Args:
        dim: dimension of density matrix
        is_real: whether the vector is real, default to be False

    Returns:
        a :math:`2^n \times 1` state vector

    """
    if is_real:
        # Generate a Haar random orthogonal matrix
        mat_orthog = haar_orthogonal(dim)
        # Perform u onto |0>, i.e., the first column of o
        phi = mat_orthog[:, 0]
    else:
        # Generate a Haar random unitary
        unitary = haar_unitary(dim)
        # Perform u onto |0>, i.e., the first column of u
        phi = unitary[:, 0]

    return phi.view([-1, 1])


def haar_density_operator(dim: int, rank: int, is_real: Optional[bool] = False) -> torch.Tensor:
    r""" randomly generate a density matrix following Haar random

    Args:
        dim: dimension of density matrix
        rank: rank of density matrix, default to be ``None`` refering to full ranks
        is_real: whether the density matrix is real, default to be False

    Returns:
        a :math:`2^n \times 2^n` density matrix
    """
    assert 0 < rank <= dim, 'rank is an invalid number'
    if is_real:
        ginibre_matrix = np.random.randn(dim, rank)
        rho = ginibre_matrix @ ginibre_matrix.T
    else:
        ginibre_matrix = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
        rho = ginibre_matrix @ ginibre_matrix.conj().T
    rho = rho / np.trace(rho)
    return torch.tensor(rho, dtype=get_dtype())


@_alias({"num_systems": "num_qubits"})
def random_channel(num_systems: int, rank: int = None, 
                   target: str = 'kraus', 
                   size: Optional[int] = 1,
                   system_dim: Union[List[int], int] = 2) -> torch.Tensor:
    r"""Generate a random channel from its Stinespring representation

    Args:
        num_systems: number of systems
        rank: rank of this Channel. Defaults to be random sampled from :math:`[1, d]`
        target: target representation, should to be ``'choi'``, ``'kraus'`` or ``'stinespring'``
        size: batch size. Defaults to 1
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
        the target representation of a random channel.

    """
    target = target.lower()
    dim = _format_total_dim(num_systems, system_dim)
    rank = np.random.randint(dim) + 1 if rank is None else rank
    assert 1 <= rank <= dim, \
        f"rank must be positive and no larger than the dimension {dim} of the channel: received {rank}"

    list_repr = []
    
    for _ in range(size):
        
        unitary = unitary_group.rvs(rank * dim)
        stinespring_mat = torch.tensor(unitary[:, :dim], dtype=get_dtype()).reshape([rank, dim, dim])
        list_kraus = stinespring_mat[:rank]

        if target == 'choi':
            list_repr.append(qkit.qinfo.channel_repr_convert(list_kraus, source='kraus', target='choi'))
        elif target == 'stinespring':
            list_repr.append(qkit.qinfo.channel_repr_convert(list_kraus, source='kraus', target='stinespring'))
        else:
            list_repr.append(list_kraus)
    
    return torch.stack(list_repr) if size > 1 else list_repr[0]