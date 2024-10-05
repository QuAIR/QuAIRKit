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

import itertools
import math
from functools import reduce
from typing import List, Union

import numpy as np
import torch
from torch.linalg import matrix_power

from .. import database
from ..core import State, get_dtype, to_state, utils
from ..core.intrinsic import _alias

__all__ = [
    "pauli_basis",
    "pauli_group",
    "pauli_str_basis",
    "pauli_str_povm",
    "qft_basis",
    "std_basis",
    "bell_basis",
    "heisenberg_weyl",
    "phase_space_point",
    "gell_mann",
]


def pauli_basis(num_qubits: int) -> torch.Tensor:
    r"""Generate a Pauli basis.

    Args:
        num_qubits: the number of qubits :math:`n`.

    Returns:
         The Pauli basis of :math:`\mathbb{C}^{2^n \times 2^n}`, where each tensor is accessible along the first dimension.

    """
    single_pauli_basis = torch.stack([database.matrix.eye(torch.complex128), 
                                      database.matrix.x(torch.complex128), 
                                      database.matrix.y(torch.complex128), 
                                      database.matrix.z(torch.complex128)]) * math.sqrt(2) / 2
    if num_qubits == 1:
        return single_pauli_basis
    return reduce(
        lambda result, index: torch.kron(result, index),
        [single_pauli_basis for _ in range(num_qubits - 2)],
        torch.kron(single_pauli_basis, single_pauli_basis),
    ).to(get_dtype())


def pauli_group(num_qubits: int) -> torch.Tensor:
    r"""Generate a Pauli group i.e., an unnormalized Pauli basis.

    Args:
        num_qubits: the number of qubits :math:`n`.

    Returns:
         The Pauli group of :math:`\mathbb{C}^{2^n \times 2^n}`, where each tensor is accessible along the first dimension.

    """
    return pauli_basis(num_qubits) * (math.sqrt(2) ** num_qubits)


def pauli_str_basis(pauli_str: Union[str, List[str]]) -> State:
    r"""Get the state basis with respect to the Pauli string
    
    Args:
        pauli_str: the string composed of 'i', 'x', 'y' and 'z' only.
    
    Returns:
        The state basis of the observable given by the Pauli string
    
    """
    x_basis = torch.tensor([[1, 1], 
                            [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    y_basis = torch.tensor([[1, 1j],
                            [1, -1j]], dtype=torch.complex128) / math.sqrt(2)
    z_basis = torch.eye(2, dtype=torch.complex128)
    i_basis = z_basis
    locals_ = locals()
    
    def __get_single_str(string: str) -> torch.Tensor:
        list_basis = [locals_[f'{p}_basis'] for p in string.lower()]
        basis = utils.linalg._nkron(*list_basis) if len(list_basis) > 1 else list_basis[0]
        return basis.unsqueeze(-1)
    
    # TODO fit batched nkron
    basis = __get_single_str(pauli_str) if isinstance(pauli_str, str) else torch.stack([__get_single_str(string) for string in pauli_str])
    return to_state(basis.to(dtype=get_dtype()))


def pauli_str_povm(pauli_str: Union[str, List[str]]) -> torch.Tensor:
    r"""Get the povm with respect to the Pauli string
    
    Args:
        pauli_str: the string composed of 'i', 'x', 'y' and 'z' only.
    
    Returns:
        The POVM of the observable given by the Pauli string
    
    """
    return pauli_str_basis(pauli_str).density_matrix


def qft_basis(num_qubits: int) -> State:
    r"""Compute the eigenvectors (eigenbasis) of the Quantum Fourier Transform (QFT) matrix.

    Args:
        num_qubits: Number of qubits :math:`n` such that :math:`N = 2^n`.

    Returns:
        A tensor where the first index gives the eigenvector of the QFT matrix.
    
    """
    #TODO numerically unstable, needs a more precise implementation instead of using eig decomposition
    _, eigvec = torch.linalg.eig(database.matrix.qft_matrix(num_qubits, dtype=torch.complex128))
    return to_state(eigvec.T.unsqueeze(-1).to(get_dtype()))


@_alias({"num_systems": "num_qubits"})
def std_basis(num_systems: int, system_dim: Union[List[int], int] = 2) -> State:
    r"""Generate all standard basis states for a given number of qubits.

    Args:
        num_systems: number of systems in this state. Alias of ``num_qubits``.
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
        A tensor where the first index gives the computational vector
    
    """
    dim = system_dim ** num_systems if isinstance(system_dim, int) else math.prod(system_dim)
    return to_state(torch.eye(dim).unsqueeze(-1).to(get_dtype()), system_dim)


def bell_basis() -> State:
    r"""Generate the Bell basis for a 2-qubit system, 
    with each basis state accessible along the first dimension of a tensor.

    Returns:
        A tensor of shape (4, 4, 1), representing the four Bell basis states.
    
    """
    mat = torch.tensor([
        [ 1,  0,  0,  1],   # |Φ+⟩ = (|00⟩ + |11⟩) / √2
        [ 1,  0,  0, -1],   # |Φ-⟩ = (|00⟩ - |11⟩) / √2
        [ 0,  1,  1,  0],   # |Ψ+⟩ = (|01⟩ + |10⟩) / √2
        [ 0,  1, -1,  0]    # |Ψ-⟩ = (|01⟩ - |10⟩) / √2
    ], dtype=torch.complex128) / math.sqrt(2)
    return to_state(mat.unsqueeze(-1).to(get_dtype()))


def heisenberg_weyl(dim: int) -> torch.Tensor:
    r"""Generate Heisenberg-Weyl operator for qudit. 
        The Heisenberg-Weyl operators are defined as T(a,b) = e^{-(d+1) \pi i a b/ d}Z^a X^b.

    Args:
        dim: dimension of qudit

    Returns:
        Heisenberg-Weyl operator for qudit
    """
    complex_dtype = get_dtype() 
    _phase = database.matrix.phase(dim)
    _shift = database.matrix.shift(dim)

    phase_pow = torch.zeros(dim, 1, dim, dim, dtype=complex_dtype)
    shift_pow = torch.zeros(1, dim, dim, dim, dtype=complex_dtype)

    a = torch.arange(dim, dtype = torch.float64).reshape(1, dim, 1,1)
    b = torch.arange(dim, dtype = torch.float64).reshape(dim, 1,1,1)

    for i in range(dim):
        phase_pow[i, 0, :, :] = matrix_power(_phase, i)
        shift_pow[0, i, :, :] = matrix_power(_shift, i)

    hw = torch.matmul(phase_pow, shift_pow)
    ab = torch.matmul(a, b)
    ab = torch.exp(-ab*1j*np.pi*(dim + 1)/dim)

    hw = ab * hw
    hw = hw.permute(1,0,2,3)
    hw = hw.reshape((dim**2, dim, dim))
    return hw


def phase_space_point(dim: int) -> torch.Tensor:
    r"""Generate phase space point operator for qudit

    Args:
        dim: dimension of qudit

    Returns:
        Phase space point operator for qudit
    """

    hw = heisenberg_weyl(dim)
    A0 = torch.sum(hw, dim=0) / dim

    hw_dagger = hw.mH
    A = A0.expand(hw.shape[0],A0.shape[0],A0.shape[1])
    A = torch.matmul(hw, A)
    A = torch.matmul(A, hw_dagger)
        
    return A

def __gell_mann(index1: int, index2: int, dim: int) -> torch.Tensor:
    if index1 == index2:
        if index1 == 0:
            mat = torch.eye(dim)
        else:
            N = math.sqrt(2 / (index1 * (index1 + 1)))
            
            diag_element = torch.zeros(dim)
            diag_element[:index1] = 1
            diag_element[index1] = -index1
            
            mat = N * torch.diag(diag_element).to(torch.complex128)
    else:
        # Off-diagonal elements
        E = torch.zeros(dim, dim)
        E[index1, index2] = 1
        mat = E + E.T if index1 < index2 else 1j * (E - E.T)
    return mat

def gell_mann(dim: int) -> torch.Tensor:
    r"""Generate a set of Gell-Mann matrices for a given dimension. These matrices span the entire space 
    dim-by-dim matrices, and they generalize the Pauli operators when dim = 2 and the Gell-Mann operators 
    when dim = 3.

    Args:
        dim: a positive integer indicating the dimension.

    Returns:
        A set of Gell-Mann matrices.
    """
    list_gell_mann = [
        __gell_mann(idx1, idx2, dim).unsqueeze(0)
        for idx1, idx2 in itertools.product(range(dim), range(dim))
        if idx1 != 0 or idx2 != 0
    ]
    list_gell_mann = torch.cat(list_gell_mann, dim=0)
    return list_gell_mann.to(get_dtype())
