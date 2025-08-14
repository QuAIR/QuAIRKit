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
from typing import List, Optional, Union

import numpy as np
import torch
from torch.linalg import matrix_power

from .. import database
from ..core import StateSimulator, get_dtype, to_state, utils
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

    Examples:
        .. code-block:: python

            num_qubits = 1
            basis = pauli_basis(num_qubits)
            print(f'The Pauli basis is:\n{basis}')

        ::

            The Pauli basis is:
            tensor([[[ 0.7071+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j,  0.7071+0.0000j]],

                    [[ 0.0000+0.0000j,  0.7071+0.0000j],
                     [ 0.7071+0.0000j,  0.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  0.0000-0.7071j],
                     [ 0.0000+0.7071j,  0.0000+0.0000j]],

                    [[ 0.7071+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j, -0.7071+0.0000j]]])
    """
    single_pauli_basis = torch.stack([database.matrix.eye(), 
                                      database.matrix.x(), 
                                      database.matrix.y(), 
                                      database.matrix.z()]) * math.sqrt(2) / 2
    if num_qubits == 1:
        return single_pauli_basis
    return reduce(
        lambda result, index: torch.kron(result, index),
        [single_pauli_basis for _ in range(num_qubits - 2)],
        torch.kron(single_pauli_basis, single_pauli_basis),
    )


def pauli_group(num_qubits: int) -> torch.Tensor:
    r"""Generate a Pauli group i.e., an unnormalized Pauli basis.

    Args:
        num_qubits: the number of qubits :math:`n`.

    Returns:
         The Pauli group of :math:`\mathbb{C}^{2^n \times 2^n}`, where each tensor is accessible along the first dimension.

    Examples:
        .. code-block:: python

            num_qubits = 1
            group = pauli_group(num_qubits)
            print(f'The Pauli group is:\n{group}')

        ::

            The Pauli group is:
            tensor([[[ 1.0000+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j,  1.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  1.0000+0.0000j],
                     [ 1.0000+0.0000j,  0.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  0.0000-1.0000j],
                     [ 0.0000+1.0000j,  0.0000+0.0000j]],

                    [[ 1.0000+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j, -1.0000+0.0000j]]])
    """
    return pauli_basis(num_qubits) * (math.sqrt(2) ** num_qubits)


def pauli_str_basis(pauli_str: Union[str, List[str]]) -> StateSimulator:
    r"""Get the state basis with respect to the Pauli string.
    
    Args:
        pauli_str: the string composed of 'i', 'x', 'y' and 'z' only.
    
    Returns:
        The state basis of the observable given by the Pauli string.
        
    Examples:
        .. code-block:: python

            pauli_str = ['x', 'z']
            state_basis = pauli_str_basis(pauli_str)
            print(f'The state basis of the observable is:\n{state_basis}')

        ::

            The state basis of the observable is:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2]
            System sequence: [0]
            Batch size: [2, 2]

            # 0:
            [0.71+0.j 0.71+0.j]
            # 1:
            [ 0.71+0.j -0.71+0.j]
            # 2:
            [1.+0.j 0.+0.j]
            # 3:
            [0.+0.j 1.+0.j]
            ---------------------------------------------------
    """
    x_basis = torch.tensor([[1, 1], 
                            [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    y_basis = torch.tensor([[1, 1j],
                            [1, -1j]], dtype=torch.complex128) / math.sqrt(2)
    z_basis = utils.matrix._eye(2)
    i_basis = z_basis
    locals_ = locals()
    
    def __get_single_str(string: str) -> torch.Tensor:
        list_basis = [locals_[f'{p}_basis'] for p in string.lower()]
        basis = utils.linalg._nkron(*list_basis) if len(list_basis) > 1 else list_basis[0]
        return basis.unsqueeze(-1)
    
    if isinstance(pauli_str, str):
        basis = __get_single_str(pauli_str)
    else:
        pauli_len = len(pauli_str[0])
        assert all(len(s) == pauli_len for s in pauli_str), \
            "All Pauli strings must have the same length."
        basis = torch.stack([__get_single_str(string) for string in pauli_str])
    return to_state(basis.to(dtype=get_dtype()))


def pauli_str_povm(pauli_str: Union[str, List[str]]) -> torch.Tensor:
    r"""Get the povm with respect to the Pauli string.
    
    Args:
        pauli_str: the string composed of 'i', 'x', 'y' and 'z' only.
    
    Returns:
        The POVM of the observable given by the Pauli string.
    
    Examples:
        .. code-block:: python

            pauli_str = ['x', 'y']
            POVM = pauli_str_povm(pauli_str)
            print(f'The POVM of the observable is:\n{POVM}')

        ::

            The POVM of the observable is:
            tensor([[[[ 0.5000+0.0000j,  0.5000+0.0000j],
                      [ 0.5000+0.0000j,  0.5000+0.0000j]],

                     [[ 0.5000+0.0000j, -0.5000+0.0000j],
                      [-0.5000+0.0000j,  0.5000+0.0000j]]],


                    [[[ 0.5000+0.0000j,  0.0000-0.5000j],
                      [ 0.0000+0.5000j,  0.5000+0.0000j]],

                     [[ 0.5000+0.0000j,  0.0000+0.5000j],
                      [ 0.0000-0.5000j,  0.5000+0.0000j]]]])
    """
    return pauli_str_basis(pauli_str).density_matrix


def qft_basis(num_qubits: int) -> StateSimulator:
    r"""Compute the eigenvectors (eigenbasis) of the Quantum Fourier Transform (QFT) matrix.

    Args:
        num_qubits: Number of qubits :math:`n` such that :math:`N = 2^n`.

    Returns:
        A tensor where the first index gives the eigenvector of the QFT matrix.
    
    Examples:
        .. code-block:: python

            num_qubits = 2
            qft_state = qft_basis(num_qubits)
            print(f'The eigenvectors of the QFT matrix is:\n{qft_state}')

        ::

            The eigenvectors of the QFT matrix is:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2]
            System sequence: [0]
            Batch size: [2]

            # 0:
            [0.92+0.j 0.38+0.j]
            # 1:
            [-0.38-0.j  0.92+0.j]
            ---------------------------------------------------
    """
    #TODO numerically unstable, needs a more precise implementation instead of using eig decomposition
    _, eigvec = torch.linalg.eig(database.matrix.qft_matrix(num_qubits))
    return to_state(eigvec.T.unsqueeze(-1).to(get_dtype()))


@_alias({"num_systems": "num_qubits"})
def std_basis(num_systems: Optional[int] = None, system_dim: Union[List[int], int] = 2) -> StateSimulator:
    r"""Generate all standard basis states for a given number of qubits.

    Args:
        num_systems: number of systems in this state. If None, inferred from system_dim. Alias of ``num_qubits``.
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
        A tensor where the first index gives the computational vector.
    
    Examples:
        .. code-block:: python

            num_systems = 2
            system_dim = [1, 2]
            basis = std_basis(num_systems, system_dim)
            print(f'The standard basis states are:\n{basis}')

        ::

            The standard basis states are:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [1, 2]
            System sequence: [0, 1]
            Batch size: [2]

            # 0:
            [1.+0.j 0.+0.j]
            # 1:
            [0.+0.j 1.+0.j]
            ---------------------------------------------------
    """
    if num_systems is None:
        num_systems = 1 if isinstance(system_dim, int) else len(system_dim)
  
    dim = system_dim ** num_systems if isinstance(system_dim, int) else math.prod(system_dim)
    return to_state(torch.eye(dim).unsqueeze(-1).to(get_dtype()), system_dim)


def bell_basis() -> StateSimulator:
    r"""Generate the Bell basis for a 2-qubit system, 
    with each basis state accessible along the first dimension of a tensor.

    Returns:
        A tensor of shape (4, 4, 1), representing the four Bell basis states.
    
    Examples:
        .. code-block:: python

            basis = bell_basis()
            print(f'The Bell basis for a 2-qubit system are:\n{basis}')

        ::

            The Bell basis for a 2-qubit system are:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2]
            System sequence: [0, 1]
            Batch size: [4]

            # 0:
            [0.71+0.j 0.  +0.j 0.  +0.j 0.71+0.j]
            # 1:
            [ 0.71+0.j  0.  +0.j  0.  +0.j -0.71+0.j]
            # 2:
            [0.  +0.j 0.71+0.j 0.71+0.j 0.  +0.j]
            # 3:
            [ 0.  +0.j  0.71+0.j -0.71+0.j  0.  +0.j]
            ---------------------------------------------------
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
    The Heisenberg-Weyl operators are defined as :math:`T(a,b) = e^{-(d+1) \pi i a b/ d}Z^a X^b`.

    Args:
        dim: dimension of qudit

    Returns:
        Heisenberg-Weyl operator for qudit.
        
    Examples:
        .. code-block:: python

            dim = 2
            operator = heisenberg_weyl(dim)
            print(f'The Heisenberg-Weyl operator for qudit is:\n{operator}')

        ::

            The Heisenberg-Weyl operator for qudit is:
            tensor([[[ 1.0000e+00+0.0000e+00j,  0.0000e+00+0.0000e+00j],
                     [ 0.0000e+00+0.0000e+00j,  1.0000e+00+0.0000e+00j]],

                    [[ 1.0000e+00+0.0000e+00j,  0.0000e+00+0.0000e+00j],
                     [ 0.0000e+00+0.0000e+00j, -1.0000e+00+1.2246e-16j]],

                    [[ 0.0000e+00+0.0000e+00j,  1.0000e+00+0.0000e+00j],
                     [ 1.0000e+00+0.0000e+00j,  0.0000e+00+0.0000e+00j]],

                    [[-0.0000e+00+0.0000e+00j, -1.8370e-16+1.0000e+00j],
                     [ 6.1232e-17-1.0000e+00j, -0.0000e+00+0.0000e+00j]]])
    """
    complex_dtype = get_dtype() 
    _phase = database.matrix.phase(dim)
    _shift = database.matrix.shift(dim)

    phase_pow = torch.zeros(dim, 1, dim, dim, dtype=complex_dtype)
    shift_pow = torch.zeros(1, dim, dim, dim, dtype=complex_dtype)

    a = torch.arange(dim, dtype=torch.float64).reshape(1, dim, 1, 1)
    b = torch.arange(dim, dtype=torch.float64).reshape(dim, 1, 1, 1)

    for i in range(dim):
        phase_pow[i, 0, :, :] = matrix_power(_phase, i)
        shift_pow[0, i, :, :] = matrix_power(_shift, i)

    hw = torch.matmul(phase_pow, shift_pow)
    ab = torch.matmul(a, b)
    ab = torch.exp(-ab * 1j * np.pi * (dim + 1) / dim)

    hw = ab * hw
    hw = hw.permute(1, 0, 2, 3)
    hw = hw.reshape((dim**2, dim, dim))
    return hw


def phase_space_point(dim: int) -> torch.Tensor:
    r"""Generate phase space point operator for qudit.

    Args:
        dim: dimension of qudit

    Returns:
        Phase space point operator for qudit.
        
    Examples:
        .. code-block:: python

            dim = 2
            operator = phase_space_point(dim)
            print(f'The phase space point operator for qudit is:\n{operator}')

        ::

            The phase space point operator for qudit is:
            tensor([[[ 1.0000+0.0000e+00j,  0.5000+5.0000e-01j],
                     [ 0.5000-5.0000e-01j,  0.0000+6.1232e-17j]],

                    [[ 1.0000+0.0000e+00j, -0.5000-5.0000e-01j],
                     [-0.5000+5.0000e-01j,  0.0000+6.1232e-17j]],

                    [[ 0.0000+6.1232e-17j,  0.5000-5.0000e-01j],
                     [ 0.5000+5.0000e-01j,  1.0000+0.0000e+00j]],

                    [[ 0.0000+6.1232e-17j, -0.5000+5.0000e-01j],
                     [-0.5000-5.0000e-01j,  1.0000+0.0000e+00j]]])
    """

    hw = heisenberg_weyl(dim)
    A0 = torch.sum(hw, dim=0) / dim

    hw_dagger = hw.mH
    A = A0.expand(hw.shape[0], A0.shape[0], A0.shape[1])
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
    of dim-by-dim matrices, and they generalize the Pauli operators when dim = 2 and the Gell-Mann operators 
    when dim = 3.

    Args:
        dim: a positive integer indicating the dimension.

    Returns:
        A set of Gell-Mann matrices.
        
    Examples:
        .. code-block:: python

            dim = 2
            matrices = gell_mann(dim)
            print(f'The Gell-Mann matrices are:\n{matrices}')

        ::

            The Gell-Mann matrices are:
            tensor([[[ 0.+0.j,  1.+0.j],
                     [ 1.+0.j,  0.+0.j]],

                    [[ 0.+0.j, -0.-1.j],
                     [ 0.+1.j,  0.+0.j]],

                    [[ 1.+0.j,  0.+0.j],
                     [ 0.+0.j, -1.+0.j]]])
    """
    list_gell_mann = [
        __gell_mann(idx1, idx2, dim).unsqueeze(0)
        for idx1, idx2 in itertools.product(range(dim), range(dim))
        if idx1 != 0 or idx2 != 0
    ]
    list_gell_mann = torch.cat(list_gell_mann, dim=0)
    return list_gell_mann.to(get_dtype())
