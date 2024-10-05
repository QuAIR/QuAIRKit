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
The library of common quantum states.
"""


import math
from typing import List, Union

import torch

from ..core import State, to_state
from ..core.intrinsic import _alias, _format_total_dim

__all__ = [
    "zero_state",
    "one_state",
    "computational_state",
    "bell_state",
    "bell_diagonal_state",
    "w_state",
    "ghz_state",
    "completely_mixed_computational",
    "r_state",
    "s_state",
    "isotropic_state",
]


@_alias({"num_systems": "num_qubits"})
def zero_state(num_systems: int, system_dim: Union[List[int], int] = 2) -> State:
    r"""The function to generate a zero state.

    Args:
        num_systems: number of systems in this state. Alias of ``num_qubits``.
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
       The generated quantum state.
    """
    return computational_state(num_systems, 0, system_dim)


@_alias({"num_systems": "num_qubits"})
def one_state(num_systems: int, system_dim: Union[List[int], int] = 2) -> State:
    r"""The function to generate a one state.

    Args:
        num_systems: number of systems in this state. Alias of ``num_qubits``.
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
       The generated quantum state.
    """
    return computational_state(num_systems, 1, system_dim)


@_alias({"num_systems": "num_qubits"})
def computational_state(num_systems: int, index: int, 
                        system_dim: Union[List[int], int] = 2) -> State:
    r"""Generate a computational state :math:`|e_{i}\rangle` , 
    whose i-th element is 1 and all the other elements are 0.

    Args:
        num_systems: number of systems in this state. Alias of ``num_qubits``.
        index:  Index :math:`i` of the computational basis state :math`|e_{i}rangle` .
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
        The generated quantum state.
    """
    dim = _format_total_dim(num_systems, system_dim)
    
    data = torch.zeros(dim)
    data[index] = 1
    return to_state(data, system_dim)


@_alias({"num_systems": "num_qubits"})
def bell_state(num_systems: int, system_dim: Union[List[int], int] = 2) -> State:
    r"""Generate a bell state.

    Its matrix form is:

    .. math::

        |\Phi_{D}\rangle=\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}

    Args:
        num_systems: number of systems in this state. Alias of ``num_qubits``.
        system_dim: dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    Returns:
        The generated quantum state.
    """
    assert num_systems % 2 == 0, \
        f"Number of systems must be even to form a Bell state. Received: {num_systems}"
    half = num_systems // 2
    
    dim = _format_total_dim(num_systems, system_dim)
    if isinstance(system_dim, int):
        local_dim = system_dim ** half
    else:
        local_dim = math.prod(system_dim[:half])
        assert dim == (local_dim ** 2), \
            f"Dimension of systems must be evenly distributed. Received: {system_dim}"
    
    data = torch.zeros(dim)
    for i in range(0, dim, local_dim + 1 ):
        data[i] = 1 / math.sqrt(local_dim)
    return to_state(data, system_dim)


def bell_diagonal_state(prob: List[float]) -> State:
    r"""Generate a bell diagonal state.

    Its matrix form is:

    .. math::

        p_{1}|\Phi^{+}\rangle\langle\Phi^{+}|+p_{2}| \Psi^{+}\rangle\langle\Psi^{+}|+p_{3}| \Phi^{-}\rangle\langle\Phi^{-}| +
        p_{4}|\Psi^{-}\rangle\langle\Psi^{-}|

    Args:
        prob: The prob of each bell state.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    p1, p2, p3, p4 = prob
    assert 0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1 and 0 <= p4 <= 1, \
        "Each probability must be in [0, 1]."
    assert abs(p1 + p2 + p3 + p4 - 1) < 1e-6, \
        "The sum of probabilities should be 1."

    phi_plus = torch.tensor([[1, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, 1]]) / 2
    phi_minus = torch.tensor([[1, 0, 0, -1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [-1, 0, 0, 1]]) / 2
    
    psi_plus = torch.tensor([[0, 0, 0, 0],
                             [0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0]]) / 2
    psi_minus = torch.tensor([[0, 0, 0, 0],
                              [0, 1, -1, 0],
                              [0, -1, 1, 0],
                              [0, 0, 0, 0]]) / 2

    data = p1 * phi_plus + p2 * psi_plus + p3 * phi_minus + p4 * psi_minus

    return to_state(data)


def w_state(num_qubits: int) -> State:
    r"""Generate a W-state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    dim = 2 ** num_qubits
    coeff = 1 / math.sqrt(num_qubits)
    data = torch.zeros(dim)

    for i in range(num_qubits):
        data[2 ** i] = coeff

    return to_state(data)


def ghz_state(num_qubits: int) -> State:
    r"""Generate a GHZ-state.

    Args:
        num_qubits: The number of qubits contained izn the quantum state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    dim = 2 ** num_qubits
    data = torch.zeros(dim)
    data[0] = 1 / math.sqrt(2)
    data[-1] = 1 / math.sqrt(2)
    return to_state(data)


def completely_mixed_computational(num_qubits: int) -> State:
    r"""Generate the density matrix of the completely mixed state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    data = torch.eye(2 ** num_qubits) / (2 ** num_qubits)
    return to_state(data)


def r_state(prob: float) -> State:
    r"""Generate an R-state.

    Its matrix form is:

    .. math::

        p|\Psi^{+}\rangle\langle\Psi^{+}| + (1 - p)|11\rangle\langle11|

    Args:
        prob: The parameter of the R-state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    phi_plus = torch.tensor([[0, 0, 0, 0],
                             [0, 1, 1, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 0]]) / 2
    
    state_11 = torch.zeros((4, 4))
    state_11[3, 3] = 1
    data = prob * phi_plus + (1 - prob) * state_11
    
    return to_state(data)


def s_state(prob: float) -> State:
    r"""Generate the S-state.

    Its matrix form is:

    .. math::

        p|\Phi^{+}\rangle\langle\Phi^{+}| + (1 - p)|00\rangle\langle00|

    Args:
        prob: The parameter of the S-state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"
    
    phi_p = bell_state(2).density_matrix
    psi0 = torch.zeros_like(phi_p)
    psi0[0, 0] = 1
    data = prob * phi_p + (1 - prob) * psi0

    return to_state(data)


def isotropic_state(num_qubits: int, prob: float) -> State:
    r"""Generate the isotropic state.

    Its matrix form is:

    .. math::

        p(\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}) + (1 - p)\frac{I}{2^n}

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        prob: The parameter of the isotropic state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    dim = 2 ** num_qubits
    phi_b = bell_state(num_qubits).density_matrix
    data = prob * phi_b + (1 - prob) * torch.eye(dim) / dim

    return to_state(data)
