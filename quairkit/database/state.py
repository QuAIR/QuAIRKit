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
from typing import List, Optional, Union

import torch

from ..core import (OperatorInfoType, StateSimulator, get_backend, get_dtype,
                    tensor_state, to_state)
from ..core.base import PRODUCT_STATE_THRESHOLD
from ..core.intrinsic import _alias, _format_total_dim, _State
from ..core.state.backend import DefaultSimulator
from ..core.state.backend.default import ProductDefaultSimulator
from ..operator import CNOT, CRY, RY, H, X

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
def zero_state(num_systems: Optional[int] = None, system_dim: Union[List[int], int] = 2) -> _State:
    r"""Generate a zero state :math:`|0\rangle^{\otimes n}`.

    Args:
        num_systems: Number of systems in this state. If None, inferred from system_dim. Alias of ``num_qubits``.
        system_dim: Dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to qubit case.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            from quairkit.database import zero_state

            # Generate a 2-qubit zero state
            state = zero_state(2)
            print(f'2-qubit zero state:\n{state}')

        ::

            2-qubit zero state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2]
            System sequence: [0, 1]
            [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
            ---------------------------------------------------

        .. code-block:: python

            # Generate a qubit-qutrit zero state
            state = zero_state(2, system_dim=[2, 3])
            print(f'Qubit-qutrit zero state:\n{state}')

        ::

            Qubit-qutrit zero state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 3]
            System sequence: [0, 1]
            [1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
            ---------------------------------------------------
    """
    return computational_state(num_systems, 0, system_dim)


@_alias({"num_systems": "num_qubits"})
def one_state(num_systems: Optional[int] = None, system_dim: Union[List[int], int] = 2) -> _State:
    r"""Generate a one state.

    Args:
        num_systems: Number of systems in this state. If None, inferred from system_dim. Alias of ``num_qubits``.
        system_dim: Dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to qubit case.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            num_systems = 2
            system_dim = [1, 3]
            state = one_state(num_systems, system_dim)
            print(f'The one state is:\n{state}')

        ::

            The one state is:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [1, 3]
            System sequence: [0, 1]
            [0.+0.j 1.+0.j 0.+0.j]
            ---------------------------------------------------
    """
    return computational_state(num_systems, 1, system_dim)


@_alias({"num_systems": "num_qubits"})
def computational_state(num_systems: Optional[int] = None, index: int = 0, 
                        system_dim: Union[List[int], int] = 2) -> _State:
    r"""Generate a computational state :math:`|e_{i}\rangle`, 
    whose i-th element is 1 and all the other elements are 0.

    Args:
        num_systems: Number of systems in this state. If None, inferred from system_dim. Alias of ``num_qubits``.
        index:  Index :math:`i` of the computational basis state :math:`|e_{i}\rangle`.
        system_dim: Dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to qubit case.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            num_systems = 2
            system_dim = [2, 3]
            index = 4
            state = computational_state(num_systems, index, system_dim)
            print(f'The state is:\n{state}')

        ::

            The state is:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 3]
            System sequence: [0, 1]
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j]
            ---------------------------------------------------
    """
    if num_systems is None:
        num_systems = 1 if isinstance(system_dim, int) else len(system_dim)
    
    backend = get_backend()
    if issubclass(backend, StateSimulator):
        if issubclass(backend, DefaultSimulator) and num_systems >= PRODUCT_STATE_THRESHOLD:
            sys_dim = [system_dim] * num_systems if isinstance(system_dim, int) else list(system_dim)
            total_dim = math.prod(sys_dim)
            if index < 0 or index >= total_dim:
                raise ValueError(
                    f"index out of range for computational basis: got {index}, "
                    f"expect in [0, {total_dim - 1}]"
                )

            digits = []
            quotient = index
            for d in reversed(sys_dim):
                digits.append(quotient % d)
                quotient //= d
            digits.reverse()

            single_states = []
            for d, digit in zip(sys_dim, digits):
                vec = torch.zeros(d, dtype=get_dtype())
                vec[digit] = 1
                single_states.append(to_state(vec, d))

            return tensor_state(single_states[0], *single_states[1:])

        dim = _format_total_dim(num_systems, system_dim)
        data = torch.zeros(dim)
        data[index] = 1
        return to_state(data, system_dim)
    else:
        system_dim = [2] * num_systems
        index_binary = bin(index)[2:].zfill(num_systems)
        x_idx = [[idx] for idx in range(num_systems) if index_binary[idx] == '1']
        data = [X(x_idx).info] if x_idx else []   
    return to_state(data, system_dim)


@_alias({"num_systems": "num_qubits"})
def bell_state(num_systems: Optional[int] = None, system_dim: Union[List[int], int] = 2) -> _State:
    r"""Generate a Bell state (maximally entangled state).

    Its matrix form is:

    .. math::

        |\Phi_{D}\rangle=\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}

    where :math:`D` is the dimension of each subsystem.

    Args:
        num_systems: Number of systems in this state. Must be even. If None, inferred from system_dim. 
            Alias of ``num_qubits``.
        system_dim: Dimension of systems. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to qubit case.

    Returns:
        The generated quantum state.

    Raises:
        AssertionError: If the number of systems is not even.

    Examples:
        .. code-block:: python

            from quairkit.database import bell_state

            # Generate a 2-qubit Bell state
            state = bell_state(2)
            print(f'2-qubit Bell state:\n{state}')

        ::

            2-qubit Bell state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2]
            System sequence: [0, 1]
            [0.71+0.j 0.  +0.j 0.  +0.j 0.71+0.j]
            ---------------------------------------------------

        .. code-block:: python

            # Generate a 4-qubit Bell state (two Bell pairs)
            state = bell_state(4)
            print(f'4-qubit Bell state:\n{state}')

        ::

            4-qubit Bell state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2, 2, 2]
            System sequence: [0, 1, 2, 3]
            [0.5+0.j 0.+0.j ... 0.5+0.j]
            ---------------------------------------------------
    """
    if num_systems is None:
        num_systems = 1 if isinstance(system_dim, int) else len(system_dim)

    assert num_systems % 2 == 0, \
        f"Number of systems must be even to form a Bell state. Received: {num_systems}"
    half = num_systems // 2

    backend = get_backend()
    if issubclass(backend, StateSimulator):
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
    else:
        if isinstance(system_dim, int):
            system_dim = [system_dim] * num_systems
        assert system_dim == [2] * num_systems,\
            f"Only qubit system is supported in circuit simulation. Received: {system_dim}"
        assert num_systems%2==0,\
            f"num_systems must be a multiple of 2. Received: {num_systems}"
        data=[]
        for i in range(num_systems//2):
            data.extend((H(i).info, CNOT([i,i+int(num_systems/2)]).info))
    return to_state(data, system_dim)


def bell_diagonal_state(prob: List[float]) -> StateSimulator:
    r"""Generate a Bell diagonal state.

    Its matrix form is:

    .. math::

        p_{1}|\Phi^{+}\rangle\langle\Phi^{+}|+p_{2}|\Psi^{+}\rangle\langle\Psi^{+}|+p_{3}|\Phi^{-}\rangle\langle\Phi^{-}| +
        p_{4}|\Psi^{-}\rangle\langle\Psi^{-}|

    Args:
        prob: The probability of each Bell state.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            prob = [0.2, 0.3, 0.4, 0.1]
            state = bell_diagonal_state(prob)
            print(f'The Bell diagonal state is:\n{state}')

        ::

            The Bell diagonal state is:
            ---------------------------------------------------
            Backend: density_matrix
            System dimension: [2, 2]
            System sequence: [0, 1]
            [[ 0.3+0.j  0. +0.j  0. +0.j -0.1+0.j]
             [ 0. +0.j  0.2+0.j  0.1+0.j  0. +0.j]
             [ 0. +0.j  0.1+0.j  0.2+0.j  0. +0.j]
             [-0.1+0.j  0. +0.j  0. +0.j  0.3+0.j]]
            ---------------------------------------------------
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


def _add_w_state_info(num_qubits: int) -> List[OperatorInfoType]:
    if num_qubits == 1:
        data = [H(0).info]
    elif num_qubits == 2:
        data = [H(0).info, CNOT([0, 1]).info]
    else:
        data=[X(0).info]        
        for i in range(num_qubits-1):
            theta_i = 2*math.acos(1 / math.sqrt(num_qubits- i))
            data.extend((CRY([i,i+1],theta_i).info,CNOT([i+1,i]).info))
    return data


def w_state(num_qubits: int) -> _State:
    r"""Generate a W-state :math:`|W_n\rangle = \frac{1}{\sqrt{n}}(|10...0\rangle + |01...0\rangle + ... + |00...1\rangle)`.

    The W-state is a symmetric superposition of all states with exactly one qubit in state :math:`|1\rangle`.

    Args:
        num_qubits: The number of qubits in the quantum state. Must be at least 1.

    Returns:
        The generated W-state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Examples:
        .. code-block:: python

            from quairkit.database import w_state

            # Generate a 3-qubit W-state
            state = w_state(3)
            print(f'3-qubit W-state:\n{state}')

        ::

            3-qubit W-state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2, 2]
            System sequence: [0, 1, 2]
            [0.  +0.j 0.58+0.j 0.58+0.j 0.  +0.j 0.58+0.j 0.  +0.j 0.  +0.j 0.  +0.j]
            ---------------------------------------------------
    """

    backend = get_backend()
    system_dim = [2] * num_qubits
    if issubclass(backend, StateSimulator):
        dim = 2 ** num_qubits
        coeff = 1 / math.sqrt(num_qubits)
        data = torch.zeros(dim)

        for i in range(num_qubits):
            data[2 ** i] = coeff
    else:
        data = _add_w_state_info(num_qubits)
    return to_state(data, system_dim)


def ghz_state(num_qubits: int) -> _State:
    r"""Generate a GHZ-state (Greenberger-Horne-Zeilinger state).

    The GHZ-state is :math:`|GHZ_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})`.

    Args:
        num_qubits: The number of qubits in the quantum state. Must be at least 2.

    Returns:
        The generated GHZ-state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Examples:
        .. code-block:: python

            from quairkit.database import ghz_state

            # Generate a 3-qubit GHZ-state
            state = ghz_state(3)
            print(f'3-qubit GHZ-state:\n{state}')

        ::

            3-qubit GHZ-state:
            ---------------------------------------------------
            Backend: state_vector
            System dimension: [2, 2, 2]
            System sequence: [0, 1, 2]
            [0.71+0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.71+0.j]
            ---------------------------------------------------
    """
    backend = get_backend()
    system_dim = [2] * num_qubits
    if issubclass(backend, StateSimulator):
        dim = 2 ** num_qubits
        data = torch.zeros(dim)
        data[0] = 1 / math.sqrt(2)
        data[-1] = 1 / math.sqrt(2)
    else:
        data=[H(0).info,
              CNOT([[i-1, i] for i in range(1, num_qubits)]).info]
    return to_state(data, system_dim)


def completely_mixed_computational(num_qubits: int) -> StateSimulator:
    r"""Generate the density matrix of the completely mixed state :math:`\rho = I/2^n`.

    The completely mixed state is the maximally mixed state with maximum von Neumann entropy.

    Args:
        num_qubits: The number of qubits in the quantum state.

    Returns:
        The generated completely mixed quantum state (density matrix).

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Examples:
        .. code-block:: python

            from quairkit.database import completely_mixed_computational

            # Generate a 2-qubit completely mixed state
            state = completely_mixed_computational(2)
            print(f'2-qubit completely mixed state:\n{state}')

        ::

            2-qubit completely mixed state:
            ---------------------------------------------------
            Backend: density_matrix
            System dimension: [2, 2]
            System sequence: [0, 1]
            [[0.25+0.j 0.  +0.j 0.  +0.j 0.  +0.j]
             [0.  +0.j 0.25+0.j 0.  +0.j 0.  +0.j]
             [0.  +0.j 0.  +0.j 0.25+0.j 0.  +0.j]
             [0.  +0.j 0.  +0.j 0.  +0.j 0.25+0.j]]
            ---------------------------------------------------
    """
    data = torch.eye(2 ** num_qubits) / (2 ** num_qubits)
    return to_state(data)


def r_state(prob: float) -> StateSimulator:
    r"""Generate an R-state.

    Its matrix form is:

    .. math::

        p|\Psi^{+}\rangle\langle\Psi^{+}| + (1 - p)|11\rangle\langle11|

    Args:
        prob: The parameter of the R-state to be generated. It should be in :math:`[0,1]`.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            prob = 0.5
            r_state_inst = r_state(prob)
            print(f'The R-state is:\n{r_state_inst}')

        ::

            The R-state is:
            ---------------------------------------------------
            Backend: density_matrix
            System dimension: [2, 2]
            System sequence: [0, 1]
            [[0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
             [0.  +0.j 0.25+0.j 0.25+0.j 0.  +0.j]
             [0.  +0.j 0.25+0.j 0.25+0.j 0.  +0.j]
             [0.  +0.j 0.  +0.j 0.  +0.j 0.5 +0.j]]
            ---------------------------------------------------
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


def s_state(prob: float) -> StateSimulator:
    r"""Generate the S-state.

    Its matrix form is:

    .. math::

        p|\Phi^{+}\rangle\langle\Phi^{+}| + (1 - p)|00\rangle\langle00|

    Args:
        prob: The parameter of the S-state to be generated. It should be in :math:`[0,1]`.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            prob = 0.5
            s_state_inst = s_state(prob)
            print(f'The S-state is:\n{s_state_inst}')

        ::

            The S-state is:
            ---------------------------------------------------
            Backend: density_matrix
            System dimension: [2, 2]
            System sequence: [0, 1]
            [[0.75+0.j 0.  +0.j 0.  +0.j 0.25+0.j]
             [0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
             [0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]
             [0.25+0.j 0.  +0.j 0.  +0.j 0.25+0.j]]
            ---------------------------------------------------
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"
    
    phi_p = bell_state(2).density_matrix
    psi0 = torch.zeros_like(phi_p)
    psi0[0, 0] = 1
    data = prob * phi_p + (1 - prob) * psi0

    return to_state(data)


def isotropic_state(num_qubits: int, prob: float) -> StateSimulator:
    r"""Generate the isotropic state.

    Its matrix form is:

    .. math::

        p\left(\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}\right) + (1 - p)\frac{I}{2^n}

    Args:
        num_qubits: The number of qubits in the quantum state.
        prob: The parameter of the isotropic state to be generated. It should be in :math:`[0,1]`.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.

    Examples:
        .. code-block:: python

            num_qubits = 2
            prob = 0.5
            isotropic_state_inst = isotropic_state(num_qubits, prob)
            print(f'The isotropic state is:\n{isotropic_state_inst}')

        ::

            The isotropic state is:
            ---------------------------------------------------
            Backend: density_matrix
            System dimension: [2, 2]
            System sequence: [0, 1]
            [[0.38+0.j 0.  +0.j 0.  +0.j 0.25+0.j]
             [0.  +0.j 0.12+0.j 0.  +0.j 0.  +0.j]
             [0.  +0.j 0.  +0.j 0.12+0.j 0.  +0.j]
             [0.25+0.j 0.  +0.j 0.  +0.j 0.38+0.j]]
            ---------------------------------------------------
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    dim = 2 ** num_qubits
    phi_b = bell_state(num_qubits).density_matrix
    data = prob * phi_b + (1 - prob) * torch.eye(dim) / dim

    return to_state(data)
