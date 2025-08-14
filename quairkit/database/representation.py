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
Representations of channels
"""

from typing import List

import torch

from ..core import State, utils
from ..core.intrinsic import (_ArrayLike, _ParamLike, _SingleParamLike,
                              _StateLike, _type_fetch, _type_transform)
from .set import pauli_group

__all__ = [
    "bit_flip_kraus",
    "phase_flip_kraus",
    "bit_phase_flip_kraus",
    "amplitude_damping_kraus",
    "generalized_amplitude_damping_kraus",
    "phase_damping_kraus",
    "depolarizing_kraus",
    "phase_damping_kraus",
    "depolarizing_kraus",
    "generalized_depolarizing_kraus",
    "pauli_kraus",
    "reset_kraus",
    "thermal_relaxation_kraus",
    "replacement_choi",
]


def bit_flip_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a bit flip channel with form

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} X.

    Args:
        prob: probability :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob = torch.tensor([0.5])
            kraus_operators = bit_flip_kraus(prob)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[0.7071+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.7071+0.j]],

                    [[0.0000+0.j, 0.7071+0.j],
                     [0.7071+0.j, 0.0000+0.j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._bit_flip_kraus(prob)
    return _type_transform(mat, type_str)


def phase_flip_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a phase flip channel with form

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} Z.

    Args:
        prob: probability :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob = torch.tensor([0.1])
            kraus_operators = phase_flip_kraus(prob)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[ 0.9487+0.j,  0.0000+0.j],
                     [ 0.0000+0.j,  0.9487+0.j]],

                    [[ 0.3162+0.j,  0.0000+0.j],
                     [ 0.0000+0.j, -0.3162+0.j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._phase_flip_kraus(prob)
    return _type_transform(mat, type_str)


def bit_phase_flip_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a bit-phase flip channel with form

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} Y.

    Args:
        prob: probability :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob = torch.tensor([0.1])
            kraus_operators = bit_phase_flip_kraus(prob)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[0.9487+0.0000j, 0.0000+0.0000j],
                     [0.0000+0.0000j, 0.9487+0.0000j]],

                    [[0.0000+0.0000j, 0.0000-0.3162j],
                     [0.0000+0.3162j, 0.0000+0.0000j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._bit_phase_flip_kraus(prob)
    return _type_transform(mat, type_str)


def amplitude_damping_kraus(gamma: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of an amplitude damping channel with form

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{bmatrix}.

    Args:
        gamma: coefficient :math:`\gamma`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            gamma = torch.tensor(0.2)
            kraus_operators = amplitude_damping_kraus(gamma)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[1.0000+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.8944+0.j]],

                    [[0.0000+0.j, 0.4472+0.j],
                     [0.0000+0.j, 0.0000+0.j]]], dtype=torch.complex128)
    """
    type_str, gamma = _type_fetch(gamma), _type_transform(gamma, "tensor")
    mat = utils.representation._amplitude_damping_kraus(gamma)
    return _type_transform(mat, type_str)


def generalized_amplitude_damping_kraus(
        gamma: _SingleParamLike,
        prob: _SingleParamLike
) -> List[_ArrayLike]:
    r"""Kraus representation of a generalized amplitude damping channel with form

    .. math::

        E_0 = \sqrt{p} \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
        E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
        E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
        E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

    Args:
        gamma: coefficient :math:`\gamma`.
        prob: probability :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            gamma = torch.tensor(0.2)
            prob = torch.tensor(0.1)
            kraus_operators = generalized_amplitude_damping_kraus(gamma, prob)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[0.3162+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.2828+0.j]],

                    [[0.0000+0.j, 0.1414+0.j],
                     [0.0000+0.j, 0.0000+0.j]],

                    [[0.8485+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.9487+0.j]],

                    [[0.0000+0.j, 0.0000+0.j],
                     [0.4243+0.j, 0.0000+0.j]]], dtype=torch.complex128)
    """
    type_str, gamma, prob = _type_fetch(gamma), _type_transform(gamma, "tensor"), _type_transform(prob, "tensor")
    mat = utils.representation._generalized_amplitude_damping_kraus(gamma, prob)
    return _type_transform(mat, type_str)


def phase_damping_kraus(gamma: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a phase damping channel with form

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{\gamma}
        \end{bmatrix}.

    Args:
        gamma: coefficient :math:`\gamma`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            gamma = torch.tensor(0.2)
            kraus_operators = phase_damping_kraus(gamma)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[1.0000+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.8944+0.j]],

                    [[0.0000+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.4472+0.j]]], dtype=torch.complex128)
    """
    type_str, gamma = _type_fetch(gamma), _type_transform(gamma, "tensor")
    mat = utils.representation._phase_damping_kraus(gamma)
    return _type_transform(mat, type_str)


def depolarizing_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a depolarizing channel with form

    .. math::

        E_0 = \sqrt{1-\frac{3p}{4}} I,
        E_1 = \sqrt{\frac{p}{4}} X,
        E_2 = \sqrt{\frac{p}{4}} Y,
        E_3 = \sqrt{\frac{p}{4}} Z.

    Args:
        prob: probability :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob = torch.tensor(0.1)
            kraus_operators = depolarizing_kraus(prob)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[ 0.9618+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j,  0.9618+0.0000j]],

                    [[ 0.0000+0.0000j,  0.1581+0.0000j],
                     [ 0.1581+0.0000j,  0.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  0.0000-0.1581j],
                     [ 0.0000+0.1581j,  0.0000+0.0000j]],

                    [[ 0.1581+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j, -0.1581+0.0000j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._depolarizing_kraus(prob)
    return _type_transform(mat, type_str)


def generalized_depolarizing_kraus(prob: _SingleParamLike, num_qubits: int) -> List[_ArrayLike]:
    r"""Kraus representation of a generalized depolarizing channel with form

    .. math::

        E_0 = \sqrt{1-\frac{(D-1)p}{D}} I, \text{ where } D = 4^n,
        E_k = \sqrt{\frac{p}{D}} \sigma_k, \text{ for } 0 < k < D.

    Args:
        prob: probability :math:`p`.
        num_qubits: number of qubits :math:`n` of this channel.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob = torch.tensor(0.1)
            num_qubits = 1
            kraus_operators = generalized_depolarizing_kraus(prob, num_qubits)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[ 1.3601+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j,  1.3601+0.0000j]],

                    [[ 0.0000+0.0000j,  0.2236+0.0000j],
                     [ 0.2236+0.0000j,  0.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  0.0000-0.2236j],
                     [ 0.0000+0.2236j,  0.0000+0.0000j]],

                    [[ 0.2236+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j, -0.2236+0.0000j]]])
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    prob = prob.view([1])

    basis = pauli_group(num_qubits).to((prob + 0j).dtype)
    I, other_elements = basis[0], basis[1:]

    dim = 4 ** num_qubits
    mat = torch.stack([I * (torch.sqrt(1 - (dim - 1) * prob / dim) + 0j)] +
                      [ele * (torch.sqrt(prob / dim) + 0j) for ele in other_elements])

    return _type_transform(mat, type_str)


def pauli_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a pauli channel

    Args:
        prob: a list of three probabilities corresponding to X, Y, Z gate :math:`p`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob_list = torch.tensor([0.1, 0.2, 0.3])
            kraus_operators = pauli_kraus(prob_list)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[ 0.6325+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j,  0.6325+0.0000j]],

                    [[ 0.0000+0.0000j,  0.3162+0.0000j],
                     [ 0.3162+0.0000j,  0.0000+0.0000j]],

                    [[ 0.0000+0.0000j,  0.0000-0.4472j],
                     [ 0.0000+0.4472j,  0.0000+0.0000j]],

                    [[ 0.5477+0.0000j,  0.0000+0.0000j],
                     [ 0.0000+0.0000j, -0.5477+0.0000j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._pauli_kraus(prob)
    return _type_transform(mat, type_str)


def reset_kraus(prob: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a reset channel with form

    .. math::

        E_0 =
        \begin{bmatrix}
            \sqrt{p} & 0 \\
            0 & 0
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{p} \\
            0 & 0
        \end{bmatrix},\\
        E_2 =
        \begin{bmatrix}
            0 & 0 \\
            \sqrt{q} & 0
        \end{bmatrix},
        E_3 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{q}
        \end{bmatrix},\\
        E_4 = \sqrt{1-p-q} I.

    Args:
        prob: list of two probabilities of resetting to state :math:`|0\rangle` and :math:`|1\rangle`.

    Returns:
        a list of Kraus operators

    Examples:
        .. code-block:: python

            prob_list = torch.tensor([0.1, 0.2])
            kraus_operators = reset_kraus(prob_list)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[0.3162+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.0000+0.j]],

                    [[0.0000+0.j, 0.3162+0.j],
                     [0.0000+0.j, 0.0000+0.j]],

                    [[0.0000+0.j, 0.0000+0.j],
                     [0.4472+0.j, 0.0000+0.j]],

                    [[0.0000+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.4472+0.j]],

                    [[0.8367+0.j, 0.0000+0.j],
                     [0.0000+0.j, 0.8367+0.j]]], dtype=torch.complex128)
    """
    type_str, prob = _type_fetch(prob), _type_transform(prob, "tensor")
    mat = utils.representation._reset_kraus(prob)
    return _type_transform(mat, type_str)


def thermal_relaxation_kraus(const_t: _ParamLike, exec_time: _SingleParamLike) -> List[_ArrayLike]:
    r"""Kraus representation of a thermal relaxation channel

    Args:
        const_t: list of :math:`T_1` and :math:`T_2` relaxation time in microseconds.
        exec_time: quantum gate execution time in the process of relaxation in nanoseconds.

    Returns:
        a list of Kraus operators.

    Examples:
        .. code-block:: python

            const_t = torch.tensor([50, 30])
            exec_time = torch.tensor([100])
            kraus_operators = thermal_relaxation_kraus(const_t, exec_time)
            print(f'The Kraus operators are:\n{kraus_operators}')

        ::

            The Kraus operators are:
            tensor([[[ 0.9987+0.j,  0.0000+0.j],
                     [ 0.0000+0.j,  0.9987+0.j]],

                    [[ 0.0258+0.j,  0.0000+0.j],
                     [ 0.0000+0.j, -0.0258+0.j]],

                    [[ 0.0447+0.j,  0.0000+0.j],
                     [ 0.0000+0.j,  0.0000+0.j]],

                    [[ 0.0000+0.j,  0.0447+0.j],
                     [ 0.0000+0.j,  0.0000+0.j]]], dtype=torch.complex128)
    """
    type_str, const_t, exec_time = _type_fetch(const_t), _type_transform(const_t, "tensor"), _type_transform(exec_time, "tensor")
    mat = utils.representation._thermal_relaxation_kraus(const_t, exec_time)
    return _type_transform(mat, type_str)


def replacement_choi(sigma: _StateLike) -> _ArrayLike:
    r"""Choi representation of a replacement channel

    Args:
        sigma: output state of this channel.

    Returns:
        a Choi operator.

    Examples:
        .. code-block:: python

            sigma = torch.tensor([[0.8, 0.0], [0.0, 0.2]])
            choi_operator = replacement_choi(sigma)
            print(f'The Choi operator is :\n{choi_operator}')

        ::

            The Choi operator is :
            tensor([[0.8000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.2000+0.j, 0.0000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.8000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.2000+0.j]], dtype=torch.complex128)
    """
    if isinstance(sigma, State):
        sigma = sigma.density_matrix
    type_str, sigma = _type_fetch(sigma), _type_transform(sigma, "tensor")
    mat = utils.representation._replacement_choi(sigma)
    return _type_transform(mat, type_str)
