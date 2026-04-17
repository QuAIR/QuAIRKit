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
from ..core.intrinsic import (
    _ArrayLike,
    _ParamLike,
    _SingleParamLike,
    _StateLike,
    _ensure_mat_3d,
    _ensure_param_2d,
    _type_fetch,
    _type_transform,
    _unflatten_batch,
)
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


def _merge_batch_shape(a: List[int], b: List[int]) -> List[int]:
    if a == b:
        return a
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    raise ValueError(f"Incompatible batch shapes: {a} vs {b}")


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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=1)
    mat_b = utils.representation._bit_flip_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=1)
    mat_b = utils.representation._phase_flip_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=1)
    mat_b = utils.representation._bit_phase_flip_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(gamma)
    gamma = _type_transform(gamma, "tensor")
    gamma2, batch_shape = _ensure_param_2d(gamma, n=1)
    mat_b = utils.representation._amplitude_damping_kraus(gamma2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(gamma)
    gamma = _type_transform(gamma, "tensor")
    prob = _type_transform(prob, "tensor")
    gamma2, gamma_bs = _ensure_param_2d(gamma, n=1)
    prob2, prob_bs = _ensure_param_2d(prob, n=1)
    B = max(gamma2.shape[0], prob2.shape[0])
    if gamma2.shape[0] not in (1, B) or prob2.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: gamma {gamma2.shape[0]} vs prob {prob2.shape[0]}")
    if gamma2.shape[0] == 1 and B > 1:
        gamma2 = gamma2.expand(B, -1)
    if prob2.shape[0] == 1 and B > 1:
        prob2 = prob2.expand(B, -1)
    batch_shape = _merge_batch_shape(gamma_bs, prob_bs)
    mat_b = utils.representation._generalized_amplitude_damping_kraus(gamma2, prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(gamma)
    gamma = _type_transform(gamma, "tensor")
    gamma2, batch_shape = _ensure_param_2d(gamma, n=1)
    mat_b = utils.representation._phase_damping_kraus(gamma2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=1)
    mat_b = utils.representation._depolarizing_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=1)
    prob2 = prob2.to(dtype=getattr(prob2, "dtype", prob2.dtype))

    basis = pauli_group(num_qubits).to((prob2 + 0j).dtype)
    I = basis[0].unsqueeze(0).unsqueeze(0)
    other = basis[1:].unsqueeze(0)

    dim = 4 ** num_qubits
    coeff0 = torch.sqrt(1 - (dim - 1) * prob2 / dim).to((prob2 + 0j).dtype)
    coeff1 = torch.sqrt(prob2 / dim).to((prob2 + 0j).dtype)
    e0 = I * coeff0.view([-1, 1, 1, 1])
    e_rest = other * coeff1.view([-1, 1, 1, 1])
    mat_b = torch.cat([e0, e_rest], dim=1)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=3)
    mat_b = utils.representation._pauli_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(prob)
    prob = _type_transform(prob, "tensor")
    prob2, batch_shape = _ensure_param_2d(prob, n=2)
    mat_b = utils.representation._reset_kraus(prob2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(const_t)
    const_t = _type_transform(const_t, "tensor")
    exec_time = _type_transform(exec_time, "tensor")

    const2, const_bs = _ensure_param_2d(const_t, n=2)
    exec2, exec_bs = _ensure_param_2d(exec_time, n=1)

    B = max(const2.shape[0], exec2.shape[0])
    if const2.shape[0] not in (1, B) or exec2.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: const_t {const2.shape[0]} vs exec_time {exec2.shape[0]}")
    if const2.shape[0] == 1 and B > 1:
        const2 = const2.expand(B, -1)
    if exec2.shape[0] == 1 and B > 1:
        exec2 = exec2.expand(B, -1)

    batch_shape = _merge_batch_shape(const_bs, exec_bs)
    mat_b = utils.representation._thermal_relaxation_kraus(const2, exec2)
    if mat_b.ndim == 3:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
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
    type_str = _type_fetch(sigma)
    sigma = _type_transform(sigma, "tensor")
    sigma3, batch_shape = _ensure_mat_3d(sigma)
    mat_b = utils.representation._replacement_choi(sigma3)
    if mat_b.ndim == 2:
        mat_b = mat_b.unsqueeze(0)
    mat = _unflatten_batch(mat_b, batch_shape)
    return _type_transform(mat, type_str)
