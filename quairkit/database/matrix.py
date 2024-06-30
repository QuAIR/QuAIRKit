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
Gate matrices.
"""

import itertools
import math
from functools import lru_cache, reduce
from typing import Callable, List, Optional

import numpy as np
import torch

from .. import database
from ..core import get_dtype
from ..core.intrinsic import _get_complex_dtype
from ..core.utils.linalg import _one, _unitary_transformation, _zero


def phase(dim: int) -> torch.Tensor:
    r"""Generate phase operator for qudit

    Args:
        dim: dimension of qudit

    Returns:
        Phase operator for qudit
    """
    w = np.exp(2 * np.pi * 1j / dim)
    return torch.from_numpy(np.diag([w ** i for i in range(dim)])).to(get_dtype())
 

def shift(dim: int) -> torch.Tensor:
    r"""Generate shift operator for qudit

    Args:
        dim: dimension of qudit

    Returns:
        Shift operator for qudit
    """
    return torch.roll(torch.eye(dim), 1, dims=0).to(get_dtype())


def grover_matrix(oracle: torch.Tensor) -> torch.Tensor:
    r"""Construct the Grover operator based on ``oracle``.

    Args:
        oracle: the input oracle :math:`A` to be rotated.

    Returns:
        Grover operator in form

    .. math::

        G = A (2 |0^n \rangle\langle 0^n| - I^n) A^\dagger \cdot (I - 2|1 \rangle\langle 1|) \otimes I^{n-1}

    """
    complex_dtype = oracle.dtype
    dimension = oracle.shape[0]
    ket_zero = torch.eye(dimension, 1).to(complex_dtype)

    diffusion_op = (2 + 0j) * ket_zero @ ket_zero.T - torch.eye(dimension).to(complex_dtype)
    reflection_op = torch.kron(torch.tensor([[1, 0], [0, -1]], dtype=complex_dtype), torch.eye(dimension // 2))

    return oracle @ diffusion_op @ oracle.conj().T @ reflection_op


def qft_matrix(num_qubits: int, dtype: torch.dtype=get_dtype()) -> torch.Tensor:
    r"""Construct the quantum fourier transpose (QFT) gate.

    Args:
        num_qubits: number of qubits :math:`n` st. :math:`N = 2^n`.
        dtype: the data type you used, default type is torch.complex64

    Returns:
        a gate in below matrix form, here :math:`\omega_N = \text{exp}(\frac{2 \pi i}{N})`

    .. math::

        \begin{align}
            QFT = \frac{1}{\sqrt{N}}
            \begin{bmatrix}
                1 & 1 & .. & 1 \\
                1 & \omega_N & .. & \omega_N^{N-1} \\
                .. & .. & .. & .. \\
                1 & \omega_N^{N-1} & .. & \omega_N^{(N-1)^2}
            \end{bmatrix}
        \end{align}

    """
    N = 2 ** num_qubits
    omega_N = np.exp(1j * 2 * math.pi / N)

    qft_mat = np.ones([N, N]).astype('complex128')
    for i in range(1, N):
        for j in range(1, N):
            qft_mat[i, j] = omega_N ** ((i * j) % N)

    return torch.tensor(qft_mat / math.sqrt(N)).to(dtype)


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are single-qubit matrices.
"""


def h(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1&1\\
                1&-1
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of H gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    element = math.sqrt(2) / 2
    gate_matrix = [
        [element, element],
        [element, -element],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def s(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        S =
            \begin{bmatrix}
                1&0\\
                0&i
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of S gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def sdg(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        S^\dagger =
            \begin{bmatrix}
                1&0\\
                0&-i
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of Sdg gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, -1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def t(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        T = \begin{bmatrix}
                1&0\\
                0&e^\frac{i\pi}{4}
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of T gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 + math.sqrt(2) / 2 * 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def tdg(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        T^\dagger =
            \begin{bmatrix}
                1&0\\
                0&e^{-\frac{i\pi}{4}}
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of Sdg gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 - math.sqrt(2) / 2 * 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def eye(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        I = \begin{bmatrix}
                1 & 0 \\
                0 & 1
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of X gate.

    """
    return torch.eye(2, dtype=get_dtype() if dtype is None else dtype)


def x(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        X = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of X gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [0, 1],
        [1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def y(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        Y = \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of Y gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [0, -1j],
        [1j, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def z(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        Z = \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of Z gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0],
        [0, -1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def p(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        P(\theta) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of P gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        _one(dtype), _zero(dtype),
        _zero(dtype), torch.cos(theta) + 1j * torch.sin(theta),
    ]
    return torch.cat(gate_matrix).view([2, 2]).to(dtype)


def rx(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        R_X(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of R_X gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        torch.cos(theta / 2).reshape([1]), -1j * torch.sin(theta / 2).reshape([1]),
        -1j * torch.sin(theta / 2).reshape([1]), torch.cos(theta / 2).reshape([1]),
    ]
    return torch.cat(gate_matrix).view([2, 2]).to(dtype)


def ry(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        R_Y(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of R_Y gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        torch.cos(theta / 2), (-torch.sin(theta / 2)),
        torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix).view([2, 2]).to(dtype)


def rz(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        R_Z(\theta) = \begin{bmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0 & e^{i\frac{\theta}{2}}
        \end{bmatrix}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of R_Z gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        torch.exp(-1j * theta / 2), _zero(dtype),
        _zero(dtype), torch.exp(1j * theta / 2)
    ]
    return torch.cat(gate_matrix).view([2, 2]).to(dtype)


def u3(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            U_3(\theta, \phi, \lambda) =
                \begin{bmatrix}
                    \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                    e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [3, 1]

    Returns:
        the matrix of U_3 gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([3, 1])
    theta, phi, lam = theta[0], theta[1], theta[2]
    gate_matrix = [
        torch.cos(theta / 2),
        -torch.exp(1j * lam) * torch.sin(theta / 2),
        torch.exp(1j * phi) * torch.sin(theta / 2),
        torch.exp(1j * (phi + lam)) * torch.cos(theta / 2)
    ]
    return torch.cat(gate_matrix).view([2, 2]).to(dtype)


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are multi-qubit matrices.
"""


def cnot(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CNOT} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of CNOT gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def cy(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CY} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -i \\
                    0 & 0 & i & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of CY gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def cz(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CZ} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & -1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of CZ gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def swap(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{SWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of SWAP gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def cp(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CP}(\theta) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of CP gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _one(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), torch.cos(theta) + 1j * torch.sin(theta),
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def crx(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_X} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                    0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of CR_X gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), torch.cos(theta / 2), -1j * torch.sin(theta / 2),
        _zero(dtype), _zero(dtype), -1j * torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def cry(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_Y} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                    0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of CR_Y gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    gate_matrix = [
        _one(dtype), _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _one(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), torch.cos(theta / 2), (-torch.sin(theta / 2)),
        _zero(dtype), _zero(dtype), torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def crz(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CR_Z} &= |0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\theta}{2}}
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of CR_Z gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    one, zero = _one(dtype).to(device=theta.device), _zero(dtype).to(device=theta.device)
    
    gate_matrix = [
        one, zero, zero, zero,
        zero, one, zero, zero,
        zero, zero, torch.cos(theta / 2) - 1j * torch.sin(theta / 2), zero,
        zero, zero, zero,torch.cos(theta / 2) + 1j * torch.sin(theta / 2),
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def cu(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CU}
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{i\gamma}\cos\frac\theta2 &-e^{i(\lambda+\gamma)}\sin\frac\theta2 \\
                    0 & 0 & e^{i(\phi+\gamma)}\sin\frac\theta2&e^{i(\phi+\lambda+\gamma)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [3,1]

    Returns:
        the matrix of CU gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([4, 1])
    one, zero = _one(dtype).to(device=theta.device), _zero(dtype).to(device=theta.device)
    
    param1 = (torch.cos(theta[3]) + 1j * torch.sin(theta[3])) * \
             (torch.cos(theta[0] / 2))
    param2 = (torch.cos(theta[2] + theta[3]) + 1j * torch.sin(theta[2] + theta[3])) * \
             (-torch.sin(theta[0] / 2))
    param3 = (torch.cos(theta[1] + theta[3]) + 1j * torch.sin(theta[1] + theta[3])) * \
        torch.sin(theta[0] / 2)
    param4 = (torch.cos(theta[1] + theta[2] + theta[3]) + 1j * \
        torch.sin(theta[1] + theta[2] + theta[3])) * torch.cos(theta[0] / 2)
    gate_matrix = [
        one, zero, zero, zero,
        zero, one, zero, zero,
        zero, zero, param1, param2,
        zero, zero, param3, param4,
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def rxx(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{XX}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of RXX gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    param1 = torch.cos(theta / 2)
    param2 = -1j * torch.sin(theta / 2)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), param2,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param2, _zero(dtype), _zero(dtype), param1,
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def ryy(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{YY}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of RYY gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    param1 = torch.cos(theta / 2)
    param2 = -1j * torch.sin(theta / 2)
    param3 = 1j * torch.sin(theta / 2)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), param3,
        _zero(dtype), param1, param2, _zero(dtype),
        _zero(dtype), param2, param1, _zero(dtype),
        param3, _zero(dtype), _zero(dtype), param1,
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def rzz(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{R_{ZZ}}(\theta) =
                    \begin{bmatrix}
                        e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                        0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                        0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                        0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                    \end{bmatrix}
        \end{align}

    Args:
        theta: the parameter of this matrix. The shape of param is [1]

    Returns:
        the matrix of RZZ gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([1])
    param1 = torch.cos(theta / 2) - 1j * torch.sin(theta / 2)
    param2 = torch.cos(theta / 2) + 1j * torch.sin(theta / 2)
    gate_matrix = [
        param1, _zero(dtype), _zero(dtype), _zero(dtype),
        _zero(dtype), param2, _zero(dtype), _zero(dtype),
        _zero(dtype), _zero(dtype), param2, _zero(dtype),
        _zero(dtype), _zero(dtype), _zero(dtype), param1,
    ]
    return torch.cat(gate_matrix).view([4, 4]).to(dtype)


def ms(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{MS} = \mathit{R_{XX}}(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                    \begin{bmatrix}
                        1 & 0 & 0 & i \\
                        0 & 1 & i & 0 \\
                        0 & i & 1 & 0 \\
                        i & 0 & 0 & 1
                    \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of MS gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    val1 = math.sqrt(2) / 2
    val2 = 1j / math.sqrt(2)
    gate_matrix = [
        [val1, 0, 0, val2],
        [0, val1, val2, 0],
        [0, val2, val1, 0],
        [val2, 0, 0, val1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def cswap(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CSWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of CSWAP gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def toffoli(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Generate the matrix

    .. math::

        \begin{align}
            \mathit{CSWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{bmatrix}
        \end{align}

    Args:
        dtype: the dtype of this matrix. Defaults to ``None``.

    Returns:
        the matrix of Toffoli gate.

    """
    dtype = get_dtype() if dtype is None else dtype
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def universal2(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    Args:
        theta: the parameter of this matrix. The shape of param is [15]

    Returns:
        the matrix of universal two qubits gate.

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([15])
    unitary = torch.eye(2 ** 2).to(dtype)
    _cnot_gate = cnot(dtype)

    unitary = _unitary_transformation(unitary, u3(theta[[0, 1, 2]]), qubit_idx=0, num_qubits=2)
    unitary = _unitary_transformation(unitary, u3(theta[[3, 4, 5]]), qubit_idx=1, num_qubits=2)
    unitary = _unitary_transformation(unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2)

    unitary = _unitary_transformation(unitary, rz(theta[[6]]), qubit_idx=0, num_qubits=2)
    unitary = _unitary_transformation(unitary, ry(theta[[7]]), qubit_idx=1, num_qubits=2)
    unitary = _unitary_transformation(unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=2)

    unitary = _unitary_transformation(unitary, ry(theta[[8]]), qubit_idx=1, num_qubits=2)
    unitary = _unitary_transformation(unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2)

    unitary = _unitary_transformation(unitary, u3(theta[[9, 10, 11]]), qubit_idx=0, num_qubits=2)
    unitary = _unitary_transformation(unitary, u3(theta[[12, 13, 14]]), qubit_idx=1, num_qubits=2)

    return unitary


def universal3(theta: torch.Tensor) -> torch.Tensor:
    r"""Generate the matrix

    Args:
        theta: the parameter of this matrix. The shape of param is [81]

    Returns:
        the matrix of universal three qubits gate. 

    """
    dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view([81])
    unitary = torch.eye(2 ** 3).to(dtype)
    _h, _s, _cnot = h(dtype), s(dtype), cnot(dtype)

    psi = torch.reshape(theta[:60], shape=[4, 15])
    phi = torch.reshape(theta[60:], shape=[7, 3])

    def __block_u(_unitary, _theta):
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, ry(_theta[0]), qubit_idx=1, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 1], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, ry(_theta[1]), qubit_idx=1, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 1], num_qubits=3)

        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _h, qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 0], num_qubits=3)

        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, rz(_theta[2]), qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 2], num_qubits=3)
        return _unitary

    def __block_v(_unitary, _theta):
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[2, 0], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[2, 1], num_qubits=3)

        _unitary = _unitary_transformation(_unitary, ry(_theta[0]), qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, ry(_theta[1]), qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 2], num_qubits=3)

        _unitary = _unitary_transformation(_unitary, _s, qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[2, 0], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 1], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[1, 0], num_qubits=3)

        _unitary = _unitary_transformation(_unitary, _h, qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 2], num_qubits=3)
        _unitary = _unitary_transformation(_unitary, rz(_theta[2]), qubit_idx=2, num_qubits=3)
        _unitary = _unitary_transformation(_unitary, _cnot, qubit_idx=[0, 2], num_qubits=3)
        return _unitary

    unitary = _unitary_transformation(unitary, universal2(psi[0]), qubit_idx=[0, 1], num_qubits=3)
    unitary = _unitary_transformation(unitary, u3(phi[0, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_u(unitary, phi[1])

    unitary = _unitary_transformation(unitary, universal2(psi[1]), qubit_idx=[0, 1], num_qubits=3)
    unitary = _unitary_transformation(unitary, u3(phi[2, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_v(unitary, phi[3])

    unitary = _unitary_transformation(unitary, universal2(psi[2]), qubit_idx=[0, 1], num_qubits=3)
    unitary = _unitary_transformation(unitary, u3(phi[4, 0:3]), qubit_idx=2, num_qubits=3)
    unitary = __block_u(unitary, phi[5])

    unitary = _unitary_transformation(unitary, universal2(psi[3]), qubit_idx=[0, 1], num_qubits=3)
    unitary = _unitary_transformation(unitary, u3(phi[6, 0:3]), qubit_idx=2, num_qubits=3)
    return unitary
    

def universal_qudit(theta: torch.Tensor, dimension: int) -> torch.Tensor:
    r"""Generalized GellMann matrix basis were used to construct the universal gate for qudits

    Args:
        theta: the parameter of this matrix. The shape of param is [dimension**2 - 1]
        dimension: the dimension of the qudit

    Returns:
        the matrix of d-dimensional unitary gate 

    References:
        [wolfram mathworld](https://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html)
    """
    complex_dtype = _get_complex_dtype(theta.dtype)
    theta = theta.view ([(dimension ** 2) - 1, 1, 1])

    generalized_gellmann_matrix = database.set.gell_mann(dimension).to(complex_dtype)
    hamiltonian = torch.eye(dimension) + torch.sum(torch.mul(theta, generalized_gellmann_matrix), dim=0)
    return torch.matrix_exp(1j * hamiltonian)


# ------------------------------------------------- Split line -------------------------------------------------
def Uf(f:Callable[[torch.Tensor], torch.Tensor], n:int) -> torch.Tensor:
    r"""Construct the unitary matrix maps :math:`|x\rangle|y\rangle` to :math:`|x\rangle|y\oplus f(x)\rangle` based on a boolean function :math:`f`.

    Args:
        f: a boolean function :math:`f` that maps :math:`\{ 0,1 \}^{n}` to :math:`\{ 0,1 \}`;
        n: the length of input :math:`\{ 0,1 \}^{n}`;
        dtype: the data type you used. Defaults to the type of current device.

    Returns:
        Unitary matrix in form

    .. math::

        U: U|x\rangle|y\rangle = |x\rangle|y\oplus f(x)\rangle

    """
    # note that for a boolean function 'f', 'x = torch.zeros(n)' is a legitimate input, but 'x = torch.zeros((n,1))' is an illegitimate input

    dtype = get_dtype() # get the default dtype

    U = torch.zeros((2**(n+1), 2**(n+1)), dtype=dtype) # initialize the unitary matrix

    for i in range(2**n):
        temp_bin_i_str = bin(i)[2:].zfill(n) # binary form of 'i' (type: string)
        temp_array = torch.zeros(n,dtype=dtype)
        for j in range(n):
            temp_array[j] = complex(temp_bin_i_str[j]) # convert the type of `i`(binary form) into torch.tensor

        f_value = f(temp_array) # compute f(i)
        f_value_int = int(f_value) # ensure that the type of 'f(i)' is int
        U[(i*2+f_value_int), (i*2)] = 1 # assignment (y=0)
        U[(i*2+((f_value_int+1)%2)), ((i*2)+1)] = 1 # assignment (y=1)

    return U


def Of(f:Callable[[torch.Tensor], torch.Tensor], n:int) -> torch.Tensor:
    r"""Construct the unitary matrix maps :math:`|x\rangle` to :math:`(-1)^{f(x)}|x\rangle` based on a boolean function :math:`f`.

    Args:
        f: a boolean function :math:`f` that maps :math:`\{ 0,1 \}^{n}` to :math:`\{ 0,1 \}`;
        n: the length of input :math:`\{ 0,1 \}^{n}`;
        dtype: the data type you used. Defaults to the type of current device.

    Returns:
        Unitary matrix in form

    .. math::

        U: U|x\rangle = (-1)^{f(x)}|x\rangle

    """
    # note that for a boolean function 'f', 'x = torch.zeros(n)' is a legitimate input, but 'x = torch.zeros((n,1))' is an illegitimate input

    dtype = get_dtype() # get the default dtype

    U = torch.zeros((2**n, 2**n), dtype=dtype) # initialize the unitary matrix

    for i in range(2**n):
        temp_bin_i_str = bin(i)[2:].zfill(n) # # binary form of 'i' (type: string)
        temp_array = torch.zeros(n, dtype=dtype)
        for j in range(n):
            temp_array[j] = complex(temp_bin_i_str[j]) # # convert the type of `i`(binary form) into torch.tensor

        f_value = f(temp_array) # compute f(i)
        f_value_int = int(f_value) # ensure that the type of 'f(i)' is int
        U[i,i] = (-1)**(f_value_int) # assignment

    return U


for name, func in list(globals().items()): 
    if callable(func) and '_' not in name:
        # Add '_gate' to the name for functions without an underscore
        new_name = f'{name}_gate'
        globals()[new_name] = func
