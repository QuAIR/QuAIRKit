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
from typing import Callable, List, Optional, Union

import torch

from .. import database
from ..core import get_dtype, utils
from ..core.intrinsic import (_ArrayLike, _ParamLike, _SingleParamLike,
                              _type_fetch, _type_transform)

__all__ = [
    "phase",
    "shift",
    "grover_matrix",
    "qft_matrix",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "eye",
    "x",
    "y",
    "z",
    "p",
    "rx",
    "ry",
    "rz",
    "u3",
    "cnot",
    "cy",
    "cz",
    "swap",
    "cp",
    "crx",
    "cry",
    "crz",
    "cu",
    "rxx",
    "ryy",
    "rzz",
    "ms",
    "cswap",
    "toffoli",
    "ccx",
    "universal2",
    "universal3",
    "universal_qudit",
    "Uf",
    "Of",
    "permutation_matrix"
]


def phase(dim: int) -> torch.Tensor:
    r"""Generate phase operator for qudit

    Args:
        dim: dimension of qudit

    Returns:
        Phase operator for qudit

    Examples:
        .. code-block:: python

            dim = 2
            phase_operator = phase(dim)
            print(f'The phase_operator is:\n{phase_operator}')

        ::

            The phase_operator is:
            tensor([[ 1.+0.0000e+00j,  0.+0.0000e+00j],
                    [ 0.+0.0000e+00j, -1.+1.2246e-16j]])
    """
    return utils.matrix._phase(dim, get_dtype())


def shift(dim: int) -> torch.Tensor:
    r"""Generate shift operator for qudit

    Args:
        dim: dimension of qudit

    Returns:
        Shift operator for qudit

    Examples:
        .. code-block:: python

            dim = 2
            shift_operator = shift(dim)
            print(f'The shift_operator is:\n{shift_operator}')

        ::

            The shift_operator is:
            tensor([[0.+0.j, 1.+0.j],
                    [1.+0.j, 0.+0.j]])
    """
    return utils.matrix._shift(dim, get_dtype())


def grover_matrix(oracle: _ArrayLike, dtype: Optional[torch.dtype] = None) -> _ArrayLike:
    r"""Construct the Grover operator based on ``oracle``.

    Args:
        oracle: the input oracle :math:`A` to be rotated.

    Returns:
        Grover operator in form

        .. math::

            G = A (2 |0^n \rangle\langle 0^n| - I^n) A^\dagger \cdot (I - 2|1 \rangle\langle 1|) \otimes I^{n-1}

    Examples:
        .. code-block:: python

            oracle = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
            grover_op = grover_matrix(oracle)
            print(f'The grover_matrix is:\n{grover_op}')

        ::

            The grover_matrix is:
            tensor([[-1.+0.j,  0.+0.j],
                    [ 0.+0.j, -1.+0.j]])
    """
    oracle = _type_transform(oracle, "tensor")
    return utils.matrix._grover(oracle)


def qft_matrix(num_qubits: int) -> torch.Tensor:
    r"""Construct the quantum Fourier transpose (QFT) gate.

    Args:
        num_qubits: number of qubits :math:`n` st. :math:`N = 2^n`.

    Returns:
        A matrix in the form

        .. math::

            QFT = \frac{1}{\sqrt{N}}
            \begin{bmatrix}
                1 & 1 & \cdots & 1 \\
                1 & \omega_N & \cdots & \omega_N^{N-1} \\
                \vdots & \vdots & \ddots & \vdots \\
                1 & \omega_N^{N-1} & \cdots & \omega_N^{(N-1)^2}
            \end{bmatrix},
            \quad \omega_N = \exp\left(\frac{2\pi i}{N}\right).

    Examples:
        .. code-block:: python

            num_qubits = 1
            qft_gate = qft_matrix(num_qubits)
            print(f'The QFT gate is:\n{qft_gate}')

        ::

            The QFT gate is:
            tensor([[ 0.7071+0.0000e+00j,  0.7071+0.0000e+00j],
                    [ 0.7071+0.0000e+00j, -0.7071+8.6596e-17j]])
    """
    return utils.matrix._qft(num_qubits, get_dtype())


def h() -> torch.Tensor:
    r"""Generate the matrix of the Hadamard gate.

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1 & 1 \\
                1 & -1
            \end{bmatrix}

    Returns:
        The matrix of the H gate.

    Examples:
        .. code-block:: python

            H = h()
            print(f'The Hadamard gate is:\n{H}')

        ::

            The Hadamard gate is:
            tensor([[ 0.7071+0.j,  0.7071+0.j],
                    [ 0.7071+0.j, -0.7071+0.j]])
    """
    return utils.matrix._h(get_dtype())


def s() -> torch.Tensor:
    r"""Generate the matrix of the S gate.

    .. math::

        S =
            \begin{bmatrix}
                1 & 0 \\
                0 & i
            \end{bmatrix}

    Returns:
        The matrix of the S gate.

    Examples:
        .. code-block:: python

            S = s()
            print(f'The S gate is:\n{S}')

        ::

            The S gate is:
            tensor([[1.+0.j, 0.+0.j],
                    [0.+0.j, 0.+1.j]])
    """
    return utils.matrix._s(get_dtype())


def sdg() -> torch.Tensor:
    r"""Generate the matrix of the Sdg (S-dagger) gate.

    .. math::

        S^\dagger =
            \begin{bmatrix}
                1 & 0 \\
                0 & -i
            \end{bmatrix}

    Returns:
        The matrix of the Sdg gate.

    Examples:
        .. code-block:: python

            Sdg = sdg()
            print(f'The dagger of S gate is:\n{Sdg}')

        ::

            The dagger of S gate is:
            tensor([[1.+0.j,  0.+0.j],
                    [0.+0.j, -0.-1.j]])
    """
    return utils.matrix._sdg(get_dtype())


def t() -> torch.Tensor:
    r"""Generate the matrix of the T gate.

    .. math::

        T =
            \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\pi/4}
            \end{bmatrix}

    Returns:
        The matrix of the T gate.

    Examples:
        .. code-block:: python

            T = t()
            print(f'The T gate is:\n{T}')

        ::

            The T gate is:
            tensor([[1.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.7071+0.7071j]])
    """
    return utils.matrix._t(get_dtype())


def tdg() -> torch.Tensor:
    r"""Generate the matrix of the Tdg (T-dagger) gate.

    .. math::

        T^\dagger =
            \begin{bmatrix}
                1 & 0 \\
                0 & e^{-i\pi/4}
            \end{bmatrix}

    Returns:
        The matrix of the Tdg gate.

    Examples:
        .. code-block:: python

            Tdg = tdg()
            print(f'The dagger of T gate is:\n{Tdg}')

        ::

            The dagger of T gate is:
            tensor([[1.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.7071-0.7071j]])
    """
    return utils.matrix._tdg(get_dtype())


def eye(dim: int = 2) -> torch.Tensor:
    r"""Generate the identity matrix.

    .. math::

        I =
            \begin{bmatrix}
                1 & 0 \\
                0 & 1
            \end{bmatrix}

    Args:
        dim: the dimension of the identity matrix (default is 2 for a qubit).

    Returns:
        The identity matrix.

    Examples:
        .. code-block:: python

            I = eye()
            print(f'The Identity Matrix is:\n{I}')

        ::

            The Identity Matrix is:
            tensor([[1.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j]])
    """
    return utils.matrix._eye(dim, get_dtype())


def x() -> torch.Tensor:
    r"""Generate the Pauli X matrix.

    .. math::

        X =
            \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

    Returns:
        The matrix of the X gate.

    Examples:
        .. code-block:: python

            X = x()
            print(f'The Pauli X Matrix is:\n{X}')

        ::

            The Pauli X Matrix is:
            tensor([[0.+0.j, 1.+0.j],
                    [1.+0.j, 0.+0.j]])
    """
    return utils.matrix._x(get_dtype())


def y() -> torch.Tensor:
    r"""Generate the Pauli Y matrix.

    .. math::

        Y =
            \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

    Returns:
        The matrix of the Y gate.

    Examples:
        .. code-block:: python

            Y = y()
            print(f'The Pauli Y Matrix is:\n{Y}')

        ::

            The Pauli Y Matrix is:
            tensor([[0.+0.j, -0.-1.j],
                    [0.+1.j,  0.+0.j]])
    """
    return utils.matrix._y(get_dtype())


def z() -> torch.Tensor:
    r"""Generate the Pauli Z matrix.

    .. math::

        Z =
            \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

    Returns:
        The matrix of the Z gate.

    Examples:
        .. code-block:: python

            Z = z()
            print(f'The Pauli Z Matrix is:\n{Z}')

        ::

            The Pauli Z Matrix is:
            tensor([[ 1.+0.j,  0.+0.j],
                    [ 0.+0.j, -1.+0.j]])
    """
    return utils.matrix._z(get_dtype())


def p(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the P gate matrix.

    .. math::

        P(\theta) =
            \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the P gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            p_matrix = p(theta)
            print(f'The P Gate is:\n{p_matrix}')

        ::

            The P Gate is:
            tensor([[1.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.7071+0.7071j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._p(theta).view(batch_dim + [2, 2])
    return _type_transform(mat, type_str)


def rx(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RX gate matrix.

    .. math::

        R_X(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RX gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_x = rx(theta)
            print(f'The R_x Gate is:\n{R_x}')

        ::

            The R_x Gate is:
            tensor([[0.9239+0.0000j, 0.0000-0.3827j],
                    [0.0000-0.3827j, 0.9239+0.0000j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._rx(theta).view(batch_dim + [2, 2])
    return _type_transform(mat, type_str)


def ry(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RY gate matrix.

    .. math::

        R_Y(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RY gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_y = ry(theta)
            print(f'The R_y Gate is:\n{R_y}')

        ::

            The R_y Gate is:
            tensor([[ 0.9239+0.j, -0.3827+0.j],
                    [ 0.3827+0.j,  0.9239+0.j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._ry(theta).view(batch_dim + [2, 2])
    return _type_transform(mat, type_str)


def rz(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RZ gate matrix.

    .. math::

        R_Z(\theta) =
            \begin{bmatrix}
                e^{-i\theta/2} & 0 \\
                0 & e^{i\theta/2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RZ gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_z = rz(theta)
            print(f'The R_z Gate is:\n{R_z}')

        ::

            The R_z Gate is:
            tensor([[0.9239-0.3827j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.9239+0.3827j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._rz(theta).view(batch_dim + [2, 2])
    return _type_transform(mat, type_str)


def u3(theta: _ParamLike) -> _ArrayLike:
    r"""Generate the U3 gate matrix.

    .. math::

        U_3(\theta, \phi, \lambda) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\
                e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameters, with shape [3, 1].

    Returns:
        The matrix of the U3 gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([[torch.pi / 4], [torch.pi / 3], [torch.pi / 6]])
            u3_matrix = u3(theta)
            print(f'The U3 Gate is:\n{u3_matrix}')

        ::

            The U3 Gate is:
            tensor([[ 9.2388e-01+0.0000j, -3.3141e-01-0.1913j],
                    [ 1.9134e-01+0.3314j, -4.0384e-08+0.9239j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._u3(theta).view(batch_dim + [2, 2])
    return _type_transform(mat, type_str)


def cnot() -> torch.Tensor:
    r"""Generate the CNOT gate matrix.

    .. math::

        CNOT = |0\rangle \langle 0| \otimes I + |1\rangle \langle 1| \otimes X =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
            \end{bmatrix}

    Returns:
        The matrix of the CNOT gate.

    Examples:
        .. code-block:: python

            CNOT = cnot()
            print(f'The CNOT Gate is:\n{CNOT}')

        ::

            The CNOT Gate is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    """
    return utils.matrix._cnot(get_dtype())


def cy() -> torch.Tensor:
    r"""Generate the CY gate matrix.

    .. math::

        CY = |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes Y =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
            \end{bmatrix}

    Returns:
        The matrix of the CY gate.

    Examples:
        .. code-block:: python

            CY = cy()
            print(f'The CY Gate is:\n{CY}')

        ::

            The CY Gate is:
            tensor([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                    [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                    [0.+0.j,  0.+0.j,  0.+0.j, -0.-1.j],
                    [0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j]])
    """
    return utils.matrix._cy(get_dtype())


def cz() -> torch.Tensor:
    r"""Generate the CZ gate matrix.

    .. math::

        CZ = |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes Z =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{bmatrix}

    Returns:
        The matrix of the CZ gate.

    Examples:
        .. code-block:: python

            CZ = cz()
            print(f'The CZ Gate is:\n{CZ}')

        ::

            The CZ Gate is:
            tensor([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
                    [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                    [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                    [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])
    """
    return utils.matrix._cz(get_dtype())


def swap() -> torch.Tensor:
    r"""Generate the SWAP gate matrix.

    .. math::

        SWAP =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}

    Returns:
        The matrix of the SWAP gate.

    Examples:
        .. code-block:: python

            SWAP = swap()
            print(f'The SWAP Gate is:\n{SWAP}')

        ::

            The SWAP Gate is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
    """
    return utils.matrix._swap(get_dtype())


def cp(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the CP gate matrix.

    .. math::

        CP(\theta) =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\theta}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the CP gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            CP = cp(theta)
            print(f'The CP Gate is:\n{CP}')

        ::

            The CP Gate is:
            tensor([[1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000+0.0000j, 1.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.7071+0.7071j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._cp(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def crx(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the CR_X gate matrix.

    .. math::

        CR_X =
            |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes R_X =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the CR_X gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            CR_X = crx(theta)
            print(f'The CR_X Gate is:\n{CR_X}')

        ::

            The CR_X Gate is:
            tensor([[1.0000+0.0000j, 0.+0.0000j, 0.+0.0000j, 0.+0.0000j],
                    [0.+0.0000j, 1.0000+0.0000j, 0.+0.0000j, 0.+0.0000j],
                    [0.+0.0000j, 0.+0.0000j, 0.9239+0.0000j, 0.+-0.3827j],
                    [0.+0.0000j, 0.+0.0000j, 0.+-0.3827j, 0.9239+0.0000j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._crx(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def cry(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the CR_Y gate matrix.

    .. math::

        CR_Y =
            |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes R_Y =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the CR_Y gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            CR_Y = cry(theta)
            print(f'The CR_Y Gate is:\n{CR_Y}')

        ::

            The CR_Y Gate is:
            tensor([[1.0000+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.0000+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.9239+0.j, -0.3827+0.j],
                    [0.+0.j, 0.+0.j, 0.3827+0.j, 0.9239+0.j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._cry(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def crz(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the CR_Z gate matrix.

    .. math::

        CR_Z =
            |0\rangle\langle0| \otimes I + |1\rangle\langle1| \otimes R_Z =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\theta/2} & 0 \\
                0 & 0 & 0 & e^{i\theta/2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the CR_Z gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            CR_Z = crz(theta)
            print(f'The CR_Z Gate is:\n{CR_Z}')

        ::

            The CR_Z Gate is:
            tensor([[1.0000+0.0000j, 0.+0.0000j, 0.+0.0000j, 0.+0.0000j],
                    [0.+0.0000j, 1.0000+0.0000j, 0.+0.0000j, 0.+0.0000j],
                    [0.+0.0000j, 0.+0.0000j, 0.9239-0.3827j, 0.+0.0000j],
                    [0.+0.0000j, 0.+0.0000j, 0.+0.0000j, 0.9239+0.3827j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._crz(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def cu(theta: _ParamLike) -> _ArrayLike:
    r"""Generate the CU gate matrix.

    .. math::

        CU =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{i\gamma}\cos\frac{\theta}{2} & -e^{i(\lambda+\gamma)}\sin\frac{\theta}{2} \\
                0 & 0 & e^{i(\phi+\gamma)}\sin\frac{\theta}{2} & e^{i(\phi+\lambda+\gamma)}\cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameters, with shape [4, 1].

    Returns:
        The matrix of the CU gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([[torch.pi / 4], [torch.pi / 3], [torch.pi / 6], [torch.pi / 6]])
            CU = cu(theta)
            print(f'The CU Gate is:\n{CU}')

        ::

            The CU Gate is:
            tensor([[1.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j],
                    [0.0000e+00+0.0000j, 1.0000e+00+0.0000j, 0.0000e+00+0.0000j, 0.0000e+00+0.0000j],
                    [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, 8.0010e-01+0.4619j, -1.9134e-01-0.3314j],
                    [0.0000e+00+0.0000j, 0.0000e+00+0.0000j, -1.6728e-08+0.3827j, -4.6194e-01+0.8001j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._cu(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def rxx(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RXX gate matrix.

    .. math::

        R_{XX}(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RXX gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_XX = rxx(theta)
            print(f'The R_XX Gate is:\n{R_XX}')

        ::

            The R_XX Gate is:
            tensor([[0.9239+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.3827j],
                    [0.0000+0.0000j, 0.9239+0.0000j, 0.0000-0.3827j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000-0.3827j, 0.9239+0.0000j, 0.0000+0.0000j],
                    [0.0000-0.3827j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9239+0.0000j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._rxx(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def ryy(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RYY gate matrix.

    .. math::

        R_{YY}(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RYY gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_YY = ryy(theta)
            print(f'The R_YY Gate is:\n{R_YY}')

        ::

            The R_YY Gate is:
            tensor([[0.9239+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.3827j],
                    [0.0000+0.0000j, 0.9239+0.0000j, 0.0000-0.3827j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000-0.3827j, 0.9239+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.3827j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9239+0.0000j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._ryy(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def rzz(theta: _SingleParamLike) -> _ArrayLike:
    r"""Generate the RZZ gate matrix.

    .. math::

        R_{ZZ}(\theta) =
            \begin{bmatrix}
                e^{-i\theta/2} & 0 & 0 & 0 \\
                0 & e^{i\theta/2} & 0 & 0 \\
                0 & 0 & e^{i\theta/2} & 0 \\
                0 & 0 & 0 & e^{-i\theta/2}
            \end{bmatrix}

    Args:
        theta: the (batched) parameter, with shape [1].

    Returns:
        The matrix of the RZZ gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([torch.pi / 4])
            R_ZZ = rzz(theta)
            print(f'The R_ZZ Gate is:\n{R_ZZ}')

        ::

            The R_ZZ Gate is:
            tensor([[0.9239-0.3827j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.9239+0.3827j, 0.0000+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000+0.0000j, 0.9239+0.3827j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9239-0.3827j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._rzz(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def ms() -> torch.Tensor:
    r"""Generate the MS gate matrix.

    .. math::

        MS = R_{XX}\left(-\frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1 & 0 & 0 & i \\
                0 & 1 & i & 0 \\
                0 & i & 1 & 0 \\
                i & 0 & 0 & 1
            \end{bmatrix}

    Returns:
        The matrix of the MS gate.

    Examples:
        .. code-block:: python

            MS = ms()
            print(f'The MS Gate is:\n{MS}')

        ::

            The MS Gate is:
            tensor([[0.7071+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.7071j],
                    [0.0000+0.0000j, 0.7071+0.0000j, 0.0000+0.7071j, 0.0000+0.0000j],
                    [0.0000+0.0000j, 0.0000+0.7071j, 0.7071+0.0000j, 0.0000+0.0000j],
                    [0.0000+0.7071j, 0.0000+0.0000j, 0.0000+0.0000j, 0.7071+0.0000j]])
    """
    return utils.matrix._ms(get_dtype())


def cswap() -> torch.Tensor:
    r"""Generate the CSWAP gate matrix.

    .. math::

        CSWAP =
            \begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
            \end{bmatrix}

    Returns:
        The matrix of the CSWAP gate.

    Examples:
        .. code-block:: python

            CSWAP = cswap()
            print(f'The CSWAP Gate is:\n{CSWAP}')

        ::

            The CSWAP Gate is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
    """
    return utils.matrix._cswap(get_dtype())


def toffoli() -> torch.Tensor:
    r"""Generate the Toffoli gate matrix.

    .. math::

        Toffoli =
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

    Returns:
        The matrix of the Toffoli gate.

    Examples:
        .. code-block:: python

            Toffoli = toffoli()
            print(f'The Toffoli Gate is:\n{Toffoli}')

        ::

            The Toffoli Gate is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    """
    return utils.matrix._toffoli(get_dtype())


ccx = toffoli  # Alias for Toffoli gate, commonly used in quantum computing libraries


def universal2(theta: _ParamLike) -> _ArrayLike:
    r"""Generate the universal two-qubit gate matrix.

    Args:
        theta: the (batched) parameter with shape [15].

    Returns:
        The matrix of the universal two-qubit gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([
                0.5, 1.0, 1.5, 2.0, 2.5,
                3.0, 3.5, 4.0, 4.5, 5.0,
                5.5, 6.0, 6.5, 7.0, 7.5])
            Universal2 = universal2(theta)
            print(f'The matrix of universal two qubits gate is:\n{Universal2}')

        ::

            The matrix of universal two qubits gate is:
            tensor([[-0.2858-0.0270j,  0.4003+0.3090j, -0.6062+0.0791j,  0.5359+0.0323j],
                    [-0.0894-0.1008j, -0.5804+0.0194j,  0.3156+0.1677j,  0.7090-0.1194j],
                    [-0.8151-0.2697j,  0.2345-0.1841j,  0.3835-0.1154j, -0.0720+0.0918j],
                    [-0.2431+0.3212j, -0.1714+0.5374j,  0.1140+0.5703j, -0.2703+0.3289j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._universal2(theta).view(batch_dim + [4, 4])
    return _type_transform(mat, type_str)


def universal3(theta: _ParamLike) -> _ArrayLike:
    r"""Generate the universal three-qubit gate matrix.

    Args:
        theta: the (batched) parameter with shape [81].

    Returns:
        The matrix of the universal three-qubit gate.

    Examples:
        .. code-block:: python

            theta = torch.tensor([
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
                5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
                6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
                7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
                8.1])
            Universal3 = universal3(theta)
            print(f'The matrix of universal three qubits gate is:\n{Universal3}')

        ::

            The matrix of universal three qubits gate is:
            tensor([[-0.0675-0.0941j, -0.4602+0.0332j,  0.2635+0.0044j,  0.0825+0.3465j,
                     -0.0874-0.3635j, -0.1177-0.4195j, -0.2735+0.3619j, -0.1760-0.1052j],
                    [ 0.0486+0.0651j, -0.1123+0.0494j,  0.1903+0.0057j, -0.2080+0.2926j,
                     -0.2099+0.0630j, -0.1406+0.5173j, -0.1431-0.3538j, -0.5460-0.1847j],
                    [ 0.0827-0.0303j,  0.1155+0.1111j,  0.5391-0.0701j, -0.4229-0.2655j,
                     -0.1546+0.1943j, -0.0455+0.1744j, -0.3242+0.3539j,  0.3118-0.0041j],
                    [-0.1222+0.3984j,  0.1647-0.1817j,  0.3294-0.1486j, -0.0293-0.1503j,
                      0.0100-0.6481j,  0.2424+0.1575j,  0.2485+0.0232j, -0.1053+0.1873j],
                    [-0.4309-0.0791j, -0.2071-0.0482j, -0.4331+0.0866j, -0.5454-0.1778j,
                     -0.1401-0.0230j,  0.0170+0.0299j,  0.0078+0.2231j, -0.2324+0.3369j],
                    [ 0.0330+0.3056j,  0.2612+0.6464j, -0.2138-0.1748j, -0.2322-0.0598j,
                      0.1387-0.1573j,  0.0914-0.2963j, -0.2712-0.1351j, -0.1272-0.1940j],
                    [ 0.0449-0.3844j,  0.1135+0.2846j, -0.0251+0.3854j,  0.0442-0.0149j,
                     -0.3671-0.1774j,  0.5158+0.1148j,  0.2151+0.1433j, -0.0188-0.3040j],
                    [-0.4124-0.4385j,  0.2306+0.0894j,  0.0104-0.2180j, -0.0180+0.2869j,
                     -0.1030-0.2991j, -0.1473+0.0931j, -0.1686-0.3451j,  0.3825+0.1480j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim = list(theta.shape[:-1])
    mat = utils.matrix._universal3(theta).view(batch_dim + [8, 8])
    return _type_transform(mat, type_str)


def universal_qudit(theta: _ParamLike, dimension: int) -> _ArrayLike:
    r"""Generate a universal gate matrix for qudits using a generalized Gell-Mann basis.

    Args:
        theta: the (batched) parameter with shape [dimension**2 - 1].
        dimension: the dimension of the qudit.

    Returns:
        The matrix of the d-dimensional unitary gate.

    Examples:
        .. code-block:: python

            dimension = 2
            theta = torch.linspace(0.1, 2.5, dimension**2 - 1)
            u_qudit_matrix = universal_qudit(theta, dimension)
            print(f'The matrix of 2-dimensional unitary gate is:\n{u_qudit_matrix}')

        ::

            The matrix of 2-dimensional unitary gate is:
            tensor([[-0.9486+0.2806j,  0.1459+0.0112j],
                    [-0.1459+0.0112j, -0.9486-0.2806j]])
    """
    type_str, theta = _type_fetch(theta), _type_transform(theta, "tensor")
    batch_dim, generator = list(theta.shape[:-1]), database.set.gell_mann(dimension)
    mat = utils.matrix._param_generator(theta, generator).view(batch_dim + [dimension, dimension])
    return _type_transform(mat, type_str)


def Uf(f: Callable[[int], int], n: int) -> torch.Tensor:
    r"""Construct the unitary matrix used in Simon's algorithm.

    Args:
        f: a 2-to-1 or 1-to-1 function :math:`f: \{0, 1\}^n \to \{0, 1\}^n`.
        n: length of the bit string.

    Returns:
        A 2n-qubit unitary matrix satisfying

        .. math::

            U|x, y\rangle = |x, y \oplus f(x)\rangle

    Examples:
        .. code-block:: python

            def f(x: int) -> int:
                return x

            unitary_matrix = Uf(f, 1)
            print(f'Unitary matrix is:\n{unitary_matrix}')

        ::

            Unitary matrix is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])
    """
    d = 2 ** n
    U = torch.zeros(d ** 2, d ** 2)
    for x, y in itertools.product(range(d), range(d)):
        y_out = y ^ f(x)
        input_index = x * d + y
        output_index = x * d + y_out
        U[output_index, input_index] = 1
    return U.to(get_dtype())


def Of(f: Callable[[int], int], n: int) -> torch.Tensor:
    r"""Construct the oracle unitary matrix for an unstructured search problem.

    Args:
        f: a function :math:`f: \{0, 1\}^n \to \{0, 1\}`.
        n: length of the bit string.

    Returns:
        An n-qubit unitary matrix satisfying

        .. math::

            U|x\rangle = (-1)^{f(x)}|x\rangle

    Examples:
        .. code-block:: python

            def f(x: int) -> int:
                return x

            unitary_matrix = Of(f, 1)
            print(f'Unitary matrix is:\n{unitary_matrix}')

        ::

            Unitary matrix is:
            tensor([[ 1.+0.j,  0.+0.j],
                    [ 0.+0.j, -1.+0.j]])
    """
    D = torch.tensor([(-1) ** f(x) for x in range(2 ** n)], dtype=get_dtype())
    return torch.diag(D)


def permutation_matrix(perm: List[int], system_dim: Union[int, List[int]] = 2):
    r"""Construct a unitary matrix representing a permutation operation on a quantum system.

    Args:
        perm: A list representing the permutation of subsystems.
              For example, [1, 0, 2] swaps the first two subsystems.
        system_dim: The dimension of each subsystem.
                    - If an integer, all subsystems are assumed to have the same dimension.
                    - If a list, it specifies the dimension of each subsystem individually.

    Returns:
        A unitary matrix representing the permutation in the Hilbert space.

    Examples:
        .. code-block:: python

            perm = [1, 0]
            U_perm = permutation_matrix(perm, system_dim=2)
            print(f'The permutation matrix is:\n{U_perm}')

        ::

            The permutation matrix is:
            tensor([[0.+0.j, 1.+0.j],
                    [1.+0.j, 0.+0.j]])
    """
    system_dim = [system_dim] * len(perm) if isinstance(system_dim, int) else system_dim
    
    assert len(perm) == len(system_dim), \
        f"perm and system_dim must have the same length, but got perm: {len(perm)} and system_dim: {len(system_dim)}"

    return utils.matrix._permutation(perm, system_dim).to(get_dtype())


for name, func in list(locals().items()):
    if callable(func) and "_" not in name:
        # Add '_gate' to the name for functions without an underscore
        new_name = f"{name}_gate"
        globals()[new_name] = func
        __all__ += [new_name]
