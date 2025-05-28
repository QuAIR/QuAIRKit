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
The source file of the classes for single-qubit gates.
"""

from typing import Iterable, Optional, Union

import torch

from ...core.utils.matrix import (_h, _p, _rx, _ry, _rz, _s, _sdg, _t, _tdg,
                                  _u3, _x, _y, _z)
from .base import Gate, ParamGate


class H(Gate):
    r"""A collection of single-qubit Hadamard gates.

    The matrix form of such a gate is:

    .. math::

        H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1&1\\
                1&-1
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _h(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None
    ):
        gate_info = {
            "name": "h",
            "tex": r'H',
            "api": "h", 
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return H.__matrix.to(self.device, dtype=self.dtype)


class S(Gate):
    r"""A collection of single-qubit S gates.

    The matrix form of such a gate is:

    .. math::

        S =
            \begin{bmatrix}
                1&0\\
                0&i
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _s(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "s",
            "tex": r'S',
            "api": "s",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return S.__matrix.to(self.device, dtype=self.dtype)


class Sdg(Gate):
    r"""A collection of single-qubit S dagger (S inverse) gates.

    The matrix form of such a gate is:

    .. math::

        S^\dagger =
            \begin{bmatrix}
                1&0\\
                0&-i
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _sdg(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "sdg",
            "tex": r'S^\dagger',
            "api": "sdg",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return Sdg.__matrix.to(self.device, dtype=self.dtype)


class T(Gate):
    r"""A collection of single-qubit T gates.

    The matrix form of such a gate is:

    .. math::

        T =
            \begin{bmatrix}
                1&0\\
                0&e^\frac{i\pi}{4}
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _t(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "t",
            "tex": r'T',
            "api": "t",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return T.__matrix.to(self.device, dtype=self.dtype)


class Tdg(Gate):
    r"""A collection of single-qubit T dagger (T inverse) gates.

    The matrix form of such a gate is:

    .. math::

        T^\dagger =
            \begin{bmatrix}
                1&0\\
                0&e^{-\frac{i\pi}{4}}
            \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _tdg(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "tdg",
            "tex": r'T^\dagger',
            "api": "tdg",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return Tdg.__matrix.to(self.device, dtype=self.dtype)


class X(Gate):
    r"""A collection of single-qubit X gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            0 & 1 \\
            1 & 0
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """

    __matrix = _x(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "x",
            "tex": r'X',
            "api": "x",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return X.__matrix.to(self.device, dtype=self.dtype)


class Y(Gate):
    r"""A collection of single-qubit Y gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            0 & -i \\
            i & 0
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _y(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "y",
            "tex": r'Y',
            "api": "y",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return Y.__matrix.to(self.device, dtype=self.dtype)


class Z(Gate):
    r"""A collection of single-qubit Z gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 \\
            0 & -1
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.

    """
    __matrix = _z(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "z",
            "tex": r'Z',
            "api": "z",
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, [2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        return Z.__matrix.to(self.device, dtype=self.dtype)


class P(ParamGate):
    r"""A collection of single-qubit P gates.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 \\
            0 & e^{i\theta}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "p",
            "tex": r'P',
            "api": "p",
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }
        super().__init__(
            _p, param, 1, param_sharing, qubits_idx, check_legality=False, gate_info=gate_info, support_batch=True)


class RX(ParamGate):
    r"""A collection of single-qubit rotation gates about the x-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
            -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "rx",
            "tex": r'R_{x}',
            "api": "rx",
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }
        super().__init__(
            _rx, param, 1, param_sharing, qubits_idx, check_legality=False, gate_info=gate_info, support_batch=True)


class RY(ParamGate):
    r"""A collection of single-qubit rotation gates about the y-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
            \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "ry",
            "tex": r'R_{y}',
            "api": "ry",
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }
        super().__init__(
            _ry, param, 1, param_sharing, qubits_idx, check_legality=False, gate_info=gate_info, support_batch=True)


class RZ(ParamGate):
    r"""A collection of single-qubit rotation gates about the z-axis.

    The matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            e^{-i\frac{\theta}{2}} & 0 \\
            0 & e^{i\frac{\theta}{2}}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "rz",
            "tex": r'R_{z}',
            "api": "rz",
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }
        super().__init__(
            _rz, param, 1, param_sharing, qubits_idx, check_legality=False, gate_info=gate_info, support_batch=True)


class U3(ParamGate):
    r"""A collection of single-qubit rotation gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            U3(\theta, \phi, \lambda) =
                \begin{bmatrix}
                    \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                    e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
            param: Optional[Union[torch.Tensor, Iterable[float]]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "u3",
            "tex": r'U',
            "api": "u3",
            "param_sharing": param_sharing,
            'plot_width': 1.65,
        }

        super().__init__(
            _u3, param, 3, param_sharing, qubits_idx, check_legality=False, gate_info=gate_info, support_batch=True)
