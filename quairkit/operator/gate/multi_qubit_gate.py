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
The source file of the classes for multi-qubit gates.
"""

from typing import Iterable, List, Optional, Tuple, Union

import matplotlib
import torch

from ...core import StateSimulator, utils
from ...core.utils.matrix import (_cnot, _cp, _crx, _cry, _crz, _cswap, _cu,
                                  _cy, _cz, _ms, _rxx, _ryy, _rzz, _swap,
                                  _toffoli, _universal2, _universal3)
from .base import Gate, ParamGate
from .visual import (_cnot_display, _crx_like_display, _cswap_display,
                     _cx_like_display, _oracle_like_display, _rxx_like_display,
                     _swap_display, _tofolli_display)


class CNOT(Gate):
    r"""A collection of CNOT gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
    
    """

    __matrix = _cnot(torch.complex128)

    def __init__(
        self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "cx",
            "tex": r'\targ{}',
            "api": "cnot",
            "num_ctrl_system": 1,
            "label": '1',
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True
    
    @property
    def matrix(self) -> torch.Tensor:
        return CNOT.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cnot_display(self, ax, x, )
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        if state._keep_dim:
            return super().forward(state)
        
        state = state.clone()
        swap_indices = utils.linalg._get_swap_indices(2, 3, self.system_idx, state.system_dim, self.device)
        state._index_select(swap_indices)
        return state

CX = CNOT


class CY(Gate):
    r"""A collection of controlled Y gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CY &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -1j \\
                0 & 0 & 1j & 0
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
    
    """

    __matrix = _cy(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "cy",
            "tex": r'Y',
            "api": "cy",
            "num_ctrl_system": 1,
            "label": '1',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True
    
    @property
    def matrix(self) -> torch.Tensor:
        return CY.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cx_like_display(self, ax, x, )


class CZ(Gate):
    r"""A collection of controlled Z gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CZ &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
    
    """

    __matrix = _cz(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "cz",
            "tex": r'Z',
            "api": "cz",
            "num_ctrl_system": 1,
            "label": '1',
            'plot_width': 0.4,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True
    
    @property
    def matrix(self) -> torch.Tensor:
        return CZ.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cx_like_display(self, ax, x)


class SWAP(Gate):
    r"""A collection of SWAP gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            SWAP =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
    
    """

    __matrix = _swap(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None,
    ):
        gate_info = {
            "name": "swap",
            "api": "swap",
            "permute": [1, 0],
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True
        
    @property
    def matrix(self) -> torch.Tensor:
        return SWAP.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _swap_display(self, ax, x, )


class CP(ParamGate):
    r"""A collection of controlled P gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{bmatrix}
            1 & 0 & 0 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & e^{i\theta}
        \end{bmatrix}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "cp",
            "tex": r'P',
            "api": "cp",
            "num_ctrl_system": 1,
            "label": '1',
            'plot_width': 0.9,
        }

        super().__init__(
            _cp, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CRX(ParamGate):
    r"""A collection of controlled rotation gates about the x-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRx &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rx\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "crx",
            "tex": r'R_{x}',
            "api": "crx",
            "num_ctrl_system": 1,
            "label": '1',
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }

        super().__init__(
            _crx, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CRY(ParamGate):
    r"""A collection of controlled rotation gates about the y-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRy &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Ry\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "cry",
            "tex": r'R_{y}',
            "api": "cry",
            "num_ctrl_system": 1,
            "label": '1',
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }

        super().__init__(
            _cry, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x, )


class CRZ(ParamGate):
    r"""A collection of controlled rotation gates about the z-axis.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CRz &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rz\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                0 & 0 & 0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "crz",
            "tex": r'R_{z}',
            "api": "crz",
            "num_ctrl_system": 1,
            "label": '1',
            "param_sharing": param_sharing,
            'plot_width': 0.9,
        }

        super().__init__(
            _crz, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class CU(ParamGate):
    r"""A collection of controlled single-qubit rotation gates.

    For a 2-qubit quantum circuit, when ``qubits_idx`` is ``[0, 1]``, the matrix form of such a gate is:

    .. math::

        \begin{align}
            CU
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{i\gamma}\cos\frac\theta2 &-e^{i(\lambda+\gamma)}\sin\frac\theta2 \\
                0 & 0 & e^{i(\phi+\gamma)}\sin\frac\theta2&e^{i(\phi+\lambda+\gamma)}\cos\frac\theta2
            \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "cu4",
            "tex": r'U',
            "api": "cu",
            "num_ctrl_system": 1,
            "label": '1',
            "param_sharing": param_sharing,
            'plot_width': 1.65,
        }
        super().__init__(
            _cu, param, 4, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _crx_like_display(self, ax, x)


class RXX(ParamGate):
    r"""A collection of RXX gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RXX(\theta) =
                \begin{bmatrix}
                    \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                    0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                    0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                    -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "rxx",
            "tex": r'R_{xx}',
            "api": "rxx",
            "param_sharing": param_sharing,
            'plot_width': 1.0,
        }
        super().__init__(
            _rxx, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class RYY(ParamGate):
    r"""A collection of RYY gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RYY(\theta) =
                \begin{bmatrix}
                    \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                    0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                    0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                    i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "ryy",
            "tex": r'R_{yy}',
            "api": "ryy",
            "param_sharing": param_sharing,
            'plot_width': 1.0,
        }

        super().__init__(
            _ryy, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class RZZ(ParamGate):
    r"""A collection of RZZ gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            RZZ(\theta) =
                \begin{bmatrix}
                    e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                    0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                    0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.

    Raises:
        ValueError: The ``param`` must be ``torch.Tensor`` or ``float``.
    
    """

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, str]] = None,
            param: Optional[Union[torch.Tensor, float]] = None, param_sharing: Optional[bool] = False
    ):
        gate_info = {
            "name": "rzz",
            "tex": r'R_{zz}',
            "api": "rzz",
            "param_sharing": param_sharing,
            'plot_width': 1.0,
        }
        super().__init__(
            _rzz, param, 1, param_sharing, qubits_idx, [2, 2], check_legality=False, gate_info=gate_info, support_batch=True)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _rxx_like_display(self, ax, x)


class MS(Gate):
    r"""A collection of Mølmer-Sørensen (MS) gates for trapped ion devices.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            MS = RXX(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1 & 0 & 0 & i \\
                    0 & 1 & i & 0 \\
                    0 & i & 1 & 0 \\
                    i & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first two qubits.
    
    """

    __matrix = _ms(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None
    ):
        gate_info = {
            "name": "ms",
            "tex": r'\text{MS}',
            "api": "ms",
            'plot_width': 0.6,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2], check_legality=False, gate_info=gate_info)

    @property
    def matrix(self) -> torch.Tensor:
        mat = MS.__matrix.to(self.device, dtype=self.dtype)
        return utils.linalg._dagger(mat) if self._is_dagger else mat

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)


class CSWAP(Gate):
    r"""A collection of CSWAP (Fredkin) gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
            CSWAP =
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
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first three qubits.
    
    """
    __matrix = _cswap(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None
    ):
        gate_info = {
            "name": "cswap",
            "api": "cswap",
            "num_ctrl_system": 1,
            "label": '1',
            "permute": [1, 0],
            'plot_width': 0.2,
        }
        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True

    @property
    def matrix(self) -> torch.Tensor:
        return CSWAP.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _cswap_display(self, ax, x)


class CCX(Gate):
    r"""A collection of CCX (Toffoli) gates.

    The matrix form of such a gate is:

    .. math::

        \begin{align}
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
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to the first three qubits.
    
    """
    __matrix = _toffoli(torch.complex128)

    def __init__(
            self, qubits_idx: Optional[Union[Iterable, int, str]] = None
    ):
        gate_info = {
            "name": "ccx",
            "tex": r'\targ{}',
            "api": "ccx",
            "num_ctrl_system": 2,
            "label": '11',
            'plot_width': 0.2,
        }

        super().__init__(
            None, qubits_idx, acted_system_dim=[2, 2, 2], check_legality=False, gate_info=gate_info)
        self._is_hermitian = True

    @property
    def matrix(self) -> torch.Tensor:
        return CCX.__matrix.to(self.device, dtype=self.dtype)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _tofolli_display(self, ax, x, )


Toffoli = CCX
