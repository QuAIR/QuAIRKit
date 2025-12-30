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
The source file of the Circuit class.
"""

import math
import shutil
import warnings
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from .ansatz import (AmplitudeEncoding, AngleEncoding, BasisEncoding,
                     ComplexBlockLayer, ComplexEntangledLayer, IQPEncoding,
                     Layer, LinearEntangledLayer, OperatorList, RealBlockLayer,
                     RealEntangledLayer, TrotterLayer, Universal2, Universal3)
from .core import (Hamiltonian, OperatorInfoType, StateSimulator, intrinsic,
                   latex, qasm2_to_info, to_state, utils)
from .core.intrinsic import _alias
from .database.state import zero_state
from .operator import (CCX, CNOT, CP, CRX, CRY, CRZ, CSWAP, CU, CY, CZ, MS, RX,
                       RXX, RY, RYY, RZ, RZZ, SWAP, U3, AmplitudeDamping,
                       BitFlip, BitPhaseFlip, ChoiRepr, Collapse,
                       ControlOracle, ControlParamOracle, Depolarizing, Gate,
                       GeneralizedAmplitudeDamping, GeneralizedDepolarizing, H,
                       KrausRepr, OneWayLOCC, Oracle, P, ParamOracle,
                       PauliChannel, Permutation, PhaseDamping, PhaseFlip,
                       QuasiOperation, ResetChannel, ResetState, S, Sdg,
                       StinespringRepr, T, Tdg, ThermalRelaxation,
                       UniversalQudits, X, Y, Z)
from .operator.gate import _circuit_plot

__all__ = ['Circuit']


class Circuit(OperatorList):
    r"""Class for quantum circuit.

    Args:
        num_systems: number of systems in the circuit. Defaults to None. Alias of ``num_qubits``.
        system_dim: dimension of systems of this circuit. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        physical_idx: physical indices of systems. Defaults to be the same as the logical indices.
    
    Note:
        when the number of system is unknown and system_dim is an int, the circuit is a dynamic quantum circuit.

    Examples:
        .. code-block:: python

            qc = Circuit(1)  # A quantum circuit with 1 qubit
            qc.h()
            print(f'The latex code of this circuit is:\n{qc.to_latex()}')

        ::

            The latex code of this circuit is:
            \lstick{} & \gate[1]{H}
    """
    @_alias({'num_systems': 'num_qubits'})
    def __init__(self, num_systems: Optional[int] = None, 
                 system_dim: Optional[Union[List[int], int]] = 2,
                 physical_idx: Optional[List[int]] = None) -> None:
        super().__init__(num_systems, system_dim, physical_idx)

        # alias
        self.toffoli = self.ccx
        self.cx = self.cnot
        self.collapse = self.measure

    def _get_drawer(self, style: str, decimal: int) -> Tuple[latex.OperatorListDrawer, Dict[int, str]]:
        r"""Return the drawer that draws the circuit in LaTeX format, and the beginning string of the circuit.

        Examples:
            .. code-block:: python

                drawer, begin_code = qc._Circuit__get_drawer('standard', 2)

            ::

                drawer: (an instance of latex.OperatorListDrawer)
                begin_code: {0: r'\lstick{}', 1: r'\lstick{}', ...}
        """
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        
        drawer = latex.OperatorListDrawer(style, decimal)
        
        for op in self.children():
            if isinstance(op, Layer):
                drawer = drawer + op._get_drawer(style, decimal)
            else:
                drawer.append(op.info)
        
        begin_code = {idx: r'\lstick{}' for idx in self.system_idx}
        self.system_idx = physical_idx
        return drawer, begin_code
        
    def to_latex(self, style: str = 'standard', decimal: int = 2) -> str:
        r"""The LaTeX representation of the circuit, written in Quantikz format.
        
        Args:
            style: the style of the plot, can be 'standard', 'compact' or 'detailed'. Defaults to ``standard``.
            decimal: number of decimal places to display. Defaults to 2.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.h()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{H} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{H} & {} & {}
        """
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        drawer, begin_code = self._get_drawer(style, decimal)
        drawer.fill_all()
        self.system_idx = physical_idx
        return latex.code_to_str(drawer.code, begin_code)
    
    def plot(self, style: str = 'standard', decimal: int = 2,
             dpi: int = 300, print_code: bool = False, 
             show_plot: bool = True, include_empty: bool = False,
             latex: bool = True, **kwargs) -> Optional[matplotlib.figure.Figure]:
        r'''Display the circuit using Quantikz if ``latex`` is True, otherwise using matplotlib.

        Args:
            style: the style of the plot, can be 'standard', 'compact' or 'detailed'. Defaults to ``standard``.
            decimal: number of decimal places to display. Defaults to 2.
            dpi: dots per inches of plot image. Defaults to 300.
            print_code: whether print the LaTeX code of the circuit, default to ``False``.
            show_plot: whether show the plotted circuit, default to ``True``.
            include_empty: whether include empty lines, default to ``False``.
            latex: whether use Quantikz, a LaTeX package, to plot circuits , default to ``True``.
            kwargs: additional parameters for matplotlib plot.
        
        Returns:
            None, or a ``matplotlib.figure.Figure`` instance depending on ``latex`` and ``output``.

        Notes:
            If ``latex`` is True, the circuit will be displayed in LaTeX format;
            if ``latex`` is False, the circuit will be displayed in matplotlib format,
            in which case we have three additional parameters:
            - output_plot: whether output the plot instance, default to ``False``.
            - save_path: the save path of image. Defaults to None.
            - scale: scale coefficient of figure. Default to 1.0.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.h()
                qc.plot()

            ::

                (Displays the plotted circuit using Quantikz or matplotlib)
                
        .. warning::
        
            Starting from QuAIRKit 2.5.0, `Circuit.plot` now defaults to using QuantiKz, 
            a LaTeX package powered by TikZ, for rendering quantum circuit diagrams 
            commonly used in academic publications. Support for plotting circuits with 
            Matplotlib will be deprecated in the near future.

            To fully utilize this feature, please ensure that a TeX distribution, such 
            as [TeX Live](https://www.tug.org/texlive), is installed on your system. 
            This will enhance your experience with QuAIRKit and quantum computing 
            visualization.
        '''
        if latex and (shutil.which("pdflatex") is None):
            warnings.warn(
                "pdflatex is mot detected on your system. Will skip the plot.", UserWarning)
            
            if print_code:
                print(self.to_latex(style, decimal))
            return

        if latex:
            drawer, begin_code = self._get_drawer(style, decimal)   
            drawer._fill_empty(list(range(self.num_systems if include_empty else 
                                          (max(drawer._code.keys()) + 1))))
            drawer.add_end()
            _fig = drawer.plot(dpi, print_code, begin_code)
            
            if show_plot:
                intrinsic._display_png(_fig)
        else:
            warnings.warn(
                "Starting from QuAIRKit 0.4.0, `Circuit.plot` now defaults to using Quantikz, a LaTeX package "
                "powered by TikZ, for rendering quantum circuit diagrams commonly used in academic publications. "
                "Support for plotting circuits with Matplotlib will be deprecated in the near future. \n"
                "To fully utilize this feature, please ensure that a TeX distribution, such as TeX Live "
                "(https://www.tug.org/texlive), is installed on your system. This will enhance your experience "
                "with QuAIRKit and quantum computing visualization.", FutureWarning)
            save_path, scale = kwargs.get('save_path', None), kwargs.get('scale', 1.0)
            output_plot = kwargs.get('output_plot', False)
            
            physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
            _fig = _circuit_plot(self, dpi=dpi, scale=scale)
            self.system_idx = physical_idx
            
            if save_path:
                plt.savefig(save_path, dpi=dpi)
            if show_plot:
                plt.show()
            if output_plot:
                return _fig

    @property
    def qasm(self) -> Union[str, List[str]]:
        r"""String representation of the circuit in qasm-like format.

        Returns:
            string representation of the operator list
        """
        qreg = f"qreg q[{self.num_systems}]; // Register dimension {self.system_dim}"
        
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        qasm_str = self.get_qasm(transpile=False)
        self.system_idx = physical_idx

        if isinstance(qasm_str, str):
            return '\n'.join([qreg, qasm_str])
        return ['\n'.join([qreg, item]) for item in qasm_str]

    @property
    def qasm2(self) -> Union[str, List[str]]:
        r"""Transpile the circuit in OpenQASM 2.0 format.

        Returns:
            Transpilation of the circuit
        """
        
        if any(dim != 2 for dim in self.system_dim):
            raise ValueError("Only qubit systems are supported in OpenQASM 2.0")
        
        header = 'OPENQASM 2.0;\ninclude "qelib1.inc";'
        qreg = f"qreg q[{self.num_systems}];"
        
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        qasm_str = self.get_qasm(transpile=True)
        self.system_idx = physical_idx
        if isinstance(qasm_str, str):
            return '\n'.join([header, qreg, qasm_str])
        return ['\n'.join([header, qreg, item]) for item in qasm_str]
    
    def unitary_matrix(self) -> torch.Tensor:
        r"""Get the unitary matrix form of the circuit.

        Returns:
            Unitary matrix form of the circuit.
        
        """
        warnings.warn(
            "Starting from QuAIRKit 0.4.0, it is recommended to use 'Circuit.matrix' instead of "
            "'Circuit.unitary_matrix()' to call the unitary matrix of the circuit.", FutureWarning)
        return self.matrix

    # ---------------------- below are common operators ---------------------- #
    def h(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit Hadamard gates.

        The matrix form of such a gate is:

        .. math::
            H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1 &  1 \\
                    1 & -1
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.h()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{H} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{H} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(H(qubits_idx))

    def s(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit S gates.

        The matrix form of such a gate is:

        .. math::
            S =
                \begin{bmatrix}
                    1 & 0 \\
                    0 & i
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.s()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{S} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{S} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(S(qubits_idx))

    def sdg(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit S dagger (S inverse) gates.

        The matrix form of such a gate is:

        .. math::
            S^\dagger =
                \begin{bmatrix}
                    1 &  0 \\
                    0 & -i
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.sdg()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{S^\dagger} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{S^\dagger} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(Sdg(qubits_idx))

    def t(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit T gates.

        The matrix form of such a gate is:

        .. math::
            T = \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\pi/4}
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.t()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{T} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{T} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(T(qubits_idx))

    def tdg(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit T dagger (T inverse) gates.

        The matrix form of such a gate is:

        .. math::
            T^\dagger = \begin{bmatrix}
                            1 & 0 \\
                            0 & e^{-i\pi/4}
                         \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.tdg()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{T^\dagger} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{T^\dagger} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(Tdg(qubits_idx))

    def x(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit X gates.

        The matrix form of such a gate is:

        .. math::
           X = \begin{bmatrix}
                    0 & 1 \\
                    1 & 0
               \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.x()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{X} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{X} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(X(qubits_idx))

    def y(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit Y gates.

        The matrix form of such a gate is:

        .. math::
            Y = \begin{bmatrix}
                    0 & -i \\
                    i &  0
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.y()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{Y} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{Y} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(Y(qubits_idx))

    def z(self, qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add single-qubit Z gates.

        The matrix form of such a gate is:

        .. math::
            Z = \begin{bmatrix}
                    1 &  0 \\
                    0 & -1
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.z()
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{Z} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{Z} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(Z(qubits_idx))

    def p(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
          param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add single-qubit P gates.

        The matrix form of such a gate is:

        .. math::
            P(\theta) = \begin{bmatrix}
                            1 & 0 \\
                            0 & e^{i\theta}
                        \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.p([0], torch.pi/2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{P(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(P(qubits_idx, param, param_sharing))

    def rx(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
           param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add single-qubit rotation gates about the x-axis.

        The matrix form of such a gate is:

        .. math::
            R_X(\theta) = \begin{bmatrix}
                                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                           \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.rx([0], torch.pi/2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{R_{x}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(RX(qubits_idx, param, param_sharing))

    def ry(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
           param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add single-qubit rotation gates about the y-axis.

        The matrix form of such a gate is:

        .. math::
            R_Y(\theta) = \begin{bmatrix}
                                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                           \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.ry([0], torch.pi/2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{R_{y}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(RY(qubits_idx, param, param_sharing))

    def rz(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
           param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add single-qubit rotation gates about the z-axis.

        The matrix form of such a gate is:

        .. math::
            R_Z(\theta) = \begin{bmatrix}
                                e^{-i\theta/2} & 0 \\
                                0 & e^{i\theta/2}
                           \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.rz([0], torch.pi/2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{R_{z}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(RZ(qubits_idx, param, param_sharing))

    def u3(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
           param: Union[torch.Tensor, Iterable[float]] = None, param_sharing: bool = False) -> None:
        r"""Add single-qubit rotation gates.

        The matrix form of such a gate is:

        .. math::
            U_3(\theta, \phi, \lambda) =
                \begin{bmatrix}
                    \cos\frac\theta2 & -e^{i\lambda}\sin\frac\theta2 \\
                    e^{i\phi}\sin\frac\theta2 & e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.u3([0], torch.pi/2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{U(1.57, 1.57, 1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(U3(qubits_idx, param, param_sharing))

    def cnot(self, qubits_idx: Union[Iterable[int], str] = 'cycle') -> None:
        r"""Add CNOT gates.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CNOT} = |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes X

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cnot([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \targ{} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CNOT(qubits_idx))

    def cy(self, qubits_idx: Union[Iterable[int], str] = 'cycle') -> None:
        r"""Add controlled Y gates.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CY} = |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes Y

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cy([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{Y} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CY(qubits_idx))

    def cz(self, qubits_idx: Union[Iterable[int], str] = 'linear') -> None:
        r"""Add controlled Z gates.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CZ} = |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes Z

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cz([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{Z} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CZ(qubits_idx))

    def swap(self, qubits_idx: Union[Iterable[int], str] = 'linear') -> None:
        r"""Add SWAP gates.

        The matrix form is:

        .. math::
            \mathit{SWAP} =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.swap([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2,style={draw=none}]{\permute{2,1}} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(SWAP(qubits_idx))

    def cp(self, qubits_idx: Union[Iterable[int], str] = 'cycle',
           param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add controlled P gates.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CP}(\theta) =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\theta}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cp([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{P(1.57)} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CP(qubits_idx, param, param_sharing))

    def crx(self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add controlled rotation gates about the x-axis.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CR_X} =
            |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes R_X

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.crx([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{R_{x}(1.57)} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CRX(qubits_idx, param, param_sharing))

    def cry(self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add controlled rotation gates about the y-axis.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CR_Y} =
            |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes R_Y

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cry([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{R_{y}(1.57)} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CRY(qubits_idx, param, param_sharing))

    def crz(self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add controlled rotation gates about the z-axis.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is:

        .. math::
            \mathit{CR_Z} =
            |0\rangle \langle 0|\otimes I + |1\rangle \langle 1|\otimes R_Z

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.crz([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{R_{z}(1.57)} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CRZ(qubits_idx, param, param_sharing))

    def cu(self, qubits_idx: Union[Iterable[int], str] = 'cycle',
           param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add controlled single-qubit rotation gates.

        For a 2-qubit circuit, when `qubits_idx` is `[0, 1]`, the matrix form is given by a controlled-U gate.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gate. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.cu([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{U(1.57, ...)} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(CU(qubits_idx, param, param_sharing))

    def rxx(self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add RXX gates.

        The matrix form is:

        .. math::
            R_{XX}(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gate. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.rxx([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{R_{xx}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(RXX(qubits_idx, param, param_sharing))

    def ryy(self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add RYY gates.

        The matrix form is:

        .. math::
            R_{YY}(\theta) =
            \begin{bmatrix}
                \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gate. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.ryy([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{R_{yy}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(RYY(qubits_idx, param, param_sharing))

    def rzz(self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add RZZ gates.

        The matrix form is:

        .. math::
            R_{ZZ}(\theta) =
            \begin{bmatrix}
                e^{-i\theta/2} & 0 & 0 & 0 \\
                0 & e^{i\theta/2} & 0 & 0 \\
                0 & 0 & e^{i\theta/2} & 0 \\
                0 & 0 & 0 & e^{-i\theta/2}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gate. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.rzz([0, 1], torch.pi / 2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{R_{zz}(1.57)} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(RZZ(qubits_idx, param, param_sharing))

    def ms(self, qubits_idx: Union[Iterable[int], str] = 'linear') -> None:
        r"""Add Mølmer-Sørensen (MS) gates.

        The matrix form is:

        .. math::
            \mathit{MS} = R_{XX}(-\pi/2) = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1 & 0 & 0 & i \\
                0 & 1 & i & 0 \\
                0 & i & 1 & 0 \\
                i & 0 & 0 & 1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.ms([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{\text{MS}} & \meter[2]{} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 2)
        self.append(MS(qubits_idx))

    def cswap(self, qubits_idx: Union[Iterable[int], str] = 'cycle') -> None:
        r"""Add CSWAP (Fredkin) gates.

        The matrix form is:

        .. math::
            \mathit{CSWAP} =
            \begin{bmatrix}
                1 & 0 & \cdots & 0 \\
                0 & 1 &         & 0 \\
                \vdots &  & \ddots  & \vdots \\
                0 & 0 & \cdots & 1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.

        Examples:
            .. code-block:: python

                qc = Circuit(3)
                qc.cswap([0, 1, 2])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{1} & \meter[3]{} & {} \\
                \lstick{} & \gate[2,style={draw=gray, dashed}]{\permute{2,1}} & {} & {} \\
                \lstick{} & {} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 3)
        self.append(CSWAP(qubits_idx))

    def ccx(self, qubits_idx: Union[Iterable[int], str] = 'cycle') -> None:
        r"""Add CCX (Toffoli) gates.

        The matrix form is:

        .. math::
            \mathit{CCX} =
            \begin{bmatrix}
                1 & 0 & \cdots & 0 \\
                0 & 1 &         & 0 \\
                \vdots &  & \ddots  & \vdots \\
                0 & 0 & \cdots & 1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.

        Examples:
            .. code-block:: python

                qc = Circuit(3)
                qc.ccx([0, 1, 2])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \ctrl[]{2} & \meter[3]{} & {} \\
                \lstick{} & \control{} & {} & {} \\
                \lstick{} & \targ{} & {} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 3)
        self.append(CCX(qubits_idx))

    def universal_two_qubits(self, qubits_idx: Union[List[int], str] = None,
                             param: Union[torch.Tensor, float] = None) -> None:
        r"""Add universal two-qubit gates. One such gate requires 15 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied.
            param: Parameters of the gates. Defaults to None.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.universal_two_qubits([0, 1])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a universal 2-qubit layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(Universal2(qubits_idx, param))

    def universal_three_qubits(self, qubits_idx: Optional[List[int]] = None,
                               param: Union[torch.Tensor, float] = None) -> None:
        r"""Add universal three-qubit gates. One such gate requires 81 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied.
            param: Parameters of the gates. Defaults to None.

        Examples:
            .. code-block:: python

                qc = Circuit(3)
                qc.universal_three_qubits([0, 1, 2])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a universal 3-qubit layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(Universal3(qubits_idx, param))
    
    @_alias({"system_idx": "qubits_idx"})
    def universal_qudits(self, system_idx: List[int],
                         param: Union[torch.Tensor, float] = None, param_sharing: bool = False) -> None:
        r"""Add universal qudit gates. One such gate requires :math:`d^2 - 1` parameters,
        where :math:`d` is the gate dimension.

        Args:
            system_idx: Indices of the systems on which the gates are applied.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Examples:
            .. code-block:: python

                qc = Circuit(1, 3)
                qc.universal_qudits([0])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\operatorname{UNI}_{3}} & \meter[1]{} & {}
        """
        system_idx = self.register_idx(system_idx, 1 if isinstance(system_idx, int) else len(system_idx))
        self.append(UniversalQudits(system_idx, [self.system_dim[idx] for idx in system_idx],
                                    param, param_sharing))
    
    @_alias({"system_idx": "qubits_idx"})
    def permute(self, perm: List[int], system_idx: List[int], control_idx: Optional[int] = None) -> None:
        r"""Add a permutation gate.

        Args:
            perm: A list representing the permutation of subsystems.
            system_idx: Indices of the systems on which the gates are applied.
            control_idx: the index that controls the permutation. Defaults to None.

        Examples:
            .. code-block:: python

                qc = Circuit(3)
                qc.permute([1, 0, 2], [0, 1, 2])
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[3,style={draw=none}]{\permute{2,1,3}} & \meter[3]{} & {} \\
                \lstick{} & {} & {} & {} \\
                \lstick{} & {} & {} & {}
        """
        if control_idx is None:
            system_idx = self.register_idx(system_idx, len(system_idx))
            self.append(Permutation(perm, system_idx, [self.system_dim[idx] for idx in system_idx]))
        else:
            if isinstance(system_idx[0], int):
                system_idx[0] = [system_idx[0]]
            _system_idx = system_idx[0] + system_idx[1:]
            _system_idx = self.register_idx(_system_idx, len(_system_idx))
            acted_system_dim = [self.system_dim[idx] for idx in _system_idx]

            ctrl_dim = math.prod(acted_system_dim[:len(system_idx[0])])
            if control_idx == -1:
                control_idx = ctrl_dim - 1
            else:
                assert control_idx < ctrl_dim, \
                    f"Control index out of range: expect < {ctrl_dim}, got {control_idx}."
            
            permutation = utils.matrix._permutation(perm, acted_system_dim[len(system_idx[0]):])
            
            gate_info = {
                "name": "cpermute",
                "api": "control_permute",
                "permute": perm,
                'plot_width': 0.2,
            }
            self.append(ControlOracle(
                permutation, system_idx, control_idx, acted_system_dim, gate_info))
    
    @_alias({"system_idx": "qubits_idx"})
    def oracle(self, oracle: torch.Tensor, system_idx: Union[List[Union[List[int], int]], int], 
               control_idx: Optional[int] = None, gate_name: Optional[str] = None,
               latex_name: Optional[str] = None) -> None:
        r"""Add an oracle gate.

        Args:
            oracle: Unitary oracle.
            system_idx: Indices of the systems on which the gate is applied.
            control_idx: the index that controls the oracle. Defaults to None.
            gate_name: name of the oracle.
            latex_name: LaTeX name of the gate. Defaults to gate_name.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.oracle(oracle=eye(2), system_idx=[0], latex_name=r'$Identity$')
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{$Identity$}
        """
        gate_info = {}
        if gate_name is not None:
            gate_info['name'] = gate_name
        if latex_name is not None:
            gate_info['tex'] = latex_name
        
        if control_idx is None:
            system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
            acted_system_dim = [self.system_dim[idx] for idx in system_idx]
            
            system_idx = self.register_idx(system_idx, len(system_idx))
            self.append(Oracle(oracle, system_idx, acted_system_dim, gate_info))
        else:
            if isinstance(system_idx[0], int):
                system_idx[0] = [system_idx[0]]
            _system_idx = system_idx[0] + system_idx[1:]
            acted_system_dim = [self.system_dim[idx] for idx in _system_idx]
            system_idx = self.register_idx(system_idx, len(_system_idx))

            ctrl_dim = math.prod(acted_system_dim[:len(system_idx[0])])
            if control_idx == -1:
                control_idx = ctrl_dim - 1
            else:
                assert control_idx < ctrl_dim, \
                    f"Control index out of range: expect < {ctrl_dim}, got {control_idx}."
            self.append(ControlOracle(
                oracle, system_idx, control_idx, acted_system_dim, gate_info))

    @_alias({"system_idx": "qubits_idx"})
    def control_oracle(self, oracle: torch.Tensor, system_idx: Union[List[Union[List[int], int]], int], control_idx: int = -1,
                        gate_name: str = 'coracle', latex_name: Optional[str] = None) -> None:
        r"""Add a controlled oracle gate.

        Args:
            oracle: Unitary oracle.
            system_idx: Indices of the systems on which the gate is applied.
            control_idx: the index that controls the oracle. Defaults to -1.
            gate_name: name of the oracle.
            latex_name: LaTeX name of the gate.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.control_oracle(oracle=eye(2), system_idx=[0, 1], control_idx=0, latex_name=r'$Identity$')
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \octrl[]{1} & \meter[2]{} & {} \\
                \lstick{} & \gate[1]{$Identity$} & {} & {}
        """
        warnings.warn(
            "Starting from QuAIRKit 0.4.1, it is recommended to use Circuit.oracle(..., control_idx=..) instead of Circuit.control_oracle", FutureWarning)
        self.oracle(oracle, system_idx, control_idx, gate_name, latex_name)
        
    @_alias({"system_idx": "qubits_idx"})
    def param_oracle(self, generator: Callable[[torch.Tensor], torch.Tensor], num_acted_param: int,
                     system_idx: Union[List[Union[List[int], int]], int], control_idx: Optional[int] = None,
                     param: Union[torch.Tensor, float] = None, gate_name: Optional[str] = None,
                     latex_name: Optional[str] = None, support_batch: bool = True) -> None:
        r"""Add a parameterized oracle gate.

        Args:
            generator: Function to generate the oracle.
            num_acted_param: Number of parameters required for a single application.
            system_idx: Indices of the systems on which the gate acts.
            control_idx: The index that controls the oracle. Defaults to None.
            param: Input parameters for the gate. Defaults to None.
            gate_name: Name of the oracle.
            latex_name: LaTeX name of the gate.
            support_batch: Whether generator supports batched input.
            
        Note:
            If the generator does not support batched input, you need to set `support_batch` to `False`.

        Examples:
            .. code-block:: python

                def rotation_generator(params: torch.Tensor) -> torch.Tensor:
                    theta = params[..., 0]
                    cos_theta = torch.cos(theta / 2).unsqueeze(-1)
                    sin_theta = torch.sin(theta / 2).unsqueeze(-1)
                    matrix = torch.cat([
                        torch.cat([cos_theta, -1j * sin_theta], dim=-1),
                        torch.cat([-1j * sin_theta, cos_theta], dim=-1)
                    ], dim=-2)
                    return matrix

                qc = Circuit(2)
                qc.param_oracle(
                    generator=rotation_generator,
                    num_acted_param=1,
                    system_idx=[0, 1],
                    control_idx=0,
                    param=None,
                    gate_name="ControlledRotation",
                    support_batch=True
                )
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \octrl[]{1} \\
                \lstick{} & \gate[1]{ControlledRotation(0.36)}
        """
        gate_info = {}
        if gate_name is not None:
            gate_info['name'] = gate_name
        if latex_name is not None:
            gate_info['tex'] = latex_name
        
        if control_idx is None:
            system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
            acted_system_dim = [self.system_dim[idx] for idx in system_idx]
            
            system_idx = self.register_idx(system_idx, len(system_idx))
            self.append(ParamOracle(
                generator, system_idx, param, num_acted_param, acted_system_dim, gate_info, support_batch))
        else:
            if isinstance(system_idx[0], int):
                system_idx[0] = [system_idx[0]]
            _system_idx = system_idx[0] + system_idx[1:]
            acted_system_dim = [self.system_dim[idx] for idx in _system_idx]
            system_idx = self.register_idx(system_idx, len(_system_idx))
            
            ctrl_dim = math.prod(acted_system_dim[:len(system_idx[0])])
            assert control_idx < ctrl_dim, f"Control index out of range: expect < {ctrl_dim}, got {control_idx}."
            self.append(ControlParamOracle(
                generator, system_idx, control_idx, param, num_acted_param, acted_system_dim, gate_info, support_batch))

    @_alias({"system_idx": "qubits_idx", "post_selection": "desired_result"})
    def measure(self, system_idx: Union[Iterable[int], int, str] = None, post_selection: Union[int, str] = None,
                if_print: bool = False, measure_basis: Optional[torch.Tensor] = None) -> None:
        r"""Perform a measurement on the specified systems.

        Args:
            system_idx: Systems to measure. Defaults to all.
            post_selection: The post-selection result. Defaults to None.
            if_print: Whether to print collapse info. Defaults to False.
            measure_basis: Measurement basis.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.measure()
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \meter[2]{} & {} \\
                \lstick{} & {} & {}
        """
        if system_idx is None:
            acted_system_dim = self.system_dim.copy()
        elif isinstance(system_idx, int):
            acted_system_dim = [self.system_dim[system_idx]]
        else:
            acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        system_idx = self.register_idx(system_idx, None)
        self.append(Collapse(system_idx, acted_system_dim, post_selection, if_print, measure_basis))
    
    @_alias({"system_idx": "qubits_idx"})
    def locc(self, local_unitary: torch.Tensor, system_idx: Union[List[Union[List[int], int]], int],
             label: str = 'M', latex_name: str = 'O') -> None:
        r"""Add a one-way LOCC protocol comprised of unitary operations.

        Args:
            local_unitary: The local unitary operation.
            system_idx: Systems on which the protocol is applied. The first element indicates the measure system.
            label: Label for measurement. Defaults to 'M'.
            latex_name: LaTeX name for the applied operator. Defaults to 'O'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.locc(local_unitary=x(), system_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \meter{} & \push{M} \wireoverride{c} & \ctrl[vertical wire=c]{1}\wireoverride{c}  \\
                \lstick{} & {} & {} & \gate[1]{O^{(M)}}
        
        """
        if isinstance(system_idx[0], int):
            system_idx[0] = [system_idx[0]]
        _system_idx = system_idx[0] + system_idx[1:]
        system_dim = [self.system_dim[idx] for idx in _system_idx]
        measure_dim, apply_dim = system_dim[:len(system_idx[0])], system_dim[len(system_idx[0]):]
        
        system_idx = self.register_idx(system_idx, len(np.unique(_system_idx)))
        measure_idx, apply_idx = system_idx[0], system_idx[1:]
        
        gate = Oracle(local_unitary, apply_idx, apply_dim)
        self.append(OneWayLOCC(gate, measure_idx, measure_dim, label=label, latex_name=latex_name))
        
    @_alias({"system_idx": "qubits_idx"})
    def param_locc(self, generator: Callable[[torch.Tensor], torch.Tensor], num_acted_param: int, 
                   system_idx: Union[List[Union[List[int], int]], int], param: Union[torch.Tensor, float] = None, 
                   label: str = 'M', latex_name: str = 'U', support_batch: bool = True) -> None:
        r"""Add a one-way LOCC protocol comprised of unitary operations, where the applied unitary is parameterized.

        Args:
            generator: Function to generate the oracle.
            num_acted_param: Number of parameters required for a single application.
            system_idx: Systems on which the protocol is applied. The first element indicates the measure system.
            param: Input parameters for the gate. Defaults to None.
            label: Label for measurement. Defaults to 'M'.
            latex_name: LaTeX name for the applied operator. Defaults to 'U'.
            support_batch: Whether generator supports batched input.
        
        Note:
            If the generator does not support batched input, you need to set `support_batch` to `False`.

        """
        if isinstance(system_idx[0], int):
            system_idx[0] = [system_idx[0]]
        _system_idx = system_idx[0] + system_idx[1:]
        system_dim = [self.system_dim[idx] for idx in _system_idx]
        measure_dim, apply_dim = system_dim[:len(system_idx[0])], system_dim[len(system_idx[0]):]
        
        system_idx = self.register_idx(system_idx, len(np.unique(_system_idx)))
        measure_idx, apply_idx = system_idx[0], system_idx[1:]
        
        if param is None:
            float_dtype = intrinsic._get_float_dtype(self.dtype)
            expect_shape = intrinsic._format_param_shape([apply_idx], num_acted_param, param_sharing=False, batch_size=np.prod(measure_dim))
            param = torch.nn.Parameter(torch.rand(expect_shape, dtype=float_dtype) * 2 * np.pi)
        param_gate = ParamOracle(
            generator, apply_idx, param, num_acted_param, apply_dim, support_batch=support_batch)
        self.append(OneWayLOCC(param_gate, measure_idx, measure_dim, label=label, latex_name=latex_name))
        
    @_alias({"system_idx": "qubits_idx"})
    def quasi(self, list_unitary: torch.Tensor, probability: Iterable[float],
              system_idx: Union[List[Union[List[int], int]], int], latex_name: str = r'\mathcal{E}'):
        r"""Add a quasi-probability operation, now only supports unitary operations.
        
        Args:
            list_unitary: list of unitary operations, each of which corresponds to a probability outcome.
            probability: (quasi-)probability distribution for applying unitary operations.
            system_idx: Systems on which the operation is applied.
            latex_name: LaTeX name for the applied operation. Defaults to '\mathcal{E}'.
        
        """
        if not torch.is_tensor(list_unitary):
            list_unitary = torch.tensor(list_unitary, dtype=self.dtype, device=self.device)
        if not torch.is_tensor(probability): 
            probability = torch.tensor(probability, dtype=self.dtype, device=self.device)
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        system_idx = self.register_idx(system_idx, len(system_idx))
        
        gate_info = {'name': 'quasi', 'tex': latex_name}
        gate = Oracle(list_unitary, system_idx, acted_system_dim, gate_info)
        self.append(QuasiOperation(gate, probability))
        
    @_alias({"system_idx": "qubits_idx"})
    def param_quasi(self, generator: Callable[[torch.Tensor], torch.Tensor], num_acted_param: int, 
                    probability: Iterable[float], system_idx: Union[List[Union[List[int], int]], int], probability_param: bool = False,
                    param: Union[torch.Tensor, float] = None, latex_name: str = r'\mathcal{E}', support_batch: bool = True):
        r"""Add a quasi-probability operation, where the applied unitary is parameterized.
        
        Args:
            generator: Function to generate the oracle.
            num_acted_param: Number of parameters required for a single application.
            probability: (quasi-)probability distribution for applying unitary operations.
            probability_param: Whether the probability is parameterized. Defaults to False.
            system_idx: Systems on which the operation is applied.
            param: Input parameters for the gate. Defaults to None.
            latex_name: LaTeX name for the applied operation. Defaults to '\mathcal{E}'.
            support_batch: Whether generator supports batched input.
        
        """
        if not torch.is_tensor(probability): 
            probability = torch.tensor(probability, dtype=self.dtype, device=self.device)
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        system_idx = self.register_idx(system_idx, len(system_idx))
        
        gate_info = {'name': 'quasi', 'tex': latex_name}
        if param is None:
            float_dtype = intrinsic._get_float_dtype(self.dtype)
            batch_size = (probability.numel() + 1) if probability_param else probability.numel()
            expect_shape = intrinsic._format_param_shape([system_idx], num_acted_param, 
                                                         param_sharing=False, batch_size=batch_size)
            param = torch.nn.Parameter(torch.rand(expect_shape, dtype=float_dtype) * 2 * np.pi)
        param_gate = ParamOracle(
            generator, system_idx, param, num_acted_param, acted_system_dim, gate_info=gate_info, support_batch=support_batch)
        
        self.append(QuasiOperation(param_gate, probability, probability_param=probability_param))
        
    @_alias({"system_idx": "qubits_idx"})
    def reset(self, system_idx: Union[List[int], int], 
              replace_state: Optional[Union[torch.Tensor, StateSimulator]] = None, 
              state_label: Optional[str] = None) -> None:
        r"""Reset the state of the specified systems to a given state.
        
        Args:
            system_idx: list of systems to be reset.
            replace_state: the state to replace the quantum state. Defaults to zero state.
            state_label: LaTeX label of the reset state, used for printing. Defaults to r'\rho' or r'\ket{0}'.
        
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        
        if isinstance(replace_state, torch.Tensor):
            replace_state = to_state(replace_state, acted_system_dim)
        system_idx = self.register_idx(system_idx, len(system_idx))

        if replace_state:
            assert acted_system_dim == replace_state.system_dim, \
                f"The system dimension of the input state {acted_system_dim} does not match the replace state {replace_state.system_dim}."
            state_label = state_label or r'\rho'
        else:
            num_systems = len(acted_system_dim)
            replace_state = zero_state(num_systems=num_systems, system_dim=acted_system_dim)
            state_label = state_label or r'\ket{0}'
        self.append(ResetState(system_idx, acted_system_dim, replace_state, state_label))

    def linear_entangled_layer(self, qubits_idx: Optional[List[int]] = None, depth: int = 1,
                               param: Union[torch.Tensor, float] = None) -> None:
        r"""Add linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

        Args:
            qubits_idx: Systems to apply the layer on. Defaults to all.
            depth: Number of layers. Defaults to 1.
            param: Parameters for the layer. Defaults to self-generated.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.linear_entangled_layer([0, 1], depth=4)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a linear entangled layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(LinearEntangledLayer(qubits_idx, depth, param))

    def real_entangled_layer(self, qubits_idx: Optional[List[int]] = None, depth: int = 1,
                             param: Union[torch.Tensor, float] = None) -> None:
        r"""Add strongly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Systems to apply the layer on. Defaults to all.
            depth: Number of layers. Defaults to 1.
            param: Layer parameters. Defaults to self-generated.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.real_entangled_layer([0, 1], depth=4)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a real entangled layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(RealEntangledLayer(qubits_idx, depth, param))

    def complex_entangled_layer(self, qubits_idx: Optional[List[int]] = None, depth: int = 1,
                                param: Union[torch.Tensor, float] = None) -> None:
        r"""Add strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Systems to apply the layer on. Defaults to all.
            depth: Number of layers. Defaults to 1.
            param: Layer parameters. Defaults to self-generated.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.complex_entangled_layer([0, 1], depth=4)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a complex entangled layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(ComplexEntangledLayer(qubits_idx, depth, param))

    def real_block_layer(self, qubits_idx: Optional[List[int]] = None, depth: int = 1,
                         param: Union[torch.Tensor, float] = None) -> None:
        r"""Add weakly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Systems to apply the layer on. Defaults to all.
            depth: Number of layers. Defaults to 1.
            param: Layer parameters. Defaults to self-generated.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.real_block_layer([0, 1], depth=4)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a real block layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(RealBlockLayer(qubits_idx, depth, param))

    def complex_block_layer(self, qubits_idx: Optional[List[int]] = None, depth: int = 1,
                            param: Union[torch.Tensor, float] = None) -> None:
        r"""Add weakly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Systems to apply the layer on. Defaults to all.
            depth: Number of layers. Defaults to 1.
            param: Layer parameters. Defaults to self-generated.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.complex_block_layer([0, 1], depth=4)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                (The output latex code shows a complex block layer.)
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(ComplexBlockLayer(qubits_idx, depth, param))
    
    def basis_encoding(self, number: Union[int, List[int], torch.Tensor],
                       qubits_idx: Optional[List[int]] = None) -> None:
        r"""Prepares a basis encoding layer that performs :math:`|0\rangle^{\otimes n} \to |x\rangle`
            
        Args:
            number: Integer to be encoded (must be in [0, 2**n_qubits - 1]).
                    If batched, must be List[int], torch.Tensor.
            qubits_idx: Indices of the qubits on which the layer is applied. Defaults to ``None``.
            
        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.basis_encoding(3, [0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{X}\gategroup[2,steps=1,style={inner sep=4pt,dashed,label={above:{Basis Encoding}}}]{} \\
                \lstick{} & \gate[1]{X}
        
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        self.append(BasisEncoding(number, qubits_idx))
        
    def amplitude_encoding(self, vector: torch.Tensor, qubits_idx: Optional[List[int]] = None) -> None:
        r"""Prepares an amplitude encoding layer that performs :math:`|0\rangle^{\otimes n} \to \sum_{i=0}^{d-1} x_i |i\rangle`
        
        Args:
            vector: Input normalized vector to be encoded. If batched, size must be 2^n_qubits * batch_size
            qubits_idx: Indices of the qubits on which the encoding is applied. Defaults to ``None``.
        
        Examples:
            .. code-block:: python

                qc = Circuit(2)
                vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
                qc.amplitude_encoding(vec, [0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{\text{Amplitude Encoding}}\gategroup[2,steps=1,style={inner sep=4pt,dashed,label={above:{Amplitude Encoding}}}]{} \\
                \lstick{} & {}
        
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=self.dtype, device=self.device)
        self.append(AmplitudeEncoding(vector, qubits_idx))
        
    def angle_encoding(self, angles: torch.Tensor, qubits_idx: Optional[List[int]] = None, rotation: str = 'RY') -> None:
        r"""Prepares an angle encoding layer that encode angles via rotation gates.
        
        Args:
            angles: Input vector of angles. If batched, size must be num_qubits * batch_size
            qubits_idx: Indices of the qubits on which the encoding is applied. Defaults to ``None``.
            rotation: Type of rotation gate ('RY', 'RZ', or 'RX').
            
        Examples:
            .. code-block:: python

                qc = Circuit(2)
                angles = torch.tensor([0.5, 1.0])
                qc.angle_encoding(angles, [0, 1], rotation='RY')
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{R_{y}(0.50)}\gategroup[2,steps=1,style={inner sep=4pt,dashed,label={above:{Angle Encoding}}}]{} \\
                \lstick{} & \gate[1]{R_{y}(1.00)}
        
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        if not isinstance(angles, torch.Tensor):
            angles = torch.tensor(angles, dtype=self.dtype, device=self.device)
        self.append(AngleEncoding(angles, qubits_idx, rotation))
    
    def iqp_encoding(self, features: torch.Tensor, set_entanglement: List[List[int]], 
                     qubits_idx: Optional[List[int]] = None, depth: int = 1) -> None:
        r"""Prepares an instantaneous quantum polynomial (IQP) layer that encode angles via a type of rotation gates.
        
        Args:
            features: Input vector for encoding. If batched, size must be num_qubits * batch_size
            set_entanglement: the set containing all pairs of qubits to be entangled using RZZ gates
            qubits_idx: Indices of the qubits on which the encoding is applied. Defaults to ``None``.
            depth: Number of depth. Defaults to 1.
            
        Examples:
            .. code-block:: python

                qc = Circuit(3)
                features = torch.tensor([0.1, 0.2, 0.3])
                entanglement = [[0, 1], [1, 2]]
                qc.iqp_encoding(features, entanglement, [0, 1, 2], depth=1)
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{H}\gategroup[3,steps=8,style={inner sep=4pt,dashed,label={above:{IQP Encoding}}}]{} & \gate[1]{R_{z}(0.10)} & \ctrl[]{1} & {} & \ctrl[]{1} & {} & {} & {} \\
                \lstick{} & \gate[1]{H} & \gate[1]{R_{z}(0.20)} & \targ{} & \gate[1]{R_{z}(0.02)} & \targ{} & \ctrl[]{1} & {} & \ctrl[]{1} \\
                \lstick{} & \gate[1]{H} & \gate[1]{R_{z}(0.30)} & {} & {} & {} & \targ{} & \gate[1]{R_{z}(0.06)} & \targ{}
        
        """
        qubits_idx = self.register_idx(qubits_idx, None)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=self.dtype, device=self.device)
        self.append(IQPEncoding(features, set_entanglement, qubits_idx, depth))
        
    def trotter(
        self, hamiltonian: Hamiltonian, time: float, qubits_idx: Optional[List[int]] = None, 
        num_steps: int = 1, order: int = 1, name: str = 'H'
    ) -> None:
        r"""Add Trotter decompositions of a Hamiltonian evolution operator.

        Args:
            hamiltonian: Hamiltonian of the system whose time evolution is to be simulated.
            time: Total evolution time.
            qubits_idx: Indices of the qubits on which the layer is applied. Defaults to ``None``.
            num_steps: Number of trotter blocks. Defaults to 1.
            order: Order of the Trotter-Suzuki decomposition. Defaults to 1.
            name: Name of the Hamiltonian. Defaults to 'H'.

        """
        qubits_idx = self.register_idx(qubits_idx, None)
        tau = time / num_steps
        self.append(TrotterLayer(hamiltonian, qubits_idx, tau, num_steps, order, name))

    def bit_flip(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add bit flip channels.

        Args:
            prob: Probability of a bit flip.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.bit_flip(prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(BF)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(BF)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(BitFlip(prob, qubits_idx))

    def phase_flip(self, prob: Union[torch.Tensor, float],
                   qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add phase flip channels.

        Args:
            prob: Probability of a phase flip.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.phase_flip(prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(PF)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(PF)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(PhaseFlip(prob, qubits_idx))

    def bit_phase_flip(self, prob: Union[torch.Tensor, float],
                       qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add bit phase flip channels.

        Args:
            prob: Probability of a bit phase flip.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.bit_phase_flip(prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(BPF)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{p = 0.5}^{\textrm{\tiny{(BPF)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(BitPhaseFlip(prob, qubits_idx))

    def amplitude_damping(self, gamma: Union[torch.Tensor, float],
                          qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add amplitude damping channels.

        Args:
            gamma: Damping probability.
            qubits_idx: Systems to apply the damping on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.amplitude_damping(gamma=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5}^{\textrm{\tiny{(AD)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5}^{\textrm{\tiny{(AD)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(AmplitudeDamping(gamma, qubits_idx))

    def generalized_amplitude_damping(self, gamma: Union[torch.Tensor, float],
                                      prob: Union[torch.Tensor, float],
                                      qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add generalized amplitude damping channels.

        Args:
            gamma: Damping probability.
            prob: Excitation probability.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.generalized_amplitude_damping(gamma=0.5, prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5, p = 0.5}^{\textrm{\tiny{(GAD)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5, p = 0.5}^{\textrm{\tiny{(GAD)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(GeneralizedAmplitudeDamping(gamma, prob, qubits_idx))

    def phase_damping(self, gamma: Union[torch.Tensor, float],
                      qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add phase damping channels.

        Args:
            gamma: Phase damping parameter.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.phase_damping(gamma=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5}^{\textrm{\tiny{(PD)}}}} \\
                \lstick{} & \gate[1]{\mathcal{E}_{\gamma = 0.5}^{\textrm{\tiny{(PD)}}}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(PhaseDamping(gamma, qubits_idx))

    def depolarizing(self, prob: Union[torch.Tensor, float],
                     qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add depolarizing channels.

        Args:
            prob: Depolarizing probability.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.depolarizing(prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{D}_{p = 0.5}} \\
                \lstick{} & \gate[1]{\mathcal{D}_{p = 0.5}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(Depolarizing(prob, qubits_idx))

    def generalized_depolarizing(self, prob: Union[torch.Tensor, float],
                                  qubits_idx: Union[Iterable[int], int, str]) -> None:
        r"""Add a general depolarizing channel.

        Args:
            prob: Probabilities for the Pauli basis.
            qubits_idx: Systems to apply the channel on.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.generalized_depolarizing(prob=0.5, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[2]{\mathcal{D}_{p = 0.5}} \\
                \lstick{} & {}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(GeneralizedDepolarizing(prob, qubits_idx))

    def pauli_channel(self, prob: Union[torch.Tensor, float],
                      qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add Pauli channels.

        Args:
            prob: Probabilities for the Pauli X, Y, and Z operators.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.pauli_channel(prob=[0.1, 0.3, 0.5])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}} \\
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(PauliChannel(prob, qubits_idx))

    def reset_channel(self, prob: Union[torch.Tensor, float],
                      qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add reset channels.

        Args:
            prob: Probabilities for resetting to the basis states.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.reset_channel(prob=[0.5, 0.4], qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}} \\
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(ResetChannel(prob, qubits_idx))

    def thermal_relaxation(self, const_t: Union[torch.Tensor, Iterable[float]], exec_time: Union[torch.Tensor, float],
                           qubits_idx: Union[Iterable[int], int, str] = 'full') -> None:
        r"""Add thermal relaxation channels.

        Args:
            const_t: The T1 and T2 relaxation times.
            exec_time: Gate execution time.
            qubits_idx: Systems to apply the channel on. Defaults to 'full'.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.thermal_relaxation(const_t=[600, 300], exec_time=500, qubits_idx=[0, 1])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}} \\
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        qubits_idx = self.register_idx(qubits_idx, 1)
        self.append(ThermalRelaxation(const_t, exec_time, qubits_idx))

    def choi_channel(self, choi_repr: Iterable[torch.Tensor],
                     system_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
                     check_legality: bool = True) -> None:
        r"""Add custom channels in the Choi representation.

        Args:
            choi_repr: Choi representation.
            system_idx: Systems to apply the channel on.
            check_legality: whether to check the legality of the input Choi operator. Defaults to ``True``.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                X = torch.tensor(x(), dtype=torch.complex64)
                choi = X.kron(X) / 2
                qc.choi_channel(choi_repr=choi, system_idx=[0])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.register_idx(system_idx, len(system_idx))
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        self.append(ChoiRepr(choi_repr, system_idx, acted_system_dim, check_legality))

    def kraus_channel(self, kraus_repr: Iterable[torch.Tensor],
                      system_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
                      check_legality: bool = True) -> None:
        r"""Add custom channels in the Kraus representation.

        Args:
            kraus_repr: Kraus operators.
            system_idx: Systems to apply the channel on.
            check_legality: whether to check the legality of the input representation. Defaults to ``True``.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.kraus_channel(kraus_repr=eye(2), system_idx=[0])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.register_idx(system_idx, len(system_idx))
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        self.append(KrausRepr(kraus_repr, system_idx, acted_system_dim, check_legality))

    def stinespring_channel(self, stinespring_repr: Iterable[torch.Tensor],
                             system_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
                             check_legality: bool = True) -> None:
        r"""Add custom channels in the Stinespring representation.

        Args:
            stinespring_repr: Stinespring representation.
            system_idx: Systems to apply the channel on.
            check_legality: whether to check the legality of the input representation. Defaults to ``True``.

        Examples:
            .. code-block:: python

                qc = Circuit(2)
                qc.stinespring_channel(stinespring_repr=eye(2), system_idx=[0])
                print(f'The latex code of this circuit is:\n{qc.to_latex()}')

            ::

                The latex code of this circuit is:
                \lstick{} & \gate[1]{\mathcal{N}}
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.register_idx(system_idx, len(system_idx))
        acted_system_dim = [self.system_dim[idx] for idx in system_idx]
        self.append(StinespringRepr(stinespring_repr, system_idx, acted_system_dim, check_legality))
        
    __1input = ['h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'cnot', 'cy', 'cz', 'swap', 'cswap', 'ccx', 'ms']
    __3input = ['p', 'rx', 'ry', 'rz', 'u3', 'cp', 'crx', 'cry', 'crz', 'cu', 'rxx', 'ryy', 'rzz']
    __custom_gate = ['permute', 'control_permute', 'oracle', 'control_oracle', 'param_oracle']
    __special = ['measure', 'locc', 'quasi', 'reset']
    __noise = ['bit_flip', 'phase_flip', 'bit_phase_flip', 'amplitude_damping', 'generalized_amplitude_damping',
               'phase_damping', 'depolarizing', 'generalized_depolarizing', 'pauli_channel', 'reset_channel','thermal_relaxation',]
    __custom_channel = ['choi_channel', 'kraus_channel', 'stinespring_channel']
    
    @classmethod
    def from_operators(cls, list_operators: List[OperatorInfoType]) -> 'Circuit':
        r"""Reconstruct a Circuit from a saved operator_history (same format as OperatorList.operator_history).
        Do not support daggered list_operators.

        Args:
            list_operators: operator_history list (or nested lists).

        Returns:
            Circuit: a Circuit instance with the same sequence of operations.
        
        Note:
            This function assumes all systems are qubits.
        """

        if not list_operators:
            raise ValueError("Input an empty list!")
        
        max_index = -1
        list_operators = [element for item in list_operators
                          for element in (item if isinstance(item, list) else [item])]
        for op_info in list_operators:
            system_idx: List[List[int]] = op_info['system_idx']
            for sub_list in system_idx:
                max_index = max([max_index] + sub_list)

        num_systems = max_index + 1 if max_index >= 0 else 1
        system_dim = [2] * num_systems
        cir = cls(num_systems, system_dim, list(range(num_systems)))
        for op_info in list_operators:
            if (op_info.get('api', '') not in ['t', 's', 'sdg', 'tdg']) and (op_info.get('tex', '').find('dagger') >= 0):
                raise NotImplementedError("Daggered OperatorList not supported!")
            
            if op_info['name'] in cls.__special:
                _load_special(cir, op_info)

            elif op_info['api'] in cls.__1input:
                method = getattr(cir, op_info['api'])
                method(op_info['system_idx'])

            elif op_info['api'] in cls.__3input:
                method = getattr(cir, op_info['api'])
                param = op_info['param'][0] if op_info['param_sharing'] else op_info['param']
                method(op_info['system_idx'], param, op_info['param_sharing'])

            elif op_info['api'] in cls.__custom_gate:
                _load_custom_gate(cir, op_info)

            elif op_info['api'] in cls.__noise:
                method = getattr(cir, op_info['api'])
                method(qubits_idx=op_info['system_idx'], **op_info['kwargs'])

            elif op_info['api'] in cls.__custom_channel:
                _load_custom_channel(cir, op_info)

            else:
                raise NotImplementedError(f'{op_info["name"]} not supported!')
        return cir
    
    @classmethod
    def from_qasm2(cls, qasm: str) -> 'Circuit':
        r"""Reconstruct a Circuit from a QASM2 string.

        Args:
            qasm: A complete OpenQASM 2.0 string representing a circuit.

        Returns:
            a Circuit instance with the same sequence of operations.
            
        Raises:
            ValueError: if the program header is missing or does not include "qelib1.inc", or malformed ops.
            NotImplementedError: if the program contains gate/opaque definitions or classically conditioned "if".
            
        Note:
            commands such as "barrier", "creg" are ignored. Multiple quantum registers are merged by order of definition. The circuit is assumed to be composed of qubits only.
        """
        return cls.from_operators(qasm2_to_info(qasm))


def _load_special(cir: Circuit, info: OperatorInfoType) -> None:
    r"""Load special operators: measure, reset, locc, quasi into a Circuit.
    """
    if info['name'] == 'measure':
        for system_idx in info['system_idx']:
            cir.measure(system_idx, info.get('label', None), **info['kwargs'])

    elif info['name'] == 'reset':
        replace_state = info['kwargs']['replace_dm']
        tex = info['tex']
        for system_idx in info['system_idx']:
            cir.reset(system_idx, replace_state, tex)

    elif info['name'] == 'locc':
        if info['kwargs']['info']['api'] == 'param_oracle':
            param = info['kwargs']['info']['param']
            cir.param_locc(info['kwargs']['info']['kwargs']['generator'], param.shape[-1],
                                        info['system_idx'][0], param=param)
        else:
            for system_idx in info['system_idx']:
                matrix = info['kwargs']['info']['matrix'].clone()
                cir.locc(matrix, system_idx)

    elif info['name'] == 'quasi':
        if info['api'] == 'oracle':
            matrix = info['matrix']
            for system_idx in info['system_idx']:
                cir.quasi(matrix, info["kwargs"]["probability"], system_idx)
        elif info['api'] == 'param_oracle':
            param = info['param']
            cir.param_quasi(info['kwargs']['generator'], param.shape[-1],
                                        info["kwargs"]["probability"], info['system_idx'][0],
                                        info["kwargs"]["probability_param"], param)
        else:
            raise NotImplementedError(f"Quasi operator with {info['api']} not implemented!")

    else:
        raise ValueError(f"{info['name']} not found in special operators")
        
def _load_custom_gate(cir: Circuit, info: OperatorInfoType) -> None:
    r"""Load custom gates: permute, control_permute, oracle, control_oracle, param_oracle into a Circuit.
    """
    if info['api'] == 'permute':
        for system_idx in info['system_idx']:
            cir.permute(info['permute'], system_idx)
    elif info['api'] == 'control_permute':
        system_idx = deepcopy(info['system_idx'][0])
        system_idx = [system_idx[:info['num_ctrl_system']]] + system_idx[info['num_ctrl_system']:]
        cir.permute(info['permute'], system_idx, int(info['label']))

    elif info['api'] == 'oracle':
        for system_idx in info['system_idx']:
            cir.oracle(info['matrix'], system_idx)

    elif info['api'] == 'control_oracle':
        system_idx = deepcopy(info['system_idx'][0])
        system_idx = [system_idx[:info['num_ctrl_system']]] + system_idx[info['num_ctrl_system']:]
        cir.oracle(info['matrix'], system_idx, int(info['label']))

    elif info['api'] == 'param_oracle':
        param = info['param']
        if 'label' in info.keys():
            system_idx = deepcopy(info['system_idx'][0])
            system_idx = [system_idx[:info['num_ctrl_system']]] + system_idx[info['num_ctrl_system']:]
            cir.param_oracle(info['kwargs']['generator'], param.shape[-1], system_idx,
                                        int(info['label']), param)
        else:
            cir.param_oracle(info['kwargs']['generator'], param.shape[-1],
                                        info['system_idx'], param=param)
    
    else:
        raise ValueError(f"{info['api']} not found in custom gates")
            
def _load_custom_channel(cir: Circuit, info: OperatorInfoType) -> None:
    r"""Load custom channels: choi_channel, kraus_channel, stinespring_channel into a Circuit.
    """
    system_idx = info['system_idx'][0]
    if info['api'] == 'choi_channel':
        cir.choi_channel(info['kwargs']['choi_repr'], system_idx)
    elif info['api'] == 'kraus_channel':
        cir.kraus_channel(info['kwargs']['kraus_repr'], system_idx)
    elif info['api'] == 'stinespring_channel':
        cir.stinespring_channel(info['kwargs']['stinespring_channel'], system_idx)
    else:
        raise ValueError(f"{info['api']} not found in custom channels")
