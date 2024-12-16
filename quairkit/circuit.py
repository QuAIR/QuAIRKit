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

import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.parameter import Parameter

from .ansatz import (ComplexBlockLayer, ComplexEntangledLayer, Layer,
                     LinearEntangledLayer, OperatorList, QAOALayer,
                     RealBlockLayer, RealEntangledLayer, SuperpositionLayer,
                     WeakSuperpositionLayer)
from .core.intrinsic import _alias, _cnot_idx_fetch, _format_circuit_idx
from .core.state import State
from .database import std_basis, zero_state
from .operator import (CCX, CNOT, CP, CRX, CRY, CRZ, CSWAP, CU, CX, CY, CZ, MS,
                       RX, RXX, RY, RYY, RZ, RZZ, SWAP, U3, AmplitudeDamping,
                       BitFlip, BitPhaseFlip, ChoiRepr, Collapse,
                       ControlOracle, Depolarizing, Gate,
                       GeneralizedAmplitudeDamping, GeneralizedDepolarizing, H,
                       KrausRepr, OneWayLOCC, Oracle, P, ParamOracle,
                       PauliChannel, PhaseDamping, PhaseFlip, ResetChannel, S,
                       Sdg, StinespringRepr, T, Tdg, ThermalRelaxation,
                       UniversalThreeQubits, UniversalTwoQubits, X, Y, Z)
from .operator.gate import _circuit_plot
from .operator.gate.custom import UniversalQudits

__all__ = ['Circuit']


class Circuit(OperatorList):
    r"""Quantum circuit.

    Args:
        num_systems: number of systems in the circuit. Defaults to None. Alias of ``num_qubits``.
        system_dim: dimension of systems of this circuit. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    
    Note:
        when the number of system is unknown and system_dim is an int, the circuit is a dynamic quantum circuit.

    """
    @_alias({'num_systems': 'num_qubits'})
    def __init__(self, num_systems: Optional[int] = None, 
                 system_dim: Optional[Union[List[int], int]] = 2):
        super().__init__()
        self.__register_circuit(num_systems, system_dim)

        # alias
        self.toffoli = self.ccx
        self.cx = self.cnot
        self.collapse = self.measure

        # TODO recover the gather logic for CNOT
        # preparing cnot index in 'cycle' case
        # num_qubits = self.num_qubits
        # if num_qubits > 1:
        #     cnot_qubits_idx = _format_system_idx("cycle", num_qubits, num_acted_system=2)
        #     self.__cnot_cycle_idx = _cnot_idx_fetch(num_qubits=num_qubits, qubits_idx=cnot_qubits_idx)

    
    def __register_circuit(self, num_systems: Union[int, None], system_dim: Union[List[int], int]) -> None:
        r"""Register the circuit and determine whether it is dynamic.
        """
        self.__isdynamic = False
        self.__equal_dim = False
        if isinstance(system_dim, int):
            self.__equal_dim = True
            if num_systems is None:
                self.__isdynamic = True
            else:
                system_dim = [system_dim] * num_systems
        elif num_systems is None:
            num_systems = len(system_dim)
        else:
            if len(set(system_dim)) == 1:
                self.__equal_dim = True
            assert num_systems == len(system_dim), \
                f"num_systems and system_dim do not agree: received {num_systems} and {system_dim}"
        self.__num_system = num_systems
        self.__system_dim = system_dim

    @property
    def num_qubits(self) -> int:
        r"""Number of qubits.
        """
        return 0 if isinstance(self.__system_dim, int) else self.__system_dim.count(2)
    
    @property
    def num_qutrits(self) -> int:
        r"""Number of qutrits.
        """
        return 0 if isinstance(self.__system_dim, int) else self.__system_dim.count(3)

    @property
    def isdynamic(self) -> bool:
        r"""Whether the circuit is dynamic
        """
        return self.__dynamic
    
    @property
    def num_systems(self) -> int:
        r"""Number of systems.
        """
        return self.__num_system
    
    @property
    def system_dim(self) -> Union[List[int], int]:
        r"""Dimension of systems.
        """
        return self.__system_dim

    def __info_update(self, systems_idx: Union[Iterable[int], int, str, None], num_acted_system: int) -> List[List[int]]:
        r"""Update circuit information according to input operator information, or report error.

        Args:
            systems_idx: input system indices
            acted_system_dim: dimension of systems that one operator acts on.
        
        Returns:
            the formatted system indices.
        
        """
        num_systems = self.num_systems

        if systems_idx is None or isinstance(systems_idx, str):
            assert self.__equal_dim, \
                    f"The circuit's systems have different dimensions. Invalid input qubit idx: {systems_idx}"
        if systems_idx is None:
            return systems_idx

        systems_idx = _format_circuit_idx(systems_idx, num_systems, num_acted_system)
        max_idx = np.max(systems_idx)

        if num_systems is None:
            self.__num_system = max_idx + 1
            self.__system_dim = [self.__system_dim]
            return systems_idx

        if self.__isdynamic:
            if max_idx >= num_systems:
                self.__num_system = max_idx + 1
                self.__system_dim = [self.__system_dim[0]] * self.__num_system
        else:
            assert max_idx < num_systems, (
                "The circuit is not a dynamic quantum circuit. "
                f"Invalid input system idx: {max_idx} for a circuit with {self.__num_system} systems.")
        return systems_idx
    
    def unitary_matrix(self) -> torch.Tensor:
        r"""Get the unitary matrix form of the circuit.

        Returns:
            Unitary matrix form of the circuit.
        """
        dim = int(np.prod(self.__system_dim))
        input_basis = std_basis(self.__num_system, self.__system_dim)
        input_basis._switch_unitary_matrix = True
        output = self.forward(input_basis)
        
        assert output.backend == 'state_vector', \
            f"The circuit seems to be a noisy circuit: expect 'state_vector', output {output_basis.backend}"
        input_basis = input_basis.bra.view([dim, 1, 1, dim])
        output_basis = output.ket.view([dim, -1, dim, 1])
        return torch.sum(output_basis @ input_basis, dim=0).view(output.batch_dim[1:] + [dim, dim])


    @property
    def gate_history(self) -> List[Dict[str, Union[str, List[int], torch.Tensor]]]:
        r"""List of gates information of circuit

        Returns:
            history of quantum gates of circuit

        """
        gate_history = []
        for op in self.children():
            if isinstance(op, Layer):
                gate_history.extend(op.gate_history)
            else:
                if op.gate_info['gatename'] is None:
                    raise NotImplementedError(
                        f"{type(op)} has no gate name and hence cannot be recorded into history.")
                op.gate_history_generation()
                gate_history.extend(op.gate_history)
        return gate_history

    @property
    def depth(self) -> int:
        r"""Depth of gate sequences.
        
        Returns:
            depth of this circuit
        
        Note:
            The measurement is omitted, and all gates are assumed to have depth 1. 
            See Niel's answer in the [StackExchange](https://quantumcomputing.stackexchange.com/a/5772).
        
        """
        system_depth = np.array([0] * self.num_systems)
        for gate_info in self.gate_history:
            sys_idx = gate_info['which_system']
            if isinstance(sys_idx, int):
                system_depth[sys_idx] += 1
            else:
                system_depth[sys_idx] = np.max(system_depth[sys_idx]) + 1
        return int(np.max(system_depth))

    def __count_history(self, history):
        # Record length of each section
        length = [5]
        n = self.__num_system
        # Record current section number for every system
        system = [0] * n
        # Number of sections
        system_max = max(system)
        # Record section number for each gate
        gate = []
        for current_gate in history:
            # Single-qubit gates with no params to print
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u', 'sdg', 'tdg'}:
                curr_system = current_gate['which_system']
                gate.append(system[curr_system])
                system[curr_system] = system[curr_system] + 1
                # A new section is added
                if system[curr_system] > system_max:
                    length.append(5)
                    system_max = system[curr_system]
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                curr_system = current_gate['which_system']
                gate.append(system[curr_system])
                if length[system[curr_system]] == 5:
                    length[system[curr_system]] = 13
                system[curr_system] = system[curr_system] + 1
                if system[curr_system] > system_max:
                    length.append(5)
                    system_max = system[curr_system]
            elif current_gate['gate'] in {
                'cnot', 'swap', 'rxx', 'ryy', 'rzz', 'ms',
                'cy', 'cz', 'cu', 'cp', 'crx', 'cry', 'crz', 'cswap', 'ccx'
            }:
                a = max(current_gate['which_system'])
                b = min(current_gate['which_system'])
                ind = max(system[b: a + 1])
                gate.append(ind)
                if length[ind] < 13 and current_gate['gate'] in {
                    'rxx', 'ryy', 'rzz', 'cp', 'crx', 'cry', 'crz'
                }:
                    length[ind] = 13
                for j in range(b, a + 1):
                    system[j] = ind + 1
                if ind + 1 > system_max:
                    length.append(5)
                    system_max = ind + 1

        return length, gate

    @property
    def system_history(self) -> List[List[Tuple[Dict[str, Union[str, List[int], torch.Tensor]], int]]]:
        r""" gate information on each system

        Returns:
            list of gate history on each system

        Note:
            The entry ``system_history[i][j][0/1]`` returns the gate information / gate index of the j-th gate
            applied on the i-th system.
        """
        history_system = [[] for _ in range(self.num_qubits)]
        for idx, i in enumerate(self.gate_history):
            systems = i["which_system"]
            if not isinstance(systems, Iterable):
                history_system[systems].append([i, idx])
            else:
                for j in systems:
                    history_system[j].append([i, idx])
        return history_system
    
    def plot(
            self,
            save_path: Optional[str] = None,
            dpi: Optional[int] = 100,
            show: Optional[bool] = True,
            output: Optional[bool] = False,
            scale: Optional[float] = 1.0,
            tex: Optional[bool] = False,
    ) -> Union[None, matplotlib.figure.Figure]:
        r'''display the circuit using matplotlib

        Args:
            save_path: the save path of image
            dpi: dots per inches, here is resolution ratio
            show: whether execute ``plt.show()``
            output: whether return the ``matplotlib.figure.Figure`` instance
            scale: scale coefficient of figure, default to 1.0
            tex: a bool flag which controls latex fonts of gate display, default to ``False``.

        Returns:
            a ``matplotlib.figure.Figure`` instance or ``None`` depends on ``output``

        Note:
            Using ``plt.show()`` may cause a distortion, but it will not happen in the figure saved.
            If the depth is too long, there will be some patches unable to display.
            Setting ``tex = True`` requires that you have TeX and the other dependencies properly 
            installed on your system. See 
            https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
            for more details.
        '''
        _fig = _circuit_plot(self, dpi=dpi, scale=scale, tex=tex)
        if save_path:
            plt.savefig(save_path, dpi=dpi, )
        if show:  # whether display in window
            plt.show()
        if output:
            return _fig  # return the ``matplotlib.pyplot.figure`` instance

    def extend(self, cir: Union['Circuit', OperatorList]) -> None:
        r""" extend for quantum circuit

        Args:
            cir: a Circuit or a OperatorList

        Returns:
            concatenation of two quantum circuits
        """
        if isinstance(cir, Circuit):
            if self.__num_system is None:
                self.__num_system = cir.num_qubits
            else:
                self.__num_system = self.__num_system if cir.num_qubits is None else max(self.__num_system,
                                                                                         cir.num_qubits)
            super().extend(cir)
        elif isinstance(cir, OperatorList):
            super().extend(cir)
        else:
            raise TypeError("the input type must be Circuit or OperatorList")
        
    
    def __call__(self, state: Optional[State] = None) -> State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    

    def forward(self, state: Optional[State] = None) -> State:
        r""" forward the input

        Args:
            state: initial state

        Returns:
            output quantum state
        """
        if state is None:
            state = zero_state(self.__num_system, self.__system_dim)
        
        if self.__num_system is None:
            warnings.warn("The circuit is empty: return the input state.", UserWarning)
            return state

        assert (
            self.__system_dim == state.system_dim
        ), f"System dimension of circuit and state does not agree: expected {self.__system_dim}, received {state.system_dim}"
            

        # TODO recover QPU history
        # if self.backend == Backend.QPU and state.backend == Backend.QPU:
        #     state.oper_history = self.oper_history
        #     return state

        state = state.clone()
        state = super().forward(state)
        return state


    # ---------------------- below are common operators ---------------------- #


    def h(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit Hadamard gates.

        The matrix form of such a gate is:

        .. math::

            H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1&1\\
                    1&-1
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(H(qubits_idx))

    def s(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit S gates.

        The matrix form of such a gate is:

        .. math::

            S =
                \begin{bmatrix}
                    1&0\\
                    0&i
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(S(qubits_idx))

    def sdg(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit S dagger (S inverse) gates.

        The matrix form of such a gate is:

        .. math::

            S^\dagger =
                \begin{bmatrix}
                    1&0\\
                    0&-i
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(Sdg(qubits_idx))

    def t(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit T gates.

        The matrix form of such a gate is:

        .. math::

            T = \begin{bmatrix}
                    1&0\\
                    0&e^\frac{i\pi}{4}
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(T(qubits_idx))

    def tdg(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit T dagger (T inverse) gates.

        The matrix form of such a gate is:

        .. math::

            T^\dagger =
                \begin{bmatrix}
                    1&0\\
                    0&e^{-\frac{i\pi}{4}}
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(Tdg(qubits_idx))

    def x(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit X gates.

        The matrix form of such a gate is:

        .. math::
           X = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(X(qubits_idx))

    def y(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit Y gates.

        The matrix form of such a gate is:

        .. math::

            Y = \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(Y(qubits_idx))

    def z(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add single-qubit Z gates.

        The matrix form of such a gate is:

        .. math::

            Z = \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(Z(qubits_idx))

    def p(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
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
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(P(qubits_idx, param, param_sharing))

    def rx(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
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
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(RX(qubits_idx, param, param_sharing))

    def ry(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
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
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(RY(qubits_idx, param, param_sharing))

    def rz(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates about the z-axis.

        The matrix form of such a gate is:

        .. math::

            R_Z(\theta) = \begin{bmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(RZ(qubits_idx, param, param_sharing))

    def u3(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full',
            param: Union[torch.Tensor, Iterable[float]] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                U_3(\theta, \phi, \lambda) =
                    \begin{bmatrix}
                        \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                        e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(U3(qubits_idx, param, param_sharing))

    def cnot(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle'
    ) -> None:
        r"""Add CNOT gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        if qubits_idx == "cycle" and not self.__isdynamic:
            self.append(CNOT(qubits_idx, self.__cnot_cycle_idx))
        else:
            self.append(CNOT(qubits_idx))

    def cy(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle'
    ) -> None:
        r"""Add controlled Y gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CY} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -1j \\
                    0 & 0 & 1j & 0
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CY(qubits_idx))

    def cz(
            self, qubits_idx: Union[Iterable[int], str] = 'linear'
    ) -> None:
        r"""Add controlled Z gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CZ(qubits_idx))

    def swap(
            self, qubits_idx: Union[Iterable[int], str] = 'linear'
    ) -> None:
        r"""Add SWAP gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(SWAP(qubits_idx))

    def cp(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled P gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CP(qubits_idx, param, param_sharing))

    def crx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the x-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CRX(qubits_idx, param, param_sharing))

    def cry(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the y-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CRY(qubits_idx, param, param_sharing))

    def crz(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the z-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CRZ(qubits_idx, param, param_sharing))

    def cu(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled single-qubit rotation gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CU}
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
                    0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CU(qubits_idx, param, param_sharing))

    def rxx(
            self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RXX gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(RXX(qubits_idx, param, param_sharing))

    def ryy(
            self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RYY gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(RYY(qubits_idx, param, param_sharing))

    def rzz(
            self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RZZ gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(RZZ(qubits_idx, param, param_sharing))

    def ms(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle'
    ) -> None:
        r"""Add Mølmer-Sørensen (MS) gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(MS(qubits_idx))

    def cswap(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle'
    ) -> None:
        r"""Add CSWAP (Fredkin) gates.

        The matrix form of such a gate is:

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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(CSWAP(qubits_idx))

    def ccx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle'
    ) -> None:
        r"""Add CCX (Toffoli) gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                    \mathit{CCX} = \begin{bmatrix}
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
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
        
        """
        qubits_idx = self.__info_update(qubits_idx, 3)
        self.append(CCX(qubits_idx))

    def universal_two_qubits(
            self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add universal two-qubit gates. One of such a gate requires 15 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        qubits_idx = self.__info_update(qubits_idx, 2)
        self.append(UniversalTwoQubits(
            qubits_idx, param, param_sharing))

    def universal_three_qubits(
            self, qubits_idx: Union[Iterable[int], str] = 'linear',
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add universal three-qubit gates. One of such a gate requires 81 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Raises:
            ValueError: The ``param`` must be torch.Tensor or float.
        """
        qubits_idx = self.__info_update(qubits_idx, 3)
        self.append(UniversalThreeQubits(
            qubits_idx, param, param_sharing))
    
    @_alias({"system_idx": "qubits_idx"})
    def universal_qudits(self, system_idx: List[int],
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add universal qudit gates. One of such a gate requires :math:`d^2 - 1` parameters,
        where :math:`d` is the gate dimension.

        Args:
            system_idx: Indices of the systems on which the gates are applied. Defaults to 'linear'.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Raises:
            ValueError: The ``param`` must be torch.Tensor or float.
        """
        system_idx = self.__info_update(system_idx, 1 if isinstance(system_idx, int) else len(system_idx))
        self.append(UniversalQudits(system_idx, [self.__system_dim[idx] for idx in system_idx],
                                    param, param_sharing))

    @_alias({"system_idx": "qubits_idx"})
    def oracle(
            self, oracle: torch.Tensor, system_idx: Union[List[int], int],
            gate_name: Optional[str] = 'O', latex_name: Optional[str] = None, plot_width: Optional[float] = None
    ) -> None:
        r"""Add an oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            system_idx: Indices of the systems on which the gates are applied.
            gate_name: name of this oracle.
            latex_name: latex name of this oracle, default to be the gate name.
            plot_width: width of this gate in circuit plot, default to be proportional with the gate name.
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.__info_update(system_idx, len(system_idx))
        gate_info = {
            'gatename': gate_name,
            'texname': f"${gate_name}$" if latex_name is None else latex_name,
            'plot_width': 0.6 * len(gate_name) if plot_width is None else plot_width}
        
        acted_system_dim = [self.__system_dim[idx] for idx in system_idx]
        self.append(Oracle(
            oracle, system_idx, acted_system_dim, gate_info))

    @_alias({"system_idx": "qubits_idx"})
    def control_oracle(
            self, oracle: torch.Tensor, system_idx: List[Union[List[int], int]], proj: Union[torch.Tensor] = None,
            gate_name: Optional[str] = 'O', latex_name: Optional[str] = None, plot_width: Optional[float] = None
    ) -> None:
        r"""Add a controlled oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            system_idx: Indices of the systems on which the gates are applied. The first element in the list is the control system, 
                defaulting to the $|d-1\rangle \langle d-1|$ state as the control qubit, 
                while the remaining elements represent the oracle system.
            proj: Projector matrix for the control qubit. Defaults to ``None``
            gate_name: name of this oracle.
            latex_name: latex name of this oracle, default to be the gate name.
            plot_width: width of this gate in circuit plot, default to be proportional with the gate name.
        """
        _system_idx = sum(([item] if isinstance(item, int) else item for item in system_idx), [])
        _system_idx = self.__info_update(_system_idx, len(_system_idx))
        gate_info = {
            'gatename': f"c{gate_name}",
            'texname': f"${gate_name}$" if latex_name is None else latex_name,
            'plot_width': 0.6 * len(gate_name) if plot_width is None else plot_width}
        
        acted_system_dim = [self.__system_dim[idx] for idx in _system_idx]
        self.append(ControlOracle(
            oracle, system_idx, acted_system_dim, proj, gate_info))
        
    @_alias({"system_idx": "qubits_idx"})
    def param_oracle(self, generator: Callable[[torch.Tensor], torch.Tensor], num_acted_param: int,
                     system_idx: Union[List[int], int], param: Union[torch.Tensor, float] = None,
                     gate_name: Optional[str] = 'P', latex_name: Optional[str] = None, plot_width: Optional[float] = None
        ) -> None:
        r"""Add a parameterized oracle gate.
        
        Args:
            generator: function that generates the oracle.
            num_acted_param: the number of parameters required for a single operation.
            system_idx: indices of the system on which this gate acts on.
            param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
            gate_name: name of this oracle.
            latex_name: latex name of this oracle, default to be the gate name.
            plot_width: width of this gate in circuit plot, default to be proportional with the gate name.
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.__info_update(system_idx, len(system_idx))
        
        gate_info = {
            'gatename': gate_name,
            'texname': f"${gate_name}$" if latex_name is None else latex_name,
            'plot_width': 1 * len(gate_name) if plot_width is None else plot_width}
        acted_system_dim = [self.__system_dim[idx] for idx in system_idx]
        self.append(ParamOracle(
            generator, param, num_acted_param, False, system_idx, acted_system_dim, gate_info))

    @_alias({"system_idx": "qubits_idx", "post_selection": "desired_result"})
    def measure(self, system_idx: Union[Iterable[int], int, str] = None,
                post_selection: Union[int, str] = None, if_print: bool = False,
                measure_basis: Optional[torch.Tensor] = None) -> None:
        r"""
        Args:
            system_idx: list of systems to be measured. Defaults to all qubits.
            post_selection: the post selection result after measurement. Defaults to ``None`` meaning preserving all measurement outcomes.
            if_print: whether print the information about the collapsed state. Defaults to ``False``.
            measure_basis: The basis of the measurement. The quantum state will collapse to the corresponding eigenstate.

        Note:
            When desired_result is `None`, collapse is equivalent to mid-circuit measurement.
        
        """
        system_idx = self.__info_update(system_idx, None)
        self.append(Collapse(
            system_idx, post_selection, if_print, measure_basis))

    def superposition_layer(
            self, qubits_idx: Iterable[int] = None
    ) -> None:
        r"""Add layers of Hadamard gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to all qubits.
        
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            SuperpositionLayer(qubits_idx, self.num_qubits))

    def weak_superposition_layer(
            self, qubits_idx: Iterable[int] = None
    ) -> None:
        r"""Add layers of Ry gates with a rotation angle :math:`\pi/4`.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to all qubits.
        
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            WeakSuperpositionLayer(qubits_idx, self.num_qubits))

    def linear_entangled_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to all qubits.
            depth: Number of layers. Defaults to 1.
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            LinearEntangledLayer(qubits_idx, self.num_qubits, depth))

    def real_entangled_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add strongly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
            depth: Number of layers. Defaults to 1.
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            RealEntangledLayer(qubits_idx, self.num_qubits, depth))

    def complex_entangled_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
            depth: Number of layers. Defaults to 1.
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            ComplexEntangledLayer(qubits_idx, self.num_qubits, depth))

    def real_block_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add weakly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
            depth: Number of layers. Defaults to 1.
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            RealBlockLayer(qubits_idx, self.num_qubits, depth))

    def complex_block_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add weakly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
            depth: Number of layers. Defaults to 1.
        """
        qubits_idx = self.__info_update(qubits_idx, None)
        self.extend(
            ComplexBlockLayer(qubits_idx, self.num_qubits, depth))

    def qaoa_layer(self, edges: Iterable, nodes: Iterable, depth: Optional[int] = 1) -> None:
        # TODO: see qaoa layer in layer.py
        self.__info_update(edges, None)
        self.__info_update(nodes, None)
        self.extend(QAOALayer(edges, nodes, depth))

    def bit_flip(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add bit flip channels.

        Args:
            prob: Probability of a bit flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(BitFlip(prob, qubits_idx))

    def phase_flip(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add phase flip channels.

        Args:
            prob: Probability of a phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(PhaseFlip(prob, qubits_idx))

    def bit_phase_flip(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add bit phase flip channels.

        Args:
            prob: Probability of a bit phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(BitPhaseFlip(prob, qubits_idx))

    def amplitude_damping(
            self, gamma: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add amplitude damping channels.

        Args:
            gamma: Damping probability.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(AmplitudeDamping(gamma, qubits_idx))

    def generalized_amplitude_damping(
            self, gamma: Union[torch.Tensor, float], prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add generalized amplitude damping channels.

        Args:
            gamma: Damping probability. Its value should be in the range :math:`[0, 1]`.
            prob: Excitation probability. Its value should be in the range :math:`[0, 1]`.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(GeneralizedAmplitudeDamping(
            gamma, prob, qubits_idx))

    def phase_damping(
            self, gamma: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add phase damping channels.

        Args:
            gamma: Parameter of the phase damping channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(PhaseDamping(gamma, qubits_idx))

    def depolarizing(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add depolarizing channels.

        Args:
            prob: Parameter of the depolarizing channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(Depolarizing(prob, qubits_idx))

    def generalized_depolarizing(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str]
    ) -> None:
        r"""Add a general depolarizing channel.

        Args:
            prob: Probabilities corresponding to the Pauli basis.
            qubits_idx: Indices of the qubits on which the channel is applied.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(GeneralizedDepolarizing(prob, qubits_idx))

    def pauli_channel(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add Pauli channels.

        Args:
            prob: Probabilities corresponding to the Pauli X, Y, and Z operators.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(PauliChannel(prob, qubits_idx))

    def reset_channel(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add reset channels.

        Args:
            prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(ResetChannel(prob, qubits_idx))

    def thermal_relaxation(
            self, const_t: Union[torch.Tensor, Iterable[float]], exec_time: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add thermal relaxation channels.

        Args:
            const_t: :math:`T_1` and :math:`T_2` relaxation time in microseconds.
            exec_time: Quantum gate execution time in the process of relaxation in nanoseconds.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        qubits_idx = self.__info_update(qubits_idx, 1)
        self.append(
            ThermalRelaxation(const_t, exec_time, qubits_idx))

    def choi_channel(
            self, choi_repr: Iterable[torch.Tensor],
            system_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Choi representation.

        Args:
            choi_repr: Choi representation of this channel.
            system_idx: Indices of the systems on which the channels are applied.
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.__info_update(system_idx, len(system_idx))
        acted_system_dim = [self.__system_dim[idx] for idx in system_idx]
        self.append(ChoiRepr(choi_repr, system_idx, acted_system_dim))

    def kraus_channel(
            self, kraus_oper: Iterable[torch.Tensor],
            system_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Kraus representation.

        Args:
            kraus_oper: Kraus representation of this channel.
            system_idx: Indices of the systems on which the channels are applied.
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.__info_update(system_idx, len(system_idx))
        acted_system_dim = [self.__system_dim[idx] for idx in system_idx]
        self.append(KrausRepr(kraus_oper, system_idx, acted_system_dim))

    def stinespring_channel(
            self, stinespring_repr: Iterable[torch.Tensor],
            system_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Stinespring representation.

        Args:
            stinespring_repr: Stinespring representation of this channel.
            system_idx: Indices of the systems on which the channels are applied.
        """
        system_idx = [system_idx] if isinstance(system_idx, int) else system_idx
        system_idx = self.__info_update(system_idx, len(system_idx))
        acted_system_dim = [self.__system_dim[idx] for idx in system_idx]
        self.append(StinespringRepr(stinespring_repr, system_idx, acted_system_dim))

    def locc(self, local_unitary: torch.Tensor, system_idx: List[Union[List[int], int]]) -> None:
        r"""Add a one-way local operation and classical communication (LOCC) protocol comprised of unitary operations.

        Args:
            measure_idx: Indices of the measured systems.
            system_idx: Indices of the systems on which the protocol is applied. The first element represents the measure system(s) and the remaining elements represent the local system(s).
        
        """
        _system_idx = (list(system_idx[0]) if isinstance(system_idx[0], int) else system_idx[0]) + system_idx[1:]
        _system_idx = self.__info_update(_system_idx, len(_system_idx))
        
        acted_system_dim = [self.__system_dim[idx] for idx in _system_idx]
        self.append(OneWayLOCC(local_unitary, system_idx, acted_system_dim))
    
    def __str__(self):
        history = self.gate_history
        num_systems = self.__num_system
        length, gate = self.__count_history(history)
        # Ignore the unused section
        total_length = sum(length) - 5
        print_list = [['-' if i % 2 == 0 else ' '] *
                      total_length for i in range(num_systems * 2)]
        for i, current_gate in enumerate(history):
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u'}:
                # Calculate starting position ind of current gate
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_system'] * 2][ind +
                                                             length[sec] // 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'sdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_system'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'tdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_system'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                line = current_gate['which_system'] * 2
                # param = self.__param[current_gate['theta'][2 if current_gate['gate'] == 'rz' else 0]]
                param = current_gate['theta']
                if current_gate['gate'] == 'p':
                    print_list[line][ind + 2] = 'P'
                    print_list[line][ind + 3] = ' '
                else:
                    print_list[line][ind + 2] = 'R'
                    print_list[line][ind + 3] = current_gate['gate'][1]
                print_list[line][ind + 4] = '('
                print_list[line][ind + 5: ind +
                                 10] = format(float(param.detach().numpy()), '.3f')[:5]
                print_list[line][ind + 10] = ')'
            # Two-qubit gates
            elif current_gate['gate'] in {'cnot', 'swap', 'rxx', 'ryy', 'rzz', 'ms', 'cz', 'cy',
                                          'cu', 'cp', 'crx', 'cry', 'crz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_system'][0]
                tqubit = current_gate['which_system'][1]
                if current_gate['gate'] in {'cnot', 'swap', 'cy', 'cz', 'cu'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = \
                        '*' if current_gate['gate'] in {'cnot',
                                                        'cy', 'cz', 'cu'} else 'x'
                    print_list[tqubit * 2][ind + length[sec] // 2] = \
                        'x' if current_gate['gate'] in {
                            'swap', 'cnot'} else current_gate['gate'][1]
                elif current_gate['gate'] == 'ms':
                    for qubit in {cqubit, tqubit}:
                        print_list[qubit * 2][ind + length[sec] // 2 - 1] = 'M'
                        print_list[qubit * 2][ind + length[sec] // 2] = '_'
                        print_list[qubit * 2][ind + length[sec] // 2 + 1] = 'S'
                elif current_gate['gate'] in {'rxx', 'ryy', 'rzz'}:
                    # param = self.__param[current_gate['theta'][0]]
                    param = current_gate['theta']
                    for line in {cqubit * 2, tqubit * 2}:
                        print_list[line][ind + 2] = 'R'
                        print_list[line][ind + 3: ind +
                                         5] = current_gate['gate'][1:3].lower()
                        print_list[line][ind + 5] = '('
                        print_list[line][ind + 6: ind +
                                         10] = format(float(param.detach().numpy()), '.2f')[:4]
                        print_list[line][ind + 10] = ')'
                elif current_gate['gate'] in {'crx', 'cry', 'crz'}:
                    param = current_gate['theta']
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + 2] = 'R'
                    print_list[tqubit * 2][ind + 3] = current_gate['gate'][2]
                    print_list[tqubit * 2][ind + 4] = '('
                    print_list[tqubit * 2][ind + 5: ind +
                                           10] = format(float(param.detach().numpy()), '.3f')[:5]
                    print_list[tqubit * 2][ind + 10] = ')'
                elif current_gate['gate'] == 'cp':
                    param = current_gate['theta']
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + 2] = ' '
                    print_list[tqubit * 2][ind + 3] = 'P'
                    print_list[tqubit * 2][ind + 4] = '('
                    print_list[tqubit * 2][ind + 5: ind +
                                           10] = format(float(param.detach().numpy()), '.3f')[:5]
                    print_list[tqubit * 2][ind + 10] = ')'
                    
                start_line = min(cqubit, tqubit)
                end_line = max(cqubit, tqubit)
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
            # Three-qubit gates
            elif current_gate['gate'] in {'cswap'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_system'][0]
                tqubit1 = current_gate['which_system'][1]
                tqubit2 = current_gate['which_system'][2]
                start_line = min(current_gate['which_system'])
                end_line = max(current_gate['which_system'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'cswap'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit1 * 2][ind + length[sec] // 2] = 'x'
                    print_list[tqubit2 * 2][ind + length[sec] // 2] = 'x'
            elif current_gate['gate'] in {'ccx'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit1 = current_gate['which_system'][0]
                cqubit2 = current_gate['which_system'][1]
                tqubit = current_gate['which_system'][2]
                start_line = min(current_gate['which_system'])
                end_line = max(current_gate['which_system'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'ccx'}:
                    print_list[cqubit1 * 2][ind + length[sec] // 2] = '*'
                    print_list[cqubit2 * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + length[sec] // 2] = 'X'
            else:
                raise NotImplementedError(
                    f"Not support to print the gate {current_gate['gate']}.")

        print_list = list(map(''.join, print_list))
        return_str = '\n'.join(print_list)

        return return_str
