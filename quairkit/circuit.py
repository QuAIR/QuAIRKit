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
from math import pi
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.parameter import Parameter

from .ansatz import (ComplexBlockLayer, ComplexEntangledLayer, Layer,
                     LinearEntangledLayer, OperatorList, QAOALayer,
                     RealBlockLayer, RealEntangledLayer, SuperpositionLayer,
                     WeakSuperpositionLayer)
from .core import Backend, get_backend, get_dtype, get_float_dtype
from .core.intrinsic import _cnot_idx_fetch, _format_qubits_idx
from .core.state import State
from .database import std_basis, zero_state
from .operator import (CCX, CNOT, CP, CRX, CRY, CRZ, CSWAP, CU, CX, CY, CZ, MS,
                       RX, RXX, RY, RYY, RZ, RZZ, SWAP, U3, AmplitudeDamping,
                       BitFlip, BitPhaseFlip, ChoiRepr, Collapse,
                       ControlOracle, Depolarizing, Gate,
                       GeneralizedAmplitudeDamping, GeneralizedDepolarizing, H,
                       KrausRepr, Oracle, P, PauliChannel, PhaseDamping,
                       PhaseFlip, ResetChannel, S, Sdg, StinespringRepr, T,
                       Tdg, ThermalRelaxation, UniversalQudits,
                       UniversalThreeQubits, UniversalTwoQubits, X, Y, Z)
from .operator.gate import _circuit_plot


class Circuit(OperatorList):
    r"""Quantum circuit.

    Args:
        num_qubits: Number of qubits. Defaults to None.
    """

    def __init__(self, num_qubits: Optional[int] = None):
        super().__init__()
        self.__num_qubits = num_qubits

        # whether the circuit is a dynamic quantum circuit
        self.__isdynamic = num_qubits is None

        # alias
        self.toffoli = self.ccx
        self.cx = self.cnot

        # preparing cnot index in 'cycle' case
        if num_qubits is not None and num_qubits > 1:
            cnot_qubits_idx = _format_qubits_idx("cycle", num_qubits, num_acted_qubits=2)
            self.__cnot_cycle_idx = _cnot_idx_fetch(num_qubits=num_qubits, qubits_idx=cnot_qubits_idx)


    @property
    def num_qubits(self) -> int:
        r"""Number of qubits.
        """
        return self.__num_qubits

    @property
    def isdynamic(self) -> bool:
        r"""Whether the circuit is dynamic
        """
        return self.__dynamic

    @num_qubits.setter
    def num_qubits(self, value: int) -> None:
        assert isinstance(value, int)
        self.__num_qubits = value

    @property
    def param(self) -> torch.Tensor:
        r"""Flattened parameters in the circuit.
        """
        assert self._modules, \
                "The circuit is empty, please add some operators first."
        if flattened_params := [
            torch.flatten(param.clone()) for param in self.parameters()
        ]:
            concatenated_params = torch.cat(flattened_params).detach()
        else:
            concatenated_params = torch.tensor([])
        return concatenated_params

    @property
    def grad(self) -> np.ndarray:
        r"""Gradients with respect to the flattened parameters.
        """
        assert self._modules, \
            "The circuit is empty, please add some operators first."
        grad_list = []
        for param in self.parameters():
            assert param.grad is not None, (
                'The gradient is None, run the backward first before calling this property, '
                'otherwise check where the gradient chain is broken.')
            grad_list.append(param.grad.detach().numpy().flatten())
        return np.concatenate(grad_list) if grad_list != [] else grad_list

    def update_param(self, theta: Union[torch.Tensor, np.ndarray, float], idx: int = None) -> None:
        r"""Replace parameters of all/one layer(s) by ``theta``.

        Args:
            theta: New parameters
            idx: Index of replacement. Defaults to None, referring to all layers.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        theta = torch.flatten(theta).to(dtype=get_float_dtype())

        if idx is None:
            assert self.param.shape == theta.shape, \
                f"the shape of input parameters is not correct: expect {self.param.shape}, received {theta.shape}"
            for layer in self:
                for name, param in layer.named_parameters():
                    num_param = int(torch.numel(param))
                    layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))

                    if num_param == theta.shape[0]:
                        return
                    theta = theta[num_param:]
        elif isinstance(idx, int):
            assert idx < len(self), f"the index is out of range, expect below {len(self)}"

            layer = self[idx]
            assert theta.shape == torch.cat([torch.flatten(param) for param in layer.parameters()]).shape, (
                "The shape of input parameters is not correct.")

            for name, param in layer.named_parameters():
                num_param = int(torch.numel(param))
                layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))
                
                if num_param == theta.shape[0]:
                    return
                theta = theta[num_param:]
        else:
            raise ValueError("idx must be an integer or None")

    def transfer_static(self) -> None:
        r"""
        set ``stop_gradient`` of all parameters of the circuit as ``True``
        """
        for layer in self:
            for name, param in layer.named_parameters():
                param.requires_grad = False
                layer.register_parameter(name, param)

    def randomize_param(self, arg0: float = 0, arg1: float = 2 * pi, initializer_type: str= 'Uniform') -> None:
        r"""Randomize parameters of the circuit based on the initializer.  Current we only support Uniform and Normal initializer. 

        Args:
            arg0: first argument of the initializer. Defaults to 0.
            arg1: first argument of the initializer. Defaults to 2 * pi.
            initializer_type: The type of the initializer. Defaults to 'Uniform'.
        """
        assert initializer_type in {
            "Uniform",
            "Normal",
        }, "The initializer should be Uniform or Normal."

        for layer in self:
            for name, param in layer.named_parameters():
                
                param = Parameter(param.normal_(mean=arg0, std=arg1))
                
                if initializer_type == 'Normal':
                    param = Parameter(param.normal_(mean=arg0, std=arg1))
                elif initializer_type == 'Uniform':
                    param = Parameter(param.uniform_(mean=arg0, std=arg1))
                else:
                    raise NotImplementedError
                
                layer.register_parameter(name, param)

    def __num_qubits_update(self, qubits_idx: Union[Iterable[int], int, str]) -> None:
        r"""Update ``self.num_qubits`` according to ``qubits_idx``, or report error.

        Args:
            qubits_idx: Input qubit indices of a quantum gate.
        """
        num_qubits = self.__num_qubits
        if isinstance(qubits_idx, str) or qubits_idx is None:
            assert num_qubits is not None, \
                f"The qubit idx cannot be a string or None when the number of qubits is unknown: received {qubits_idx}"
            return

        if isinstance(qubits_idx, Iterable):
            max_idx = np.max(qubits_idx)
        else:
            max_idx = qubits_idx

        if num_qubits is None:
            self.__num_qubits = max_idx + 1
            return

        assert max_idx + 1 <= num_qubits or self.__isdynamic, (
            "The circuit is not a dynamic quantum circuit. "
            f"Invalid input qubit idx: {max_idx} num_qubit: {self.__num_qubits}")
        self.__num_qubits = int(max(max_idx + 1, num_qubits))
    
    def unitary_matrix(self) -> torch.Tensor:
        r"""Get the unitary matrix form of the circuit.

        Returns:
            Unitary matrix form of the circuit.
        """
        dim = 2 ** self.num_qubits
        input_basis = std_basis(self.num_qubits)
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
        qubit_depth = np.array([0] * self.num_qubits)
        for gate_info in self.gate_history:
            qubits_idx = gate_info['which_qubits']
            if isinstance(qubits_idx, int):
                qubit_depth[qubits_idx] += 1
            else:
                qubit_depth[qubits_idx] = np.max(qubit_depth[qubits_idx]) + 1
        return int(np.max(qubit_depth))

    def __count_history(self, history):
        # Record length of each section
        length = [5]
        n = self.__num_qubits
        # Record current section number for every qubit
        qubit = [0] * n
        # Number of sections
        qubit_max = max(qubit)
        # Record section number for each gate
        gate = []
        for current_gate in history:
            # Single-qubit gates with no params to print
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u', 'sdg', 'tdg'}:
                curr_qubit = current_gate['which_qubits']
                gate.append(qubit[curr_qubit])
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                # A new section is added
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                curr_qubit = current_gate['which_qubits']
                gate.append(qubit[curr_qubit])
                if length[qubit[curr_qubit]] == 5:
                    length[qubit[curr_qubit]] = 13
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            elif current_gate['gate'] in {
                'cnot', 'swap', 'rxx', 'ryy', 'rzz', 'ms',
                'cy', 'cz', 'cu', 'cp', 'crx', 'cry', 'crz', 'cswap', 'ccx'
            }:
                a = max(current_gate['which_qubits'])
                b = min(current_gate['which_qubits'])
                ind = max(qubit[b: a + 1])
                gate.append(ind)
                if length[ind] < 13 and current_gate['gate'] in {
                    'rxx', 'ryy', 'rzz', 'cp', 'crx', 'cry', 'crz'
                }:
                    length[ind] = 13
                for j in range(b, a + 1):
                    qubit[j] = ind + 1
                if ind + 1 > qubit_max:
                    length.append(5)
                    qubit_max = ind + 1

        return length, gate

    @property
    def qubit_history(self) -> List[List[Tuple[Dict[str, Union[str, List[int], torch.Tensor]], int]]]:
        r""" gate information on each qubit

        Returns:
            list of gate history on each qubit

        Note:
            The entry ``qubit_history[i][j][0/1]`` returns the gate information / gate index of the j-th gate
            applied on the i-th qubit.
        """
        history_qubit = [[] for _ in range(self.num_qubits)]
        for idx, i in enumerate(self.gate_history):
            qubits = i["which_qubits"]
            if not isinstance(qubits, Iterable):
                history_qubit[qubits].append([i, idx])
            else:
                for j in qubits:
                    history_qubit[j].append([i, idx])
        return history_qubit
    
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

    def extend(self, cir):
        r""" extend for quantum circuit

        Args:
            cir: a Circuit or a Sequential

        Returns:
            concatenation of two quantum circuits
        """
        if isinstance(cir, Circuit):
            if self.__num_qubits is None:
                self.__num_qubits = cir.num_qubits
            else:
                self.__num_qubits = self.__num_qubits if cir.num_qubits is None else max(self.__num_qubits,
                                                                                         cir.num_qubits)
            super().extend(cir)
        elif isinstance(cir, OperatorList):
            super().extend(cir)
        else:
            raise TypeError("the input type must be Circuit or Sequential")
        
    
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
        assert self.__num_qubits is not None, "Information about num_qubits is required before running the circuit"

        if state is None:
            state = zero_state(self.__num_qubits)
        else:
            assert self.__num_qubits == state.num_qubits, \
                f"num_qubits does not agree: expected {self.__num_qubits}, received {state.num_qubits}"

        if self.backend == Backend.QPU and state.backend == Backend.QPU:
            state.oper_history = self.oper_history
            return state

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
        self.__num_qubits_update(qubits_idx)
        self.append(H(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(S(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(Sdg(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(T(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(Tdg(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(X(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(Y(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(Z(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(P(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RX(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RY(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RZ(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(U3(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        if qubits_idx == "cycle" and not self.__isdynamic:
            self.append(CNOT(qubits_idx, self.num_qubits, self.__cnot_cycle_idx))
        else:
            self.append(CNOT(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CY(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CZ(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(SWAP(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CP(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CRX(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CRY(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CRZ(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CU(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RXX(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RYY(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(RZZ(qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(MS(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CSWAP(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(CCX(qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(UniversalTwoQubits(
            qubits_idx, self.num_qubits, param, param_sharing))

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
        self.__num_qubits_update(qubits_idx)
        self.append(UniversalThreeQubits(
            qubits_idx, self.num_qubits, param, param_sharing))
    
    def universal_qudits(self, qubits_idx: List[int],
            param: Union[torch.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""TODO add description
        """
        self.__num_qubits_update(qubits_idx)
        self.append(UniversalQudits(qubits_idx, param, param_sharing))

    def oracle(
            self, oracle: torch.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            gate_name: Optional[str] = 'O', latex_name: Optional[str] = None, plot_width: Optional[float] = None
    ) -> None:
        """Add an oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            gate_name: name of this oracle.
            latex_name: latex name of this oracle, default to be the gate name.
            plot_width: width of this gate in circuit plot, default to be proportional with the gate name.
        """
        self.__num_qubits_update(qubits_idx)
        gate_info = {
            'gatename': gate_name,
            'texname': f"${gate_name}$" if latex_name is None else latex_name,
            'plot_width': 0.6 * len(gate_name) if plot_width is None else plot_width}
        self.append(Oracle(
            oracle, qubits_idx, self.num_qubits, gate_info))

    def control_oracle(
            self, oracle: torch.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
            gate_name: Optional[str] = 'O', latex_name: Optional[str] = None, plot_width: Optional[float] = None
    ) -> None:
        """Add a controlled oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            gate_name: name of this oracle.
            latex_name: latex name of this oracle, default to be the gate name.
            plot_width: width of this gate in circuit plot, default to be proportional with the gate name.
        """
        self.__num_qubits_update(qubits_idx)
        gate_info = {
            'gatename': f"c{gate_name}",
            'texname': f"${gate_name}$" if latex_name is None else latex_name,
            'plot_width': 0.6 * len(gate_name) if plot_width is None else plot_width}
        self.append(ControlOracle(
            oracle, qubits_idx, self.num_qubits, gate_info))

    def collapse(self, qubits_idx: Union[Iterable[int], int, str] = 'full',
                 desired_result: Union[int, str] = None, if_print: bool = False,
                 measure_basis: Union[Iterable[torch.Tensor], str] = 'z') -> None:
        r"""
        Args:
            qubits_idx: list of qubits to be collapsed. Defaults to ``'full'``.
            desired_result: The desired result you want to collapse. Defaults to ``None`` meaning randomly choose one.
            if_print: whether print the information about the collapsed state. Defaults to ``False``.
            measure_basis: The basis of the measurement. The quantum state will collapse to the corresponding eigenstate.

        Raises:
            NotImplementedError: If the basis of measurement is not z. Other bases will be implemented in future.
            TypeError: cannot get probability of state when the backend is unitary_matrix.

        Note:
            When desired_result is `None`, Collapse does not support gradient calculation
        """
        self.__num_qubits_update(qubits_idx)
        self.append(Collapse(
            qubits_idx, self.num_qubits, desired_result, if_print, measure_basis))

    def superposition_layer(
            self, qubits_idx: Iterable[int] = None
    ) -> None:
        r"""Add layers of Hadamard gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
        
        """
        self.__num_qubits_update(qubits_idx)
        self.extend(
            SuperpositionLayer(qubits_idx, self.num_qubits))

    def weak_superposition_layer(
            self, qubits_idx: Iterable[int] = None
    ) -> None:
        r"""Add layers of Ry gates with a rotation angle :math:`\pi/4`.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
        
        """
        self.__num_qubits_update(qubits_idx)
        self.extend(
            WeakSuperpositionLayer(qubits_idx, self.num_qubits))

    def linear_entangled_layer(
            self, qubits_idx: Iterable[int] = None, depth: int = 1
    ) -> None:
        r"""Add linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
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
        self.__num_qubits_update(qubits_idx)
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
        self.__num_qubits_update(qubits_idx)
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
        self.__num_qubits_update(qubits_idx)
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
        self.__num_qubits_update(qubits_idx)
        self.extend(
            ComplexBlockLayer(qubits_idx, self.num_qubits, depth))

    def qaoa_layer(self, edges: Iterable, nodes: Iterable, depth: Optional[int] = 1) -> None:
        # TODO: see qaoa layer in layer.py
        self.__num_qubits_update(edges)
        self.__num_qubits_update(nodes)
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
        self.__num_qubits_update(qubits_idx)
        self.append(BitFlip(prob, qubits_idx, self.num_qubits))

    def phase_flip(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add phase flip channels.

        Args:
            prob: Probability of a phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PhaseFlip(prob, qubits_idx,
                    self.num_qubits))

    def bit_phase_flip(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add bit phase flip channels.

        Args:
            prob: Probability of a bit phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(BitPhaseFlip(prob, qubits_idx, self.num_qubits))

    def amplitude_damping(
            self, gamma: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add amplitude damping channels.

        Args:
            gamma: Damping probability.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(AmplitudeDamping(gamma, qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(GeneralizedAmplitudeDamping(
            gamma, prob, qubits_idx, self.num_qubits))

    def phase_damping(
            self, gamma: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add phase damping channels.

        Args:
            gamma: Parameter of the phase damping channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PhaseDamping(gamma, qubits_idx, self.num_qubits))

    def depolarizing(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add depolarizing channels.

        Args:
            prob: Parameter of the depolarizing channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(Depolarizing(prob, qubits_idx, self.num_qubits))

    def generalized_depolarizing(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str]
    ) -> None:
        r"""Add a general depolarizing channel.

        Args:
            prob: Probabilities corresponding to the Pauli basis.
            qubits_idx: Indices of the qubits on which the channel is applied.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(GeneralizedDepolarizing(prob, qubits_idx, self.num_qubits))

    def pauli_channel(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add Pauli channels.

        Args:
            prob: Probabilities corresponding to the Pauli X, Y, and Z operators.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PauliChannel(prob, qubits_idx, self.num_qubits))

    def reset_channel(
            self, prob: Union[torch.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full'
    ) -> None:
        r"""Add reset channels.

        Args:
            prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(ResetChannel(prob, qubits_idx, self.num_qubits))

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
        self.__num_qubits_update(qubits_idx)
        self.append(
            ThermalRelaxation(const_t, exec_time, qubits_idx, self.num_qubits))

    def choi_channel(
            self, choi_repr: Iterable[torch.Tensor],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Choi representation.

        Args:
            choi_repr: Choi representation of this channel.
            qubits_idx: Indices of the qubits on which the channels are applied.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(ChoiRepr(choi_repr, qubits_idx, self.num_qubits))

    def kraus_channel(
            self, kraus_oper: Iterable[torch.Tensor],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Kraus representation.

        Args:
            kraus_oper: Kraus representation of this channel.
            qubits_idx: Indices of the qubits on which the channels are applied.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(KrausRepr(kraus_oper, qubits_idx, self.num_qubits))

    def stinespring_channel(
            self, stinespring_repr: Iterable[torch.Tensor],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
    ) -> None:
        r"""Add custom channels in the Stinespring representation.

        Args:
            stinespring_repr: Stinespring representation of this channel.
            qubits_idx: Indices of the qubits on which the channels are applied.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            StinespringRepr(stinespring_repr, qubits_idx, self.num_qubits))

    def __str__(self):
        history = self.gate_history
        num_qubits = self.__num_qubits
        length, gate = self.__count_history(history)
        # Ignore the unused section
        total_length = sum(length) - 5
        print_list = [['-' if i % 2 == 0 else ' '] *
                      total_length for i in range(num_qubits * 2)]
        for i, current_gate in enumerate(history):
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u'}:
                # Calculate starting position ind of current gate
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][ind +
                                                             length[sec] // 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'sdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'tdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                line = current_gate['which_qubits'] * 2
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
                cqubit = current_gate['which_qubits'][0]
                tqubit = current_gate['which_qubits'][1]
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
                cqubit = current_gate['which_qubits'][0]
                tqubit1 = current_gate['which_qubits'][1]
                tqubit2 = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'cswap'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit1 * 2][ind + length[sec] // 2] = 'x'
                    print_list[tqubit2 * 2][ind + length[sec] // 2] = 'x'
            elif current_gate['gate'] in {'ccx'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit1 = current_gate['which_qubits'][0]
                cqubit2 = current_gate['which_qubits'][1]
                tqubit = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
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
