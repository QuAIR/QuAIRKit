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
The source file of the oracle class and the control oracle class.
"""

from typing import Callable, Iterable, List, Union

import matplotlib
import torch

from .base import Gate, ParamGate
from .visual import _c_oracle_like_display, _oracle_like_display


class Oracle(Gate):
    """An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, oracle: torch.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            num_qubits: int = None, gate_info: dict = None
    ):
        super().__init__(oracle, qubits_idx, gate_info, num_qubits)
    
    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _oracle_like_display(self, ax, x)


class ControlOracle(Gate):
    """A controlled oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        qubits_idx: Indices of the qubits on which the gates are applied.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    
    def __init__(
            self, oracle: torch.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
            num_qubits: int = None, gate_info: dict = None
    ) -> None:
        #TODO: support more control types
        __zero = torch.tensor([[1, 0], [0, 0]])
        __one = torch.tensor([[0, 0], [0, 1]])
        __eye = torch.eye(oracle.shape[-1])
        if len(oracle.shape) > 2:
            __zero = __zero.view([1, 2, 2])
            __one = __one.view([1, 2, 2])
            __eye = __eye.expand_as(oracle)
        oracle = torch.kron(__zero, __eye) + torch.kron(__one, oracle)

        default_gate_info = {
            'gatename': 'cO',
            'texname': r'$O$',
            'plot_width': 0.6,
        }
        if gate_info is not None:
            default_gate_info |= gate_info
        super().__init__(oracle, qubits_idx, default_gate_info, num_qubits)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _c_oracle_like_display(self, ax, x)


class ParamOracle(ParamGate):
    """An parameterized oracle as a gate

    Args:
        generator: function that generates the oracle.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        qubits_idx: indices of the qubits on which this gate acts on. Defaults to ``None`` i.e. list(range(num_qubits)).
        depth: number of layers. Defaults to ``1``.
        num_acted_param: the number of parameters required for a single operation.
        param_sharing: whether all operations are shared by the same parameter set.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
        Defaults to ``None``.
        num_qubits: total number of qubits. Defaults to ``None``.

    """
    def __init__(
            self, generator: Callable[[torch.Tensor], torch.Tensor], param: Union[torch.Tensor, float, List[float]] = None,
            num_acted_param: int = 1, param_sharing: bool = False,
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            gate_info: dict = None, num_qubits: int = None
    ):
        super().__init__(generator, param, num_acted_param, param_sharing, qubits_idx, gate_info, num_qubits)
