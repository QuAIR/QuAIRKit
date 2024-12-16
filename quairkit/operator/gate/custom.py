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

import math
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Union

import matplotlib
import torch

from ...core import get_device, get_dtype, utils
from ...database.set import gell_mann
from .base import Gate, ParamGate
from .visual import _c_oracle_like_display, _oracle_like_display


class Oracle(Gate):
    """An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        system_idx: Indices of the systems on which the gates are applied.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    """
    def __init__(
            self, oracle: torch.Tensor, system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            acted_system_dim: Union[List[int], int] = 2, gate_info: Dict = None,
    ):
        super().__init__(oracle, system_idx, acted_system_dim, gate_info=gate_info)
    
    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _oracle_like_display(self, ax, x)


class ControlOracle(Gate):
    """A controlled oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        system_idx: Indices of the systems on which the gates are applied. The first element in the list is the control system, 
            defaulting to :math:`|d-1\rangle \langle d-1|` as the control qubit, 
            while the remaining elements represent the oracle system.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        proj: Projector matrix for the control qubit. Defaults to ``None``.
    """
    def __init__(
            self, oracle: torch.Tensor, system_idx: List[Union[List[int], int]],
            acted_system_dim: Union[List[int], int] = 2, proj: Union[torch.Tensor] = None, gate_info: Dict = None,
    ) -> None:
        if isinstance(acted_system_dim, int):
            ctrl_dim = acted_system_dim
        elif isinstance(system_idx[0], int):
            ctrl_dim = acted_system_dim[0]
        else:
            ctrl_dim = math.prod(acted_system_dim[:len(system_idx[0])])
            system_idx = system_idx[0] + system_idx[1:]

        if proj is None:
            proj = torch.zeros([ctrl_dim, ctrl_dim])
            proj[-1, -1] += 1
        else:
            assert proj.shape == (ctrl_dim, ctrl_dim), \
                f"Input project does not match the control dimension: expected {ctrl_dim}, received {proj.shape}"
            assert utils.check._is_projector(proj), \
                "Input matrix is not a projector."

        _eye = torch.eye(oracle.shape[-1]).expand_as(oracle)
        oracle = utils.linalg._kron(torch.eye(ctrl_dim) - proj, _eye) + utils.linalg._kron(proj, oracle)

        default_gate_info = {
            'gatename': 'cO',
            'texname': r'$O$',
            'plot_width': 0.6,
        }
        if gate_info is not None:
            default_gate_info |= gate_info
        super().__init__(oracle, system_idx, acted_system_dim, gate_info=gate_info)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _c_oracle_like_display(self, ax, x)


class ParamOracle(ParamGate):
    """An parameterized oracle as a gate

    Args:
        generator: function that generates the oracle.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        system_idx: indices of the system on which this gate acts on. Defaults to ``None`` i.e. list(range(num_systems)).
        num_acted_param: the number of parameters required for a single operation.
        param_sharing: whether all operations are shared by the same parameter set.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
            Defaults to ``None``.

    """
    def __init__(
            self, generator: Callable[[torch.Tensor], torch.Tensor], param: Union[torch.Tensor, float, List[float]] = None,
            num_acted_param: int = 1, param_sharing: bool = False,
            system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            acted_system_dim: Union[List[int], int] = 2, gate_info: Dict = None
    ):
        super().__init__(generator, param, num_acted_param, param_sharing, system_idx, acted_system_dim, gate_info)


def _universal_matrix(param: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    r"""Generate a universal matrix with the given parameters and bases.
    """
    h = torch.sum(torch.mul(param.view([-1, 1, 1]), bases), dim=-3)
    return torch.matrix_exp(1j * h)

class UniversalQudits(ParamGate):
    r"""A collection of universal qudit gates. One of such a gate requires :math:`d^2 - 1` parameters.

    Args:
        system_idx: Indices of the qubits on which the gates are applied. Defaults to the first qubit.
        param: Parameters of the gates. Defaults to ``None``.
        param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    """
    def __init__(
        self, system_idx: Optional[Union[Iterable[int], str]], acted_system_dim: Iterable[int],
        param: Optional[Union[torch.Tensor, float]] = None,
        param_sharing: Optional[bool] = False,

    ):
        assert not isinstance(acted_system_dim, int), \
            f"system dimensions for UniversalQudits cannot be a integer: received {acted_system_dim}"

        dim = math.prod(acted_system_dim)
        bases = gell_mann(dim).to(get_device(), dtype=get_dtype())
        matrix_func = partial(_universal_matrix, bases=bases)

        gate_info = {
            'gatename': 'uni qudit',
            'texname': r'$\text{UNI}_{' + str(dim) + r'}$',
            'plot_width': 0.8,
        }
        super().__init__(
            matrix_func, param, dim ** 2 - 1, param_sharing, system_idx, acted_system_dim,
            check_legality=False, gate_info=gate_info)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)
    
