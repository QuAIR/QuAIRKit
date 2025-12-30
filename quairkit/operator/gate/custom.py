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

from ...core import (OperatorInfoType, StateSimulator, get_device, get_dtype,
                     utils)
from ...core.intrinsic import _int_to_digit
from ...core.operator.base import OperatorInfoType
from ...database.set import gell_mann
from .base import Gate, ParamGate
from .visual import _c_oracle_like_display, _oracle_like_display


class Oracle(Gate):
    r"""An oracle as a gate.

    Args:
        oracle: Unitary oracle to be implemented.
        system_idx: Indices of the systems on which the gates are applied.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    
    Note:
        The forward function of this class will not create a new instance of state.
    """
    def __init__(
            self, oracle: torch.Tensor, system_idx: Union[List[int], int],
            acted_system_dim: Union[List[int], int] = 2, gate_info: Dict = None,
    ):
        default_gate_info = {
            "name": "oracle",
            "tex": r'\mathtt{oracle}',
            "api": "oracle",
            'plot_width': 0.6,
        }
        default_gate_info.update(gate_info or {})
        super().__init__(oracle, system_idx, acted_system_dim, gate_info=default_gate_info)

    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this gate.
        """
        oracle = self.matrix
        info = super().info
        info.update({
            'matrix': oracle,
        })
        return info
    
    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _oracle_like_display(self, ax, x)


class ControlOracle(Gate):
    r"""A controlled oracle as a gate.
    
    Args:
        oracle: Unitary oracle to be implemented.
        system_idx: Indices of the systems on which the gates are applied. The first element in the list is the control system, 
            represented by a int or a list of ints, and the remaining elements represent the oracle system.
        index: control index that activates the matrix.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
            Defaults to ``None``.

    Note:
        The forward function of this class will not create a new instance of state.

    """
    def __init__(
        self, oracle: torch.Tensor, system_idx: List[Union[List[int], int]], index: int, 
        acted_system_dim: Union[List[int], int] = 2, gate_info: Optional[Dict] = None
    ) -> None:
        ctrl_system_idx = [system_idx[0]] if isinstance(system_idx[0], int) else system_idx[0]
        
        num_ctrl_system = len(ctrl_system_idx)
        ctrl_system_dim = [acted_system_dim] * num_ctrl_system if isinstance(acted_system_dim, int) else acted_system_dim[:num_ctrl_system]
        default_gate_info = {
            "name": "coracle",
            "tex": r'\mathtt{oracle}',
            "api": "control_oracle",
            "num_ctrl_system": len(ctrl_system_idx),
            "label": _int_to_digit(index, ctrl_system_dim).zfill(num_ctrl_system),
            'plot_width': 0.6,
        }
        default_gate_info.update(gate_info or {})
        super().__init__(None, ctrl_system_idx + system_idx[1:], acted_system_dim, gate_info=default_gate_info)
        
        self.__system_idx = [ctrl_system_idx] + system_idx[1:]
        self.__apply_matrix = oracle.to(dtype=self.dtype, device=self.device)
        self.__index = index
        
    @property
    def _apply_matrix(self) -> torch.Tensor:
        r"""Unitary matrix of the control part
        """
        mat = self.__apply_matrix
        return utils.linalg._dagger(mat) if self._is_dagger else mat
    
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this gate.
        """
        info = super().info
        if (perm := info.get('permute', None)) and len(perm) > 2:
            perm = utils.linalg._perm_of_list(perm, list(range(len(perm))))
            info['permute'] = perm
        info['matrix'] = self.__apply_matrix
        return info
        
    @property
    def matrix(self) -> torch.Tensor:
        ctrl_dim = math.prod(self.system_dim[:len(self.__system_idx[0])])
        index = self.__index
        
        proj = torch.zeros([ctrl_dim, ctrl_dim])
        proj[index, index] = 1
        
        matrix = self._apply_matrix
        _eye = torch.eye(matrix.shape[-1]).expand_as(matrix)
        return utils.linalg._kron(proj, matrix) + utils.linalg._kron(torch.eye(ctrl_dim) - proj, _eye)
    
    def forward(self, state: StateSimulator) -> StateSimulator: 
        state._evolve_ctrl(self._apply_matrix, self.__index, self.__system_idx)
        return state
    
    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float,) -> float:
        return _c_oracle_like_display(self, ax, x)


class ParamOracle(ParamGate):
    r"""An parameterized oracle as a gate

    Args:
        generator: function that generates the oracle.
        system_idx: indices of the system on which this gate acts on.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        num_acted_param: the number of parameters required for a single operation.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
            Defaults to ``None``.
        support_batch: whether the generator supports batched input. Defaults to ``True``.

    Note:
        The forward function of this class will not create a new instance of state.
    """
    def __init__(
        self, generator: Callable[[torch.Tensor], torch.Tensor], system_idx: Union[List[int], int],
        param: Union[torch.Tensor, float, List[float]] = None, num_acted_param: int = 1,
        acted_system_dim: Union[List[int], int] = 2, gate_info: Dict = None, support_batch: bool = True,
    ):
        default_gate_info = {
            "name": "oracle",
            "tex": r'\mathtt{oracle}',
            "param_sharing": False,
            "api": "param_oracle",
            "kwargs": {"generator": generator},
            'plot_width': 0.6,
        }
        default_gate_info.update(gate_info or {})
        super().__init__(generator, param, num_acted_param, False, system_idx, acted_system_dim, True, default_gate_info, support_batch)


class ControlParamOracle(ParamGate):
    r"""An parameterized oracle as a gate

    Args:
        generator: function that generates the oracle.
        system_idx: Indices of the systems on which the gates are applied. The first element in the list is the control system, 
            represented by a int or a list of ints, and the remaining elements represent the oracle system.
        index: control index that activates the matrix.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        num_acted_param: the number of parameters required for a single operation.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
            Defaults to ``None``.
        support_batch: whether the generator supports batched input. Defaults to ``True``.
    
    Note:
        The forward function of this class will not create a new instance of state.

    """
    def __init__(
        self, generator: Callable[[torch.Tensor], torch.Tensor], system_idx: Union[List[int], int], index: int, 
        param: Union[torch.Tensor, float, List[float]] = None, num_acted_param: int = 1,
        acted_system_dim: Union[List[int], int] = 2, gate_info: Dict = None, support_batch: bool = True,
    ):
        ctrl_system_idx = [system_idx[0]] if isinstance(system_idx[0], int) else system_idx[0]
        
        num_ctrl_system = len(ctrl_system_idx)
        ctrl_system_dim = [acted_system_dim] * num_ctrl_system if isinstance(acted_system_dim, int) else acted_system_dim[:num_ctrl_system]
        default_gate_info = {
            "name": "coracle",
            "tex": r'\mathtt{oracle}',
            "param_sharing": False,
            "api": "param_oracle",
            "num_ctrl_system": len(ctrl_system_idx),
            "label": _int_to_digit(index, ctrl_system_dim).zfill(num_ctrl_system),
            'plot_width': 0.6,
            "kwargs": {"generator": generator},
        }
        default_gate_info.update(gate_info or {})
        super().__init__(generator, param, num_acted_param, False, ctrl_system_idx + system_idx[1:], 
                         acted_system_dim, True, default_gate_info, support_batch)
        
        self.__system_idx = [ctrl_system_idx] + system_idx[1:]
        self.__index = index
        
    @property
    def matrix(self) -> torch.Tensor:
        ctrl_dim = math.prod(self.system_dim[:len(self.__system_idx[0])])
        index = self.__index
        
        proj = torch.zeros([ctrl_dim, ctrl_dim])
        proj[index, index] = 1
        
        matrix = super().matrix
        _eye = torch.eye(matrix.shape[-1]).expand_as(matrix)
        return utils.linalg._kron(proj, matrix) + utils.linalg._kron(torch.eye(ctrl_dim) - proj, _eye)
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        state._evolve_ctrl(super().matrix, self.__index, self.__system_idx)
        return state


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

    Note:
        The forward function of this class will not create a new instance of state.
    """
    def __init__(
        self, system_idx: Optional[Union[Iterable[int], str]], acted_system_dim: Iterable[int],
        param: Optional[Union[torch.Tensor, float]] = None,
        param_sharing: Optional[bool] = False,

    ):
        assert not isinstance(acted_system_dim, int), \
            f"system dimensions for UniversalQudits cannot be a integer: received {acted_system_dim}"

        dim = math.prod(acted_system_dim)
        generator = gell_mann(dim).to(get_device(), dtype=get_dtype())
        matrix_func = partial(utils.matrix._param_generator, generator=generator)

        gate_info = {
            "name": "universal",
            "tex": r'\operatorname{UNI}_{' + str(dim) + r'}',
            "api": "universal_qudits",
            "param_sharing": param_sharing,
            'plot_width': 0.8,
        }
        super().__init__(
            matrix_func, param, dim ** 2 - 1, param_sharing, system_idx, acted_system_dim,
            check_legality=False, gate_info=gate_info)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        return _oracle_like_display(self, ax, x)


class Permutation(Gate):
    r"""permutation matrix as a gate.

    Args:
        perm: A list representing the permutation of subsystems.
            For example, [1, 0, 2] swaps the first two subsystems.
        system_idx: Indices of the systems on which the gates are applied.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.

    """
    def __init__(
            self, perm: List[int], system_idx: List[Union[List[int], int]],
            acted_system_dim: Union[List[int], int] = 2
        ):
        gate_info = {
            "name": "permute",
            "api": "permute",
            'plot_width': 0.2,
        }
        
        self.perm = perm
        super().__init__(None, system_idx, acted_system_dim, check_legality=False, gate_info=gate_info)
    
    @property
    def matrix(self) -> torch.Tensor:
        return utils.matrix._permutation(self.perm, self.system_dim)
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this gate.
        """
        info = super().info
        info.update({
            'permute': self.perm
        })
        return info
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        perm, system_idx = self.perm, self.system_idx[0]
        key_map = {system_idx[i]: system_idx[perm[i]] for i in range(len(system_idx))}
        target_seq = [(key_map[idx] if idx in system_idx else idx) for idx in range(state.num_systems)]
        return state.permute(target_seq)
    
    def dagger(self) -> None:
        self.system_idx = list(reversed(self.system_idx))
        if len(self.perm) <= 2:
            return
        
        perm = self.perm
        self.perm = utils.linalg._perm_of_list(perm, list(range(len(perm))))
