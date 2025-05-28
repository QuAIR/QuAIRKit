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
The source file of the class for the special quantum operator.
"""

from typing import Iterable, List, Optional, Union

import numpy as np
import torch

from ..core import Operator, OperatorInfoType, State, utils
from ..core.intrinsic import _alias, _digit_to_int, _int_to_digit
from . import Channel, Gate


class ResetState(Operator):
    r"""The class to reset the quantum state. It will be implemented soon.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **kwargs):
        r"""The forward function.

        Returns:
            NotImplemented.
        """
        return NotImplemented


class Collapse(Operator):
    r"""The class to compute the collapse of the quantum state.

    Args:
        system_idx: list of systems to be collapsed.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case. Used for collapse function.
        desired_result: The desired result you want to collapse. Defaults to ``None`` meaning preserving all results, 
            and activate probabilistic computation.
        if_print: whether print the information about the collapsed state. Defaults to ``False``.
        measure_basis: The basis of the measurement. Defaults to the computational basis.

    Raises:
        NotImplementedError: If the basis of measurement is not z. Other bases will be implemented in future.
    
    """
    @_alias({"system_idx": "qubits_idx"})
    def __init__(self, system_idx: Union[int, Iterable[int]], acted_system_dim: List[int], 
                 desired_result: Union[int, str] = None, if_print: bool = False, measure_basis: Optional[torch.Tensor] = None):
        super().__init__()
        self.measure_basis = []

        if isinstance(system_idx, int):
            self.system_idx = [system_idx]
        else:
            self.system_idx = list(system_idx)
        
        self.if_print = if_print
    
        if measure_basis:
            assert len(measure_basis) > 1, \
                "The input measurement op should be a list of measurement operators."
            assert utils.check.is_pvm(measure_basis), \
                "The input measurement op do not form a projection-valued measurement."
        self.measure_basis = measure_basis
        self.system_dim = acted_system_dim
        
        if desired_result is None:
            digits_str = None
        else:
            if isinstance(desired_result, str):
                desired_result = _digit_to_int(desired_result, acted_system_dim)
            digits_str = _int_to_digit(desired_result, acted_system_dim)
            desired_result = torch.tensor(desired_result)
        
        self.desired_result = desired_result
        self.digits_str = digits_str

        _info = {"name": "measure",
                 "system_idx": self.system_idx,
                 "type": "channel",
                 "kwargs": {"basis": measure_basis, "if_print": if_print}}
        if desired_result is not None:
            _info["label"] = digits_str
        self._info.update(_info)
        
    def __call__(self, state: State) -> State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this channel
        """
        info = super().info
        info.update({
            'system_idx': self.system_idx,
        })
        return info
        
    def forward(self, state: State) -> State:
        r"""Compute the collapse of the input state.

        Args:
            state: The input state, which will be collapsed

        Returns:
            The collapsed quantum state.
        """
        prob_array, measured_state = state.measure(self.measure_basis, self.system_idx, keep_state=True)
        
        desired_result = self.desired_result
        if desired_result is None:
            return measured_state
        
        digits_str = self.digits_str
        state_str = '>'.join(f'|{d}' for d in digits_str) + '>'
        prob_collapse = prob_array.index_select(-1, desired_result)
        assert torch.all(prob_collapse > 1e-10).item(), (
            f"It is computationally infeasible for some states in systems {self.system_idx} "
            f"to collapse to state {state_str}")

        # whether print the collapsed result
        if self.if_print:
            prob = prob_collapse.mean().item()
            print(f"systems {self.system_idx} collapse to the state {state_str} with (average) probability {prob}")

        return measured_state.prob_select(desired_result)


class OneWayLOCC(Operator):
    r"""A one-way LOCC protocol, where quantum measurement is modelled by a PVM and all channels are unitary channels.
    
    Args:
        gate: a Gate operator.
        measure_basis: basis of the measurement. Defaults to the computational basis.
        label: name of the measured label. Defaults to ``'M'``.
        latex_name: latex name of the applied operator. Defaults to ``'O'``.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self, gate: Gate,
        measure_idx: List[int], measure_dim: Union[List[int], int],
        measure_basis: Optional[torch.Tensor] = None, label: str = 'M', latex_name: str = 'O'
    ):
        super().__init__()
        num_measure_system = len(measure_idx)
        measure_dim = [measure_dim] * num_measure_system if isinstance(measure_dim, int) else measure_dim
        
        assert (batch_dim := gate.matrix.shape[0]) == np.prod(measure_dim), \
            f"Batch dimension mismatch: expected {np.prod(measure_dim)}, received {batch_dim} unitaries"
        
        self.measure = Collapse(measure_idx, measure_dim, measure_basis=measure_basis)
        self.list_gate = gate
        
        self._info.update({'name': "locc",
                           'system_idx': measure_idx + gate.system_idx,
                           'tex': latex_name,
                           'label': label,
                           'type': "locc",
                           'num_ctrl_system': len(measure_idx),
                           'kwargs': {'measure_basis': measure_basis,
                                      'label_name': label}})
        
    def __call__(self, state: State) -> State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    @property
    def system_idx(self) -> List[int]:
        r"""The system indices of the LOCC protocol.
        """
        return self.measure.system_idx + self.list_gate.system_idx[0]
    
    @system_idx.setter
    def system_idx(self, value: List[int]):
        r"""Set the system indices of the LOCC protocol.
        """
        assert len(value) == len(self.system_idx), \
             f"Length of system_idx should be {len(self.system_idx)}, but got {len(value)}"
        num_measure_system = len(self.measure.system_idx)
        self.measure.system_idx = value[:num_measure_system]
        self.list_gate.system_idx = [value[num_measure_system:]]
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this channel
        """
        info = super().info
        info.update({
            'system_idx': self.system_idx,
        })
        return info
    
    def forward(self, state: State) -> State:
        r"""Compute the input state passing through the LOCC protocol.

        Args:
            state: The input state.

        Returns:
            The collapsed quantum state.
        
        """
        matrix, sys_idx = self.list_gate.matrix, self.list_gate.system_idx[0]
        
        measured_state = self.measure(state)
        
        dim = len(matrix.shape[:-2])
        matrix = matrix.repeat(measured_state._prob_dim[:-dim] + [1] * (dim + 2))
        measured_state._evolve(matrix, sys_idx, on_batch=False)
        return measured_state


class QuasiOperation(Operator):
    r"""A quantum protocol containing quasi-operations.
    
    Args:
        list_channels: a batched tensor that represents all unitaries.
        quasi_prob: the quasi-probability distribution for this quasi-operation.
        system_idx: indices of the systems on which the protocol is applied. 
        type_repr: one of ``'choi'``, ``'kraus'``, ``'stinespring'`` or ``'gate'``. Defaults to ``'gate'``.
        acted_system_dim: dimension of systems that these channels act on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    """
    def __init__(
        self, list_channels: torch.Tensor, quasi_prob: torch.Tensor, 
        system_idx: Union[Iterable[int], int] = None,
        type_repr: str = 'gate', acted_system_dim: Union[List[int], int] = 2
    ):
        super().__init__()
        
        # TODO: support other types of representations
        if type_repr != 'gate':
            raise NotImplementedError("Only 'gate' type is supported for now.")
        
        assert np.abs((s := quasi_prob.sum().item()) - 1) < 1e-4, \
            f"The quasi-probability distribution should sum to 1: received sum {s}"
        
        if type_repr == 'gate':
            self.channel = Gate(list_channels, system_idx, acted_system_dim)
        else:
            self.channel = Channel(type_repr, list_channels, system_idx, acted_system_dim)
        self.prob = quasi_prob.view([-1])
    
    def forward(self, state: State) -> State:
        r"""Compute the input state passing through the quasi-operation.
        
        Args:
            state: The input state.
        
        Returns:
            The collapsed quantum state.
        """
        state, prob = state.clone(), self.prob
        
        if state._prob:
            last_prob = state._prob[-1]
            prob = prob.view([1] * last_prob.ndim() + [-1])
        state._prob.append(self.prob)
        state._evolve(self.channel.matrix, self.channel.system_idx[0], on_batch=False)
        return state
