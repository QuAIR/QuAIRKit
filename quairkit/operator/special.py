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

import copy
import warnings
from typing import Iterable, List, Optional, Union

import numpy as np
import torch

from ..core import Operator, OperatorInfoType, StateSimulator, utils
from ..core.intrinsic import _alias, _digit_to_int, _int_to_digit
from . import Channel, Gate


class ResetState(Operator):
    r"""The class to reset the quantum state.
    
    Args:
        system_idx: list of systems to be reset.
        acted_system_dim: dimension of systems that this gate acts on.
        replace_state: the state to replace the quantum state.
        state_label: LaTeX label of the reset state, used for printing.
    """
    @_alias({"system_idx": "qubits_idx"})
    def __init__(self, system_idx: Union[int, Iterable[int]], acted_system_dim: List[int],
                 replace_state: StateSimulator, state_label: str) -> None:
        super().__init__()
        self.system_idx = [[system_idx]] if isinstance(system_idx, int) else [list(system_idx)]
        self.system_dim = acted_system_dim
        self.replace_state = replace_state
        
        _info = {"name": "reset",
                 "system_idx": self.system_idx,
                 "type": "channel",
                 "tex": state_label}
        self._info.update(_info)
        
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this operation
        """
        info = super().info
        info.update({
            'system_idx': copy.deepcopy(self.system_idx),
        })
        return info
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        r"""Perform the reset operation on the input quantum state.
        
        Args:
            state: The input quantum state to be reset.
            
        Returns:
            The reset quantum state.
        """
        return state.reset(self.system_idx[0], self.replace_state)


class Collapse(Operator):
    r"""The class to compute the collapse of the quantum state.

    Args:
        system_idx: list of systems to be collapsed.
        acted_system_dim: dimension of systems that this gate acts on.
        desired_result: The desired result you want to collapse. Defaults to ``None`` meaning preserving all results, 
            and activate probabilistic computation.
        if_print: whether print the information about the collapsed state. Defaults to ``False``.
        measure_op: The measurement operators of the measurement. Defaults to the computational measure.

    """
    @_alias({"system_idx": "qubits_idx"})
    def __init__(self, system_idx: Union[int, Iterable[int]], acted_system_dim: List[int], 
                 desired_result: Union[int, str] = None, if_print: bool = False, measure_op: Optional[torch.Tensor] = None):
        super().__init__()
        self.measure_op = []

        system_idx = [system_idx] if isinstance(system_idx, int) else list(system_idx)
        if measure_op is None:
            dim = int(np.prod(acted_system_dim))
            identity = torch.eye(dim, dtype=self.dtype, device=self.device).unsqueeze(-1)
            measure_op = identity @ identity.mH
        else:
            assert len(measure_op) > 1, \
                    "The input measurement op should be a list of measurement operators."
            assert utils.check.is_pvm(measure_op), \
                    "The input measurement op do not form a projection-valued measurement."

        if desired_result is None:
            digits_str = None
        else:
            if isinstance(desired_result, str):
                desired_result = _digit_to_int(desired_result, acted_system_dim)
            digits_str = _int_to_digit(desired_result, acted_system_dim)
            desired_result = torch.tensor(desired_result)
            measure_op = torch.index_select(measure_op, -3, desired_result)

        self.system_idx = [system_idx]
        self.system_dim = acted_system_dim
        self.measure_op = measure_op

        self.if_print = if_print
        self.digits_str = digits_str
        self.desired_result = desired_result


        _info = {"name": "measure",
                 "system_idx": self.system_idx,
                 "type": "channel",
                 "kwargs": {"basis": measure_op, "if_print": if_print}}
        if desired_result is not None:
            _info["label"] = digits_str
        self._info.update(_info)
        
    def __call__(self, state: StateSimulator) -> StateSimulator:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this channel
        """
        info = super().info
        info.update({
            'system_idx': copy.deepcopy(self.system_idx),
        })
        return info
        
    def forward(self, state: StateSimulator) -> StateSimulator:
        r"""Compute the collapse of the input state.

        Args:
            state: The input state, which will be collapsed

        Returns:
            The collapsed quantum state.
        """
        prob_array, measured_state = state._measure(self.measure_op, self.system_idx[0])
        
        desired_result = self.desired_result
        if desired_result is None:
            return measured_state
        
        digits_str = self.digits_str
        state_str = '>'.join(f'|{d}' for d in digits_str) + '>'
        
        if torch.any(prob_array < 1e-10).item():
            warnings.warn(
                f"It is computationally infeasible for some states in systems {self.system_idx} "
                f"to collapse to state {state_str}")

        # whether print the collapsed result
        if self.if_print:
            prob = prob_array.mean().item()
            print(f"systems {self.system_idx[0]} collapse to the state {state_str} with (average) probability {prob}")

        return measured_state


class OneWayLOCC(Operator):
    r"""A one-way LOCC protocol, where quantum measurement is modelled by a PVM and all channels are unitary channels.
    
    Args:
        gate: a batched Gate instance.
        measure_op: The measurement operators of the measurement. Defaults to the computational measure.
        label: name of the measured label. Defaults to ``'M'``.
        latex_name: latex name of the applied operator. Defaults to ``'O'``.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self, gate: Gate,
        measure_idx: List[int], measure_dim: Union[List[int], int],
        measure_op: Optional[torch.Tensor] = None, label: str = 'M', latex_name: str = 'O'
    ):
        super().__init__()
        num_measure_system = len(measure_idx)
        measure_dim = [measure_dim] * num_measure_system if isinstance(measure_dim, int) else measure_dim
        
        assert gate.matrix.ndim > 2, \
            f"The local operation must be batched, received shape {gate.matrix.shape}"
        assert (batch_dim := gate.matrix.shape[0]) == np.prod(measure_dim), \
            f"Batch dimension mismatch: expected {np.prod(measure_dim)}, received {batch_dim} unitaries"
        
        self.measure = Collapse(measure_idx, measure_dim, measure_op=measure_op)
        self.local_op = gate
        
        self._info.update({'name': "locc",
                           'system_idx': self.system_idx,
                           'tex': latex_name,
                           'label': label,
                           'type': "locc",
                           'num_ctrl_system': len(measure_idx),
                           'kwargs': {'measure_basis': measure_op,
                                      'label_name': label}})
        
    def __call__(self, state: StateSimulator) -> StateSimulator:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    @property
    def system_idx(self) -> List[List[int]]:
        r"""The system indices of the LOCC protocol.
        """
        return [self.measure.system_idx[0] + self.local_op.system_idx[0]]
    
    @system_idx.setter
    def system_idx(self, value: List[List[int]]):
        r"""Set the system indices of the LOCC protocol.
        """
        value, system_idx = value[0], self.system_idx[0]
        assert len(value) == len(system_idx), \
             f"Length of system_idx should be {len(system_idx)}, but got {len(value)}"
        num_measure_system = len(self.measure.system_idx[0])
        self.measure.system_idx = [value[:num_measure_system]]
        self.local_op.system_idx = [value[num_measure_system:]]
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this operation
        """
        info = super().info
        info.update({
            'system_idx': copy.deepcopy(self.system_idx),
        })
        return info
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        r"""Compute the input state passing through the LOCC protocol.

        Args:
            state: The input state.

        Returns:
            The collapsed quantum state.
        
        """
        matrix, sys_idx = self.local_op.matrix, self.local_op.system_idx[0]
        
        measured_state = self.measure(state)
        
        dim = len(matrix.shape[:-2])
        matrix = matrix.repeat(measured_state._prob_dim[:-dim] + [1] * (dim + 2))
        measured_state._evolve(matrix, sys_idx, on_batch=False)
        return measured_state


class QuasiOperation(Operator):
    r"""A quantum protocol containing quasi-operations.
    
    Args:
        channel: a batched Channel instance. Currently, we only support the gate channel.
        prob: a 1D (quasi-)probability distribution for the channel argument.
        probability_param: whether the probability is parameterized. Defaults to ``False``.
    
    """
    def __init__(
        self, channel: Channel, prob: torch.Tensor, probability_param: bool = False
    ):
        super().__init__()
        assert isinstance(channel, Gate), \
            f"Currently we only support quasi simulations with gates, received {channel.type_repr} representation"
        
        if (not probability_param) and np.abs((s := prob.sum().item()) - 1) > 1e-4:
            warnings.warn(
                f"The quasi-probability distribution should sum to 1, but received sum {s}. ", UserWarning)
        
        matrix = channel.matrix
        assert matrix.ndim == 3, \
            f"The input gate should be a batched gate, received shape {matrix.shape}."
            
        num_outcomes = prob.numel() + 1 if probability_param else prob.numel() 
        assert num_outcomes == (batch_num := matrix.shape[0]), \
            f"The prob distribution does not match the operator #: expect of num {num_outcomes}, received {batch_num}."
        
        prob = prob.flatten().to(dtype=self.dtype, device=self.device)
        if probability_param:
            prob = torch.nn.Parameter(prob)
            self.register_parameter('_prob', prob)
        else:
            self._prob = prob
        self.quasi_op = channel
        self.probability_param = probability_param
        
    def __call__(self, state: StateSimulator) -> StateSimulator:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    @property
    def probability(self) -> torch.Tensor:
        r"""The (quasi-)probability distribution of the quasi-operation.
        """
        if self.probability_param:
            return torch.cat([self._prob, (1 - self._prob.sum()).view([-1])], dim=0)
        return self._prob
    
    @property
    def system_idx(self) -> List[int]:
        r"""The system indices of the quasi-operation.
        """
        return self.quasi_op.system_idx[0]
    
    @system_idx.setter
    def system_idx(self, value: List[int]):
        r"""Set the system indices of the quasi-operation.
        """
        assert len(value) == len(self.system_idx), \
             f"Length of system_idx should be {len(self.system_idx)}, but got {len(value)}"
        self.quasi_op.system_idx = [value]
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this operation
        """
        return self.quasi_op.info
    
    def forward(self, state: StateSimulator) -> StateSimulator:
        r"""Compute the input state passing through the quasi-operation.
        
        Args:
            state: The input state.
        
        Returns:
            The collapsed quantum state.
        """
        state = state.clone()
        state.add_probability(self.probability)
        
        matrix = self.quasi_op.matrix
        matrix = matrix.repeat(state._prob_dim[:-1] + [1, 1, 1])
        state._evolve(matrix, self.system_idx, on_batch=False)
        return state
