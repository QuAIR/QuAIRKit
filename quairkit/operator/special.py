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

from ..core import Operator, State
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
        desired_result: The desired result you want to collapse. Defaults to ``None`` meaning preserving all results, 
        and activate probabilistic computation.
        if_print: whether print the information about the collapsed state. Defaults to ``False``.
        measure_basis: The basis of the measurement. Defaults to the computational basis.

    Raises:
        NotImplementedError: If the basis of measurement is not z. Other bases will be implemented in future.
        
    Note:
        When desired_result is `None`, Collapse does not support gradient calculation
    """
    @_alias({"system_idx": "qubits_idx"})
    def __init__(self, system_idx: Union[int, Iterable[int]],
                 desired_result: Union[int, str] = None, if_print: bool = False,
                 measure_basis: Optional[torch.Tensor] = None):
        super().__init__()
        self.measure_basis = []

        if isinstance(system_idx, int):
            self.system_idx = [system_idx]
        else:
            self.system_idx = list(system_idx)

        self.desired_result = desired_result
        self.if_print = if_print

        self.measure_basis = measure_basis
        
    def __call__(self, state: State) -> State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
        
    def forward(self, state: State) -> State:
        r"""Compute the collapse of the input state.

        Args:
            state: The input state, which will be collapsed

        Returns:
            The collapsed quantum state.
        """
        system_dim = [state.system_dim[idx] for idx in self.system_idx]
        desired_result = self.desired_result

        prob_array, measured_state = state.measure(self.measure_basis, self.system_idx, keep_state=True)
        
        if desired_result is None:
            return measured_state
        
        if isinstance(desired_result, str):
            digits_str = desired_result
            desired_result = _digit_to_int(desired_result, system_dim)
        else:
            digits_str = _int_to_digit(desired_result, base=system_dim)
        desired_result = torch.tensor(desired_result)
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
        list_unitary: a batched tensor that represents all unitaries.
        system_idx: Indices of the systems on which the protocol is applied. The first element in the list 
        indexes systems to be measured.
        acted_system_dim: dimension of systems that unitary channels act on. Can be a list of system dimensions 
        or an int representing the dimension of all systems. Defaults to be qubit case.
        measure_basis: The basis of the measurement. Defaults to the computational basis.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self, list_unitary: torch.Tensor, system_idx: Union[Iterable[int], int] = None,
        acted_system_dim: Union[List[int], int] = 2, measure_basis: Optional[torch.Tensor] = None
    ):
        super().__init__()
        measure_idx, act_idx = system_idx[0], system_idx[1:]
        acted_system_dim = acted_system_dim if isinstance(acted_system_dim, int) else acted_system_dim[len(measure_idx):]
        
        self.measure = Collapse(measure_idx, measure_basis=measure_basis)
        self.list_gate = Gate(list_unitary, act_idx, acted_system_dim)
        
    def __call__(self, state: State) -> State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)
    
    def forward(self, state: State) -> State:
        r"""Compute the input state passing through the LOCC protocol.

        Args:
            state: The input state.

        Returns:
            The collapsed quantum state.
        
        """
        matrix, sys_idx = self.list_gate.matrix, self.list_gate.system_idx[0]
        
        measured_state = self.measure(state)
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
