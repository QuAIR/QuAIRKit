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
The source file of the state_vector backend.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ... import utils
from .. import MixedState, State


class PureState(State):
    r"""The pure state class.

    Args:
        data: tensor array in vector representation for quantum pure state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 1 to n.
        probability: tensor array for state distributions. Defaults to be 1.

    Note:
        The data is stored in the vector-form with shape :math:`(-1, d)`

    """
    def __init__(self, data: torch.Tensor, sys_dim: List[int], 
                 system_seq: Optional[List[int]] = None,
                 probability: Optional[List[torch.Tensor]] = None):
        dim = int(np.prod(sys_dim))
        self._prob = [] if probability is None else probability
        
        non_batch_len = 2 + len(self._prob)
        self._batch_dim = list(data.shape[:-non_batch_len])
        
        data = data.reshape([-1, dim])
        super().__init__(data, sys_dim, system_seq)
        
        self._switch_unitary_matrix = False # the flag to switch on the unitary matrix recording.
        
    def __getitem__(self, key: Union[int, slice]) -> 'PureState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return PureState(self.ket[key], self.system_dim, self.system_seq, 
                         [prob.clone() for prob in self._prob])
    
    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> 'PureState':
        num_prob = len(self._prob)
        if prob_idx > 0:
            prob_idx -= num_prob

        new_prob = []
        for idx, prob in enumerate(self._prob):
            if num_prob + prob_idx > idx:
                new_prob.append(prob.clone())
            else:
                new_prob.append(prob.index_select(dim=prob_idx, index=outcome_idx).squeeze(prob_idx))
        
        data_idx = prob_idx - 2
        data = self.ket.index_select(dim=data_idx, index=outcome_idx).squeeze(data_idx)
        return PureState(data, self.system_dim, self.system_seq, new_prob)
    
    def expand(self, batch_dim: List[int]) -> 'PureState':
        if self._prob != []:
            raise NotImplementedError(
                "Batch expansion of pure state with probabilistic computation is not supported."
            )
        
        expand_state = self.clone()
        expand_state._batch_dim = batch_dim
        expand_state._prob = []
        
        if np.prod(batch_dim) == np.prod(self._batch_dim):
            return expand_state
        
        non_batch_len = 1 + len(self._prob)
        expand_state._data = self._data.expand(batch_dim + [-1] * non_batch_len).reshape([-1, self.dim])
        return expand_state
    
    @property
    def backend(self) -> str:
        return "state_vector"
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        # formalize the input data with [d], [1, d], [d, 1], [..., d, 1]
        if data.ndim == 1:
            data = data.view([-1, 1])
        else:
            data = data.view(list(data.shape[:-2]) + [-1, 1])

        if isinstance(sys_dim, int):
            num_sys = int(math.log(data.shape[-2], sys_dim))
            expected_dimension = sys_dim ** num_sys
        else:
            num_sys = len(sys_dim)
            expected_dimension = int(np.prod(sys_dim))
        
        assert data.shape[-2] == expected_dimension, \
            f"Input data shape {data.shape} does not match the input system dimension {sys_dim}."
        
        if eps:
            assert torch.all(is_vec := utils.check._is_state_vector(data, eps)), \
                f"(Some) data is not a state vector: asserted {is_vec.tolist()}"
        
        return num_sys
        
    @property
    def ket(self) -> torch.Tensor:
        self.reset_sequence()
        return self._data.view(self.batch_dim + [self.dim, 1]).clone()

    @property
    def density_matrix(self) -> torch.Tensor:
        ket = self.ket
        return torch.matmul(ket, ket.mH)
    
    def to_mixed(self) -> MixedState:
        r"""Convert the pure state to mixed state computation.
        """
        return MixedState(self.density_matrix, self.system_dim, self.system_seq, 
                           [prob.clone() for prob in self._prob])
    
    def _trace(self, sys_idx: List[int]) -> MixedState:
        return self.to_mixed()._trace(sys_idx)
    
    def _transpose(self, sys_idx: List[int]) -> MixedState:
        return self.to_mixed()._transpose(sys_idx)

    @property
    def rank(self) -> int:
        return 1
    
    def normalize(self):
        return torch.div(self._data, torch.norm(self._data, dim=-1))

    def clone(self) -> 'PureState':
        data = self._data.view(self.batch_dim + [self.dim, 1]).clone()
        state = PureState(data, self.system_dim, self.system_seq, [prob.clone() for prob in self._prob])
        state._switch_unitary_matrix = self._switch_unitary_matrix
        return state

    def fit(self, backend: str) -> torch.Tensor:
        if backend == self.backend:
            return self.ket
        
        if backend == 'density_matrix':
            return self.density_matrix
        
        raise NotImplementedError(
            f"Unsupported state conversion from {self.backend} to {backend}.")

    @State.system_seq.setter
    def system_seq(self, target_seq: List[int]) -> None:
        if target_seq == self._system_seq:
            return
        
        perm_map = utils.linalg._perm_of_list(self._system_seq, target_seq)
        current_system_dim = [self._sys_dim[x] for x in self._system_seq]
        
        self._data = utils.linalg._permute_sv(self._data, perm_map, current_system_dim).contiguous()
        self._system_seq = target_seq
        
    def _record_unitary(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        r"""The function that records the input unitary for computing the overall 
        unitary matrix of the circuit.
        
        Note:
            This function is for calling `Circuit.matrix` only
        """
        if self._batch_dim == [self.dim]:
            self._evolve_keep_dim(unitary, sys_idx)
            return
        
        applied_dim, dim = unitary.shape[-1], self.dim
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([dim, -1, applied_dim, dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, dim])
        
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        if self._switch_unitary_matrix:
            self._record_unitary(unitary, sys_idx)
            return
        dim, _shape = self.dim, self._squeeze_shape
        
        if on_batch:
            self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        unitary = unitary.view(evolve_axis + [applied_dim, applied_dim])
        data = self._data.view(_shape + [applied_dim, dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, dim])
    
    def _evolve_ctrl(self, unitary: torch.Tensor, index: int, sys_idx: List[Union[int, List[int]]], on_batch: bool = True) -> None:
        ctrl_idx = sys_idx[0]
        ctrl_dim = np.prod([self.system_dim[idx] for idx in ctrl_idx])
        
        sys_idx = ctrl_idx + sys_idx[1:]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        if self._switch_unitary_matrix:
            proj = torch.zeros([ctrl_dim, ctrl_dim])
            proj[index, index] = 1
            unitary = torch.kron(proj, unitary) + torch.kron(torch.eye(ctrl_dim) - proj, torch.eye(unitary.shape[-1]).expand_as(unitary))
            self._record_unitary(unitary, sys_idx)
            return
        dim, _shape = self.dim, self._squeeze_shape
        
        if on_batch:
            self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
        
        applied_dim = unitary.shape[-1]
        unitary = unitary.view(evolve_axis + [applied_dim, applied_dim])
        
        other_dim = dim // ctrl_dim
        data = self._data.squeeze().expand(self.batch_dim + [dim])
        
        data = data.view(_shape + [ctrl_dim, applied_dim, other_dim // applied_dim]).clone()
        data[:, :, index] = torch.matmul(unitary, data[:, :, index])
        
        self._data = data.view([-1, dim])
        
    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        unitary_batch_dim = list(unitary.shape[:-2])
        dim, _shape = self.dim, self._squeeze_shape
        
        if on_batch:
            self._batch_dim = (self._batch_dim or unitary_batch_dim[:-1]) + unitary_batch_dim[-1:]
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
        
        applied_dim, num_unitary = unitary.shape[-1], int(np.prod(unitary_batch_dim[-1:]))
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        unitary = unitary.view(evolve_axis + [num_unitary, applied_dim, applied_dim])
        data = self._data.view(_shape + [1, applied_dim, dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, dim])
    
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        dim = self.dim
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        applied_dim, num_obs = obs.shape[-1], obs.shape[-3]
        state = self._data.view([-1, 1, applied_dim, dim // applied_dim])
        measured_state = torch.matmul(obs, state).view([-1, num_obs, dim])

        return torch.linalg.vecdot(self._data.unsqueeze(-2), measured_state).view(self.batch_dim + [num_obs])
    
    def _expec_state(self, prob_idx: List[int]) -> 'PureState':
        dim, num_prob = self.dim, len(self._prob)
        batch_prob_len = len(self._prob[-1].shape[:-num_prob])
        prob = self._joint_probability(prob_idx)
        
        states = self._data.view([-1] + self._prob_dim + [dim])
        prob = prob.view(list(prob.shape) + [1] * (self._prob[-1].dim() - prob.dim() + 1))
        prob_state = torch.mul(prob, states)
        
        sum_idx = [idx + 1 for idx in prob_idx]
        expectation = prob_state.sum(sum_idx)

        new_prob = []
        if len(prob_idx) != num_prob:
            new_prob = [prob.clone() for idx, prob in enumerate(self._prob) if idx not in prob_idx]
            expectation = expectation.view(self._batch_dim + list(new_prob[-1].shape[batch_prob_len:]) + [dim, 1])
        else:
            expectation = expectation.view(self._batch_dim + [dim, 1])
        return PureState(expectation, self.system_dim, self.system_seq, new_prob)
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'PureState']:
        new_state = self.clone()
        new_state._evolve_keep_dim(measure_op, sys_idx)
        
        data = new_state._data.view([-1, self.dim, 1])
        measure_prob = (data.mH @ data).real
        collapsed_data = data / torch.sqrt(measure_prob)
        collapsed_data[collapsed_data != collapsed_data] = 0
        
        measure_prob = measure_prob.view(new_state._batch_dim[:-1] + self._prob_dim + [-1])
        new_state._data = collapsed_data.view([-1, self.dim])
        new_state._batch_dim = new_state._batch_dim[:-1]
        new_state._prob.append(measure_prob)
        return measure_prob, new_state

    def _transform(self, *args) -> None:
        raise NotImplementedError(
            "The state vector backend does not internally support the channel conversion. \
                Please call the 'transform' function directly instead of '_transform'.")
    
    def transform(self, op: torch.Tensor, sys_idx: List[int] = None, repr_type: str = 'kraus') -> MixedState:
        return self.to_mixed().transform(op, sys_idx, repr_type)
    
    def product_trace(self, trace_state: 'PureState', trace_idx: List[int]) -> 'PureState':
        r"""Partial trace over this state, when this state is a product state
        """
        remain_seq, remain_system_dim, trace_system_dim = [], [], []
        for x in self._system_seq:
            if x in trace_idx:
                trace_system_dim.append(self.system_dim[x])
            else:
                remain_seq.append(x)
                remain_system_dim.append(self.system_dim[x])

        assert trace_state.system_dim == trace_system_dim, \
                f"Traced state does not match with the target trace system: received {trace_state.system_dim}, expect {trace_system_dim}."
        assert trace_state.numel() in [1, self.numel()], \
            f"# of traced state mismatch: received {trace_state.numel()}, expect 1 or {self.numel()}."

        trace_state = trace_state.ket.squeeze(-1)
        self.system_seq = trace_idx + remain_seq

        data = self._data.view(self.batch_dim + [self.dim])
        data = utils.linalg._ptrace_1(data, trace_state).unsqueeze(-1)
        # convert remaining sequence
        value_to_index = {value: index for index, value in enumerate(sorted(remain_seq))}
        remain_seq = [value_to_index[i] for i in remain_seq]
        return PureState(data, remain_system_dim, remain_seq, self._prob)
    
    def sqrt(self) -> torch.Tensor:
        return self.density_matrix
    
    def log(self) -> None:
        raise TypeError(
            "Cannot apply matrix logarithm to a pure state.")
