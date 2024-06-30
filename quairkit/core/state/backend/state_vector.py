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

    Note:
        The data is stored in the vector-form with shape :math:`(-1, d)`

    """
    def __init__(self, data: torch.Tensor, sys_dim: List[int], system_seq: Optional[List[int]] = None):
        dim = np.prod(sys_dim)
        
        self.batch_dim = list(data.shape[:-2])
        data = data.reshape([-1, dim])
        super().__init__(data, sys_dim, system_seq)
        
        self._switch_unitary_matrix = False # the flag to switch on the unitary matrix recording.
        
    def __getitem__(self, key: Union[int, slice]) -> 'PureState':
        return PureState(self.ket[key], self.system_dim, self.system_seq)
    
    def index_select(self, dim: int, index: torch.Tensor) -> 'PureState':
        dim = dim - 2 if dim < 0 else dim 
        return PureState(torch.index_select(self.ket, dim=dim, index=index), self.system_dim, self.system_seq)
    
    def expand(self, batch_dim: List[int]) -> 'PureState':
        expand_state = self.clone()
        expand_state.batch_dim = batch_dim
        
        if np.prod(batch_dim) == np.prod(self.batch_dim):
            return expand_state
        
        expand_state._data = expand_state._data.expand(batch_dim + [-1]).reshape([-1, self.dim])
        return expand_state
    
    @property
    def backend(self) -> str:
        return "state_vector"
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: List[int]) -> Tuple[bool, str]:
        data = data.squeeze()
        data = data.view([-1, data.shape[-1]])
        
        #TODO more check for batch case
        for vec in data:
            
            is_valid, msg = utils.check._is_state_vector(vec, sys_dim=sys_dim)
            
            assert is_valid, \
                f"The input data is not legal for PureState: error message {msg}"
        
        return msg
        
    @property
    def ket(self) -> torch.Tensor:
        self.reset_sequence()
        return self._data.view(self.batch_dim + [self.dim, 1]).clone()

    @property
    def density_matrix(self) -> torch.Tensor:
        return torch.matmul(self.ket, self.bra)
    
    def _trace(self, sys_idx: List[int]) -> MixedState:
        # TODO requires qudit and batch support for partial trace discontiguous
        raise NotImplementedError

    @property
    def rank(self) -> int:
        return 1
    
    def normalize(self):
        return torch.div(self._data, torch.norm(self._data, dim=-1))

    def clone(self) -> 'PureState':
        
        # print(self._data.shape, self.batch_dim)
        state = PureState(self.ket, self.system_dim, self.system_seq)
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

        self._data = utils.linalg._base_transpose(self._data, perm_map).contiguous()
        self._system_seq = target_seq
        
    def _record_unitary(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        r"""The function that records the input unitary for computing the overall 
        unitary matrix of the circuit.
        
        Note:
            This function is for calling `Circuit.unitary_matrix()` only
        """
        if self.batch_dim == [self.dim]:
            self._evolve_keep_dim(unitary, sys_idx)
            return
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([self.dim, -1, applied_dim, self.dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, self.dim])
        
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        if self._switch_unitary_matrix:
            self._record_unitary(unitary, sys_idx)
            return
        self.batch_dim = self.batch_dim or list(unitary.shape[:-2])
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([-1, applied_dim, self.dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, self.dim])

    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        self.batch_dim = (self.batch_dim or list(unitary.shape[:-3])) + list(unitary.shape[-3:-2])
        unitary = unitary.view((list(unitary.shape[:-2]) or [-1]) + list(unitary.shape[-2:]))
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([-1, 1, applied_dim, self.dim // applied_dim])
        self._data = torch.matmul(unitary, data).view([-1, self.dim])
    
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        applied_dim, num_obs = obs.shape[-1], obs.shape[-3]
        state = self._data.view([-1, 1, applied_dim, self.dim // applied_dim])
        measured_state = torch.matmul(obs, state).view([-1, num_obs, self.dim])
    
        return torch.linalg.vecdot(self._data.unsqueeze(-2), measured_state).view(self.batch_dim + [num_obs])
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'PureState']:
        origin_batch_dim, origin_data = self.batch_dim.copy(), self._data.clone()
        self._evolve_keep_dim(measure_op, sys_idx)
        data = self._data.view([-1, self.dim, 1])
        
        prob = data.mH @ data
        collapsed_state = data / torch.sqrt(prob)
        collapsed_state[collapsed_state != collapsed_state] = 0
        
        new_batch_dim = self.batch_dim
        collapsed_state = PureState(collapsed_state.view(new_batch_dim + [self.dim, 1]),
                                     self.system_dim, self.system_seq)
        prob = prob.view(new_batch_dim)
        
        self.batch_dim, self._data = origin_batch_dim, origin_data
        return prob.real, collapsed_state

    def _transform(self, *args) -> None:
        raise NotImplementedError(
            "The state vector backend does not support the channel conversion. \
                Please call the 'transform' function directly instead of '_transform'.")
    
    def transform(self, op: torch.Tensor, sys_idx: List[int] = None, repr_type: str = 'kraus') -> MixedState:
        state = MixedState(self.fit('density_matrix'), self.system_dim, self.system_seq)
        return state.transform(op, sys_idx, repr_type)
