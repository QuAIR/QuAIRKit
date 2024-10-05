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
The source file of the density_matrix backend.
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ... import utils
from .. import State


class MixedState(State):
    r"""The mixed state class. 

    Args:
        data: tensor array (in density matrix representation) for quantum mixed state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 1 to n.

    Note:
        The data is stored in the matrix-form with shape :math:`(-1, d, d)`

    """
    def __init__(self, data: torch.Tensor, sys_dim: List[int], system_seq: Optional[List[int]] = None):
        dim = int(np.prod(sys_dim))
        
        self._batch_dim = list(data.shape[:-2])
        data = data.reshape([-1, dim, dim])
        super().__init__(data, sys_dim, system_seq)
    
    def __getitem__(self, key: Union[int, slice]) -> 'MixedState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return MixedState(self.density_matrix[key], self.system_dim, self.system_seq)
    
    def index_select(self, dim: int, index: torch.Tensor) -> 'MixedState':
        dim = dim - 2 if dim < 0 else dim 
        return MixedState(torch.index_select(self.density_matrix, dim=dim, index=index).squeeze(), 
                          self.system_dim, self.system_seq)
    
    def expand(self, batch_dim: List[int]) -> 'MixedState':
        expand_state = self.clone()
        expand_state._batch_dim = batch_dim
        
        if np.prod(batch_dim) == np.prod(self._batch_dim):
            return expand_state
        
        expand_state._data = self._data.expand(batch_dim + [-1, -1]).reshape([-1, self.dim, self.dim])
        return expand_state
    
    @property
    def backend(self) -> str:
        return "density_matrix"
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        if isinstance(sys_dim, List):
            num_sys = len(sys_dim)
            expected_dimension = int(np.prod(sys_dim))
        else:
            num_sys = int(math.log(data.shape[-2], sys_dim))
            expected_dimension = sys_dim ** num_sys
        
        assert data.shape[-2] == expected_dimension, \
            f"Input data shape {data.shape} does not match the input system dimension {sys_dim}."
        
        if eps:
            assert torch.all(is_density := utils.check._is_density_matrix(data, eps)), \
                f"(Some) data is not a density matrix: asserted {is_density.tolist()}"
        
        return num_sys

    @property
    def ket(self) -> torch.Tensor:
        raise NotImplementedError(
            "Mixed state does not support state vector representation." +
            "If you are looking for the vectorization of the mixed state, please call the 'vec' property.")

    @property
    def density_matrix(self) -> torch.Tensor:
        self.reset_sequence()
        return self._data.view(self._batch_dim + [self.dim, self.dim]).clone()
    
    def _trace(self, trace_idx: List[int]) -> 'MixedState':
        remain_seq, remain_system_dim = [], []
        for x in self._system_seq:
            if x not in trace_idx:
                remain_seq.append(x)
                remain_system_dim.append(self.system_dim[x])
        trace_dim = math.prod([self.system_dim[x] for x in trace_idx])
        self.system_seq = trace_idx + remain_seq
        
        data = self._data.clone().view(self._batch_dim + [self.dim, self.dim])
        data = utils.linalg._trace_1(data, trace_dim)
        
        # convert remaining sequence
        value_to_index = {value: index for index, value in enumerate(sorted(remain_seq))}
        remain_seq = [value_to_index[i] for i in remain_seq]
        return MixedState(data, remain_system_dim, remain_seq)
    
    def _transpose(self, transpose_idx: List[int]) -> 'MixedState':
        state = self.clone()
        state.system_seq = transpose_idx + [x for x in self._system_seq if x not in transpose_idx]
        
        transpose_dim = math.prod([self.system_dim[x] for x in transpose_idx])
        state._data = utils.linalg._transpose_1(state._data, transpose_dim)
        return state

    def normalize(self) -> None:
        self._data = torch.div(self._data, utils.linalg._trace(self._data, -2, -1).view([-1, 1, 1]))

    def clone(self) -> 'MixedState':
        data = self._data.view(self._batch_dim + [self.dim, self.dim]).clone()
        return MixedState(data, self.system_dim, self.system_seq)

    def fit(self, backend: str) -> torch.Tensor:
        data = self.density_matrix
        if backend == self.backend:
            return data
        
        if backend == 'state_vector':
            # TODO: add batch support
            return utils.linalg._density_to_vector(data)
        
        raise NotImplementedError(
            f"Unsupported state conversion from 'density_matrix' to {backend}.")

    @State.system_seq.setter
    def system_seq(self, target_seq: List[int]) -> None:
        if target_seq == self._system_seq:
            return
        
        perm_map = utils.linalg._perm_of_list(self._system_seq, target_seq)
        current_system_dim = [self._sys_dim[x] for x in self._system_seq]

        self._data = utils.linalg._base_transpose_for_dm(self._data, perm_map, current_system_dim).contiguous()
        self._system_seq = target_seq
    
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([-1, applied_dim, (self.dim ** 2) // applied_dim])
        data = torch.matmul(unitary, data)
        
        data = data.view([-1, self.dim, applied_dim, self.dim // applied_dim])
        self._data = torch.matmul(unitary.unsqueeze(-3).conj().clone(), data).view([-1, self.dim, self.dim])
    
    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        self._batch_dim = (self._batch_dim or list(unitary.shape[:-3])) + list(unitary.shape[-3:-2])
        unitary = unitary.view((list(unitary.shape[:-2]) or [-1]) + list(unitary.shape[-2:]))
        
        applied_dim, num_unitary = unitary.shape[-1], unitary.shape[-3]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        data = self._data.view([-1, 1, applied_dim, (self.dim ** 2) // applied_dim])
        data = torch.matmul(unitary, data)
        
        data = data.view([-1, num_unitary, self.dim, applied_dim, self.dim // applied_dim])
        unitary = unitary.unsqueeze(-3)
        data = torch.matmul(unitary.conj(), data)
        self._data = data.view([-1, self.dim, self.dim])
    
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        applied_dim, num_obs = obs.shape[-1], obs.shape[-3]
        state = self._data.view([-1, 1, applied_dim, (self.dim ** 2) // applied_dim])
        state = torch.matmul(obs, state).view([-1, num_obs, self.dim, self.dim])
    
        return utils.linalg._trace(state, -2, -1).view(self._batch_dim + [num_obs])
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'MixedState']:
        origin_batch_dim, origin_data = self._batch_dim.copy(), self._data.clone()
        self._evolve_keep_dim(measure_op, sys_idx)
        
        data = self._data.view([-1, self.dim, self.dim])
        
        prob = utils.linalg._trace(data, -2, -1).view([-1, 1, 1])
        collapsed_state = data / prob
        collapsed_state[collapsed_state != collapsed_state] = 0
        
        new_batch_dim = self._batch_dim
        collapsed_state = MixedState(collapsed_state.view(new_batch_dim + [self.dim, self.dim]), 
                                     self.system_dim, self.system_seq)
        prob = prob.view(new_batch_dim)
        
        self._batch_dim, self._data = origin_batch_dim, origin_data
        return prob.real, collapsed_state
    
    def __kraus_transform(self, list_kraus: torch.Tensor, sys_idx: List[int]) -> None:
        r"""Apply the Kraus operators to the state.
        
        Args:
            list_kraus: the Kraus operators.
            sys_idx: the system index list.
        """
        self._evolve_keep_dim(list_kraus, sys_idx)
        self._batch_dim, rank = self._batch_dim[:-1], self._batch_dim[-1]
        self._data = self._data.view([-1, rank, self.dim, self.dim]).sum(dim=-3)
    
    def __choi_transform(self, choi: torch.Tensor, sys_idx: List[int]) -> None:
        r"""Apply the Choi operator to the state.
        
        Args:
            choi: the Choi operator.
            sys_idx: the system index list.
        """
        self._batch_dim = self._batch_dim or list(choi.shape[:-2])
        
        refer_sys_idx = [x for x in self._system_seq if x not in sys_idx]
        dim_refer = int(np.prod([self.system_dim[x] for x in refer_sys_idx]))
        dim_in = int(np.prod([self.system_dim[x] for x in sys_idx]))
        dim_out = choi.shape[-1] // dim_in
        
        self.system_seq = refer_sys_idx + sys_idx
        data = self._data.view([-1, dim_refer, dim_in, dim_refer, dim_in]).transpose(-1, -3)
        data = torch.matmul(data.reshape([-1, (dim_refer ** 2) * dim_in, dim_in]), 
                            choi.view([-1, dim_in, dim_in * (dim_out ** 2)]))
        data = torch.transpose(data.view([-1, dim_in, dim_refer, dim_out, dim_in, dim_out]), -3, -4)
        data = utils.linalg._trace(data, -2, -5)
        self._data = data.view([-1, dim_refer * dim_out, dim_refer * dim_out])
    
    def _transform(self, op: torch.Tensor, sys_idx: List[int], repr_type: str) -> None:
        self.__kraus_transform(op, sys_idx) if repr_type == 'kraus' else self.__choi_transform(op, sys_idx)
    
    def sqrt(self) -> torch.Tensor:
        return utils.linalg._sqrtm(self.density_matrix)
    
    def log(self) -> torch.Tensor:
        if self.rank < self.dim:
            warnings.warn(
                "The state is not full-rank, thus the matrix logarithm may not be accurate.", UserWarning)
        
        return utils.linalg._logm(self.density_matrix)
