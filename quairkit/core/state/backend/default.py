# !/usr/bin/env python3
# Copyright (c) 2025 QuAIR team. All Rights Reserved.
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
The module that implements the default backend of the state.
"""

r"""
Default simulations for States in QuAIRKit obeys the following broadcast rule:

| Batch size of Operators | Batch size of Input State | Batch size of Output State |
|:-----------------------:|:-------------------------:|:--------------------------:|
|           1             |             1             |             1            |
|           n             |             1             |             n            |
|           1             |             n             |             n            |
|           n             |             n             |             n            |
|           m             |             n             |          (m, n)          |

Note that the last rule currently is only applicable to the `evolve_keep_dim` method.
"""

import math
import warnings
from abc import abstractmethod
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch

from ... import utils
from .base import State
from .simulator import StateSimulator

__all__ = ['DefaultSimulator', 'MixedState', 'PureState']


def _slice_len(key: Union[int, slice], N: int) -> int:
    r"""Return how many items a key (int or slice) would select from a sequence of length N.
    """
    if isinstance(key, int):
        i = key if key >= 0 else N + key
        return 1 if 0 <= i < N else 0
    if isinstance(key, slice):
        start, stop, step = key.indices(N)
        return len(range(start, stop, step))
    raise TypeError("key must be int or slice")


class DefaultSimulator(StateSimulator):
    r"""The abstract base class for default simulators of quantum states in QuAIRKit.
    """
    backend: str = 'default'
    
    @staticmethod
    def fetch_initializer(shape: List[int]) -> Union[Type['PureState'], Type['MixedState']]:
        r"""Determine whether the input data should be a (batch of) pure/mixed state in ``'default'`` backend.
        
        Note:
            When the ``backend`` is set as ``'default'``, the default initializer of this state is determined by the table
            
            +----------------+---------------------+---------------------+
            |                | single              | batch               |
            +================+=====================+=====================+
            | state_vector   | [d], [1, d], [d, 1] | [d1, ..., dn, d, 1] |
            +----------------+---------------------+---------------------+
            | density_matrix | [d, d]              | [d1, ..., dn, d, d] |
            +----------------+---------------------+---------------------+
        """
        if len(shape) == 1:
            return PureState

        if len(shape) == 2:
            return MixedState if shape[0] == shape[1] else PureState
        
        # batch case
        if shape[-1] == 1:   # Pure state batch
            return PureState
        if shape[-1] == shape[-2]:  # Mixed state batch
            return MixedState
        
        raise ValueError(
            f"The input data shape does not match the 'default' backend: received shape {shape}, "
            "expect one of [d], [1, d], [d, 1], [d, d], [..., d, 1] or [..., d, d].")
    
    def __new__(cls, data: torch.Tensor, sys_dim: List[int], 
                system_seq: Optional[List[int]], probability: Optional[List[torch.Tensor]]):
        subclass = cls.fetch_initializer(data.shape)
        instance = super(DefaultSimulator, subclass).__new__(subclass)
        if instance is not None and hasattr(instance, '__init__'):
            instance.__init__(data, sys_dim, system_seq, probability)
        return instance
        
    @abstractmethod
    def _expand(self, batch_dim: List[int]) -> 'DefaultSimulator':
        r"""Expand the batch dimension of the State.
        
        Args:
            batch_dim: the new batch dimension
            
        Note:
            For internal use only, see torch.expand() for more information about expand.
        
        """

    def expand_as(self, other: 'DefaultSimulator') -> 'DefaultSimulator':
        assert len(self._prob) == 0, \
            "Does not support expanding a State with probability"

        other_batch_dim = other.batch_dim
        expand_state = self._expand(other_batch_dim)
        expand_state._batch_dim = other._batch_dim.copy()

        for prob in other._prob.list:
            expand_state._prob.append(prob.clone(), normalize=False)
        return expand_state

    def __str__(self) -> str:
        split_line = "\n-----------------------------------------------------\n"
        s = f"{split_line} Backend: {self.backend}\n"
        s += f" System dimension: {self._sys_dim}\n"
        s += f" System sequence: {self._system_seq}\n"

        data = np.round(self.numpy(), decimals=2)
        interrupt_num = 5
        if not self.batch_dim:
            s += str(data.squeeze(0))
            s += split_line
            return s

        s += f" Batch size: {self.batch_dim}\n"
        for i, mat in enumerate(data):
            s += f"\n # {i}:\n{mat}"
            
            if i > interrupt_num:
                break_line = ("\n----------skipped for the rest of " + 
                              f"{list(data.shape)[0] - interrupt_num} states----------\n")
                s+= break_line
                return s
        
        s += split_line
        return s

    @property
    def data(self) -> torch.Tensor:
        r"""The data tensor of this state
        """
        warnings.warn(
            'The data property is depreciated, use ket or density_matrix instead', DeprecationWarning)
        return self._data
    
    def kron(self, other: 'DefaultSimulator') -> 'DefaultSimulator':
        system_dim = self.system_dim + other.system_dim
        system_seq = self.system_seq + [x + self.num_systems for x in other.system_seq]

        if self._prob:
            prob = self._prob.clone_list()
            if other._prob:
                warnings.warn(
                    "Detect tensor product of two probabilistic states: will discard prob info of the 2nd one", UserWarning)
        else:
            prob = other._prob.clone_list() if other._prob else None

        if self.backend == 'default-pure' and other.backend == 'default-pure':
            data = utils.linalg._kron(self.ket, other.ket)
            return PureState(data, system_dim, system_seq, prob)
        else:
            data = utils.linalg._kron(self.density_matrix, other.density_matrix)
            return MixedState(data, system_dim, system_seq, prob)
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        initializer = DefaultSimulator.fetch_initializer(data.shape)
        return initializer.check(data, sys_dim, eps)
    
    @property
    def batch_dim(self) -> List[int]:
        return self._batch_dim + self._prob.shape
    
    @property
    def _squeeze_shape(self) -> List[int]:
        r"""The squeezed shape of this state batch
        """
        return [-1, self._prob.product_dim]
    
    @property
    def shape(self) -> torch.Size:
        return self._data.shape
    
    @State.system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        self.reset_sequence()
        super(DefaultSimulator, DefaultSimulator).system_dim.__set__(self, sys_dim)
        self._system_seq = list(range(len(sys_dim)))

    @property
    def rank(self) -> Union[int, List[int]]:
        r"""The rank of the state.
        """
        dtype = self._data.dtype
        tol = 1e-8 if dtype == torch.complex64 else 1e-12
        tol *= self.dim
        result = torch.linalg.matrix_rank(self.density_matrix, 
                                          tol=tol, hermitian=True)
        return result.tolist() if self.batch_dim else int(result)
    
    def numpy(self) -> np.ndarray:
        return self._data.detach().numpy()
    
    def to(self, dtype: str = None, device: str = None) -> 'DefaultSimulator':
        new_state = self.clone()
        new_state._data = new_state._data.to(dtype=dtype, device=device)
        return new_state
    
    @property
    def system_seq(self) -> List[int]:
        r"""The system order of this state
        """
        return self._system_seq.copy()

    @system_seq.setter
    @abstractmethod
    def system_seq(self, target_seq: List[int]) -> None:
        r"""Set the system order of the state.

        Args:
            target_seq: the target systems order.

        """
    
    def permute(self, target_seq: List[int]) -> 'DefaultSimulator':
        new_state = self.clone()
        new_state.system_seq = target_seq
        new_state._system_seq = list(range(self.num_systems))
        return new_state

    def reset_sequence(self) -> None:
        r"""reset the system order to default sequence i.e. from 0 to n - 1.
        """
        self.system_seq = list(range(self.num_systems))
    

class MixedState(DefaultSimulator):
    r"""The mixed state class. 

    Args:
        data: tensor array (in density matrix representation) for quantum mixed state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 0 to n - 1.
        probability: list of state probability distributions. Defaults to be 1.
    
    Note:
        The data is stored in the matrix-form with shape :math:`(-1, d, d)`

    """
    backend: str = 'default-mixed'
    
    def __init__(self, data: torch.Tensor, sys_dim: List[int], 
                 system_seq: Optional[List[int]] = None,
                 probability: Optional[List[torch.Tensor]] = None):
        super().__init__(data, sys_dim, system_seq, probability)

        dim = self.dim
        self._data = data.reshape([-1, dim, dim])
    
    def __getitem__(self, key: Union[int, slice]) -> 'MixedState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return MixedState(self.density_matrix[key], self.system_dim, self.system_seq,
                          self._prob.clone_list())
    
    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> 'MixedState':
        num_prob = len(self._prob)
        if prob_idx > 0:
            prob_idx -= num_prob

        new_prob = []
        for idx, prob in enumerate(self._prob.list):
            if num_prob + prob_idx > idx:
                new_prob.append(prob.clone())
            else:
                new_prob.append(prob.index_select(dim=prob_idx, index=outcome_idx).squeeze(prob_idx))

        data_idx = prob_idx - 2
        data = self.density_matrix.index_select(dim=data_idx, index=outcome_idx).squeeze(data_idx)
        return MixedState(data, self.system_dim, self.system_seq, new_prob)
    
    def _expand(self, batch_dim: List[int]) -> 'MixedState':
        if original_batch_dim := self.batch_dim:
            assert batch_dim[-len(original_batch_dim):] == original_batch_dim, \
                f"Expand dimension mismatch: expected {original_batch_dim}, received {batch_dim[-len(original_batch_dim):]}."

        expand_state = self.clone()
        expand_state._data = self.density_matrix.expand(batch_dim + [-1, -1]).reshape([-1, self.dim, self.dim])
        return expand_state
    
    def add_probability(self, prob: torch.Tensor) -> None:
        shaped = self._prob.prepare_new(prob, dtype=self.dtype, device=self.device, real_only=True)
        data = self._data.view(self.batch_dim + [1, self.dim, self.dim])
        self._data = data.repeat([1] * len(self.batch_dim) + [shaped.shape[-1], 1, 1]).view([-1, self.dim, self.dim])
        self._prob.append(shaped, normalize=False)
    
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
        return self._data.view(self.batch_dim + [self.dim, self.dim]).clone()
    
    def _trace(self, trace_idx: List[int]) -> 'MixedState':
        remain_seq, remain_system_dim = [], []
        for x in self._system_seq:
            if x not in trace_idx:
                remain_seq.append(x)
                remain_system_dim.append(self.system_dim[x])
        trace_dim = math.prod([self.system_dim[x] for x in trace_idx])
        self.system_seq = trace_idx + remain_seq

        data = self._data.view(self.batch_dim + [self.dim, self.dim])
        data = utils.linalg._trace_1(data, trace_dim)

        # convert remaining sequence
        value_to_index = {value: index for index, value in enumerate(sorted(remain_seq))}
        remain_seq = [value_to_index[i] for i in remain_seq]
        return MixedState(data, remain_system_dim, remain_seq, self._prob.clone_list())
    
    def _reset(self, reset_idx: List[int], replace_state: 'DefaultSimulator') -> 'MixedState':
        remain_seq = [x for x in self._system_seq if x not in reset_idx]
        remain_seq.sort()

        trace_dim = math.prod([self.system_dim[x] for x in reset_idx])
        self.system_seq = reset_idx + remain_seq

        data = self._data.view(self.batch_dim + [self.dim, self.dim])
        data = utils.linalg._kron(utils.linalg._trace_1(data, trace_dim), replace_state.density_matrix)
        return MixedState(data, self.system_dim, self.system_seq, self._prob.clone_list())
    
    def _transpose(self, transpose_idx: List[int]) -> 'MixedState':
        self.system_seq = transpose_idx + [x for x in self._system_seq if x not in transpose_idx]
        transpose_dim = math.prod([self.system_dim[x] for x in transpose_idx])
        
        state = self.clone()
        state._data = utils.linalg._transpose_1(state._data, transpose_dim)
        return state

    def normalize(self) -> None:
        self._data = torch.div(self._data, utils.linalg._trace(self._data, -2, -1).view([-1, 1, 1]))

    def clone(self) -> 'MixedState':
        dim = self.dim
        data = self._data.view(self.batch_dim + [dim, dim]).clone()
        return MixedState(data, self.system_dim, self.system_seq, [prob.clone() for prob in self._prob])

    @DefaultSimulator.system_seq.setter
    def system_seq(self, target_seq: List[int]) -> None:
        if target_seq == self._system_seq:
            return
        
        perm_map = utils.linalg._perm_of_list(self._system_seq, target_seq)
        current_system_dim = [self._sys_dim[x] for x in self._system_seq]

        self._data = utils.linalg._permute_dm(self._data, perm_map, current_system_dim).contiguous()
        self._system_seq = target_seq
        
    def _index_select(self, new_indices: torch.Tensor) -> None:
        self.reset_sequence()
        self._data = self._data.index_select(-2, new_indices).index_select(-1, new_indices)
    
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        dim, _shape = self.dim, self._squeeze_shape
        if on_batch:
            self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
        
        applied_dim = unitary.shape[-1]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        unitary = unitary.view(evolve_axis + [applied_dim, applied_dim])
        data = self._data.view(_shape + [applied_dim, (dim ** 2) // applied_dim])
        data = torch.matmul(unitary, data)
        
        data = data.view(_shape + [dim, applied_dim, dim // applied_dim])
        self._data = torch.matmul(unitary.unsqueeze(-3).conj(), data).view([-1, dim, dim])
    
    def _evolve_ctrl(self, unitary: torch.Tensor, index: int, 
                     sys_idx: List[Union[int, List[int]]], on_batch: bool = True) -> None:
        ctrl_idx = sys_idx[0]
        ctrl_dim = np.prod([self.system_dim[idx] for idx in ctrl_idx])
        
        sys_idx = ctrl_idx + sys_idx[1:]
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        dim, _shape = self.dim, self._squeeze_shape
        data = self._data.view(self.batch_dim + [dim, dim]).clone()
        if on_batch:
            self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
        
        applied_dim = unitary.shape[-1]
        unitary = unitary.view(evolve_axis + [applied_dim, applied_dim])
        
        other_dim = dim // ctrl_dim   
        data = data.expand(self.batch_dim + [dim, dim]).clone()
             
        data = data.view(_shape + [ctrl_dim, applied_dim, (other_dim ** 2) * ctrl_dim // applied_dim]).clone()
        data[:, :, index] = torch.matmul(unitary, data[:, :, index].clone())
        
        data = data.view(_shape + [dim, ctrl_dim, applied_dim, other_dim // applied_dim]).clone()
        data[:, :, :, index] = torch.matmul(unitary.unsqueeze(-3).conj(), data[:, :, :, index].clone())
        
        self._data = data.view([-1, dim, dim])
    
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
        data = self._data.view(_shape + [1, applied_dim, (dim ** 2) // applied_dim])
        data = torch.matmul(unitary, data)
        
        data = data.view(_shape + [num_unitary, dim, applied_dim, dim // applied_dim])
        unitary = unitary.unsqueeze(-3)
        data = torch.matmul(unitary.conj(), data)
        self._data = data.view([-1, dim, dim])
    
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        dim = self.dim
        self.system_seq = sys_idx + [x for x in self._system_seq if x not in sys_idx]
        
        applied_dim, num_obs = obs.shape[-1], obs.shape[-3]
        state = self._data.view([-1, 1, applied_dim, (dim ** 2) // applied_dim])
        state = torch.matmul(obs, state).view([-1, dim, dim])
    
        return utils.linalg._trace(state, -2, -1).view(self.batch_dim + [num_obs])
    
    def _expec_state(self, prob_idx: List[int]) -> 'MixedState':
        dim, num_prob = self.dim, len(self._prob)
        batch_prob_len = len(self._prob.list[-1].shape[:-num_prob])
        joint = self._joint_probability(prob_idx)

        states = self._data.view([-1] + self._prob.shape + [dim ** 2])
        joint = joint.view(list(joint.shape) + [1] * (self._prob.list[-1].dim() - joint.dim() + 1))
        prob_state = torch.mul(joint, states)

        sum_idx = [idx + 1 for idx in prob_idx]
        expectation = prob_state.sum(sum_idx)

        new_prob: List[torch.Tensor] = []
        if len(prob_idx) != num_prob:
            new_prob = [p.clone() for idx, p in enumerate(self._prob.list) if idx not in prob_idx]
            expectation = expectation.view(self._batch_dim + list(new_prob[-1].shape[batch_prob_len:]) + [dim, dim])
        else:
            expectation = expectation.view(self._batch_dim + [dim, dim])
        return MixedState(expectation, self.system_dim, self.system_seq, new_prob)
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'MixedState']:
        new_state = self.clone()
        new_state._evolve_keep_dim(measure_op, sys_idx)

        data = new_state._data
        measure_prob = utils.linalg._trace(data, -2, -1).real.view([-1, 1, 1])
        mask = torch.abs(measure_prob) >= 1e-10
        collapsed_data = data / torch.where(mask, measure_prob, torch.ones_like(measure_prob))
        collapsed_data *= mask.to(collapsed_data.dtype)
        collapsed_data[torch.isnan(collapsed_data)] = 0

        measure_prob = measure_prob.view(new_state._batch_dim[:-1] + self._prob.shape + [-1])
        new_state._data = collapsed_data
        new_state._batch_dim = new_state._batch_dim[:-1]
        new_state._prob.append(measure_prob, normalize=False)
        return measure_prob, new_state
    
    def __kraus_transform(self, list_kraus: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        r"""Apply the Kraus operator to the state.
        
        Args:
            list_kraus: the Kraus operators.
            sys_idx: the system index list.
            on_batch: whether this operator evolves on batch axis. Defaults to True.
        
        """
        self._evolve_keep_dim(list_kraus, sys_idx)
        self._batch_dim, rank = self._batch_dim[:-1], self._batch_dim[-1]
        self._data = self._data.view([-1, rank, self.dim, self.dim]).sum(dim=-3)
    
    def __choi_transform(self, choi: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        r"""Apply the Choi operator to the state.
        
        Args:
            choi: the Choi operator.
            sys_idx: the system index list.
            on_batch: whether this operator evolves on batch axis. Defaults to True.
        
        """
        _shape = self._squeeze_shape
        if on_batch:
            self._batch_dim = self._batch_dim or list(choi.shape[:-2])
            transform_axis = [-1, 1]
        else:
            transform_axis = [1, -1]
        
        refer_sys_idx = [x for x in self._system_seq if x not in sys_idx]
        dim_refer = int(np.prod([self.system_dim[x] for x in refer_sys_idx]))
        dim_in = int(np.prod([self.system_dim[x] for x in sys_idx]))
        dim_out = choi.shape[-1] // dim_in
        
        self.system_seq = refer_sys_idx + sys_idx
        data = self._data.view(_shape + [dim_refer, dim_in, dim_refer, dim_in])
        choi = choi.view(transform_axis + [dim_in, dim_in * (dim_out ** 2)])
        
        data = torch.matmul(data.transpose(-1, -3).reshape(_shape + [(dim_refer ** 2) * dim_in, dim_in]), choi)
        data = torch.transpose(data.view(_shape + [dim_in, dim_refer, dim_out, dim_in, dim_out]), -3, -4)
        data = utils.linalg._trace(data, -2, -5)
        self._data = data.view([-1, dim_refer * dim_out, dim_refer * dim_out])
    
    def _transform(self, op: torch.Tensor, sys_idx: List[int], repr_type: str) -> None:
        self.__kraus_transform(op, sys_idx) if repr_type == 'kraus' else self.__choi_transform(op, sys_idx)
    
    def sqrt(self) -> torch.Tensor:
        return utils.linalg._sqrtm(self.density_matrix)
    
    def log(self) -> torch.Tensor:
        max_rank = max(self.rank) if self.batch_dim else self.rank
        if max_rank < self.dim:
            warnings.warn(
                f"The matrix logarithm may not be accurate: expect rank {self.dim}, received {self.rank}", UserWarning)
        return utils.linalg._logm(self.density_matrix)


class PureState(DefaultSimulator):
    r"""The pure state class.

    Args:
        data: tensor array in vector representation for quantum pure state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 1 to n.
        probability: tensor array for state distributions. Defaults to be 1.

    Note:
        The data is stored in the vector-form with shape :math:`(-1, d)`

    """
    backend: str = 'default-pure'
    
    def __init__(self, data: torch.Tensor, sys_dim: List[int], 
                 system_seq: Optional[List[int]] = None,
                 probability: Optional[List[torch.Tensor]] = None):
        super().__init__(data, sys_dim, system_seq, probability)
        
        dim = self.dim
        self._data = data.reshape([-1, dim])
        
    def __getitem__(self, key: Union[int, slice]) -> 'PureState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return PureState(self.ket[key], self.system_dim, self.system_seq,
                         self._prob.clone_list())
    
    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> 'PureState':
        num_prob = len(self._prob)
        if prob_idx > 0:
            prob_idx -= num_prob

        new_prob = []
        for idx, prob in enumerate(self._prob.list):
            if num_prob + prob_idx > idx:
                new_prob.append(prob.clone())
            else:
                new_prob.append(prob.index_select(dim=prob_idx, index=outcome_idx).squeeze(prob_idx))

        data_idx = prob_idx - 2
        data = self.ket.index_select(dim=data_idx, index=outcome_idx).squeeze(data_idx)
        return PureState(data, self.system_dim, self.system_seq, new_prob)
    
    def _expand(self, batch_dim: List[int]) -> 'PureState':
        if original_batch_dim := self.batch_dim:
            assert batch_dim[-len(original_batch_dim):] == original_batch_dim, \
                    f"Expand dimension mismatch: expected {original_batch_dim}, received {batch_dim[-len(original_batch_dim):]}."

        expand_state = self.clone()
        expand_state._data = self.ket.expand(batch_dim + [-1, -1]).reshape([-1, self.dim])
        return expand_state
    
    def add_probability(self, prob: torch.Tensor) -> None:
        shaped = self._prob.prepare_new(prob)
        data = self._data.view(self.batch_dim + [1, self.dim])
        self._data = data.repeat([1] * len(self.batch_dim) + [shaped.shape[-1], 1]).view([-1, self.dim])
        self._prob.append(shaped, normalize=False)
    
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
                          self._prob.clone_list())
    
    def _trace(self, sys_idx: List[int]) -> MixedState:
        return self.to_mixed()._trace(sys_idx)
    
    def _reset(self, reset_idx: List[int], replace_state: 'DefaultSimulator') -> 'MixedState':
        return self.to_mixed()._reset(reset_idx, replace_state)
    
    def _transpose(self, sys_idx: List[int]) -> MixedState:
        return self.to_mixed()._transpose(sys_idx)

    @property
    def rank(self) -> Union[int, List[int]]:
        return torch.ones(self.batch_dim).tolist() if self.batch_dim else 1
    
    def normalize(self) -> None:
        self._data = torch.div(self._data, torch.norm(self._data, dim=-1))

    def clone(self) -> 'PureState':
        data = self._data.view(self.batch_dim + [self.dim, 1]).clone()
        state = PureState(data, self.system_dim, self.system_seq, [prob.clone() for prob in self._prob])
        state._keep_dim = self._keep_dim
        return state

    @DefaultSimulator.system_seq.setter
    def system_seq(self, target_seq: List[int]) -> None:
        if target_seq == self._system_seq:
            return
        
        perm_map = utils.linalg._perm_of_list(self._system_seq, target_seq)
        current_system_dim = [self._sys_dim[x] for x in self._system_seq]
        
        self._data = utils.linalg._permute_sv(self._data, perm_map, current_system_dim).contiguous()
        self._system_seq = target_seq
        
    def _index_select(self, new_indices: torch.Tensor) -> None:
        self.reset_sequence()
        self._data = self._data.index_select(dim=-1, index=new_indices)
        
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
        if self._keep_dim:
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
        
        if self._keep_dim:
            proj = torch.zeros([ctrl_dim, ctrl_dim])
            proj[index, index] = 1
            unitary = torch.kron(proj, unitary) + torch.kron(torch.eye(ctrl_dim) - proj, torch.eye(unitary.shape[-1]).expand_as(unitary))
            self._record_unitary(unitary, sys_idx)
            return
        dim, _shape = self.dim, self._squeeze_shape
        
        data = self._data.view(self.batch_dim + [dim])
        if on_batch:
            self._batch_dim = self._batch_dim or list(unitary.shape[:-2])
            evolve_axis = [-1, 1]
        else:
            evolve_axis = [1, -1]
            
        applied_dim = unitary.shape[-1]
        unitary = unitary.view(evolve_axis + [applied_dim, applied_dim])
        
        other_dim = dim // ctrl_dim
        data = data.expand(self.batch_dim + [dim]).clone()
        
        data = data.view(_shape + [ctrl_dim, applied_dim, other_dim // applied_dim]).clone()
        data[:, :, index] = torch.matmul(unitary, data[:, :, index].clone())
        
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
    
    def _expec_state(self, prob_idx: List[int]) -> 'MixedState':
        return self.to_mixed()._expec_state(prob_idx)
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'PureState']:
        new_state = self.clone()
        new_state._evolve_keep_dim(measure_op, sys_idx)

        data = new_state._data.view([-1, self.dim, 1])
        measure_prob = (data.mH @ data).real
        mask = torch.abs(measure_prob) >= 1e-10
        collapsed_data = data / torch.sqrt(torch.where(mask, measure_prob, torch.ones_like(measure_prob)))
        collapsed_data *= mask.to(collapsed_data.dtype)
        collapsed_data[torch.isnan(collapsed_data)] = 0

        measure_prob = measure_prob.view(new_state._batch_dim[:-1] + self._prob.shape + [-1])
        new_state._data = collapsed_data.view([-1, self.dim])
        new_state._batch_dim = new_state._batch_dim[:-1]
        new_state._prob.append(measure_prob, normalize=False)
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
        if trace_state.batch_dim:
            assert trace_state.batch_dim == self.batch_dim[-len(trace_state.batch_dim):], \
                f"# of traced states mismatch: received {trace_state.batch_dim}, expect {self.batch_dim[-len(trace_state.batch_dim):]}."
            num_trace = np.prod(trace_state.batch_dim)
        else:
            num_trace = 1

        trace_state = trace_state.ket.squeeze(-1)
        self.system_seq = trace_idx + remain_seq

        data = self._data.view([-1] + [num_trace, self.dim])
        data = utils.linalg._ptrace_1(data, trace_state)
        data = data.view(self.batch_dim + [-1, 1])

        # convert remaining sequence
        value_to_index = {value: index for index, value in enumerate(sorted(remain_seq))}
        remain_seq = [value_to_index[i] for i in remain_seq]
        return PureState(data, remain_system_dim, remain_seq, self._prob.clone_list())
    
    def sqrt(self) -> torch.Tensor:
        return self.density_matrix
    
    def log(self) -> torch.Tensor:
        return torch.zeros_like(self.density_matrix)

