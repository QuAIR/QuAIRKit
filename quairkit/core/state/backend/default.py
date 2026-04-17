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

import importlib
import itertools
import math
import warnings
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from ... import utils
from .base import State
from .simulator import ProbabilityData, StateSimulator

__all__ = ['DefaultSimulator', 'MixedState', 'PureState', 'ProductDefaultSimulator']


_CPP_MOD = importlib.import_module("quairkit._C")
_CPP_STATE = getattr(_CPP_MOD, "state")
_CPP_LINALG = getattr(_CPP_MOD, "linalg")


def _validate_subgroup_indices(subgroup_indices: List[List[int]], num_systems: int) -> List[List[int]]:
    if not subgroup_indices:
        raise ValueError("subgroup_indices must be non-empty.")
    normalized = [[int(i) for i in group] for group in subgroup_indices]
    if any(not group for group in normalized):
        raise ValueError("Each subgroup in subgroup_indices must be non-empty.")
    flat = [idx for group in normalized for idx in group]
    expected = list(range(num_systems))
    if sorted(flat) != expected:
        raise ValueError(
            f"subgroup_indices must be a partition of [0..{num_systems - 1}], "
            f"but got flattened indices {flat}."
        )
    return normalized

def _resolve_prod_sum_tol(tol: Optional[float], dtype: torch.dtype, dim: int) -> float:
    if tol is not None:
        if tol < 0:
            raise ValueError(f"tol must be non-negative, but got {tol}.")
        return float(tol)
    base = 1e-8 if dtype == torch.complex64 else 1e-12
    return float(base * max(1, dim))


def _reshape_dense_batch_tensor(
    data: torch.Tensor, batch_shape: List[int], tail_shape: List[int]
) -> torch.Tensor:
    if batch_shape:
        return data.view(batch_shape + tail_shape)

    tail_numel = int(np.prod(tail_shape))
    inferred = int(data.numel() // tail_numel)
    return data.view(tail_shape if inferred == 1 else [inferred] + tail_shape)


def _prepare_dense_to_prod_sum(
    state: "DefaultSimulator",
    subgroup_indices: List[List[int]],
    *,
    tol: Optional[float],
    is_matrix: bool,
) -> Tuple["DefaultSimulator", List[List[int]], List[List[int]], List[int], float, torch.Tensor]:
    normalized = _validate_subgroup_indices(subgroup_indices, state.num_systems)
    target_seq = [idx for group in normalized for idx in group]

    state = state.clone()
    state.reset_sequence()
    if target_seq != list(range(state.num_systems)):
        state = state.permute(target_seq)

    subgroup_system_dims = [[state.system_dim[i] for i in group] for group in normalized]
    local_dims = [int(np.prod(group_dim)) for group_dim in subgroup_system_dims]
    tol_value = _resolve_prod_sum_tol(tol, state.dtype, state.dim)
    tail_shape = [state.dim, state.dim] if is_matrix else [state.dim]
    tensor = _reshape_dense_batch_tensor(state._cpp_state.data(), state.batch_dim, tail_shape)
    return state, normalized, subgroup_system_dims, local_dims, tol_value, tensor


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
        if shape[-1] == 1:
            return PureState
        if shape[-1] == shape[-2]:
            return MixedState
        raise ValueError(
            f"The input data shape does not match the 'default' backend: received shape {shape}, "
            "expect one of [d], [1, d], [d, 1], [d, d], [..., d, 1] or [..., d, d]."
        )
    
    def __new__(cls, data: torch.Tensor, sys_dim: List[int],
                system_seq: Optional[List[int]], probability: Optional[List[torch.Tensor]]):
        cpp_state = _CPP_STATE.create_state(
            data, sys_dim, system_seq or [], probability or []
        )
        subclass = PureState if isinstance(cpp_state, _CPP_STATE.PureState) else MixedState
        instance = super(DefaultSimulator, subclass).__new__(subclass)
        instance._cpp_state = cpp_state
        instance._initialized_from_cpp = True
        return instance

    def __init__(self, data: torch.Tensor, sys_dim: List[int],
                 system_seq: Optional[List[int]], probability: Optional[List[torch.Tensor]]) -> None:
        self._cpp_state = _CPP_STATE.create_state(data, sys_dim, system_seq or [], probability or [])
        State.__init__(self, sys_dim)
        self._prob = ProbabilityData(probability or [])
        self._prob._impl = self._cpp_state.prob
        self._keep_dim = False
    
    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> 'DefaultSimulator':
        if len(self._cpp_state.prob) == 0:
            raise ValueError("prob_select is not applicable: this state has no probability variables.")
        num_prob = len(self._prob)
        normalized_idx = num_prob + prob_idx if prob_idx < 0 else prob_idx
        if normalized_idx < 0 or normalized_idx >= num_prob:
            raise IndexError(f"Probability index out of range: got {prob_idx}, expected in [-{num_prob}, {num_prob - 1}].")
        cpp_prob_idx = normalized_idx - num_prob
        selected = self._cpp_state.prob_select(outcome_idx, cpp_prob_idx)
        return DefaultSimulator._wrap_cpp_state(selected)
    
    def add_probability(self, prob: torch.Tensor) -> None:
        self._cpp_state.add_probability(prob)
    
    def clone(self) -> 'DefaultSimulator':
        return DefaultSimulator._wrap_cpp_state(self._cpp_state.clone())
    
    def _index_select(self, new_indices: torch.Tensor, system_idx_pairs: Optional[List[List[int]]] = None) -> None:
        self._cpp_state.index_select(new_indices)

    def _adopt_cpp_state(self, cpp_state: Union["_CPP_STATE.PureState", "_CPP_STATE.MixedState"]) -> None:
        """Replace internal storage with a new authoritative C++ state (in-place)."""
        self._cpp_state = cpp_state
        self._prob = ProbabilityData()
        self._prob._impl = self._cpp_state.prob

    @abstractmethod
    def to_prod_sum(self, subgroup_indices: List[List[int]], tol: Optional[float] = None) -> "ProductDefaultSimulator":
        r"""Convert the state into a subgroup-level product-sum container."""

    def _evolve_many(
        self,
        unitary: torch.Tensor,
        sys_idx_list: List[List[int]],
        on_batch: bool = True,
    ) -> None:
        if self._keep_dim:
            super()._evolve_many(unitary, sys_idx_list, on_batch)
            return

        self._cpp_state.evolve_many(unitary, sys_idx_list, on_batch)
        return None

    def _evolve_many_batched_groups(
        self,
        unitary_groups: List[torch.Tensor],
        sys_idx_groups: List[List[List[int]]],
        on_batch: bool = True,
    ) -> None:
        if self._keep_dim:
            super()._evolve_many_batched_groups(unitary_groups, sys_idx_groups, on_batch)
            return

        self._cpp_state.evolve_many_batched(unitary_groups, sys_idx_groups, on_batch)

    @staticmethod
    def _wrap_cpp_state(
        cpp_state: Union["_CPP_STATE.PureState", "_CPP_STATE.MixedState"],
    ) -> "DefaultSimulator":
        """Wrap a C++ state object into the corresponding thin Python class."""
        subclass = PureState if isinstance(cpp_state, _CPP_STATE.PureState) else MixedState
        inst = super(DefaultSimulator, subclass).__new__(subclass)
        inst._cpp_state = cpp_state
        State.__init__(inst, list(cpp_state.system_dim))
        init_probs = list(cpp_state.prob.list)
        inst._prob = ProbabilityData(init_probs)
        inst._prob._impl = cpp_state.prob
        inst._keep_dim = False
        return inst

    def __str__(self) -> str:
        split_line = "\n-----------------------------------------------------\n"
        s = f"{split_line} Backend: {self.backend}\n"
        s += f" System dimension: {self._sys_dim}\n"
        s += f" System sequence: {list(self._cpp_state.system_seq)}\n"

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
    
    def kron(self, other: 'DefaultSimulator | ProductDefaultSimulator') -> 'ProductDefaultSimulator | DefaultSimulator':
        if len(self._prob) and len(getattr(other, "_prob", [])):
            raise ValueError("Cannot take kron of two probabilistic states.")

        left = self.clone()
        left.reset_sequence()
        idx_left = list(range(left.num_systems))
        shift = left.num_systems

        base_prob = ProbabilityData()
        if len(left._prob):
            base_prob = left._prob.clone()
        elif len(getattr(other, "_prob", [])):
            base_prob = other._prob.clone()

        if isinstance(other, ProductDefaultSimulator):
            new_states = [left] + [s.clone() for s in other._list_state]
            new_idxs = [idx_left] + [[i + shift for i in idx] for idx in other._list_state_idx]
            keep_dim = self._keep_dim or other._keep_dim
            roles = [ProductDefaultSimulator._ROLE_CLASSICAL] * len(base_prob) if len(left._prob) else other.roles
            return ProductDefaultSimulator(
                list_state=new_states,
                list_state_idx=new_idxs,
                _prob=base_prob,
                _roles=roles,
                _keep_dim=keep_dim,
            )

        right = other.clone()
        right.reset_sequence()
        idx_right = [i + shift for i in range(right.num_systems)]

        return ProductDefaultSimulator(
            list_state=[left, right],
            list_state_idx=[idx_left, idx_right],
            _prob=base_prob,
            _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * len(base_prob),
            _keep_dim=False,
        )
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        kind = _CPP_STATE.create_state_type(data)
        initializer = PureState if kind == "pure" else MixedState
        return initializer.check(data, sys_dim, eps)
    
    @property
    def batch_dim(self) -> List[int]:
        return list(self._cpp_state.batch_dim) + self._prob.shape
    
    @property
    def _squeeze_shape(self) -> List[int]:
        r"""The squeezed shape of this state batch
        """
        return [-1, self._prob.product_dim]
    
    @property
    def shape(self) -> torch.Size:
        return self._cpp_state.data().shape
    
    @State.system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        self.reset_sequence()
        super(DefaultSimulator, DefaultSimulator).system_dim.__set__(self, sys_dim)

    @property
    def rank(self) -> Union[int, List[int]]:
        r"""The rank of the state.
        """
        dtype = self._cpp_state.data().dtype
        tol = 1e-8 if dtype == torch.complex64 else 1e-12
        tol *= self.dim
        result = torch.linalg.matrix_rank(self.density_matrix, 
                                          tol=tol, hermitian=True)
        return result.tolist() if self.batch_dim else int(result)
    
    def numpy(self) -> np.ndarray:
        return self._cpp_state.data().detach().cpu().numpy()
    
    def to(self, dtype: str = None, device: str = None) -> 'DefaultSimulator':
        new_state = self.clone()
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        dev = torch.device(device) if isinstance(device, str) else device
        new_state._cpp_state.to(dtype=dt, device=dev)
        return new_state
    
    @property
    def system_seq(self) -> List[int]:
        r"""The system order of this state
        """
        return list(self._cpp_state.system_seq)

    @system_seq.setter
    def system_seq(self, target_seq: List[int]) -> None:
        r"""Set the system order of the state.

        Args:
            target_seq: the target systems order.

        """
        self._cpp_state.set_system_seq(target_seq)

    @property
    def system_dim(self) -> List[int]:
        return self._sys_dim.copy()

    @system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        assert int(np.prod(sys_dim)) == self.dim, \
            f"The input system dim {sys_dim} does not match the original state dim {self.dim}."
        self._sys_dim = sys_dim.copy()
        self._cpp_state.set_system_dim(sys_dim)
    
    def permute(self, target_seq: List[int]) -> 'DefaultSimulator':
        n = self.num_systems
        assert sorted(target_seq) == list(range(n)), \
            f"target_seq must be a permutation of [0..{n-1}], got {target_seq}"
        if target_seq == list(range(n)):
            return self.clone()

        state = self.clone()
        state.reset_sequence()
        state.system_seq = target_seq
        state._cpp_state.set_system_seq_metadata(list(range(n)))
        state._keep_dim = self._keep_dim
        return state

    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        self._cpp_state.evolve(unitary, sys_idx, on_batch)
    
    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        self._cpp_state.evolve_keep_dim(unitary, sys_idx, on_batch)
    
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        return self._cpp_state.expectation_value(obs, sys_idx)

    def _expec_val_pauli_terms(
        self,
        pauli_words_r: List[str],
        sites: List[List[int]],
    ) -> torch.Tensor:
        return self._cpp_state.expec_val_pauli_terms(pauli_words_r, sites).real
    
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'DefaultSimulator']:
        prob, cpp_state = self._cpp_state.measure(measure_op, sys_idx)
        collapsed = DefaultSimulator._wrap_cpp_state(cpp_state)
        return prob, collapsed
    
    def _measure_many(self, measure_op: torch.Tensor, sys_idx_list: List[List[int]]) -> Tuple[torch.Tensor, 'DefaultSimulator']:
        prob, cpp_state = self._cpp_state.measure_many(measure_op, sys_idx_list)
        collapsed = DefaultSimulator._wrap_cpp_state(cpp_state)
        return prob, collapsed
    
    def __measure_by_state_pure(
        self, measure_basis: 'PureState', sys_idx: List[int], keep_state: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, 'ProductDefaultSimulator']]:
        r"""Measure using PureState basis (single block of kets).

        Args:
            measure_basis: PureState containing batched measurement basis kets.
            sys_idx: System indices to measure.
            keep_state: If ``False``, return only the probability tensor.

        Returns:
            Probability tensor, or ``(prob, collapsed ProductDefaultSimulator)`` when ``keep_state`` is ``True``.
        """
        measured_sys_idx = sys_idx
        rest_sys_idx = [i for i in range(self.num_systems) if i not in measured_sys_idx]

        prob, rest_cpp_opt = self._cpp_state.measure_by_state(
            measure_basis.ket, sys_idx, keep_state
        )
        if not keep_state:
            return prob
        rest_cpp_state = rest_cpp_opt
        
        measured_state = measure_basis.clone()
        num_outcomes = int(prob.shape[-1])
        data_batch = list(self.batch_dim)
        target_batch = data_batch + [num_outcomes]
        
        measured_state._expand_data_batch_dim(target_batch)
        measured_state._cpp_state.prob.clear()
        
        if rest_sys_idx:
            rest_state = DefaultSimulator._wrap_cpp_state(rest_cpp_state)
            rest_state._cpp_state.set_batch_dim(target_batch)
            rest_state._cpp_state.prob.clear()
            
            prior_probs = list(self._cpp_state.prob.list)
            collapsed = ProductDefaultSimulator(
                [measured_state, rest_state],
                [measured_sys_idx, rest_sys_idx],
                ProbabilityData(prior_probs + [prob]),
                _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * (len(prior_probs) + 1),
                _keep_dim=self._keep_dim,
            )
        else:
            prior_probs = list(self._cpp_state.prob.list)
            collapsed = ProductDefaultSimulator(
                [measured_state],
                [measured_sys_idx],
                ProbabilityData(prior_probs + [prob]),
                _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * (len(prior_probs) + 1),
                _keep_dim=self._keep_dim,
            )
        
        collapsed._sync_data_batch_dim()
        
        return prob, collapsed
    
    def __measure_by_state_product(
        self, measure_basis: 'ProductDefaultSimulator', sys_idx: List[int], keep_state: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, 'ProductDefaultSimulator']]:
        r"""Measure using ProductDefaultSimulator basis (multi-block).

        Args:
            measure_basis: ProductDefaultSimulator containing measurement basis blocks.
            sys_idx: System indices to measure (maps to measure_basis block structure).
            keep_state: If ``False``, return only the probability tensor.

        Returns:
            Probability tensor, or ``(prob, collapsed ProductDefaultSimulator)`` when ``keep_state`` is ``True``.

        Note:
            `measure_basis._list_state_idx` is treated as **local labels** for the measured
            subsystems and must follow the convention ``0..k-1`` (where ``k = len(sys_idx)``),
            in the same order as `sys_idx`. In other words, local label ``i`` corresponds to
            the measured system `sys_idx[i]`.
        """
        mapped_sys_idx = []
        flat_idx = 0
        for group in measure_basis._list_state_idx:
            mapped_group = [sys_idx[flat_idx + i] for i in range(len(group))]
            mapped_sys_idx.append(mapped_group)
            flat_idx += len(group)
        
        measured_sys_idx = [x for group in mapped_sys_idx for x in group]
        rest_sys_idx = [i for i in range(self.num_systems) if i not in measured_sys_idx]
        
        prob, rest_cpp_opt = self._cpp_state.measure_by_state_product(
            [state.ket for state in measure_basis._list_state],
            mapped_sys_idx,
            keep_state,
        )
        if not keep_state:
            return prob
        rest_cpp_state = rest_cpp_opt
        
        measured_state = measure_basis.clone()
        num_outcomes = int(prob.shape[-1])
        data_batch = list(self.batch_dim)
        target_batch = data_batch + [num_outcomes]
        
        for s in measured_state._list_state:
            s._expand_data_batch_dim(target_batch)
            s._cpp_state.prob.clear()
        
        measured_blocks = measured_state._list_state
        
        if rest_sys_idx:
            rest_state = DefaultSimulator._wrap_cpp_state(rest_cpp_state)
            rest_state._cpp_state.set_batch_dim(target_batch)
            rest_state._cpp_state.prob.clear()
            
            prior_probs = list(self._cpp_state.prob.list)
            collapsed = ProductDefaultSimulator(
                measured_blocks + [rest_state],
                mapped_sys_idx + [rest_sys_idx],
                ProbabilityData(prior_probs + [prob]),
                _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * (len(prior_probs) + 1),
                _keep_dim=self._keep_dim,
            )
        else:
            prior_probs = list(self._cpp_state.prob.list)
            collapsed = ProductDefaultSimulator(
                measured_blocks,
                mapped_sys_idx,
                ProbabilityData(prior_probs + [prob]),
                _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * (len(prior_probs) + 1),
                _keep_dim=self._keep_dim,
            )
        
        collapsed._sync_data_batch_dim()
        return prob, collapsed
    
    def _measure_by_state(
        self,
        measure_basis: Union['PureState', 'ProductDefaultSimulator'],
        sys_idx: List[int],
        keep_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, 'ProductDefaultSimulator']]:
        r"""Measure the state using measure_basis.

        The collapsed state is composed of:
        - The measured part: directly from measure_basis (known outcome states)
        - The unmeasured part: rest state from C++ computation (if any)

        This avoids explicit kron by utilizing ProductDefaultSimulator's lazy evaluation.

        Args:
            measure_basis: PureState or ProductDefaultSimulator containing measurement basis.
            sys_idx: System indices to measure.
            keep_state: If ``False``, return only the probability tensor.

        Returns:
            Probability tensor, or ``(prob, collapsed ProductDefaultSimulator)`` when ``keep_state`` is ``True``.
        """
        if isinstance(measure_basis, ProductDefaultSimulator):
            return self.__measure_by_state_product(measure_basis, sys_idx, keep_state)
        return self.__measure_by_state_pure(measure_basis, sys_idx, keep_state)

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
    
    def __getitem__(self, key: Union[int, slice]) -> 'MixedState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return MixedState(
            self.density_matrix[key],
            self.system_dim,
            self.system_seq,
            self._prob.clone_list(),
        )

    def _expand_data_batch_dim(self, target_batch_dim: List[int]) -> None:
        r"""Expand data to target batch dimension (memory-sharing via expand).
        
        Args:
            target_batch_dim: Target data batch dimensions to expand to.
        """
        cur_bd = list(self._cpp_state.batch_dim)
        if cur_bd == target_batch_dim or not target_batch_dim:
            return
        
        d = int(self.dim)
        
        if not cur_bd:
            cur_shape = [1] * len(target_batch_dim) + [d, d]
            target_shape = target_batch_dim + [d, d]
            data = self._cpp_state.data().view(cur_shape)
        else:
            pad_len = len(target_batch_dim) - len(cur_bd)
            if pad_len > 0:
                if target_batch_dim[:len(cur_bd)] == cur_bd:
                    cur_bd_padded = cur_bd + [1] * pad_len
                else:
                    cur_bd_padded = [1] * pad_len + cur_bd
            else:
                cur_bd_padded = cur_bd
            cur_shape = cur_bd_padded + [d, d]
            target_shape = target_batch_dim + [d, d]
            data = self._cpp_state.data().view(cur_shape)
        
        expanded = data.expand(target_shape).contiguous()
        self._cpp_state.set_data(expanded.view(-1, d, d))
        self._cpp_state.set_batch_dim(list(target_batch_dim))
    
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
            if data.ndim == 2:
                data_check = data
            else:
                d = int(data.shape[-1])
                data_check = data.reshape([-1, d, d])
            assert torch.all(is_density := utils.check._is_density_matrix(data_check, eps)), \
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
        density = self._cpp_state.density_matrix()
        if self.batch_dim:
            density = density.view(self.batch_dim + list(density.shape[-2:]))
        if not self.batch_dim and not self._keep_dim and density.ndim > 2 and density.shape[0] == 1:
            density = density.squeeze(0)
        return density

    def _trace(self, trace_idx: List[int]) -> 'MixedState':
        return DefaultSimulator._wrap_cpp_state(self._cpp_state.trace(trace_idx))
    
    def _reset(self, reset_idx: List[int], replace_state: 'DefaultSimulator | ProductDefaultSimulator') -> 'MixedState':
        return DefaultSimulator._wrap_cpp_state(
            self._cpp_state.reset(reset_idx, replace_state.density_matrix)
        )
    
    def _transpose(self, transpose_idx: List[int]) -> 'MixedState':
        return DefaultSimulator._wrap_cpp_state(self._cpp_state.transpose(transpose_idx))

    def normalize(self) -> None:
        data = self._cpp_state.data()
        self._cpp_state.set_data(
            torch.div(data, utils.linalg._trace(data, -2, -1).view([-1, 1, 1]))
        )
    
    def _evolve_ctrl(self, unitary: torch.Tensor, index: int, 
                     sys_idx: List[Union[int, List[int]]], on_batch: bool = True) -> None:
        normalized: List[List[int]] = []
        for v in sys_idx:
            if isinstance(v, list):
                normalized.append(v)
            else:
                normalized.append([int(v)])
        ctrl_idx = normalized[0]
        ctrl_dim = np.prod([self.system_dim[idx] for idx in ctrl_idx])
        index = int(index) % int(ctrl_dim)
        self._cpp_state.evolve_ctrl(unitary, index, normalized, on_batch)
    
    def _expec_state(self, prob_idx: List[int]) -> 'MixedState':
        return DefaultSimulator._wrap_cpp_state(self._cpp_state.expec_state(prob_idx))

    def to_prod_sum(self, subgroup_indices: List[List[int]], tol: Optional[float] = None) -> "ProductDefaultSimulator":
        state, subgroup_indices, subgroup_system_dims, local_dims, tol_value, matrix = (
            _prepare_dense_to_prod_sum(self, subgroup_indices, tol=tol, is_matrix=True)
        )

        factors, coeffs = utils.linalg._matrix_to_prod_sum(matrix, local_dims, tol_value)
        cpp_state = _CPP_STATE.build_product_from_matrix_prod_sum(
            factors,
            coeffs,
            subgroup_indices,
            subgroup_system_dims,
            state._prob.clone_list(),
            state._keep_dim,
            tol_value,
        )
        return ProductDefaultSimulator._wrap_cpp_product_state(cpp_state)
    
    def _transform(self, op: torch.Tensor, sys_idx: List[int], repr_type: str, on_batch: bool = True) -> None:
        if repr_type == "kraus":
            self._cpp_state.transform_kraus(op, sys_idx, on_batch)
        else:
            self._cpp_state.transform_choi(op, sys_idx, on_batch)

    def _transform_many(
        self,
        op: torch.Tensor,
        sys_idx_list: Union[Iterable[Union[int, List[int]]], List[List[int]], List[int], int],
        repr_type: str,
        on_batch: bool = True,
    ) -> None:
        checked: List[List[int]] = []
        for sys_idx in sys_idx_list:
            sys_idx_checked = self._check_sys_idx(sys_idx)
            if repr_type == 'kraus':
                self._check_op_dim(op, sys_idx_checked)
            checked.append(sys_idx_checked)

        if repr_type == 'kraus':
            self._cpp_state.transform_many_kraus(op, checked, on_batch)
        else:
            self._cpp_state.transform_many_choi(op, checked, on_batch)
        return None
    
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
        
    def __getitem__(self, key: Union[int, slice]) -> 'PureState':
        assert self.batch_dim, \
            f"This state is not batched and hence cannot be indexed: received key {key}."
        return PureState(
            self.ket[key],
            self.system_dim,
            self.system_seq,
            self._prob.clone_list(),
        )

    def _expand_data_batch_dim(self, target_batch_dim: List[int]) -> None:
        r"""Expand data to target batch dimension (memory-sharing via expand).
        
        Args:
            target_batch_dim: Target data batch dimensions to expand to.
        """
        cur_bd = list(self._cpp_state.batch_dim)
        if cur_bd == target_batch_dim or not target_batch_dim:
            return
        
        d = int(self.dim)
        
        if not cur_bd:
            cur_shape = [1] * len(target_batch_dim) + [d]
            target_shape = target_batch_dim + [d]
            data = self._cpp_state.data().view(cur_shape)
        else:
            pad_len = len(target_batch_dim) - len(cur_bd)
            if pad_len > 0:
                if target_batch_dim[:len(cur_bd)] == cur_bd:
                    cur_bd_padded = cur_bd + [1] * pad_len
                else:
                    cur_bd_padded = [1] * pad_len + cur_bd
            else:
                cur_bd_padded = cur_bd
            cur_shape = cur_bd_padded + [d]
            target_shape = target_batch_dim + [d]
            data = self._cpp_state.data().view(cur_shape)
        
        expanded = data.expand(target_shape).contiguous()
        self._cpp_state.set_data(expanded.view(-1, d))
        self._cpp_state.set_batch_dim(list(target_batch_dim))
    
    @staticmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        if data.ndim == 1:
            data = data.reshape([1, -1, 1])
        elif data.ndim == 2:
            if data.shape[-1] == 1:
                data = data.reshape([1, -1, 1])
            else:
                data = data.unsqueeze(-1)
        else:
            if data.shape[-1] == 1:
                d = int(data.shape[-2])
                data = data.reshape([-1, d, 1])
            else:
                d = int(data.shape[-1])
                data = data.reshape([-1, d, 1])

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
        data = self._cpp_state.data()
        d = int(data.shape[-1])
        if self.batch_dim:
            return data.view(self.batch_dim + [d, 1])
        ket = data.view(1, d, 1)
        return ket.squeeze(0) if not self._keep_dim else ket

    @property
    def density_matrix(self) -> torch.Tensor:
        ket = self.ket
        density = ket @ _CPP_LINALG.dagger(ket)
        if not self.batch_dim and not self._keep_dim and density.shape[0] == 1:
            return density.squeeze(0)
        return density

    def _to_mixed(self) -> MixedState:
        r"""Convert the state to mixed state representation.
        """
        return MixedState(
            self.density_matrix,
            self.system_dim,
            self.system_seq,
            self._prob.clone_list(),
        )
    
    def _trace(self, sys_idx: List[int]) -> MixedState:
        return self._to_mixed()._trace(sys_idx)
    
    def _reset(self, reset_idx: List[int], replace_state: 'DefaultSimulator | ProductDefaultSimulator') -> 'MixedState':
        return self._to_mixed()._reset(reset_idx, replace_state)
    
    def _transpose(self, sys_idx: List[int]) -> MixedState:
        return self._to_mixed()._transpose(sys_idx)

    @property
    def rank(self) -> Union[int, List[int]]:
        return torch.ones(self.batch_dim).tolist() if self.batch_dim else 1
    
    def normalize(self) -> None:
        data = self._cpp_state.data()
        self._cpp_state.set_data(torch.div(data, torch.norm(data, dim=-1)))

    def clone(self) -> 'PureState':
        state = super().clone()
        state._keep_dim = self._keep_dim
        return state
        
    def _record_unitary(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        r"""The function that records the input unitary for computing the overall 
        unitary matrix of the circuit.
        
        Note:
            This function is for calling `Circuit.matrix` only
        """
        if self.batch_dim == [self.dim]:
            self._evolve_keep_dim(unitary, sys_idx)
            return
        
        applied_dim, dim = unitary.shape[-1], self.dim
        cur_seq = self.system_seq
        self.system_seq = sys_idx + [x for x in cur_seq if x not in sys_idx]
        data = self._cpp_state.data().view([dim, -1, applied_dim, dim // applied_dim])
        self._cpp_state.set_data(torch.matmul(unitary, data).view([-1, dim]))
        
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        if self._keep_dim:
            self._record_unitary(unitary, sys_idx)
            return
        super()._evolve(unitary, sys_idx, on_batch)
    
    def _evolve_ctrl(self, unitary: torch.Tensor, index: int, sys_idx: List[Union[int, List[int]]], on_batch: bool = True) -> None:
        normalized: List[List[int]] = []
        for v in sys_idx:
            if isinstance(v, list):
                normalized.append(v)
            else:
                normalized.append([int(v)])
        ctrl_idx = normalized[0]
        ctrl_dim = np.prod([self.system_dim[idx] for idx in ctrl_idx])
        index = int(index) % int(ctrl_dim)

        if self._keep_dim:
            flat_sys = [x for group in normalized for x in group]
            u_ctrl = _CPP_STATE.make_controlled_unitary(unitary, int(ctrl_dim), int(index))
            self._record_unitary(u_ctrl, flat_sys)
            return
        self._cpp_state.evolve_ctrl(unitary, index, normalized, on_batch)
    
    def _expec_state(self, prob_idx: List[int]) -> 'MixedState':
        return self._to_mixed()._expec_state(prob_idx)

    def to_prod_sum(self, subgroup_indices: List[List[int]], tol: Optional[float] = None) -> "ProductDefaultSimulator":
        state, subgroup_indices, subgroup_system_dims, local_dims, tol_value, vec = (
            _prepare_dense_to_prod_sum(self, subgroup_indices, tol=tol, is_matrix=False)
        )

        factors, coeffs = utils.linalg._vector_to_prod_sum(vec, local_dims, tol_value)
        cpp_state = _CPP_STATE.build_product_from_vector_prod_sum(
            factors,
            coeffs,
            subgroup_indices,
            subgroup_system_dims,
            state._prob.clone_list(),
            state._keep_dim,
            tol_value,
        )
        return ProductDefaultSimulator._wrap_cpp_product_state(cpp_state)
        
    def _transform(self, *args) -> None:
        raise NotImplementedError(
            "The state vector backend does not internally support the channel conversion. \
                Please call the 'transform' function directly instead of '_transform'.")
    
    def transform(self, op: torch.Tensor, sys_idx: List[int] = None, repr_type: str = 'kraus') -> MixedState:
        return self._to_mixed().transform(op, sys_idx, repr_type)

    def _transform_many(
        self,
        op: torch.Tensor,
        sys_idx_list: Union[Iterable[Union[int, List[int]]], List[List[int]], List[int], int],
        repr_type: str = 'kraus',
        on_batch: bool = True,
    ) -> MixedState:
        return self._to_mixed()._transform_many(op, sys_idx_list, repr_type, on_batch)
    
    def sqrt(self) -> torch.Tensor:
        return self.density_matrix
    
    def log(self) -> torch.Tensor:
        return torch.zeros_like(self.density_matrix)


def _relabel_block_indices(list_state_idx: List[List[int]], target_seq: List[int]) -> List[List[int]]:
    n = max((g for idxs in list_state_idx for g in idxs), default=-1) + 1
    if sorted(target_seq) != list(range(n)):
        raise ValueError("target_seq must be a permutation of [0..n-1].")
    old_of_new = {new: int(old) for new, old in enumerate(target_seq)}
    new_of_old = {old: new for new, old in old_of_new.items()}
    return [[new_of_old[int(g)] for g in idxs] for idxs in list_state_idx]


def _nested_to_list(sys_idx_list: Union[Iterable[int], int]) -> List[int]:
    r"""Recursively flattens a nested list/structure of integer indices into a sorted list of unique indices.
    """
    def _flatten(x):
        if isinstance(x, int):
            return [x]
        elif isinstance(x, (list, tuple)):
            ret = []
            for item in x:
                ret.extend(_flatten(item))
            return ret
        else:
            raise TypeError(f"Invalid index type: {type(x)}")

    flat = _flatten(sys_idx_list)
    return list(set(int(i) for i in flat))


def _convert_ctrl_sys_idx_to_local(
    sys_idx: List[Union[int, List[int]]], global_to_local: dict[int, int]
) -> List[Union[int, List[int]]]:
    r"""Convert control/target sys_idx (mixed int/list) to local indices while keeping structure."""
    local: List[Union[int, List[int]]] = []
    for item in sys_idx:
        if isinstance(item, list):
            local.append([global_to_local[int(g)] for g in item])
        else:
            local.append(global_to_local[int(item)])
    return local


def _extract_prod_sum_branch_local(
    factor: torch.Tensor,
    branch: Tuple[int, ...],
    group_idx: int,
    num_groups: int,
    is_matrix: bool,
) -> torch.Tensor:
    if is_matrix:
        if num_groups == 1:
            return factor.squeeze(-4).squeeze(-1)
        if group_idx == 0:
            return factor.squeeze(-4)[..., :, :, branch[0]]
        if group_idx == num_groups - 1:
            return factor.squeeze(-1)[..., branch[group_idx - 1], :, :]
        return factor[..., branch[group_idx - 1], :, :, branch[group_idx]]

    if num_groups == 1:
        return factor.squeeze(-3).squeeze(-1)
    if group_idx == 0:
        return factor.squeeze(-3)[..., :, branch[0]]
    if group_idx == num_groups - 1:
        return factor.squeeze(-1)[..., branch[group_idx - 1], :]
    return factor[..., branch[group_idx - 1], :, branch[group_idx]]


def _fold_prod_sum_chain_to_shared_branch(
    factors: List[torch.Tensor],
    coeffs: List[torch.Tensor],
    is_matrix: bool,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], List[int]]:
    if not factors:
        raise ValueError("factors must be non-empty.")

    batch_shape = list(factors[0].shape[: -(4 if is_matrix else 3)])
    num_groups = len(factors)
    if not coeffs:
        local_tensors = [
            _extract_prod_sum_branch_local(factor, (), group_idx, num_groups, is_matrix)
            for group_idx, factor in enumerate(factors)
        ]
        return local_tensors, None, batch_shape

    stack_dim = len(batch_shape)
    branches = list(itertools.product(*[range(int(coeff.shape[-1])) for coeff in coeffs]))
    coeff_base = factors[0].new_ones(batch_shape if batch_shape else ())
    shared_coeff: List[torch.Tensor] = []
    local_tensors_by_group: List[List[torch.Tensor]] = [[] for _ in range(num_groups)]

    for branch in branches:
        branch_coeff = coeff_base
        for idx, coeff in zip(branch, coeffs):
            branch_coeff = branch_coeff * coeff[(..., idx)]
        shared_coeff.append(branch_coeff)

        for group_idx, factor in enumerate(factors):
            local_tensors_by_group[group_idx].append(
                _extract_prod_sum_branch_local(factor, branch, group_idx, num_groups, is_matrix)
            )

    return (
        [torch.stack(group_items, dim=stack_dim) for group_items in local_tensors_by_group],
        torch.stack(shared_coeff, dim=stack_dim),
        batch_shape,
    )


def _first_nonzero_phase(local_tensor: torch.Tensor, tol: float) -> torch.Tensor:
    flat = local_tensor.reshape(-1)
    nz = torch.nonzero(torch.abs(flat) > tol, as_tuple=False)
    if nz.numel() == 0:
        return local_tensor.new_tensor(1.0)
    value = flat[int(nz[0].item())]
    return value / torch.abs(value)


def _canonicalize_shared_branch(
    branch_locals: List[torch.Tensor],
    branch_coeff: torch.Tensor,
    tol: float,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    coeff = branch_coeff
    canonical_locals: List[torch.Tensor] = []

    for local in branch_locals:
        norm = torch.linalg.norm(local.reshape(-1))
        if float(torch.abs(norm).item()) <= tol:
            return coeff.new_zeros(()), [torch.zeros_like(item) for item in branch_locals]

        local = local / norm
        coeff = coeff * norm

        phase = _first_nonzero_phase(local, tol)
        local = local / phase
        coeff = coeff * phase
        canonical_locals.append(local)

    return coeff, canonical_locals


def _compress_shared_prod_sum_branches(
    local_tensors: List[torch.Tensor],
    coeff_tensor: Optional[torch.Tensor],
    tol: float,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    if coeff_tensor is None:
        return local_tensors, coeff_tensor

    branch_axis = coeff_tensor.ndim - 1
    flat_coeff = coeff_tensor.reshape(-1, coeff_tensor.shape[-1])
    keep_mask = torch.amax(torch.abs(flat_coeff), dim=0) > tol
    if not torch.any(keep_mask):
        keep_mask[0] = True
    if not torch.all(keep_mask):
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        coeff_tensor = coeff_tensor.index_select(branch_axis, keep_idx)
        local_tensors = [local.index_select(branch_axis, keep_idx) for local in local_tensors]

    if coeff_tensor.ndim != 1:
        return local_tensors, coeff_tensor

    merge_tol = max(tol, 1e-12 if coeff_tensor.dtype == torch.complex128 else 1e-6)
    merged: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
    num_branches = int(coeff_tensor.shape[-1])
    for branch_idx in range(num_branches):
        branch_coeff = coeff_tensor[branch_idx]
        if float(torch.abs(branch_coeff).item()) <= merge_tol:
            continue

        branch_locals = [local[branch_idx] for local in local_tensors]
        branch_coeff, branch_locals = _canonicalize_shared_branch(
            branch_locals, branch_coeff, merge_tol
        )
        if float(torch.abs(branch_coeff).item()) <= merge_tol:
            continue

        matched = False
        for i, (existing_coeff, existing_locals) in enumerate(merged):
            if all(
                torch.allclose(local, existing, atol=merge_tol, rtol=merge_tol)
                for local, existing in zip(branch_locals, existing_locals)
            ):
                merged[i] = (existing_coeff + branch_coeff, existing_locals)
                matched = True
                break

        if not matched:
            merged.append((branch_coeff, branch_locals))

    merged = [
        (branch_coeff, branch_locals)
        for branch_coeff, branch_locals in merged
        if float(torch.abs(branch_coeff).item()) > merge_tol
    ]
    if not merged:
        return local_tensors, coeff_tensor

    coeff_tensor = torch.stack([branch_coeff for branch_coeff, _ in merged], dim=0)
    local_tensors = [
        torch.stack([branch_locals[group_idx] for _, branch_locals in merged], dim=0)
        for group_idx in range(len(local_tensors))
    ]
    return local_tensors, coeff_tensor


def _build_product_from_prod_sum_factors(
    factors: List[torch.Tensor],
    coeffs: List[torch.Tensor],
    subgroup_indices: List[List[int]],
    subgroup_system_dims: List[List[int]],
    base_prob: ProbabilityData,
    keep_dim: bool,
    is_matrix: bool,
    compress_tol: float,
) -> "ProductDefaultSimulator":
    local_tensors, shared_coeff, batch_shape = _fold_prod_sum_chain_to_shared_branch(
        factors, coeffs, is_matrix
    )
    local_tensors, shared_coeff = _compress_shared_prod_sum_branches(
        local_tensors, shared_coeff, compress_tol
    )

    target_batch_shape = (
        batch_shape + [int(shared_coeff.shape[-1])] if shared_coeff is not None else batch_shape
    )
    block_cls = MixedState if is_matrix else PureState

    list_state: List[DefaultSimulator] = []
    for local_tensor, group_dim in zip(local_tensors, subgroup_system_dims):
        local_dim = int(np.prod(group_dim))
        block_data_tail_shape = [local_dim, local_dim] if is_matrix else [local_dim, 1]
        data = local_tensor.reshape([-1] + block_data_tail_shape)
        block = block_cls(data, group_dim, None, None)
        if target_batch_shape:
            block._cpp_state.set_batch_dim(target_batch_shape)
        list_state.append(block)

    prob = base_prob.clone()
    if shared_coeff is not None:
        prob.append(shared_coeff, normalize=False)

    return ProductDefaultSimulator(
        list_state=list_state,
        list_state_idx=[idx.copy() for idx in subgroup_indices],
        _prob=prob,
        _roles=[ProductDefaultSimulator._ROLE_CLASSICAL] * len(base_prob) +
        ([ProductDefaultSimulator._ROLE_PROD_SUM] if shared_coeff is not None else []),
        _keep_dim=keep_dim,
    )


def _build_product_from_vector_prod_sum(
    factors: List[torch.Tensor],
    coeffs: List[torch.Tensor],
    subgroup_indices: List[List[int]],
    subgroup_system_dims: List[List[int]],
    base_prob: ProbabilityData,
    keep_dim: bool,
    compress_tol: float,
) -> "ProductDefaultSimulator":
    return _build_product_from_prod_sum_factors(
        factors,
        coeffs,
        subgroup_indices,
        subgroup_system_dims,
        base_prob,
        keep_dim,
        is_matrix=False,
        compress_tol=compress_tol,
    )


def _build_product_from_matrix_prod_sum(
    factors: List[torch.Tensor],
    coeffs: List[torch.Tensor],
    subgroup_indices: List[List[int]],
    subgroup_system_dims: List[List[int]],
    base_prob: ProbabilityData,
    keep_dim: bool,
    compress_tol: float,
) -> "ProductDefaultSimulator":
    return _build_product_from_prod_sum_factors(
        factors,
        coeffs,
        subgroup_indices,
        subgroup_system_dims,
        base_prob,
        keep_dim,
        is_matrix=True,
        compress_tol=compress_tol,
    )


class ProductDefaultSimulator(StateSimulator):
    r"""Default-backend-only product state container composed of dense blocks.

    Args:
        list_state: Dense component states. Each element must be a single-block
            ``DefaultSimulator`` subclass, i.e. ``PureState`` or ``MixedState``.
        list_state_idx: Global subsystem indices covered by each block in ``list_state``.
            The i-th entry gives the subsystem labels represented by
            ``list_state[i]``.
        _prob: Shared probability history stored at the container level. Dense blocks
            inside ``list_state`` must not carry their own probability history.
        _roles: Per-probability-dimension metadata used only by
            ``ProductDefaultSimulator`` to distinguish classical dimensions from
            internal product-sum coefficient dimensions.
        _keep_dim: Whether to preserve singleton batch dimensions when converting
            back to dense states.

    Note:
        For internal usage only.
    """
    _ROLE_CLASSICAL: str = "classical"
    _ROLE_PROD_SUM: str = "prod_sum"
    _VALID_ROLES = {_ROLE_CLASSICAL, _ROLE_PROD_SUM}

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


    @classmethod
    def _validate_role(cls, role: str) -> str:
        role = str(role).lower()
        if role not in cls._VALID_ROLES:
            raise ValueError(
                f"Unsupported product probability role '{role}': expected one of {sorted(cls._VALID_ROLES)}."
            )
        return role

    @classmethod
    def _normalize_roles(cls, roles: Optional[List[str]], prob: ProbabilityData) -> List[str]:
        num_prob = len(prob)
        if roles is None:
            return [cls._ROLE_CLASSICAL] * num_prob
        if len(roles) != num_prob:
            raise ValueError(
                f"roles length must match probability length: got {len(roles)} and {num_prob}."
            )
        return [cls._validate_role(role) for role in roles]

    def _normalize_classical_prob_idx(self, prob_idx: int) -> int:
        classical_indices = self.classical_indices
        num_classical = len(classical_indices)
        if num_classical == 0:
            raise ValueError("This product state has no classical probability dimensions.")
        normalized = num_classical + prob_idx if prob_idx < 0 else prob_idx
        if normalized < 0 or normalized >= num_classical:
            raise IndexError(
                f"Classical probability index out of range: got {prob_idx}, "
                f"expected in [-{num_classical}, {num_classical - 1}]."
            )
        return normalized

    def _normalize_classical_prob_indices(
        self, prob_idx: Optional[Union[int, List[int]]]
    ) -> List[int]:
        classical_indices = self.classical_indices
        if not classical_indices:
            if prob_idx is None:
                return []
            raise ValueError("This product state has no classical probability dimensions.")
        if prob_idx is None:
            return classical_indices.copy()
        if isinstance(prob_idx, int):
            return [classical_indices[self._normalize_classical_prob_idx(prob_idx)]]

        normalized_public: List[int] = []
        for idx in prob_idx:
            normalized_public.append(self._normalize_classical_prob_idx(int(idx)))
        return [classical_indices[idx] for idx in sorted(normalized_public)]

    @staticmethod
    def check(data, sys_dim, eps) -> None:
        raise NotImplementedError(
            "ProductDefaultSimulator.check should not be called")


    def _refresh_from_cpp_state(self) -> None:
        r"""Refresh Python-side mirrors from the authoritative C++ ProductState."""
        exported = self._cpp_state.export_blocks(True)
        self._list_state = [
            DefaultSimulator._wrap_cpp_state(cpp_block) for cpp_block, _ in exported
        ]
        self._list_state_idx = [list(idx) for _, idx in exported]

        prob_list = list(self._cpp_state.prob_list)
        self._prob = ProbabilityData(prob_list)
        self._roles = list(self._cpp_state.roles)

        full_batch = list(self._cpp_state.batch_dim)
        num_prob = len(prob_list)
        if num_prob > 0 and len(full_batch) >= num_prob:
            self._batch_dim = full_batch[:-num_prob]
        else:
            self._batch_dim = full_batch

        self._keep_dim = bool(self._cpp_state.keep_dim)
        self._sys_dim = list(self._cpp_state.system_dim)
        if self._list_state:
            self._dtype = self._list_state[0].dtype
            self._device = self._list_state[0].device

    @staticmethod
    def _create_cpp_product_state(
        blocks: List[DefaultSimulator],
        block_indices: List[List[int]],
        prob: ProbabilityData,
        roles: List[str],
        keep_dim: bool,
    ) -> "_CPP_STATE.ProductState":
        r"""Create a C++ ProductState from wrapped dense blocks and metadata."""
        return _CPP_STATE.create_product_state(
            [state._cpp_state for state in blocks],
            [idx.copy() for idx in block_indices],
            prob.clone_list(),
            roles,
            keep_dim,
        )

    @staticmethod
    def _wrap_cpp_product_state(
        cpp_state: "_CPP_STATE.ProductState",
    ) -> "ProductDefaultSimulator":
        r"""Wrap an existing C++ ProductState without rebuilding it."""
        inst = object.__new__(ProductDefaultSimulator)
        inst._cpp_state = cpp_state

        exported = cpp_state.export_blocks(True)
        if exported:
            first = DefaultSimulator._wrap_cpp_state(exported[0][0])
            State.__init__(
                inst,
                list(cpp_state.system_dim),
                dtype=first.dtype,
                device=first.device,
            )
        else:
            State.__init__(inst, list(cpp_state.system_dim))
        inst._refresh_from_cpp_state()
        return inst

    def _adopt_cpp_product_state(self, cpp_state: "_CPP_STATE.ProductState") -> None:
        self._cpp_state = cpp_state
        self._refresh_from_cpp_state()


    def __init__(
        self,
        list_state: List[DefaultSimulator],
        list_state_idx: List[List[int]],
        _prob: Optional[ProbabilityData] = None,
        _roles: Optional[List[str]] = None,
        _keep_dim: bool = False,
    ) -> None:
        for state in list_state:
            if isinstance(state, ProductDefaultSimulator):
                raise ValueError(
                    "Cannot create ProductDefaultSimulator from nested ProductDefaultSimulator."
                )

        prob = _prob if _prob is not None else ProbabilityData()
        probs = prob.clone_list()
        roles = self._normalize_roles(_roles, prob)
        self._cpp_state = self._create_cpp_product_state(
            list_state,
            list_state_idx,
            ProbabilityData(probs),
            roles,
            _keep_dim,
        )

        exported = self._cpp_state.export_blocks(True)
        if exported:
            first = DefaultSimulator._wrap_cpp_state(exported[0][0])
            State.__init__(
                self,
                list(self._cpp_state.system_dim),
                dtype=first.dtype,
                device=first.device,
            )
        else:
            State.__init__(self, list(self._cpp_state.system_dim))
        self._refresh_from_cpp_state()


    def clone(self) -> "ProductDefaultSimulator":
        return self._wrap_cpp_product_state(self._cpp_state.clone())

    @property
    def backend(self) -> str:
        return str(self._cpp_state.backend)

    @property
    def roles(self) -> List[str]:
        return list(self._cpp_state.roles)

    @property
    def classical_indices(self) -> List[int]:
        return list(self._cpp_state.classical_indices)

    @property
    def prod_sum_indices(self) -> List[int]:
        return list(self._cpp_state.prod_sum_indices)

    @property
    def system_dim(self) -> List[int]:
        return list(self._cpp_state.system_dim)

    @system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        raise RuntimeError(
            "ProductDefaultSimulator does not support setting system_dim directly."
        )

    @property
    def batch_dim(self) -> List[int]:
        return list(self._cpp_state.batch_dim)

    @property
    def shape(self) -> torch.Size:
        return self._merged().shape

    @property
    def probability(self) -> torch.Tensor:
        return self._cpp_state.probability()

    @property
    def ket(self) -> torch.Tensor:
        return self._merged().ket

    @property
    def density_matrix(self) -> torch.Tensor:
        return self._cpp_state.density_matrix()

    @property
    def rank(self) -> Union[int, Iterable[int]]:
        return self._merged().rank

    def numpy(self) -> np.ndarray:
        return self._merged().numpy()

    def __getitem__(self, key: Union[int, slice]) -> "ProductDefaultSimulator":
        assert self.batch_dim, (
            f"This state is not batched and hence cannot be indexed: received key {key}."
        )
        return ProductDefaultSimulator(
            [state[key] for state in self._list_state],
            [idx.copy() for idx in self._list_state_idx],
            _prob=self._prob.clone(),
            _roles=self.roles,
            _keep_dim=self._keep_dim,
        )

    def _merged(self) -> DefaultSimulator:
        dense = DefaultSimulator._wrap_cpp_state(self._cpp_state.merged_dense())
        dense._keep_dim = self._keep_dim
        return dense


    def _export_blocks(
        self, clone_state: bool = True
    ) -> List[Tuple[DefaultSimulator, List[int]]]:
        r"""Export C++ blocks as wrapped dense states with their global indices."""
        out = []
        for cpp_block, idx in self._cpp_state.export_blocks(clone_state):
            out.append((DefaultSimulator._wrap_cpp_state(cpp_block), list(idx)))
        return out

    def _adopt_blocks(
        self,
        blocks: List[DefaultSimulator],
        block_indices: List[List[int]],
    ) -> None:
        r"""Rebuild the authoritative C++ ProductState from exported dense blocks."""
        cpp_state = self._create_cpp_product_state(
            blocks,
            block_indices,
            self._prob,
            self.roles,
            self._keep_dim,
        )
        self._adopt_cpp_product_state(cpp_state)

    def _merge_exported_blocks(
        self,
        blocks: List[DefaultSimulator],
        block_indices: List[List[int]],
        block_ids: List[int],
    ) -> Tuple[DefaultSimulator, List[int]]:
        r"""Merge selected exported blocks into one dense block for local rewrites."""
        if not block_ids:
            raise ValueError("block_ids must be non-empty.")

        merged_global = sorted([g for b in block_ids for g in block_indices[b]])
        block_ids_sorted = sorted(block_ids, key=lambda b: min(block_indices[b]))

        concat_global: List[int] = []
        ordered_items: List[Tuple[List[int], DefaultSimulator]] = []
        for block_id in block_ids_sorted:
            idxs = block_indices[block_id]
            state = blocks[block_id]
            perm = sorted(range(len(idxs)), key=lambda i: idxs[i])
            if perm != list(range(len(perm))):
                state = state.permute(perm)
                idxs = [idxs[i] for i in perm]
            ordered_items.append((idxs, state))
            concat_global.extend(idxs)

        all_pure = all(state.backend.endswith("pure") for _, state in ordered_items)
        data = _CPP_LINALG.nkron(
            [state.ket if all_pure else state.density_matrix for _, state in ordered_items]
        )

        merged = DefaultSimulator(
            data,
            [self.system_dim[g] for g in concat_global],
            list(range(len(concat_global))),
            None,
        )
        if concat_global != merged_global:
            target_seq = [concat_global.index(g) for g in merged_global]
            merged = merged.permute(target_seq)
            merged = DefaultSimulator(
                merged.ket if all_pure else merged.density_matrix,
                [self.system_dim[g] for g in merged_global],
                list(range(len(merged_global))),
                None,
            )

        return merged, merged_global

    def _sync_data_batch_dim(self) -> None:
        r"""Broadcast exported blocks to a common batch shape and rebuild if needed."""
        if not self._list_state:
            return

        target_batch: List[int] = []
        for state in self._list_state:
            batch_dim = list(state._cpp_state.batch_dim)
            if not target_batch:
                target_batch = batch_dim
                continue

            left, right = target_batch, batch_dim
            if len(right) > len(left):
                left, right = right, left
            result = list(left)
            for i, (x, y) in enumerate(zip(reversed(right), reversed(left))):
                idx = len(left) - 1 - i
                if x == 1:
                    continue
                if y == 1:
                    result[idx] = x
                    continue
                if x != y:
                    raise ValueError(
                        f"Block batch dims not broadcastable: {target_batch} vs {batch_dim}"
                    )
            target_batch = result

        changed = False
        for state in self._list_state:
            if list(state._cpp_state.batch_dim) != target_batch:
                state._expand_data_batch_dim(target_batch)
                changed = True

        if changed:
            self._adopt_blocks(
                self._list_state,
                [idx.copy() for idx in self._list_state_idx],
            )

    @property
    def num_blocks(self) -> int:
        return int(self._cpp_state.num_blocks)


    def to(
        self, dtype: Optional[Union[str, torch.dtype]] = None, device: Optional[Union[str, torch.device]] = None
    ) -> "ProductDefaultSimulator":
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        dev = torch.device(device) if isinstance(device, str) else device
        return self._wrap_cpp_product_state(self._cpp_state.to(dt, dev))

    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> DefaultSimulator:
        try:
            return DefaultSimulator._wrap_cpp_state(
                self._cpp_state.prob_select_dense(outcome_idx, prob_idx)
            )
        except RuntimeError as err:
            msg = str(err).lower()
            if "out of range" in msg:
                raise IndexError(str(err)) from err
            if "no classical" in msg or "no probability" in msg:
                raise ValueError(str(err)) from err
            raise

    def expec_state(
        self, prob_idx: Optional[Union[int, List[int]]] = None
    ) -> Union[DefaultSimulator, "ProductDefaultSimulator"]:
        raw_prob_idx = self._normalize_classical_prob_indices(prob_idx)
        if not raw_prob_idx:
            return self.clone()
        return DefaultSimulator._wrap_cpp_state(
            self._cpp_state.expec_state_dense(raw_prob_idx)
        )

    def _expec_state(self, prob_idx: List[int]) -> DefaultSimulator:
        return DefaultSimulator._wrap_cpp_state(
            self._cpp_state.expec_state_dense(prob_idx)
        )

    def add_probability(self, prob: torch.Tensor) -> None:
        try:
            self._cpp_state.add_probability(prob)
            self._refresh_from_cpp_state()
        except RuntimeError as err:
            if "not supported" in str(err).lower():
                raise NotImplementedError(str(err)) from err
            raise

    def kron(
        self, other: "DefaultSimulator | ProductDefaultSimulator"
    ) -> "ProductDefaultSimulator":
        if isinstance(other, ProductDefaultSimulator):
            cpp_out = self._cpp_state.kron_product(other._cpp_state)
        else:
            cpp_out = self._cpp_state.kron_dense(other._cpp_state)
        return self._wrap_cpp_product_state(cpp_out)

    def export_block(self) -> DefaultSimulator:
        try:
            return DefaultSimulator._wrap_cpp_state(self._cpp_state.export_block_dense())
        except RuntimeError as err:
            raise ValueError(str(err)) from err

    def to_prod_sum(
        self, subgroup_indices: List[List[int]], tol: Optional[float] = None
    ) -> "ProductDefaultSimulator":
        if self.num_blocks == 1:
            return self.export_block().to_prod_sum(subgroup_indices, tol=tol)
        warnings.warn(
            "Cannot call to_prod_sum() on a multi-block ProductDefaultSimulator because it already "
            "represents a product-state structure. Returning a clone instead.",
            UserWarning,
        )
        return self.clone()

    def normalize(self) -> None:
        self._cpp_state.normalize()
        self._refresh_from_cpp_state()


    def permute(self, target_seq: List[int]) -> "ProductDefaultSimulator":
        if getattr(self, "_keep_dim", False):
            return self._merged().permute(target_seq)
        return self._wrap_cpp_product_state(self._cpp_state.permute(target_seq))

    def _evolve(
        self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True
    ) -> None:
        self._cpp_state.evolve(unitary, sys_idx, on_batch)
        self._refresh_from_cpp_state()

    def _evolve_many(
        self,
        unitary: torch.Tensor,
        sys_idx_list: List[List[int]],
        on_batch: bool = True,
    ) -> None:
        checked = [self._check_sys_idx(sys_idx) for sys_idx in sys_idx_list]
        if not checked:
            return None
        self._cpp_state.evolve_many(unitary, checked, on_batch)
        self._refresh_from_cpp_state()
        return None

    def _evolve_many_batched_groups(
        self,
        unitary_groups: List[torch.Tensor],
        sys_idx_groups: List[List[List[int]]],
        on_batch: bool = True,
    ) -> None:
        checked_groups = []
        for sys_idx_list in sys_idx_groups:
            checked_groups.append([self._check_sys_idx(sys_idx) for sys_idx in sys_idx_list])
        if not checked_groups:
            return None
        self._cpp_state.evolve_many_batched(unitary_groups, checked_groups, on_batch)
        self._refresh_from_cpp_state()
        return None

    def _evolve_keep_dim(
        self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True
    ) -> None:
        self._cpp_state.evolve_keep_dim(unitary, sys_idx, on_batch)
        self._refresh_from_cpp_state()

    def _evolve_ctrl(
        self,
        unitary: torch.Tensor,
        index: int,
        sys_idx: List[Union[int, List[int]]],
    ) -> None:
        normalized = []
        for v in sys_idx:
            normalized.append(v if isinstance(v, list) else [int(v)])
        ctrl_idx = normalized[0]
        ctrl_dim = int(np.prod([self.system_dim[idx] for idx in ctrl_idx]))
        clamped = int(index) % ctrl_dim
        self._cpp_state.evolve_ctrl(unitary, clamped, normalized, True)
        self._refresh_from_cpp_state()

    def _transform(
        self,
        op: torch.Tensor,
        sys_idx: List[int],
        repr_type: str,
        on_batch: bool = True,
    ) -> None:
        self._cpp_state.transform(op, sys_idx, repr_type, on_batch)
        self._refresh_from_cpp_state()

    def _transform_many(
        self,
        op: torch.Tensor,
        sys_idx_list: List[List[int]],
        repr_type: str,
        on_batch: bool = True,
    ) -> None:
        repr_type = repr_type.lower()
        checked = [self._check_sys_idx(sys_idx) for sys_idx in sys_idx_list]
        if not checked:
            return None
        if repr_type == "kraus":
            for sys_idx in checked:
                self._check_op_dim(op, sys_idx)
        self._cpp_state.transform_many(op, checked, repr_type, on_batch)
        self._refresh_from_cpp_state()
        return None

    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        return self._cpp_state.expec_val(obs, sys_idx)

    def _expec_val_pauli_terms(
        self,
        pauli_words_r: List[str],
        sites: List[List[int]],
    ) -> torch.Tensor:
        return self._cpp_state.expec_val_pauli_terms(pauli_words_r, sites).real

    def _measure(
        self, measure_op: torch.Tensor, sys_idx: List[int]
    ) -> Tuple[torch.Tensor, "ProductDefaultSimulator"]:
        prob, cpp_state = self._cpp_state.measure(measure_op, sys_idx)
        return prob, self._wrap_cpp_product_state(cpp_state)

    def _measure_many(
        self, measure_op: torch.Tensor, sys_idx_list: List[List[int]]
    ) -> Tuple[torch.Tensor, "ProductDefaultSimulator"]:
        prob, cpp_state = self._cpp_state.measure_many(measure_op, sys_idx_list)
        return prob, self._wrap_cpp_product_state(cpp_state)

    def _measure_by_state(
        self,
        measure_basis: Union[PureState, "ProductDefaultSimulator"],
        sys_idx: List[int],
        keep_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, "ProductDefaultSimulator"]]:
        cpp_basis = measure_basis._cpp_state
        prob, cpp_opt = self._cpp_state.measure_by_state(cpp_basis, sys_idx, keep_state)
        if not keep_state:
            return prob
        return prob, self._wrap_cpp_product_state(cpp_opt)

    def _trace(self, trace_idx: List[int]) -> "ProductDefaultSimulator":
        return self._wrap_cpp_product_state(self._cpp_state.trace(trace_idx))

    def _reset(
        self,
        reset_idx: List[int],
        replace_state: "DefaultSimulator | ProductDefaultSimulator",
    ) -> "ProductDefaultSimulator":
        cpp_replace = replace_state._cpp_state
        return self._wrap_cpp_product_state(self._cpp_state.reset(reset_idx, cpp_replace))

    def _transpose(self, transpose_idx: List[int]) -> "ProductDefaultSimulator":
        return self._wrap_cpp_product_state(self._cpp_state.transpose(transpose_idx))

    def _to_mixed(self) -> "ProductDefaultSimulator":
        exported = self._export_blocks(clone_state=True)
        blocks = []
        idxs = []
        for state, idx in exported:
            blocks.append(state._to_mixed() if state.backend == "default-pure" else state)
            idxs.append(idx)
        return ProductDefaultSimulator(
            blocks,
            idxs,
            _prob=self._prob.clone(),
            _roles=self.roles,
            _keep_dim=self._keep_dim,
        )

    def sqrt(self) -> torch.Tensor:
        return self._merged().sqrt()

    def log(self) -> torch.Tensor:
        return self._merged().log()

    def __str__(self) -> str:
        return str(self._merged())


    def _index_select(
        self,
        new_indices: torch.Tensor,
        system_idx_pairs: Optional[List[List[int]]] = None,
    ) -> None:
        if system_idx_pairs is None:
            merged = self._merged()
            merged._index_select(new_indices, None)
            self._adopt_blocks([merged], [list(range(self.num_systems))])
            return

        exported = self._export_blocks(clone_state=True)
        blocks = [state for state, _ in exported]
        block_indices = [idx.copy() for _, idx in exported]

        all_sys = sorted({int(idx) for pair in system_idx_pairs for idx in pair})
        involved = sorted(
            {
                b
                for b, idxs in enumerate(block_indices)
                if any(global_idx in all_sys for global_idx in idxs)
            }
        )

        if len(involved) > 1:
            merged, merged_global = self._merge_exported_blocks(
                blocks, block_indices, involved
            )
            keep_blocks: List[DefaultSimulator] = []
            keep_indices: List[List[int]] = []
            insert_pos = min(involved)
            removed = set(involved)
            inserted = False
            for i, (state, idxs) in enumerate(zip(blocks, block_indices)):
                if i in removed:
                    if i == insert_pos and not inserted:
                        keep_blocks.append(merged)
                        keep_indices.append(merged_global)
                        inserted = True
                    continue
                keep_blocks.append(state)
                keep_indices.append(idxs)

            blocks = keep_blocks
            block_indices = keep_indices

        target_block = None
        for i, idxs in enumerate(block_indices):
            if all(global_idx in idxs for global_idx in all_sys):
                target_block = i
                break
        if target_block is None:
            raise ValueError(
                f"Failed to localize systems {all_sys} in ProductDefaultSimulator."
            )

        block_global = block_indices[target_block]
        local_pairs = [[block_global.index(global_idx) for global_idx in pair] for pair in system_idx_pairs]
        local_indices = utils.linalg._get_swap_indices(
            2, 3, local_pairs, blocks[target_block].system_dim, new_indices.device
        ).long()
        blocks[target_block]._index_select(local_indices, None)
        self._adopt_blocks(blocks, block_indices)

