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
Source file for base class that simulates quantum states.
"""


import importlib
import warnings
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from ... import Hamiltonian, utils
from .base import State


def _get_cpp_probability_data():
    """Return the C++ ProbabilityData class if the extension is available."""
    try:
        mod = importlib.import_module("quairkit._C")
        state_mod = getattr(mod, "state", None)
        if state_mod is None:
            return None
        return getattr(state_mod, "ProbabilityData", None)
    except Exception:
        return None


_CPP_ProbabilityData = _get_cpp_probability_data()


class ProbabilityData:
    r"""
    A small helper that owns and manages the probability history attached to a StateSimulator.

    It stores a list of torch.Tensor objects, each representing the distribution of a random
    variable X_{t=n}. The shape rules follow the original design:

    - When adding a fresh distribution for time n (i.e., after n previous distributions),
      the canonical shape is [1, 1, ..., 1, d_n] with exactly n leading 1's.
    - When a distribution already carries the broadcasted batch/prob dimensions (e.g., from
      measurement), we can append it as-is.

    It also tracks the per-variable outcome counts (dims) and provides utilities to compute
    the joint distribution over a subset of variables.
    """

    def __init__(self, probs: Optional[List[torch.Tensor]] = None) -> None:
        self._impl = _CPP_ProbabilityData(probs or []) if _CPP_ProbabilityData is not None else None

        self._list: List[torch.Tensor] = []
        self._dims: List[int] = []
        if self._impl is None and probs:
            for p in probs:
                self._list.append(p)
                self._dims.append(int(p.shape[-1]))

    def __len__(self) -> int:
        return len(self._impl) if self._impl is not None else len(self._list)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._impl is not None:
            return self._impl.list[idx]
        return self._list[idx]

    @property
    def list(self) -> List[torch.Tensor]:
        return self._impl.list if self._impl is not None else self._list

    def clone_list(self) -> List[torch.Tensor]:
        if self._impl is not None:
            return self._impl.clone_list()
        return [p.clone() for p in self._list]

    @property
    def shape(self) -> List[int]:
        if self._impl is not None:
            return list(self._impl.shape)
        return self._dims.copy()
    
    @property
    def non_prob_dim(self) -> List[int]:
        if self._impl is not None:
            return list(self._impl.non_prob_dim)
        num_prob = len(self._list)
        return [] if num_prob == 0 else list(self._list[-1].shape[:-num_prob])

    @property
    def product_dim(self) -> int:
        if self._impl is not None:
            return int(self._impl.product_dim)
        return int(np.prod(self._dims))

    def clear(self) -> None:
        if self._impl is not None:
            self._impl.clear()
        else:
            self._list.clear()
            self._dims.clear()

    def prepare_new(self, prob: torch.Tensor, dtype: Optional[torch.dtype] = None,
                    device: Optional[torch.device] = None, real_only: bool = False) -> torch.Tensor:
        r"""
        Canonicalize a fresh probability tensor to shape [1]*num_prev + [-1].
        This does not append; it only prepares the shaped tensor.
        """
        if self._impl is not None:
            if isinstance(device, str):
                device = torch.device(device)
            return self._impl.prepare_new(prob, dtype=dtype, device=device, real_only=real_only)

        num_prev = len(self._list)
        p = prob.view([1] * num_prev + [-1])
        if dtype is not None or device is not None:
            p = p.to(dtype=dtype, device=device)
        if real_only:
            p = p.real
        return p

    def append(self, prob: torch.Tensor, normalize: bool = False,
               dtype: Optional[torch.dtype] = None,
               device: Optional[torch.device] = None,
               real_only: bool = False) -> None:
        r"""
        Append a probability tensor into the history.

        Args:
            prob: The tensor to append.
            normalize: If True, reshape to [1]*num_prev + [-1] before appending.
            dtype, device: Optional dtype/device cast.
            real_only: If True, keep only the real part.
        """
        if self._impl is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self._impl.append(prob, normalize=normalize, dtype=dtype, device=device, real_only=real_only)
        else:
            p = self.prepare_new(prob, dtype=dtype, device=device, real_only=real_only) if normalize else prob
            if dtype is not None or device is not None:
                p = p.to(dtype=dtype, device=device)
            if real_only:
                p = p.real
            self._list.append(p)
            self._dims.append(int(p.shape[-1]))

    def joint(self, prob_idx: List[int]) -> torch.Tensor:
        r"""
        Compute the joint distribution across the selected indices (with broadcasting).

        Returns:
            A tensor shaped to the broadcasted product of the selected distributions.
        """
        if self._impl is not None:
            return self._impl.joint(prob_idx)

        if len(self.list) == 0:
            return torch.tensor(1.0)

        first = self.list[0]
        dtype = first.dtype
        device = first.device
        result = torch.tensor(1.0, dtype=dtype, device=device)
        for idx in sorted(prob_idx):
            p = self[idx]
            if p.dim() > result.dim():
                result = result.view(list(result.shape) + [1] * (p.dim() - result.dim()))
            result = result * p
        return result

    def clone(self) -> 'ProbabilityData':
        if self._impl is not None:
            out = ProbabilityData()
            out._impl = self._impl.clone()
            return out
        return ProbabilityData(self.clone_list())


class StateSimulator(State):
    r"""The abstract base class for simulating quantum states in QuAIRKit.

    Args:
        data: tensor array for quantum state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 0 to n - 1.
        probability: list of state probability distributions. Defaults to be 1.

    Note:
        This class is designed to be inherited by specific state simulator classes.
        Such class should simulate quantum computing with batch, qudit and probabilities.

    """
    def __init__(self, data: torch.Tensor, sys_dim: List[int],
                 system_seq: Optional[List[int]],
                 probability: Optional[List[torch.Tensor]]) -> None:
        super().__init__(sys_dim)

        self._prob = ProbabilityData(probability)
        self._data = data.contiguous().to(dtype=self.dtype, device=self.device)

    def _get_data_tensor(self) -> torch.Tensor:
        r"""Internal helper to access the underlying tensor without relying on public `_data` access.

        For the default backend, the real storage is `_data_tensor` and `_data` may be a
        deprecated property. For other backends, `_data` is typically the real storage.
        """
        return getattr(self, "_data_tensor", self._data)

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]) -> 'StateSimulator':
        r"""Indexing of the State class
        """

    @abstractmethod
    def prob_select(self, outcome_idx: torch.Tensor, prob_idx: int = -1) -> 'StateSimulator':
        r"""Indexing probability outcome from the State batch along the given dimension.

        Args:
            outcome_idx: the 1-D tensor containing the outcomes to index
            prob_idx: the int that indexing which probability distribution. Defaults to be the last distribution.

        Returns:
            States that are selected by the specific probability outcomes.
        
        """
    
    @abstractmethod
    def add_probability(self, prob: torch.Tensor) -> None:
        r"""Add one probability distribution to the state.
        
        Args:
            prob: the probability distribution to be added, which should be a 1-D tensor.
        
        """

    def __copy__(self) -> 'StateSimulator':
        return self.clone()

    @abstractmethod
    def __str__(self) -> str:
        r"""String representation of the state.
        """
        
    def __repr__(self) -> None:
        return self.__str__()

    def __matmul__(self, other: 'StateSimulator') -> torch.Tensor:
        r"""Matrix product between the density matrix representation of two states

        Args:
            other: a quantum state

        Returns:
            the product of these two states

        """
        return torch.matmul(self.density_matrix, other.density_matrix)

    def _check_op_dim(self, op: torch.Tensor, sys_idx: List[int]) -> None:
        applied_dim = op.shape[-1]
        assert applied_dim == int(np.prod((list_dim := [self._sys_dim[i] for i in sys_idx]))), \
            f"The operator's dimension {applied_dim} does not match the dimensions of its acting systems {list_dim}"

    @abstractmethod
    def kron(self, other: 'StateSimulator') -> 'StateSimulator':
        r"""Take the tensor product with other state.
        """

    @staticmethod
    @abstractmethod
    def check(data: torch.Tensor, sys_dim: Union[int, List[int]], eps: Optional[float] = 1e-4) -> int:
        r"""Assert whether the input data is valid for the specific State class.

        Args:
            data: tensor array for quantum state(s).
            sys_dim: (list of) dimension(s) of the systems, can be a list of integers or an integer.
            eps: the tolerance for the numerical check. Can be None means the check is overridden.

        Returns:
            the number of systems.

        """

    def _joint_probability(self, prob_idx: List[int]) -> torch.Tensor:
        r"""The joint probability distribution of these states' occurrences
        """
        return self._prob.joint(prob_idx)

    @property
    def probability(self) -> torch.Tensor:
        r"""The probability distribution(s) of these states' occurrences
        """
        return self._prob.joint(list(range(len(self._prob))))

    @property
    @abstractmethod
    def batch_dim(self) -> List[int]:
        r"""The batch dimension of this state
        """

    def numel(self) -> int:
        r"""The number of elements in this data
        """
        return int(np.prod(self.batch_dim))

    @property
    def _squeeze_shape(self) -> List[int]:
        r"""The squeezed shape of this state batch
        """
        return [-1, self._prob.product_dim]

    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        r"""The recognize shape of this state
        """

    @property
    @abstractmethod
    def ket(self) -> torch.Tensor:
        r"""The ket form of this state

        Note:
            If the state is pure, the ket form is the statevector representation.
            If the state is mixed, the ket form is the vectorization of its density matrix representation.

        """

    @property
    def bra(self) -> torch.Tensor:
        r"""Dagger of the ket form.
        """
        return self.ket.mH

    @property
    @abstractmethod
    def density_matrix(self) -> torch.Tensor:
        r"""The density matrix representation of this state.
        """

    @property
    def vec(self) -> torch.Tensor:
        r"""Vectorization of the state: :math:`\textrm{vec}(|i \rangle\langle j|)=|j, i \rangle`
        """
        data = self.density_matrix.view([1, -1, self.dim, self.dim])
        
        newshape = [data.shape[0], data.shape[1], -1, 1]
        rev_src = tuple(range(data.ndim - 1, -1, -1))
        rev_dst = tuple(reversed(newshape))
        data = data.permute(rev_src).reshape(rev_dst).permute(tuple(range(len(newshape) - 1, -1, -1)))
        return data.view(self.batch_dim + [-1, 1])

    @abstractmethod
    def _trace(self, trace_idx: List[int]) -> 'StateSimulator':
        r"""Partial trace of the state

        Args:
            trace_idx: the subsystem indices to be traced out.

        Returns:
            the partial trace of the state as a new state.

        """
        
    @abstractmethod
    def _reset(self, reset_idx: List[int], replace_state: 'StateSimulator') -> 'StateSimulator':
        r"""Reset the state to a new state.

        Args:
            reset_idx: the subsystem indices to be reset.
            replace_state: the state to replace the quantum state.
            
        Note:
            reset_idx should not include all systems, otherwise call `StateSimulator.reset`.
        
        """

    @abstractmethod
    def _transpose(self, transpose_idx: List[int]) -> 'StateSimulator':
        r"""Partial transpose of the state

        Args:
            transpose_idx: the subsystem indices to be transposed.

        Returns:
            the transposed state as a new state.

        """

    @property
    @abstractmethod
    def rank(self) -> Union[int, List[int]]:
        r"""The rank of the state.
        """

    @abstractmethod
    def normalize(self) -> None:
        r"""Normalize this state to the correct format
        """

    @abstractmethod
    def numpy(self) -> np.ndarray:
        r"""Get the data in numpy.

        Returns:
            The numpy array of the data for the quantum state.
        """

    @abstractmethod
    def clone(self) -> 'StateSimulator':
        r"""Return a copy of the quantum state.
        """

    @abstractmethod
    def to(self, dtype: str = None, device: str = None) -> 'StateSimulator':
        r"""Change the property of the data tensor, and return a copy of this State

        Args:
            dtype: the new data type of the state.
            device: the new device of the state.

        """

    @abstractmethod
    def _index_select(self, new_indices: torch.Tensor, system_idx_pairs: Optional[List[List[int]]] = None) -> None:
        r"""Select the entries stored in this state.

        Args:
            new_indices: the indices of the entries to be selected.
            system_idx_pairs: list of system index pairs involved (e.g. [[0, 1], [2, 3]]).
                Used to determine if the operation can be performed locally on a single block.
        """

    def index_select(self, new_indices: Iterable[int]) -> 'StateSimulator':
        r"""Select the entries stored in this state.

        Args:
            new_indices: the new indices of the entries to be selected.

        Returns:
            a new state with selected indices.

        """
        if not isinstance(new_indices, torch.Tensor):
            new_indices = torch.tensor(new_indices, dtype=torch.long, device=self.device)

        assert new_indices.numel() == self.dim, \
            f"The number of new indices {new_indices.numel()} does not match the state dimension {self.dim}."

        new_state = self.clone()
        new_state._index_select(new_indices)
        return new_state

    @abstractmethod
    def permute(self, target_seq: List[int]) -> 'StateSimulator':
        r"""Permute the systems order of the state.

        Args:
            target_seq: the target systems order.

        """

    @abstractmethod
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        r"""Evolve this state with unitary operators.

        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on.
            on_batch: whether this unitary operator evolves on batch axis. Defaults to True.

        Note:
            The difference between `State.evolve` and `State._evolve` is that the former returns a new state 
            by calling an extra clone.

            Support the following transformation
            - (), () -> ()
            - (), (n) -> (n)
            - (n), () -> (n)
            - (n), (n) -> (n)
            - (m), (n) -> error

        """

    @abstractmethod
    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int], on_batch: bool = True) -> None:
        r"""Evolve this state with unitary operators, while record the size of unitary batch

        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on.
            on_batch: whether this unitary operator evolves on batch axis. Defaults to True.

        Note:
            Support the following transformation
            - (), () -> (1)
            - (m), () -> (m)
            - (), (n) -> (n, 1)
            - (m), (n) -> (n, m)
            - (n, m), (n) -> (n, m)
            - (l, m), (n) -> error
        """

    @abstractmethod
    def _evolve_ctrl(self, unitary: torch.Tensor, index: int, sys_idx: List[Union[int, List[int]]]) -> None:
        r"""Evolve this state with unitary operators controlled by a computational state

        Args:
            unitary: the unitary operator.
            index: the index of the computational state that activates the unitary.
            sys_idx: indices of the systems on which the whole unitary is applied. The first element in the list is 
                a list that gives the control system, while the remaining elements are int that give the applied system.

        Note:
            Support the following transformation
            - (), () -> ()
            - (), (n) -> (n)
            - (n), () -> (n)
            - (n), (n) -> (n)
            - (m), (n) -> error

        """

    @abstractmethod
    def _transform(self, op: torch.Tensor, sys_idx: List[int], repr_type: str, on_batch: bool = True) -> None:
        r"""Apply a general linear operator to the state.

        Args:
            op: the input operator.
            sys_idx: the subsystem indices to be applied.
            repr_type: the representation type of input operator, can be 'kraus' or 'choi'.
            on_batch: whether this operator evolves on batch axis. Defaults to True.

        Note:
            The difference between `State.transform` and `State._transform` is that the former returns a new state 
            by calling an extra clone.

            Support the following transformation
            - (), (n) -> (n)
            - (n), () -> (n)
            - (n), (n) -> (n)
            - (m), (n) -> error

        """

    @abstractmethod
    def _expec_val(self, obs: torch.Tensor, sys_idx: List[int]) -> torch.Tensor:
        r"""The expectation value of observables.

        Args:
            obs: the (list of) input observable.
            sys_idx: the system indices to be measured.

        Returns:
            the expectation value of the input observable for the quantum state.

        Note:
            Support the following transformation
            - (), (n) -> (n)
            - (n), () -> (n)
            - (n), (n) -> (n)
            - (m), (n) -> error

        """

    def _expec_val_pauli_terms(
        self,
        pauli_words_r: List[str],
        sites: List[List[int]],
    ) -> torch.Tensor:
        r"""Compute exact expectation values of non-identity Pauli terms.

        Args:
            pauli_words_r: List of Pauli words in right-order form.
            sites: List of acted subsystem indices for each Pauli word.

        Returns:
            Real-valued expectation values for each input term before coefficients,
            with shape ``[num_terms, batch_product]``.
        """
        if len(pauli_words_r) != len(sites):
            raise ValueError(
                "pauli_words_r and sites must have the same length: "
                f"got {len(pauli_words_r)} and {len(sites)}."
            )
        if not pauli_words_r:
            batch_product = int(self.trace().reshape(-1).numel())
            return torch.empty((0, batch_product), dtype=self.dtype, device=self.device)

        pauli_ops = {
            "I": torch.eye(2, dtype=self.dtype, device=self.device),
            "X": torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device),
            "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device),
            "Z": torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device),
        }

        expec_vals = []
        for pauli_word, qubits_idx in zip(pauli_words_r, sites):
            if len(pauli_word) != len(qubits_idx):
                raise ValueError(
                    "Each Pauli word length must match its site length: "
                    f"got word='{pauli_word}' with sites={qubits_idx}."
                )
            obs_factors = []
            for p in pauli_word:
                key = str(p).upper()
                if key not in pauli_ops:
                    raise ValueError(
                        "Unsupported Pauli character in _expec_val_pauli_terms: "
                        f"received '{p}' in '{pauli_word}'."
                    )
                obs_factors.append(pauli_ops[key])

            obs = utils.linalg._nkron(*obs_factors)
            term_expec = self._expec_val(obs.unsqueeze(0), [int(q) for q in qubits_idx])
            expec_vals.append(term_expec.real.reshape(-1))

        return torch.stack(expec_vals, dim=0)

    @abstractmethod
    def _expec_state(self, prob_idx: List[int]) -> 'StateSimulator':
        r"""The expectation with respect to the specific probability distribution(s) of states

        Args:
            prob_idx: indices of probability distributions. Defaults to all distributions.
        """

    @abstractmethod
    def _measure(
        self,
        measure_op: torch.Tensor,
        sys_idx: List[int]
    ) -> Tuple[torch.Tensor, 'StateSimulator']:
        r"""Measure the quantum state with the measured operators.

        Args:
            measure_op: the measurement operators, where the first dimension is the number of operators.
            sys_idx: the system indices to be measured.

        Returns:
            The probability and collapsed states of each measurement result.

        Note:
            Support the following transformation
            - (r), () -> (r)
            - (m, r), () -> (m, r)
            - (r), (n) -> (n, r)
            - (n, r), (n) -> (n, r)
            - (m, r), (n) -> error

        """
        
    def _measure_many(self, measure_op: torch.Tensor, sys_idx_list: List[List[int]]) -> Tuple[torch.Tensor, 'StateSimulator']:
        r"""Measure the quantum state with the measured operators for many measurement operators.
        
        Args:
            measure_op: stacked measurement operators of shape
                ``[num_measure, num_measure_ops, d, d]``.
                Here ``measure_op[i]`` is the operator set used in the i-th measurement, and
                all measurements in this call must share the same ``num_measure_ops`` and ``d``.
            sys_idx_list: a list of system index lists to be measured for each measurement in order.
                Must satisfy ``len(sys_idx_list) == num_measure``. For each i, the dimension implied by
                ``sys_idx_list[i]`` must match the operator dimension ``d`` of ``measure_op[i]``.
            
        Returns:
            A tuple ``(prob, state)`` where:

            - ``prob``: the (broadcasted) **joint** probability distribution of the measurement
              variables added in this call, with the last ``num_measure`` dimensions corresponding
              to the outcomes of each measurement **in order**. It may carry leading broadcast/batch
              dimensions (including existing probability dimensions when measuring a probabilistic
              state).
            - ``state``: the collapsed state after all measurements.
        """
        num_measure = int(measure_op.shape[0])
        if len(sys_idx_list) != num_measure:
            raise ValueError(
                "sys_idx_list length must match measure_op.shape[0]: "
                f"got len(sys_idx_list)={len(sys_idx_list)} and num_measure={num_measure}."
            )

        state = self.clone()
        prob_start = len(state._prob)
        for i in range(num_measure):
            _, state = state._measure(measure_op[i], sys_idx_list[i])

        prob_idx = list(range(prob_start, len(state._prob)))
        prob = state._joint_probability(prob_idx)
        return prob, state
    
    def _measure_by_state(
        self,
        measure_basis: 'StateSimulator',
        sys_idx: List[int],
        keep_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, 'StateSimulator']]:
        r"""Measure the quantum state with the measured basis, which is a state basis represented by a batched pure StateSimulator (i.e., has ket property)

        Args:
            measure_basis: the measurement basis.
            sys_idx: the system indices to be measured.
            keep_state: if ``False``, return only probabilities (delegates to :meth:`measure` with the same flag).

        Returns:
            Probabilities, or ``(prob, collapsed_state)`` when ``keep_state`` is ``True``.
        """
        return self.measure(
            system_idx=sys_idx,
            measure_op=measure_basis.density_matrix,
            keep_state=keep_state,
        )

    def evolve(self, unitary: torch.Tensor, sys_idx: Optional[Union[int, List[int]]] = None) -> 'StateSimulator':
        r"""Evolve this state with unitary operators.

        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on. Defaults to all systems.

        Returns:
            the evolved state.

        Note:
            Evolve support the broadcast rule.

        """
        sys_idx = self._check_sys_idx(sys_idx)
        self._check_op_dim(unitary, sys_idx)

        new_state = self.clone()
        new_state._evolve(unitary, sys_idx)
        return new_state

    def _evolve_many(
        self,
        unitary: torch.Tensor,
        sys_idx_list: List[List[int]],
        on_batch: bool = True,
    ) -> None:
        r"""Evolve this state by applying the same unitary on many subsystem index groups.

        This is a batching helper to reduce Python overhead. Backends may override this
        method to fuse or vectorize the underlying kernel calls.

        Args:
            unitary: The unitary operator.
            sys_idx_list: A list of subsystem indices. Each element is a list of ints (multi-subsystem).
            on_batch: Whether the operator batch dimension aligns with the state batch.

        Note:
            This method is **in-place** and returns ``None``.
        """
        for sys_idx in sys_idx_list:
            self._evolve(unitary, sys_idx, on_batch)
        return None

    def _evolve_many_batched_groups(
        self,
        unitary_groups: List[torch.Tensor],
        sys_idx_groups: List[List[List[int]]],
        on_batch: bool = True,
    ) -> None:
        r"""Internal helper for applying varying unitaries in groups.

        This default implementation is a Python for-loop so that all backends can
        support Layer fast paths without runtime capability checks.

        Args:
            unitary_groups: list of tensors. Each element is either a single unitary
                (broadcast) shaped [..., d, d], or a per-op stack shaped
                [N, ..., d, d] where N == len(sys_idx_list) for the corresponding group.
            sys_idx_groups: list of sys_idx_list, each is a list of sys_idx (List[int]).
            on_batch: whether unitary batch dims align with state batch dims.
        """
        if len(unitary_groups) != len(sys_idx_groups):
                raise ValueError(
                    "unitary_groups and sys_idx_groups must have the same number of groups: "
                    f"got {len(unitary_groups)} and {len(sys_idx_groups)}."
                )
                
        for unitary_group, sys_idx_list in zip(unitary_groups, sys_idx_groups):
            n_ops = len(sys_idx_list)
            per_op = unitary_group.dim() >= 3 and unitary_group.size(0) == n_ops
            for i, sys_idx in enumerate(sys_idx_list):
                u = unitary_group[i] if per_op else unitary_group
                self._evolve(u, self._check_sys_idx(sys_idx), on_batch)

    def transform(self, op: torch.Tensor, sys_idx: Optional[Union[int, List[int]]] = None,
                  repr_type: str = 'kraus') -> 'StateSimulator':
        r"""Apply a general linear operator to the state.

        Args:
            op: the input operator.
            sys_idx: the qubit indices to be applied. Defaults to all systems.
            repr_type: the representation type of input operator. Defaults to 'kraus'.

        Returns:
            the transformed state.

        Note:
            Transform support the broadcast rule similar to evolve.

        """
        sys_idx = self._check_sys_idx(sys_idx)

        repr_type = repr_type.lower()
        if repr_type == 'kraus':
            self._check_op_dim(op, sys_idx)
            if len(op.shape) == 2:
                op = op.unsqueeze(0)
        elif repr_type == 'choi':
            pass
        else:
            raise ValueError(
                f"Unsupported representation type {repr_type}: expected 'kraus' or 'choi'.")

        new_state = self.clone()
        new_state._transform(op, sys_idx, repr_type)
        return new_state

    def _transform_many(
        self,
        op: torch.Tensor,
        sys_idx_list: List[List[int]],
        repr_type: str,
        on_batch: bool = True,
    ) -> None:
        r"""Transform this state by applying the same channel/operator on many subsystem index groups.

        Args:
            op: The input operator. For 'kraus', supports [num_kraus, d, d] or
                [batch, num_kraus, d, d]. For 'choi', supports [d_out^2, d_in^2]
                or [batch, d_out^2, d_in^2].
            sys_idx_list: A list of subsystem indices. Each element can be an int
                (single subsystem) or a list of ints (multi-subsystem).
            repr_type: Representation type of the operator, 'kraus' or 'choi'.
            on_batch: Whether the operator batch dimension aligns with the state batch.

        Note:
            This method is **in-place** and returns ``None``.
        """
        for sys_idx in sys_idx_list:
            self._transform(op, sys_idx, repr_type, on_batch)
        return None

    def expec_val(self, hamiltonian: Hamiltonian, shots: Optional[int] = None, decompose: bool = False) -> torch.Tensor:
        r"""The expectation value of the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            shots: The number of shots to measure the observable. Should not be used in simulation mode.
            decompose: If decompose is ``True``, it will return the expectation value of each term.

        Returns:
            The expectation value (per term) of the input observable for the quantum state.

        Note:
            currently only run in qubit case.
            
        Raises:
            NotImplementedError: If `shots` is specified, since simulators do not support shot-based computations.
            NotImplementedError: If the state is not a qubit state.
            AssertionError: If the number of qubits in the Hamiltonian is greater than the number of state qubits.

        """
        if shots:
            raise NotImplementedError(
                "The shots argument is not supported in simulators. "
                "Please use the subclass of `StateOperator` for shot-based computations.")
        
        if not self.are_qubits():
            raise NotImplementedError(
                f"Currently only support qubit computation in Hamiltonian tasks: received {self._sys_dim}")

        assert hamiltonian.n_qubits <= self.num_qubits, \
            f"The number of qubits in the Hamiltonian {hamiltonian.n_qubits} is greater the number of state qubits {self.num_qubits}"

        num_terms, list_qubits_idx = hamiltonian.n_terms, hamiltonian.sites
        list_coef = hamiltonian.coefficients
        list_pauli_words_r = hamiltonian.pauli_words_r

        expec_val_terms: List[torch.Tensor] = [None] * num_terms
        pauli_words_non_id: List[str] = []
        sites_non_id: List[List[int]] = []
        coef_list_non_id = []
        non_id_indices = []

        for i in range(num_terms):
            qubits_idx = list_qubits_idx[i]
            if qubits_idx == ['']:
                expec_val_terms[i] = list_coef[i] * self.trace()
                continue

            pauli_words_non_id.append(list_pauli_words_r[i])
            sites_non_id.append([int(q) for q in qubits_idx])
            coef_list_non_id.append(list_coef[i])
            non_id_indices.append(i)

        if pauli_words_non_id:
            expec_vals = self._expec_val_pauli_terms(pauli_words_non_id, sites_non_id)
            coef_tensor = torch.tensor(coef_list_non_id, dtype=expec_vals.dtype, device=expec_vals.device).unsqueeze(-1)
            expec_vals = expec_vals * coef_tensor
            for idx, term_idx in enumerate(non_id_indices):
                expec_val_terms[term_idx] = expec_vals[idx]
            expec_val_terms = [
                term.reshape(-1) if term.ndim != 1 else term
                for term in expec_val_terms
            ]

        expec_val_each_term = torch.stack(expec_val_terms)
        return expec_val_each_term if decompose else torch.sum(expec_val_each_term, dim=0)

    def expec_state(self, prob_idx: Optional[Union[int, List[int]]] = None) -> 'StateSimulator':
        r"""The expectation with respect to the specific probability distribution(s) of states

        Args:
            prob_idx: indices of probability distributions. Defaults to all distributions.

        Returns:
            The expected State obtained from the taken probability distributions.

        """
        num_prob = len(self._prob)
        if num_prob == 0:
            return self.clone()

        if prob_idx is None:
            prob_idx = list(range(num_prob))
        elif isinstance(prob_idx, int):
            normalized = num_prob + prob_idx if prob_idx < 0 else prob_idx
            prob_idx = [normalized]
        else:
            prob_idx = [num_prob + idx if idx < 0 else idx for idx in sorted(prob_idx)]
        if not prob_idx:
            return self.clone()
        for idx in prob_idx:
            if idx < 0 or idx >= num_prob:
                raise IndexError(f"Probability index out of range: got {idx}, expected in [0, {num_prob - 1}].")
        return self._expec_state(prob_idx)

    def to_prod_sum(self, subgroup_indices: List[List[int]], tol: Optional[float] = None) -> 'StateSimulator':
        r"""Convert the state into a subgroup-level product-sum form.

        Args:
            subgroup_indices: Partition of system indices into ordered subgroups.
            tol: Truncation tolerance. If None, backend-specific default is used.
        """
        raise NotImplementedError(f"Backend '{self.backend}' does not implement to_prod_sum().")

    def measure(self, system_idx: Optional[Union[int, List[int]]] = None, shots: Optional[int] = None,
                measure_op: Optional[torch.Tensor] = None, 
                is_povm: bool = False, keep_state: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, 'StateSimulator']]:
        r"""Measure the quantum state

        Args:
            system_idx: the system indices to be measured. Defaults to all systems.
            shots: the number of shots to measure. Should not be used in simulation mode.
            measure_op: measurement operator. Defaults to the computational basis.
            is_povm: whether the measurement is a POVM.
            keep_state: whether to keep the state after measurement.

        Returns:
            The probability of each measurement result. The corresponding collapsed state will also be returned if
            `is_povm` is False or `keep_state` is True.

        Raises:
            NotImplementedError: If `shots` is specified, since simulators do not support shot-based computations.
            UserWarning: `is_povm` and `keep_state` cannot be both True,
            since a general POVM does not distinguish states.

        """
        if shots:
            raise NotImplementedError(
                "The shots argument is not supported in simulators. "
                "Please use the subclass of `StateOperator` for shot-based computations.")
        
        system_idx = self._check_sys_idx(system_idx)

        if measure_op is None:
            dim = int(np.prod([self._sys_dim[i] for i in system_idx]))
            identity = torch.eye(dim, dtype=self.dtype, device=self.device).unsqueeze(-1)
            measure_op = identity @ identity.mH
        else:
            self._check_op_dim(measure_op, system_idx)

        if is_povm:
            if keep_state:
                raise ValueError(
                    "`is_povm` and `keep_state` cannot be both True, " +
                    "since a general POVM does not distinguish states.")
            return self._expec_val(measure_op, system_idx).real

        if keep_state:
            prob, collapsed_state = self._measure(measure_op, system_idx)
            return prob, collapsed_state
        return self._expec_val(measure_op, system_idx).real

    @abstractmethod
    def sqrt(self) -> torch.Tensor:
        r"""Matrix square root of the state.
        """

    @abstractmethod
    def log(self) -> torch.Tensor:
        r"""Matrix logarithm of the state.
        """

    def trace(self, trace_idx: Optional[Union[List[int], int]] = None) -> Union[torch.Tensor, 'StateSimulator']:
        r"""(Partial) trace of the state

        Args:
            trace_idx: the subsystem indices to be traced out. Defaults to all systems.

        Returns:
            the trace of the state as a Tensor, or a new state if sys_idx is not None.
        """
        if trace_idx is None:
            return utils.linalg._trace(self.density_matrix, -2, -1)
        if isinstance(trace_idx, int):
            trace_idx = [trace_idx]
        assert max(trace_idx) < self.num_systems, \
            f"The trace index {trace_idx} should be smaller than number of systems {self.num_systems}"

        return self._trace(trace_idx)
    
    def reset(self, reset_idx: Union[List[int], int], replace_state: 'StateSimulator') -> 'StateSimulator':
        r"""Reset the state to a new state.

        Args:
            reset_idx: the subsystem indices to be reset.
            replace_state: the state to replace the quantum state.
            
        Returns:
            the new state after resetting.
        
        """
        if isinstance(reset_idx, int):
            reset_idx = [reset_idx]
        system_dim = [self.system_dim[i] for i in reset_idx]
        assert replace_state.system_dim == system_dim, \
            f"The system dimension of the replace state {replace_state.system_dim} does not match the reset system dimension {system_dim}."
        if len(reset_idx) == self.num_systems and self._get_data_tensor().requires_grad:
            warnings.warn("All systems will be reset: gradient break here.", UserWarning)
            return replace_state.clone()
        if replace_state.batch_dim:
            raise NotImplementedError(
                "A batched replace state is not supported")
        
        return self._reset(reset_idx, replace_state)
    
    def transpose(self, transpose_idx: Optional[Union[List[int], int]] = None) -> 'StateSimulator':
        r"""(Partial) transpose of the state

        Args:
            transpose_idx: the subsystem indices to be transposed. Defaults to all systems.

        Returns:
            the transpose of the state as a Tensor, or a new state if sys_idx is not None.
        """
        if transpose_idx is None:
            transpose_idx = list(range(self.num_systems))
        elif isinstance(transpose_idx, int):
            transpose_idx = [transpose_idx]
        assert max(transpose_idx) < self.num_systems, \
            f"The transpose index {transpose_idx} should be smaller than number of systems {self.num_systems}"

        return self._transpose(transpose_idx)

