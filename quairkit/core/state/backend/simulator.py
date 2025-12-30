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


import warnings
from abc import abstractmethod
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from ... import Hamiltonian, utils
from .base import State


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
        self._list: List[torch.Tensor] = []
        self._dims: List[int] = []
        if probs:
            for p in probs:
                self._list.append(p)
                self._dims.append(int(p.shape[-1]))

    def __len__(self) -> int:
        return len(self._list)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._list[idx]

    @property
    def list(self) -> List[torch.Tensor]:
        return self._list

    def clone_list(self) -> List[torch.Tensor]:
        return [p.clone() for p in self._list]

    @property
    def shape(self) -> List[int]:
        return self._dims.copy()
    
    @property
    def non_prob_dim(self) -> List[int]:
        num_prob = len(self._list)
        return [] if num_prob == 0 else list(self._list[-1].shape[:-num_prob])

    @property
    def product_dim(self) -> int:
        return int(np.prod(self._dims))

    def clear(self) -> None:
        self._list.clear()
        self._dims.clear()

    def prepare_new(self, prob: torch.Tensor, dtype: Optional[torch.dtype] = None,
                    device: Optional[torch.device] = None, real_only: bool = False) -> torch.Tensor:
        r"""
        Canonicalize a fresh probability tensor to shape [1]*num_prev + [-1].
        This does not append; it only prepares the shaped tensor.
        """
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
        if len(self._list) == 0:
            # Keep previous behavior: return a CPU scalar 1.0
            return torch.tensor(1.0)

        dtype = self._list[0].dtype
        device = self._list[0].device
        result = torch.tensor(1.0, dtype=dtype, device=device)
        for idx in sorted(prob_idx):
            p = self._list[idx]
            if p.dim() > result.dim():
                result = result.view(list(result.shape) + [1] * (p.dim() - result.dim()))
            result = result * p
        return result

    def clone(self) -> 'ProbabilityData':
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
        
        non_batch_len = 2 + len(self._prob)
        self._batch_dim = list(data.shape[:-non_batch_len])

        self._data = data.contiguous().to(dtype=self.dtype, device=self.device)
        self._system_seq = list(range(len(sys_dim))) if system_seq is None else system_seq

        self._keep_dim = False # the flag to switch on the unitary matrix recording.

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

    @abstractmethod
    def expand_as(self, other: 'StateSimulator') -> 'StateSimulator':
        r"""Expand this tensor to the same size as other.

        Args:
            other: the state to be expanded to.

        Note:
            See torch.Tensor.expand_as() for more information about expand logic.

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
    def _index_select(self, new_indices: torch.Tensor) -> None:
        r"""Select the entries stored in this state.

        Args:
            new_indices: the indices of the entries to be selected.
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

    @abstractmethod
    def _expec_state(self, prob_idx: List[int]) -> 'StateSimulator':
        r"""The expectation with respect to the specific probability distribution(s) of states

        Args:
            prob_idx: indices of probability distributions. Defaults to all distributions.
        """

    @abstractmethod
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'StateSimulator']:
        r"""Measure the quantum state with the measured operators.

        Args:
            measure_op: the measurement operators, where the first dimension is the number of operators.

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

    def transform(self, op: torch.Tensor, sys_idx: Optional[Union[int, List[int]]] = None,
                  repr_type: str = 'kraus') -> 'StateSimulator':
        r"""Apply a general linear operator to the state.

        Args:
            op: the input operator.
            sys_idx: the qubit indices to be applied. Defaults to all systems.
            repr_type: the representation type of input operator. Defaults to 'kraus'.

        Returns:
            the transformed state.

        """
        # TODO add assertion for the dimension of input channels
        sys_idx = self._check_sys_idx(sys_idx)

        # TODO add support for batched operators
        repr_type = repr_type.lower()
        if repr_type == 'kraus':
            self._check_op_dim(op, sys_idx)
            if len(op.shape) == 2:
                op = op.unsqueeze(0)
            elif len(op.shape) > 3:
                raise NotImplementedError(
                    'consider the batched Kraus operators in the upcoming future')

        elif repr_type == 'choi':
            if len(op.shape) > 2:
                raise NotImplementedError(
                    'consider the batched Choi operators in the upcoming future')
        else:
            raise ValueError(
                f"Unsupported representation type {repr_type}: expected 'kraus' or 'choi'.")

        new_state = self.clone()
        new_state._transform(op, sys_idx, repr_type)
        return new_state

    def product_trace(self, trace_state: 'StateSimulator', trace_idx: List[int]) -> 'StateSimulator':
        r"""Partial trace over this state, when this state is a product state

        Args:
            trace_state: the state for the subsystem to be traced out.
            trace_idx: the subsystem indices to be traced out.

        Note:
            This function only works when the state is a product state represented by PureState

        """
        return self.trace(trace_idx)

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
        list_coef, list_matrices = hamiltonian.coefficients, hamiltonian.pauli_words_matrix

        expec_val_each_term = []
        for i in range(num_terms):
            qubits_idx = list_qubits_idx[i]
            if qubits_idx == ['']:
                expec_val_each_term.append(list_coef[i] * self.trace())
                continue
            matrix = list_matrices[i]

            #TODO this assertion should be done by Hamiltonian, not State
            assert 2 ** len(qubits_idx) == matrix.shape[0], \
                f"The qubit index {qubits_idx} does not match the matrix dimension {matrix.shape[0]}"

            expec_val = self._expec_val(matrix.unsqueeze(0), qubits_idx).squeeze(-1).real * list_coef[i]
            expec_val_each_term.append(expec_val)

        expec_val_each_term = torch.stack(expec_val_each_term)
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
            prob_idx = [prob_idx]
        else:
            prob_idx = [num_prob + idx if idx < 0 else idx for idx in sorted(prob_idx)]
        return self._expec_state(prob_idx)

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
            return self._expec_val(measure_op, system_idx)

        prob, collapsed_state = self._measure(measure_op, system_idx)
        return (prob, collapsed_state) if keep_state else prob

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
        if len(reset_idx) == self.num_systems and self._data.requires_grad:
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
