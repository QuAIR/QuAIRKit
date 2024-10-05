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
The module that implements various backends of the state.
"""

r"""
States in QuAIRKit obeys the following broadcast rule:

| Batch size of Operators | Batch size of Input State | Batch size of Output State |
|:-----------------------:|:-------------------------:|:--------------------------:|
|           1             |             1             |             1            |
|           n             |             1             |             n            |
|           1             |             n             |             n            |
|           n             |             n             |             n            |
|           m             |             n             |          (m, n)          |

Note that the last rule currently is only applicable to the `evolve_keep_dim` method.
"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ... import Hamiltonian, base, utils


class State(ABC):
    r"""The abstract base class for quantum states in QuAIRKit.

    Args:
        data: tensor array for quantum state(s).
        sys_dim: a list of dimensions for each system.
        system_seq: the system order of this state. Defaults to be from 0 to n - 1.

    """
    def __init__(self, data: torch.Tensor, sys_dim: List[int], system_seq: Optional[List[int]] = None):
        self._data, self._sys_dim = data.contiguous().to(dtype=base.get_dtype(), device=base.get_device()), sys_dim
        self._system_seq = list(range(len(sys_dim))) if system_seq is None else system_seq

        self.is_swap_back = True # TODO: depreciated, will be removed in the future
        self._keep_dim = False

    @abstractmethod
    def __getitem__(self, key: Union[int, slice]) -> 'State':
        r"""Indexing of the State class
        """
    
    @abstractmethod
    def index_select(self, dim: int, index: torch.Tensor) -> 'State':
        r"""Indexing elements from the State batch along the given dimension.
        
        Args:
            dim: the dimension in which we index
            index: the 1-D tensor containing the indices to index
        
        Note:
            Here `dim` refers to the dimension in batch_dim, dimensions for data are not considered.
        
        """
        
    @abstractmethod
    def expand(self, batch_dim: List[int]) -> 'State':
        r"""Expand the batch dimension of the State.
        
        Args:
            batch_dim: the new batch dimension
            
        Note:
            See torch.expand() for more information about expand. 
            This expand function, however, may change the format of _data to keep save memory.
        
        """

    def expand_as(self, other: Union['State', torch.Tensor]) -> 'State':
        r"""Expand this tensor to the same size as other.
        
        Args:
            other: the state/tensor to be expanded to. Note that if other is a
             torch.Tensor, the batch dimension of the state will be expanded to the same size as other.
        
        Note:
            See torch.Tensor.expand_as() for more information about expand.
            This expand function, however, may change the format of _data to keep save memory.
         
        """
        other_batch_dim = other.batch_dim if isinstance(other, State) else list(other.size())
        return self.expand(other_batch_dim)

    def __copy__(self) -> 'State':
        return self.clone()

    def __str__(self) -> str:
        split_line = '\n---------------------------------------------------\n'
        s = f"{split_line} Backend: {self.backend}\n"
        s += f" System dimension: {self._sys_dim}\n"
        s += f" System sequence: {self._system_seq}\n"

        data = np.round(self.numpy(), decimals=2)
        if not self.batch_dim:
            s += str(data.squeeze(0))
            s += split_line
            return s

        s += f" Batch size: {self.batch_dim}\n"
        for i, mat in enumerate(data):
            s += f"\n # {i}:\n{mat}"
        s += split_line
        return s

    def __matmul__(self, other: 'State') -> torch.Tensor:
        r"""Matrix product between the density matrix representation of two states

        Args:
            other: a quantum state

        Returns:
            the product of these two states

        """
        return torch.matmul(self.density_matrix, other.density_matrix)

    @property
    @abstractmethod
    def backend() -> str:
        r"""The backend of this state.
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

    @property
    def data(self) -> torch.Tensor:
        r"""The data tensor of this state
        """
        warnings.warn(
            'The data property is depreciated, use ket or density_matrix instead', DeprecationWarning)
        return self._data
    
    @property
    def batch_dim(self) -> List[int]:
        r"""The batch dimension of this state
        """
        if hasattr(self, '_batch_dim'):
            return self._batch_dim
        else:
            raise KeyError(f"The state class {self.backend} does not have batch functional")
    
    @property
    def shape(self) -> torch.Size:
        r"""The shape of this state
        """
        return self._data.shape
    
    @property
    def dtype(self) -> torch.dtype:
        r"""The data type of this state
        """
        return self._data.dtype
    
    @property
    def device(self) -> torch.device:
        r"""The device of this state
        """
        return self._data.device

    def numel(self) -> int:
        r"""The number of elements in this data
        """
        return int(np.prod(self.batch_dim)) if self.batch_dim else 1

    @property
    def dim(self) -> int:
        r"""The dimension of this state
        """
        return self._data.shape[-1]
    
    @property
    def system_dim(self) -> List[int]:
        r"""The list of dimensions for each system
        """
        return self._sys_dim.copy()
    
    @system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        r"""Set the system dimensions of the state
        
        Args:
            sys_dim: the target system dimensions.
        
        """
        self.reset_sequence()
        assert int(np.prod(sys_dim)) == self.dim, \
            f"The input system dim {sys_dim} does not match the original state dim {self.dim}."
        self._sys_dim = sys_dim.copy()
        self._system_seq = list(range(len(sys_dim)))
    
    def are_qubits(self) -> bool:
        r"""Whether all systems are qubits
        """
        return all(x == 2 for x in self._sys_dim)
    
    def are_qutrits(self) -> bool:
        r"""Whether all systems are qutrits
        """
        return all(x == 3 for x in self._sys_dim)
        
    @property
    def num_systems(self) -> int:
        r"""The number of systems
        """
        return len(self._sys_dim)

    @property
    def num_qubits(self) -> int:
        r"""The number of qubits of this state, when all systems are qubits
        """
        assert self.are_qubits, \
            f"Not all systems are qubits: received {self._sys_dim}"
        return len(self._sys_dim)

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
        data = np.reshape(data, [data.shape[0], data.shape[1], -1, 1], order='F')
        return data.squeeze((0,1)).view(self.batch_dim + [-1, 1])
    
    @abstractmethod 
    def _trace(self, trace_idx: List[int]) -> 'State':
        r"""Partial trace of the state
        
        Args:
            trace_idx: the subsystem indices to be traced out.
        
        Returns:
            the partial trace of the state as a new state.
        
        """
        
    @abstractmethod
    def _transpose(self, transpose_idx: List[int]) -> 'State':
        r"""Partial transpose of the state
        
        Args:
            transpose_idx: the subsystem indices to be transposed.
        
        Returns:
            the transposed state as a new state.
        
        """

    @property
    def rank(self) -> List[int]:
        r"""The rank of the state.
        """
        dtype = self._data.dtype
        tol = 1e-8 if dtype == torch.complex64 else 1e-12
        tol *= self.dim
        return torch.linalg.matrix_rank(self.density_matrix, 
                                        tol=tol, hermitian=True).tolist()

    @abstractmethod
    def normalize(self) -> None:
        r"""Normalize this state
        """

    def numpy(self) -> np.ndarray:
        r"""Get the data in numpy.

        Returns:
            The numpy array of the data for the quantum state.
        """
        return self._data.detach().numpy()

    @abstractmethod
    def clone(self) -> 'State':
        r"""Return a copy of the quantum state.
        """
    
    @abstractmethod
    def fit(self, state_backend: str) -> torch.Tensor:
        r"""Convert the data to the specified backend.
        
        Args:
            state_backend: the target backend of the state. Available options see quairkit.Backend.

        Returns:
            a copy of the fitted data.
        
        """
    
    def to(self, dtype: str = None, device: str = None) -> 'State':
        r"""Change the property of the data tensor, and return a copy of this State

        Args:
            dtype: the new data type of the state.
            device: the new device of the state.

        """
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
    
    def permute(self, target_seq: List[int]) -> 'State':
        r"""Permute the systems order of the state.
        
        Args:
            target_seq: the target systems order.
        
        """
        new_state = self.clone()
        new_state.system_seq = target_seq
        new_state._system_seq = list(range(self.num_systems))
        return new_state

    def reset_sequence(self) -> None:
        r"""reset the system order to default sequence i.e. from 1 to n.
        """
        self.system_seq = list(range(self.num_systems))
    
    @abstractmethod
    def _evolve(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        r"""Evolve this state with unitary operators.

        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on.
        
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
    def _evolve_keep_dim(self, unitary: torch.Tensor, sys_idx: List[int]) -> None:
        r"""Evolve this state with unitary operators, while record the size of unitary batch
        
        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on.
            
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
    def _transform(self, op: torch.Tensor, sys_idx: List[int], repr_type: str) -> None:
        r"""Apply a general linear operator to the state.

        Args:
            op: the input operator.
            sys_idx: the subsystem indices to be applied.
            repr_type: the representation type of input operator.

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
    def _measure(self, measure_op: torch.Tensor, sys_idx: List[int]) -> Tuple[torch.Tensor, 'State']:
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
        
    def evolve(self, unitary: torch.Tensor, sys_idx: Optional[Union[int, List[int]]] = None) -> 'State':
        r"""Evolve this state with unitary operators.

        Args:
            unitary: the unitary operator.
            sys_idx: the system indices to be acted on. Defaults to all systems.

        Returns:
            the evolved state.

        Note:
            Evolve support the broadcast rule.
        
        """
        if sys_idx is None:
            sys_idx = list(range(self.num_systems))
        if isinstance(sys_idx, int):
            sys_idx = [sys_idx]
        
        applied_dim = unitary.shape[-1]
        assert applied_dim == int(np.prod((list_dim := [self._sys_dim[i] for i in sys_idx]))), \
            f"The unitary's dimension {applied_dim} does not match the dimensions of its acting systems {list_dim}"
        
        new_state = self.clone()
        new_state._evolve(unitary, sys_idx)
        return new_state

    def transform(self, op: torch.Tensor, sys_idx: Optional[Union[int, List[int]]] = None, 
                  repr_type: str = 'kraus') -> 'State':
        r"""Apply a general linear operator to the state.

        Args:
            op: the input operator.
            sys_idx: the qubit indices to be applied. Defaults to all systems.
            repr_type: the representation type of input operator. Defaults to 'kraus'.

        Returns:
            the transformed state.

        """
        # TODO add assertion for the dimension of input channels
        if sys_idx is None:
            sys_idx = list(range(self.num_systems))
        if isinstance(sys_idx, int):
            sys_idx = [sys_idx]
        
        # TODO add support for batched operators
        repr_type = repr_type.lower()
        if repr_type == 'kraus':
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

    def expec_val(self, hamiltonian: Hamiltonian, shots: Optional[int] = 0, 
                  decompose: Optional[bool] = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""The expectation value of the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            decompose: If decompose is ``True``, it will return the expectation value of each term.

        Returns:
            The expectation value (per term) of the input observable for the quantum state.
        
        Note:
            currently only run in qubit case.
        
        """
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
                expec_val_each_term.append(list_coef[i] * self.trace)
                continue
            matrix = list_matrices[i]
            
            #TODO this assertion should be done by Hamiltonian, not State
            assert 2 ** len(qubits_idx) == matrix.shape[0], \
                f"The qubit index {qubits_idx} does not match the matrix dimension {matrix.shape[0]}"
            
            expec_val = self._expec_val(matrix.unsqueeze(0), qubits_idx).squeeze(-1).real * list_coef[i]
            expec_val_each_term.append(expec_val)
        
        expec_val_each_term = torch.stack(expec_val_each_term)
        return expec_val_each_term if decompose else torch.sum(expec_val_each_term, dim=0)

    def measure(self, measured_op: torch.Tensor = None, sys_idx: Optional[Union[int, List[int]]] = None, 
                is_povm: Optional[bool] = False, keep_state: Optional[bool] = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, 'State']]:
        r"""Measure the quantum state

        Args:
            measured_op: measurement operator. Defaults to the computational basis.
            sys_idx: the system indices to be measured. Defaults to all systems.
            is_povm: whether the measurement is a POVM.
            keep_state: whether to keep the state after measurement.

        Returns:
            The probability of each measurement result. The corresponding collapsed state will also be returned if
            `is_povm` is False or `keep_state` is True.
        
        Raises:
            UserWarning: `is_povm` and `keep_state` cannot be both True,
            since a general POVM does not distinguish states.
        
        """
        # TODO add assertion for the dimension of measured operators
        if sys_idx is None:
            sys_idx = list(range(self.num_systems))
        if isinstance(sys_idx, int):
            sys_idx = [sys_idx]
        
        if measured_op is None:
            dim = int(np.prod([self._sys_dim[i] for i in sys_idx]))
            identity = torch.eye(dim, dtype=self.dtype, device=self.device).unsqueeze(-1)
            measured_op = identity @ identity.mH
        
        if is_povm:
            if keep_state:
                raise ValueError(
                    "`is_povm` and `keep_state` cannot be both True, " + 
                    "since a general POVM does not distinguish states.")
            return self._expec_val(measured_op, sys_idx)
            
        prob, collapsed_state = self._measure(measured_op, sys_idx)
        return (prob, collapsed_state) if keep_state else prob
    
    @abstractmethod
    def sqrt(self) -> torch.Tensor:
        r"""Matrix square root of the state.
        """
    
    @abstractmethod
    def log(self) -> torch.Tensor:
        r"""Matrix logarithm of the state.
        """
    
    def trace(self, trace_idx: Optional[Union[List[int], int]] = None) -> Union[torch.Tensor, 'State']:
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
    
    def transpose(self, transpose_idx: Optional[Union[List[int], int]] = None) -> 'State':
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
