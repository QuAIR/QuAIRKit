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


import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch

from ... import Hamiltonian, base

__all__ = ['State']


class State(ABC):
    r"""The abstract base class for all quantum state backends in QuAIRKit.
    
    Args:
        sys_dim: a list of dimensions for each system.
    
    """
    backend: Optional[str] = None
    
    def __init__(self, sys_dim: List[int], 
                 dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> None:
        assert all(d >= 2 for d in sys_dim), \
            f"All systems must have dimension >= 2: received {sys_dim}"
        
        self._dtype = dtype or base.get_dtype()
        self._device = device or base.get_device()
        
        self._sys_dim = sys_dim.copy()

    @property
    def dtype(self) -> torch.dtype:
        r"""The data type of this state
        """
        return self._dtype

    @property
    def device(self) -> torch.device:
        r"""The device of this state
        """
        return self._device

    @property
    def system_dim(self) -> List[int]:
        r"""The list of dimensions for each system
        """
        return self._sys_dim.copy()

    @system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        assert int(np.prod(sys_dim)) == self.dim, \
            f"The input system dim {sys_dim} does not match the original state dim {self.dim}."
        self._sys_dim = sys_dim.copy()

    @property
    def dim(self) -> int:
        r"""The dimension of this state
        """
        return math.prod(self.system_dim)

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
        return sum(x == 2 for x in self._sys_dim)

    def _check_sys_idx(self, sys_idx: Optional[Union[int, List[int]]]) -> List[int]:
        r"""Check the system dimension of the input system indices

        Args:
            sys_idx: the system indices to be checked.

        Returns:
            the list of system dimensions.

        """
        if sys_idx is None:
            sys_idx = list(range(self.num_systems))
        elif isinstance(sys_idx, int):
            sys_idx = [sys_idx]
        else:
            sys_idx = list(sys_idx)
            assert (
                all(isinstance(x, int) for x in sys_idx)
                and min(sys_idx) >= 0
                and max(sys_idx) < self.num_systems
            ), f"The input system indices {sys_idx} should be a list of integers in range [0, {self.num_systems - 1}]"
        return sys_idx
    
    @abstractmethod
    def clone(self) -> 'State':
        r"""Return a copy of the quantum state.
        """
        
    @abstractmethod
    def measure(self, system_idx: Optional[Union[int, List[int]]], 
                shots: Optional[int], measure_op: Union[str, Optional[torch.Tensor]]) -> torch.Tensor:
        r"""Measure the quantum state with the given measurement operator.
        
        Args:
            system_idx: the system indices to be measured.
            shots: the number of measurement shots.
            measure_op: the measurement operator.
            
        Returns:
            A tensor containing the measurement results.
        
        """
        
    @abstractmethod
    def expec_val(self, hamiltonian: Hamiltonian, shots: Optional[int], decompose: bool) -> torch.Tensor:
        r"""The expectation value of the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            shots: the total number of shots to measure.
            decompose: If decompose is ``True``, it will return the expectation value of each term.

        Returns:
            The measured expectation value (per term) of the input observable for the quantum state.
        
        """

