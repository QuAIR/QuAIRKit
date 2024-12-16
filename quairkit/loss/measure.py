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
The source file of the class for the measurement.
"""

import math
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..core import Hamiltonian, Operator, utils
from ..core.intrinsic import _alias, _digit_to_int
from ..core.state import State
from ..database import pauli_str_basis, std_basis

__all__ = ["Measure", "ExpecVal"]


# Debug: torch.trace
class ExpecVal(Operator):
    r"""The class of the loss function to compute the expectation value for the observable.

    This interface can make you using the expectation value for the observable as the loss function.

    Args:
        hamiltonian: The input observable.
    """
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__()
        self.hamiltonian = hamiltonian
    
    def __call__(self, state: State, decompose: Optional[bool] = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.forward(state, decompose)

    def forward(self, state: State, decompose: Optional[bool] = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""Compute the expectation value of the observable with respect to the input state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the expectation value.
            decompose: Defaults to ``False``.  If decompose is ``True``, it will return the expectation value of each term.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The expectation value. If the backend is QuLeaf, it is computed by sampling.
        """
        return state.expec_val(self.hamiltonian, decompose=decompose)

class Measure(Operator):
    r"""Compute the probability of the specified measurement result.

    Args:
        measure_op: Specify the basis of the measurement. Defaults to ``'computational'``.
        
    Note:
        the allowable input for `measure_op` are:
        - None, i.e., a 'computational' basis
        - a string composed of 'i', 'x', 'y', and 'z'
        - a projection-valued measurement (PVM) in torch.Tensor
    
    """
    def __init__(self, measure_op: Optional[Union[str, List[str], torch.Tensor]] = None) -> None:
        super().__init__()

        self.measure_basis = None
        if measure_op is None:
            self.measure_op = measure_op
        
        elif isinstance(measure_op, (str, List)):
            self.measure_basis = pauli_str_basis(measure_op).to(self.dtype)
            self.measure_op = self.measure_basis.density_matrix
        
        elif isinstance(measure_op, torch.Tensor):
            if not all(check := utils.check._is_positive(measure_op).tolist()):
                warnings.warn(
                    f"Some elements of the input PVM is not positive: received {check}", UserWarning)

            sum_basis = torch.sum(measure_op, dim=-3)
            if (err := torch.norm(sum_basis - torch.eye(sum_basis.shape[-1]).expand_as(sum_basis))) > 1e-6:
                warnings.warn(
                    f"The input PVM is not complete: sum distance to identity is {err}", UserWarning)

            #TODO check whether the input is orthogonal
            self.measure_op = measure_op.to(self.dtype)

        else:
            raise ValueError(
                f"Unsupported type for measure_basis: receive type {type(measure_op)}")
    
    @_alias({"system_idx": "qubits_idx"})
    def __call__(self, state: State, system_idx: Optional[Union[Iterable[int], int, str]] = 'full',
                 desired_result: Optional[Union[Iterable[str], str]] = None, keep_state: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, State]]:
        return self.forward(state, system_idx, desired_result, keep_state)
    
    def __check_measure_op(self, state: State, system_idx: List[int]) -> bool:
        r"""Check whether the shape of input measurement operator is valid,
        and return whether measurement is performed across all qubits.
        """
        system_dim = [state.system_dim[idx] for idx in system_idx]
        measure_op = self.measure_op
        num_measure_systems = len(system_idx)
        
        if measure_op is None:
            self.measure_basis = std_basis(num_measure_systems, system_dim)
            return num_measure_systems == state.num_systems
        
        dim = math.prod(system_dim)
        expected_shape = [dim, dim]
        if measure_op.shape[-2:] != tuple(expected_shape):
            raise ValueError(
                f"The dimension of the PVM does not match the number of qubits: "
                f"received shape {measure_op.shape}, expected {expected_shape}"
            )
        if ((measure_batch_dim := list(measure_op.shape[:-3])) and 
            state._batch_dim and 
            (state._batch_dim != measure_batch_dim)):
            raise ValueError(
                f"The batch dimensions of input state do not match with measurement operator: "
                f"expected None or {measure_batch_dim}, received {state.batch_dim}"
            )
        
        return bool(
            self.measure_basis
            and dim == measure_op.shape[-3]
            and num_measure_systems == state.num_systems
        )

    @_alias({"system_idx": "qubits_idx"})
    def forward(
            self, state: State, system_idx: Optional[Union[Iterable[int], int, str]] = 'full',
            desired_result: Optional[Union[List[Union[int, str]], int, str]] = None, keep_state: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, State], Tuple[Dict, State]]:
        r"""Compute the probability of measurement to the input state.

        Args:
            state: The quantum state to be measured.
            system_idx: The index of the systems to be measured. Defaults to ``'full'`` which means measure all the qubits.
            desired_result: Specify the results of the measurement to return. Defaults to ``None`` which means return the probability of all the results.
            keep_state: Whether return the measured state. Defaults to ``False``.

        Returns:
            The probability of the measurement.
        
        """
        num_systems = state.num_systems
        if system_idx == 'full':
            system_idx = list(range(num_systems))
        elif isinstance(system_idx, int):
            system_idx = [system_idx]
        else:
            system_idx = list(system_idx)
        system_dim = [state.system_dim[idx] for idx in system_idx]
        measure_all_sys = self.__check_measure_op(state, system_idx)
        
        prob_array, measured_state = state.measure(self.measure_op, system_idx, keep_state=True)
        
        if measure_all_sys:
            measured_state = self.measure_basis.expand_as(measured_state)
        
        if desired_result:
            if isinstance(desired_result, int):
                desired_result = [desired_result]
            elif isinstance(desired_result, str):
                desired_result = [_digit_to_int(desired_result, system_dim)]
            else:
                desired_result = [(_digit_to_int(res, system_dim) if isinstance(res, str) else res)
                                  for res in desired_result]
            
            prob_array = torch.index_select(prob_array, dim=-1, index=torch.tensor(desired_result))
            measured_state = measured_state.prob_select(torch.tensor(desired_result))
        
        return (prob_array, measured_state) if keep_state else prob_array
