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
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..core import Hamiltonian, Operator, StateOperator, StateSimulator, utils
from ..core.intrinsic import _alias, _digit_to_int, _State
from ..database import pauli_str_basis, std_basis

__all__ = ["Measure", "ExpecVal"]


def _select_measure_results(prob_array: torch.Tensor, desired_result: torch.Tensor) -> torch.Tensor:
    r"""Select desired measurement outcomes from dense or sparse probabilities."""
    if prob_array.layout == torch.strided:
        return torch.index_select(prob_array, dim=-1, index=desired_result)

    sparse_prob = prob_array.coalesce() if prob_array.layout == torch.sparse_coo else prob_array.to_sparse_coo().coalesce()
    desired_cpu = desired_result.to(device="cpu", dtype=torch.int64)
    desired_map = {}
    for pos, idx in enumerate(desired_cpu.tolist()):
        desired_map.setdefault(int(idx), []).append(pos)

    if sparse_prob.ndim == 1:
        result = torch.zeros((desired_result.numel(),), dtype=sparse_prob.dtype, device=sparse_prob.device)
        indices = sparse_prob.indices()[0]
        values = sparse_prob.values()
        for p in range(values.numel()):
            mapped_positions = desired_map.get(int(indices[p].item()))
            if mapped_positions is None:
                continue
            for pos in mapped_positions:
                result[pos] = values[p]
        return result

    if sparse_prob.ndim != 2:
        raise ValueError(
            f"Expected a 1D/2D measurement tensor, received shape {list(prob_array.shape)}."
        )

    result = torch.zeros(
        (sparse_prob.shape[0], desired_result.numel()),
        dtype=sparse_prob.dtype,
        device=sparse_prob.device,
    )
    indices = sparse_prob.indices()
    values = sparse_prob.values()
    for p in range(values.numel()):
        mapped_positions = desired_map.get(int(indices[1, p].item()))
        if mapped_positions is None:
            continue
        row = int(indices[0, p].item())
        for pos in mapped_positions:
            result[row, pos] = values[p]
    return result


class ExpecVal(Operator):
    r"""The class of the loss function to compute the expectation value for an observable.

    This interface allows you to use the expectation value of an observable as a loss function for training quantum circuits.
    The expectation value is defined as :math:`\langle H \rangle = \text{tr}(\rho H)`.

    Args:
        hamiltonian: The input observable (Hamiltonian) to compute the expectation value for.

    Note:
        This class supports batch operations. When input states have batch dimensions,
        the expectation value is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import random_hamiltonian_generator, random_state
            from quairkit.loss import ExpecVal

            # Create observable and compute expectation value
            observable = random_hamiltonian_generator(1)
            print(f'The input observable is:\n{observable}')
            expec_val = ExpecVal(observable)

            input_state = random_state(num_systems=1, rank=2)
            result = expec_val(input_state, decompose=True)
            print('The expectation value of the observable:', result)

        ::

            The input observable is:
            -0.28233465254251144 Z0
            0.12440505341821817 X0
            -0.2854054036807161 Y0
            The expectation value of the observable: tensor([-0.1162,  0.0768, -0.0081])

        .. code-block:: python

            # Batch expectation value example
            input_state_batch = random_state(num_systems=1, size=2)
            batch_result = expec_val(input_state_batch, decompose=False)
            print('The expectation value of the observable:', batch_result)

        ::

            The expectation value of the observable: tensor([0.1748, 0.0198])
    """
    
    def __init__(self, hamiltonian: Hamiltonian):
        super().__init__()
        self.hamiltonian = hamiltonian
    
    def __call__(self, state: _State, shots: Optional[int] = None, decompose: Optional[bool] = False) -> torch.Tensor:
        return self.forward(state, decompose)

    def forward(self, state: _State, shots: Optional[int] = None, decompose: Optional[bool] = False) -> torch.Tensor:
        r"""Compute the expectation value of the observable with respect to the input state.

        The value computed by this function can be used as a loss function to optimize quantum circuits.

        Args:
            state: The input quantum state which will be used to compute the expectation value.
                Can be a single state or a batch of states.
            shots: The number of shots to measure the observable. Defaults to None, which means exact computation.
            decompose: If True, returns the expectation value of each term in the observable separately.
                Defaults to False, which returns the total expectation value.

        Returns:
            The expectation value of the observable. If ``decompose=True``, returns a tensor with one value per term.
            If ``decompose=False``, returns a scalar for single states or a tensor for batched states.

        Raises:
            NotImplementedError: If the backend is wrong or not supported.
        """
        return state.expec_val(self.hamiltonian, shots=shots, decompose=decompose)


class Measure(Operator):
    r"""Compute the probability of the specified measurement result.

    Args:
        measure_op: Specify the basis of the measurement. Defaults to ``'computational'``.
        
    Note:
        the allowable input for `measure_op` are:
        - None, i.e., a 'computational' basis
        - a string composed of 'i', 'x', 'y', and 'z'
        - a projection-valued measurement (PVM) in torch.Tensor
    
    .. code-block:: python
    
        from quairkit.database import random_state, std_basis

        # Define measurement basis using a string (e.g., 'xy' denotes eigen-basis of X ⊗ Y)
        op = Measure("xy")
        # Define a custom measurement basis using a user-specified PVM tensor
        pvm_tensor = std_basis(2).density_matrix
        op = Measure(pvm_tensor)
        # Use default computational basis
        op = Measure()

        state = random_state(num_qubits=2)

        # Full measurement: probability distribution over all measurement outcomes.
        result = op(state)
        print("Probabilities for measuring all qubits:", result)

        # Measurement on a targeted subsystem (e.g., the first qubit)
        result = op(state, system_idx=0)
        print("Probabilities for measuring the first qubit:", result)

        # Compute probability for a specific full-system outcome ('10')
        result = op(state, desired_result='10')
        print("Probability for measuring all qubits with outcome 10:", result)

        # Compute probabilities for selected outcomes (e.g., outcomes corresponding to indices 0 and 3)
        result = op(state, desired_result=[0, 3])
        print("Probabilities for measuring all qubits with outcomes 00 and 11:", result)

        # Retrieve both the probability and the post-measurement state.
        prob, post_state = op(state, keep_state=True)
        print("Post-measurement state:", post_state)

        # Batched measurement on multiple states.
        state_batch = random_state(num_systems=2, size=2)
        result = op(state_batch)
        print(f"Probabilities for measuring two states:\n{result}")
                
    ::
    
        Probabilities for measuring all qubits: tensor([0.1273, 0.0956, 0.3312, 0.4459])
        Probabilities for measuring the first qubit: tensor([0.2229, 0.7771])
        Probability for measuring all qubits with outcome 10: tensor([0.3312])
        Probabilities for measuring all qubits with outcomes 00 and 11: tensor([0.1273, 0.4459])
        Post-measurement state:
        -----------------------------------------------------
        Backend: state_vector
        System dimension: [2, 2]
        System sequence: [0, 1]
        Batch size: [4]

        # 0:
        [1.+0.j 0.+0.j 0.+0.j 0.+0.j]
        # 1:
        [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
        # 2:
        [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
        # 3:
        [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
        -----------------------------------------------------
        
        Probabilities for measuring two states:
        tensor([[0.2846, 0.1826, 0.2665, 0.2663],
                [0.1513, 0.4921, 0.1676, 0.1891]])
    """
    
    def __init__(self, measure_op: Optional[Union[str, torch.Tensor]] = None) -> None:
        r"""
        
        Note:
            the allowable input for `measure_op` are:
            - None, i.e., a computational basis
            - a string composed of 'i', 'x', 'y', and 'z'
            - a projection-valued measurement (PVM) in torch.Tensor
        """
        super().__init__()

        if measure_op is None or isinstance(measure_op, str):
            self.measure_op: Union[torch.Tensor, StateSimulator] = measure_op
        
        elif isinstance(measure_op, torch.Tensor):
            assert torch.all(utils.check._is_pvm(measure_op)).item(), \
                "The input measurement operators do not form a projection-valued measurement (PVM)."
            self.measure_op = measure_op.to(self.dtype)

        else:
            raise ValueError(
                f"Unsupported type for measure_op: receive type {type(measure_op)}")
    
    @_alias({"system_idx": "qubits_idx"})
    def __call__(self, state: _State, system_idx: Optional[Union[Iterable[int], int, str]] = 'full', shots: Optional[int] = None,
                 desired_result: Optional[Union[Iterable[str], str]] = None, keep_state: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, _State]]:
        return self.forward(state, system_idx, shots, desired_result, keep_state)
    
    def __check_measure_op(self, state: _State, system_idx: List[int]) -> None:
        r"""Check whether the shape of input measurement operator is valid,
        and return whether measurement is performed across all qubits.
        
        Note:
            used for simulator backend only.
        """
        system_dim = [state.system_dim[idx] for idx in system_idx]
        num_measure_systems = len(system_idx)

        if self.measure_op is None:
            self.measure_op = std_basis(num_measure_systems, system_dim)
            return num_measure_systems == state.num_systems
        elif isinstance(self.measure_op, str):
            if len(self.measure_op) == 1:
                self.measure_op *= num_measure_systems
            self.measure_op = pauli_str_basis(self.measure_op)
            return num_measure_systems == state.num_systems

        dim = math.prod(system_dim)
        expected_shape = [dim, dim]
        if self.measure_op.shape[-2:] != tuple(expected_shape):
            raise ValueError(
                f"The dimension of the PVM does not match the number of qubits: "
                f"received shape {self.measure_op.shape}, expected {expected_shape}"
            )
        
    def __measure_simulator_backend(
        self, state: StateSimulator, system_idx: List[int],
        desired_result: Optional[torch.Tensor], keep_state: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, StateSimulator]]:
        self.__check_measure_op(state, system_idx)
        measure_op = self.measure_op

        if desired_result is not None:
            if isinstance(measure_op, StateSimulator):
                measure_op = measure_op[desired_result]
            else:
                measure_op = torch.index_select(measure_op, dim=-3, index=desired_result)

        if isinstance(measure_op, StateSimulator):
            if keep_state:
                prob_array, measured_state = state._measure_by_state(
                    measure_op, system_idx, keep_state=True
                )
                return prob_array, measured_state
            return state._measure_by_state(measure_op, system_idx, keep_state=False)
        else:
            if keep_state:
                return state._measure(measure_op, system_idx)
            return state._expec_val(measure_op, system_idx).real
    
    def __measure_operator_backend(
        self, state: StateOperator, system_idx: List[int], shots: Optional[int],
        desired_result: Optional[torch.Tensor], keep_state: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, StateOperator]]:
        assert not keep_state, \
            "The post-measurement state is not supported for state operators."
        
        prob_array = state.measure(system_idx, shots=shots, measure_op=self.measure_op)
        if desired_result is not None:
            prob_array = _select_measure_results(prob_array, desired_result)
        return prob_array

    @_alias({"system_idx": "qubits_idx"})
    def forward(
            self, state: _State, system_idx: Optional[Union[Iterable[int], int, str]] = 'full', shots: Optional[int] = None,
            desired_result: Optional[Union[List[Union[int, str]], int, str]] = None, keep_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, _State], Tuple[Dict, _State]]:
        r"""Compute the probability of measurement to the input state.

        Args:
            state: The quantum state to be measured.
            system_idx: The index of the systems to be measured. Defaults to ``'full'`` which means measure all the qubits.
            shots: The number of shots for the measurement. Defaults to None which means the default behavior.
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
        if desired_result is not None:
            if isinstance(desired_result, int):
                desired_result = [desired_result]
            elif isinstance(desired_result, str):
                desired_result = [_digit_to_int(desired_result, system_dim)]
            else:
                desired_result = [(_digit_to_int(res, system_dim) if isinstance(res, str) else res)
                                for res in desired_result]
            desired_result = torch.tensor(desired_result, dtype=torch.long)

        if isinstance(state, StateSimulator):
            return self.__measure_simulator_backend(
                state, system_idx, desired_result, keep_state
            )
        
        return self.__measure_operator_backend(
            state, system_idx, shots, desired_result, keep_state)
