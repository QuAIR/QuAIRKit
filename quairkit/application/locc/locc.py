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
The source file of the LOCCNet class.
"""

import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import ModuleDict, ModuleList

from quairkit import Circuit
from quairkit.core import StateSimulator, get_dtype, intrinsic
from quairkit.core.intrinsic import _alias
from quairkit.database import bell_state, std_basis, zero_state
from quairkit.operator import OneWayLOCC, Oracle, ParamOracle
from quairkit.qinfo import nkron

_LogicalIndex = Tuple[str, int]
_PartyInfo = Dict[str, Union[int, Dict[str, int]]]


class OneWayLOCCNet(torch.nn.Module):
    r"""Network for general one-way LOCC (Local Operations and Classical Communication) protocol.
    Party that is measured will be traced out, and the party that performs the action will be the only one left.
    
    Args:
        party_info: a dictionary containing the information of the parties involved in the LOCC protocol.
        
    Note:
        Information of each party in `party_info` should be one of the following format:
        - {'Alice': 2} (Alice has 2 qubits)
        - {'Alice': {'num_systems': 2, 'system_dim': 3}} (Alice has 2 qutrits)
        - {'Alice': {'num_systems': 2, 'system_dim': [2, 3]}} (Alice has a qubit and a qutrit)
    
    """
    def __init__(self, party_info: _PartyInfo) -> None:
        super().__init__()
        self._cir_map = ModuleDict()
        self._state_map: Dict[Tuple[int, ...], StateSimulator] = {}
        self._list_locc: Iterable[OneWayLOCC] = ModuleList()

        current_idx = 0
        for name, info in party_info.items():
            if isinstance(info, int):
                num_systems = info
                system_dim = [2] * num_systems
            else:
                num_systems = info['num_systems']
                dim_info = info.get('system_dim', 2)
                system_dim = dim_info if isinstance(dim_info, List) else [dim_info] * num_systems
            
            idx_map = list(range(current_idx, current_idx + num_systems))
            self._cir_map[name] = Circuit(num_systems, system_dim, physical_idx=idx_map)
            current_idx += num_systems
        
    def __getitem__(self, key: str) -> Circuit:
        return self._cir_map[key]
    
    def __setitem__(self, key: str, cir: Circuit) -> None:
        r"""Set a party's circuit in the LOCC protocol.
        
        Args:
            key: The name of the party.
            cir: The circuit of the party.
        
        """
        if not isinstance(cir, Circuit):
            raise TypeError("Value must be an instance of Circuit")

        if key in self._cir_map:
            existing_cir = self._cir_map[key]
            if cir.num_systems != existing_cir.num_systems or cir.system_dim != existing_cir.system_dim:
                raise ValueError(
                    f"Circuit mismatch for '{key}': expected systems {existing_cir.num_systems}, dimensions {existing_cir.system_dim}; "
                    f"got systems {cir.num_systems}, dimensions {cir.system_dim}"
                )
        self._cir_map[key] = cir
        
    def __delitem__(self, key: str) -> None:
        raise NotImplementedError(
            "Deleting parties from the LOCC protocol is not supported.")
    
    def __len__(self) -> int:
        return len(self._cir_map)
    
    def __repr__(self):
        return repr(self._cir_map)
    
    def __str__(self):
        return str(self._cir_map)
    
    def keys(self) -> Iterable[str]:
        r"""Get the names of the parties in the LOCC protocol.
        """
        return self._cir_map.keys()
    
    def values(self) -> Iterable[Circuit]:
        r"""Get the circuits in the LOCC protocol.
        """
        return self._cir_map.values()
    
    def items(self) -> Iterable[Tuple[str, Circuit]]:
        return self._cir_map.items()
        
    def __check_party(self, *party_names: str) -> None:
        r"""Check if the party is registered in the LOCC protocol.
        """
        available_parties = set(self.keys())
        for name in party_names:
            if name not in available_parties:
                raise KeyError(f"Party '{name}' is not registered. Available parties: {available_parties}")
    
    @property
    def party_info(self) -> _PartyInfo:
        r"""Get the information of the parties in the LOCC protocol.
        """
        return {
            name: {'num_systems': circuit.num_systems, 'system_dim': circuit.system_dim}
            for name, circuit in self.items()
        }
    
    @property
    def physical_circuit(self) -> Circuit:
        r"""Get the complete physical circuit of the LOCC protocol.
        """
        system_dim = sum((cir.system_dim for cir in self.values()), [])
        combined_circuit = Circuit(system_dim=system_dim)
        for cir in self.values():
            combined_circuit += cir
            
        for op in self._list_locc:
            combined_circuit.append(op)
        return combined_circuit
    
    @_alias({"system_idx": "qubits_idx"})
    def set_init_state(self, system_idx: Union[Union[List[_LogicalIndex], _LogicalIndex]], state: Optional[StateSimulator] = None) -> None:
        r"""Set the initial (Bell) state of the LOCC protocol.
        
        Args:
            system_idx: a list of logical indices of where this state is an input
            state: the initial state of the LOCC protocol. Defaults to the (generalized) Bell state pair
        
        """
        if isinstance(system_idx, Tuple):
            system_idx = [system_idx]
        
        physical_indices, indexed_system_dim = [], []
        for name, idx in system_idx:
            self.__check_party(name)
            circuit = self._cir_map[name]
            physical_indices.append(circuit.system_idx[idx])
            indexed_system_dim.append(circuit.system_dim[idx])

        indices_tuple = tuple(physical_indices)
        if indices_tuple in self._state_map:
            raise ValueError(f"Initial state for indices {indices_tuple} already set.")

        if state is None:
            if len(physical_indices) != 2:
                raise ValueError("Bell state initialization requires exactly two systems.")
            state = bell_state(2, system_dim=indexed_system_dim)
        elif state.system_dim != indexed_system_dim:
            raise ValueError(f"State dimension mismatch: expected {indexed_system_dim}, got {state.system_dim}")

        state.reset_sequence()
        self._state_map[indices_tuple] = state
        
    def clean_init_state(self) -> None:
        r"""Clear all the initial states set in the LOCC protocol.
        """
        self._state_map.clear()
        
    def __prepare_init_state(self) -> StateSimulator:
        r"""Prepare the physical initial state for the LOCC protocol.
        """
        num_total_systems = sum(cir.num_systems for cir in self.values())
        total_system_dim = sum((cir.system_dim for cir in self.values()), [])

        settled_indices = [idx for indices in self._state_map for idx in indices]
        unsettled_indices = sorted(set(range(num_total_systems)) - set(settled_indices))

        states = [self._state_map[indices] for indices in self._state_map]
        if unsettled_indices:
            unsettled_dims = [total_system_dim[idx] for idx in unsettled_indices]
            states.append(zero_state(len(unsettled_indices), system_dim=unsettled_dims))

        combined_state = nkron(*states)
        combined_state._system_seq = settled_indices + unsettled_indices
        return combined_state
    
    def __check_locc_idx(self, system_idx: List[Union[List[_LogicalIndex]]]) -> List[List[int]]:
        r"""Check if the logical indices are valid and not measured before. Return formatted information.
        """
        if isinstance(system_idx[0], Tuple):
            system_idx[0] = [system_idx[0]]
        measure_idx, apply_idx = [], []
        measure_dim, apply_dim = [], []
        
        for name, idx in system_idx[0]:
            cir = self[name]
            measure_idx.append(cir.system_idx[idx])
            measure_dim.append(cir.system_dim[idx])
            
        for idx, op in enumerate(self._list_locc):
            settled_idx = op.measure.system_idx[0]
            assert set(measure_idx).isdisjoint(set(settled_idx)), \
                f"Measurement system {measure_idx} has been used in the {idx}-th LOCC protocols {settled_idx}"
            
        for name, idx in system_idx[1:]:
            cir = self[name]
            apply_idx.append(cir.system_idx[idx])
            apply_dim.append(cir.system_dim[idx])
            
        return measure_idx, measure_dim, apply_idx, apply_dim
    
    @_alias({"system_idx": "qubits_idx"})
    def locc(self, local_unitary: torch.Tensor, system_idx: List[Union[List[_LogicalIndex], _LogicalIndex]],
             label: str = 'M', latex_name: str = 'O') -> None:
        r"""Set a (non-parameterized) one-way LOCC protocol to the network.
        
        Args:
            local_unitary: The local unitary operation.
            system_idx: Systems on which the protocol is applied. The first element indicates the measure system.
            label: Label for measurement. Defaults to 'M'.
            latex_name: LaTeX name for the applied operator. Defaults to 'O'.
        
        """
        measure_idx, measure_dim, apply_idx, apply_dim = self.__check_locc_idx(system_idx)
        
        gate = Oracle(local_unitary, apply_idx, apply_dim)
        self._list_locc.append(OneWayLOCC(gate, measure_idx, measure_dim, label=label, latex_name=latex_name))
       
    @_alias({"system_idx": "qubits_idx"})
    def param_locc(self, generator: Callable[[torch.Tensor], torch.Tensor], num_acted_param: int, 
                   system_idx: List[Union[List[_LogicalIndex], _LogicalIndex]], param: Union[torch.Tensor, float] = None, 
                   label: str = 'M', latex_name: str = 'U', support_batch: bool = True) -> None:
        r"""Add a one-way LOCC protocol comprised of unitary operations, where the applied unitary is parameterized.

        Args:
            generator: Function to generate the oracle.
            num_acted_param: Number of parameters required for a single application.
            system_idx: Systems on which the protocol is applied. The first element indicates the measure system.
            param: Input parameters for the gate. Defaults to None.
            label: Label for measurement. Defaults to 'M'.
            latex_name: LaTeX name for the applied operator. Defaults to 'U'.
            support_batch: Whether generator supports batched input.
        
        Note:
            If the generator does not support batched input, you need to set `support_batch` to `False`.

        """
        measure_idx, measure_dim, apply_idx, apply_dim = self.__check_locc_idx(system_idx)
        
        if param is None:
            float_dtype = intrinsic._get_float_dtype(get_dtype())
            expect_shape = intrinsic._format_param_shape([apply_idx], num_acted_param, param_sharing=False, batch_size=np.prod(measure_dim))
            param = torch.nn.Parameter(torch.rand(expect_shape, dtype=float_dtype) * 2 * np.pi)
        param_gate = ParamOracle(
            generator, apply_idx, param, num_acted_param, apply_dim, support_batch=support_batch)
        self._list_locc.append(OneWayLOCC(param_gate, measure_idx, measure_dim, label=label, latex_name=latex_name))
    
    def __call__(self) -> StateSimulator:
        return self.forward()
        
    def forward(self) -> StateSimulator:
        r"""Run the one-way LOCC protocol and return the final state.
        
        Returns:
            The final state of the LOCC protocol.
        
        """
        state = self.__prepare_init_state()
        for cir in self.values():
            state = cir(state)
        
        trace_idx, trace_dim = [], []
        for op in self._list_locc:
            state = op(state)
            trace_idx.extend(op.measure.system_idx[0])
            trace_dim.append(np.prod(op.measure.system_dim))
        trace_state = std_basis(len(trace_idx), system_dim=[state.system_dim[idx] for idx in trace_idx])
        trace_state._batch_dim = trace_dim
        return state.product_trace(trace_state, trace_idx)
