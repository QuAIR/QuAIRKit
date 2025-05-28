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
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import ModuleDict

from quairkit import Circuit, State
from quairkit.database import bell_state, zero_state
from quairkit.operator import Collapse
from quairkit.qinfo import is_pvm, nkron

_LogicalIndex = Tuple[str, int]

class OneWayLOCCNet(torch.nn.Module):
    r"""Network for general one-way LOCC (Local Operations and Classical Communication) protocol.
    TO BE RELEASED.
    
    Args:
        party_info: a dictionary containing the information of the parties involved in the LOCC protocol.
        
    Note:
        Information of each party in `party_info` should be one of the following format:
        - {'Alice': 2} (Alice has 2 qubits)
        - {'Alice': {'num_systems': 2, 'system_dim': 3}} (Alice has 2 qutrits)
        - {'Alice': {'num_systems': 2, 'system_dim': [2, 3]}} (Alice has a qubit and a qutrit)
    
    """
    def __init__(self, party_info: Dict[str, Union[int, Dict[str, int]]]) -> None:
        super().__init__()
        
        self.__party_map: Dict[str, List[int]] = {}
        self.__state_map: Dict[List[int], State] = {}
        self.cir: ModuleDict = ModuleDict({})
        
        self.__register_party(party_info)
        
    def __register_party(self, party_info: Dict[str, Union[int, Dict[str, int]]]) -> None:
        r"""Check and process the input party information of LOCC. Generate corresponding quantum circuits.
        """
        party_map = {}, {}

        current_idx = 0
        for name, kwargs in party_info.items():
            idx_map = list(range(current_idx, current_idx + cir.num_qubits))
            cir = Circuit(kwargs) if isinstance(kwargs, int) else Circuit(**kwargs)

            party_map[name], self.cir[name] = idx_map, cir

        self.__party_map = party_map
        self.__system_dim = sum((cir.system_dim for cir in self.cir.values()), [])
        
    def __check_party(self, *party_name: str) -> None:
        r"""Check if the party is registered in the LOCC protocol.
        """
        print_str = "Available parties: " + ", ".join(self.__party_map.keys())

        for name in party_name:
            assert (
                name in self.__party_map
            ), f"{name} is not registered in the parties in this LOCC protocol\n{print_str}"
    
    def set_init_state(self, logical_indices: List[_LogicalIndex], 
                       state: Optional[State] = None) -> None:
        r"""Set the initial (Bell) state of the LOCC protocol.
        
        Args:
            logical_indices: a list of logical indices of where this state is an input
            state: the initial state of the LOCC protocol. Defaults to the (generalized) Bell state pair
        
        """
        self.__check_party(*logical_idx.keys())
        
        physical_indices, indexed_system_dim = [], []
        for name, logical_idx in logical_indices:
            physical_idx = self.__party_map[name][logical_idx]
            physical_indices.append(physical_idx)
            indexed_system_dim.append(self.__system_dim[physical_idx])

        settled_indices = {idx for indices in self.__state_map.keys() for idx in indices}
        assert set(physical_indices).isdisjoint(settled_indices), \
            "Some of input logical indices have been setup, check your codes"

        if state is None:
            assert len(physical_indices) == 2, \
                        f"Bell state can only be created for a pair of systems: received {len(physical_indices)} systems"
            state = bell_state(2, system_dim=indexed_system_dim)
        else:
            assert state.system_dim == indexed_system_dim, \
                        f"Input state dimension mismatch: expected {indexed_system_dim}, got {state.system_dim}"

        self.__state_map[physical_indices] = state
        
    def __prepare_init_state(self) -> None:
        r"""Prepare the physical initial state for the LOCC protocol.
        """
        assert self.__state_map, \
                "Initial states for all parties have not been set, please call `set_init_state` at least once"

        for indices, state in self.__state_map.items():
            self.__state_map[indices] = state.to_matrix()

        list_state, settled_indices = [], []
        for indices, state in self.__state_map.items():
            settled_indices.extend(indices)
            list_state.append(state)
        
        num_systems = len(self.__system_dim)
        if unsettled_indices := (set(range(num_systems)) - set(settled_indices)):
            # TODO wrong grammar for unsettled_indices
            warnings.warn(
                "Initial states of some systems in this protocol are not specified," + 
                "taken as zero states by default", UserWarning)
            unsettled_dim = [self.__system_dim[idx] for idx in unsettled_indices]
            list_state.append(zero_state(len(unsettled_dim), system_dim=unsettled_dim))
        unsettled_indices = sorted(list(unsettled_indices))

        init_state = nkron(*list_state)
        init_state._system_seq = settled_indices + unsettled_indices
        self.__physical_state: State = init_state
        
    def __prepare_circuit(self) -> None:
        r"""Prepare the physical initial state for the LOCC protocol.
        """
        measure_cir = Circuit(system_dim=self.__system_dim)
        for name, cir in self.cir.items():
            cir.set_system_seq(self.__party_map[name])
            cir.set_system_dim(self.__system_dim)
            cir.set_system_state(self.__physical_state)
        
        
    def set_locc(self, measure_indices: List[_LogicalIndex], act_party: str, 
                 measure_basis: Optional[torch.Tensor] = None) -> None:
        r"""Set a one-way LOCC protocol to the network.
        
        Args:
            measure_indices: a list of logical indices to be measured
            act_party: the party that performs the action w.r.t. the measurement result
            measure_idx: the index of the system that `measure_party` will measure, defaults to be all systems in the party.
            measure_op: the basis of the measurement. Defaults to the computational basis.
        
        Note:
            This function is the last step before protocol execution,
            as it will also prepare the physical initial state and quantum circuit for this protocol.
        
        """
        physical_measure_indices, measure_parties = [], []
        for name, idx in measure_indices:
            self.__check_party(name, act_party)
            physical_measure_indices.append(self.__party_map[name][idx])
            measure_parties.append(name)
        
        assert act_party not in measure_parties, \
            "The party that performs the action corresponding to the measurement should not be self-measured"
        
        physical_indices = self.__party_map[act_party]
        if measure_idx is None:
            measure_idx = physical_indices
        else:
            measure_idx = [physical_indices[idx] for idx in measure_idx]
        self.measure = Collapse(measure_idx, measure_basis=measure_basis)
        
        self.__prepare_circuit()
        
        
        
        self.__prepare_init_state()
        
    def __call__(self) -> State:
        return self.forward()
        
    def forward(self) -> State:
        r"""Run the one-way LOCC protocol and return the final state.
        
        Returns:
            The final state of the LOCC protocol.
        
        """
        return self.__physical_circuit(self.__physical_state)
