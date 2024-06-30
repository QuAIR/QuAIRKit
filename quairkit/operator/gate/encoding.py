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
The source file of the classes for quantum encoding.
"""

from typing import Iterable, Optional, Union

import torch

from ...core import Operator, State, get_float_dtype, to_state
from ...core.intrinsic import _format_qubits_idx
from ...database.matrix import cnot, h, rx, ry, rz, x
from ...database.state import zero_state


class BasisEncoding(Operator):
    r"""Basis encoding gate for encoding input classical data into quantum states.

    In basis encoding, the input classical data can only consist of 0's and 1's. If the input data are 1101,
    then the quantum state after encoding is :math:`|1101\rangle`. Note that the quantum state before encoding is
    assumed to be :math:`|00\ldots 0\rangle`.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    
    __x = x(torch.complex128)
    
    def __init__(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_info = {
            'gatename': 'BasisEnc',
            'texname': r'$\text{BasisEnc}$',
            'plot_width': 1.5,
        }

    def forward(self, feature: torch.Tensor, state: State = None) -> State:
        x = BasisEncoding.__x.to(self.dtype)
        if state is None:
            state = zero_state(self.num_qubits)
        feature = feature.to('int32')
        gate_history = []
        for idx, element in enumerate(feature):
            if element:
                state = simulation(state, [x], self.qubits_idx[idx])
                gate_history.append({'gate': 'x', 'which_qubits': self.qubits_idx[idx], 'theta': None})
        self.gate_history = gate_history
        return state
    
    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")        


class AmplitudeEncoding(Operator):
    r"""Amplitude encoding gate for encoding input classical data into quantum states.

    Args:
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    """
    def __init__(
            self, qubits_idx: Optional[Union[Iterable[int], int, str]] = 'full', num_qubits: Optional[int] = None
    ) -> None:
        if num_qubits is None:
            num_qubits = max(qubits_idx)
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        self.gate_info = {
            'gatename': 'AmpEnc',
            'texname': r'$\text{AmpEnc}$',
            'plot_width': 1.5,
        }

    def forward(self, feature: torch.Tensor) -> State:
        def calc_location(location_of_bits_list):
            if len(location_of_bits_list) <= 1:
                result_list = [0, location_of_bits_list[0]]
            else:
                current_tmp = location_of_bits_list[0]
                inner_location_of_qubits_list = calc_location(location_of_bits_list[1:])
                current_list_len = len(inner_location_of_qubits_list)
                for each in range(current_list_len):
                    inner_location_of_qubits_list.append(inner_location_of_qubits_list[each] + current_tmp)
                result_list = inner_location_of_qubits_list
            return result_list

        def encoding_location_list(which_qubits):
            location_of_bits_list = []
            for qubit_idx in which_qubits:
                tmp = 2 ** (self.num_qubits - qubit_idx - 1)
                location_of_bits_list.append(tmp)
            result_list = calc_location(location_of_bits_list)

            return sorted(result_list)

        # Get the specific position of the code, denoted by sequence number (list)
        location_of_qubits_list = encoding_location_list(self.qubits_idx)
        # Classical data preprocessing
        feature = feature.to(get_float_dtype())
        feature = torch.flatten(feature)
        length = torch.norm(feature, p=2)
        # Normalization
        feature = torch.divide(feature, length)
        # Create a quantum state with all zero amplitudes
        data = torch.zeros((2 ** self.num_qubits,), feature.dtype)
        # The value of the encoded amplitude is filled into the specified qubits
        for idx in range(len(feature)):
            data[location_of_qubits_list[idx]] = feature[idx]
        encoding_state = to_state(data)
        return encoding_state
    
    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")


class AngleEncoding(Operator):
    r"""Angle encoding gate for encoding input classical data into quantum states.

    Args:
        feature: Vector to be encoded.
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        encoding_gate: The type of quantum gates used for encoding, which should be one of ``"rx"``, ``"ry"``,
            and ``"rz"``. Defaults to ``None``.
    """
    def __init__(
            self, feature: torch.Tensor, qubits_idx: Optional[Union[Iterable[int], int, str]] = 'full', 
            num_qubits: Optional[int] = None, encoding_gate: Optional[str] = None,
    ) -> None:
        if num_qubits is None:
            num_qubits = max(qubits_idx)
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits)
        
        if encoding_gate == 'rx':
            self.encoding_gate = rx
        elif encoding_gate == 'ry':
            self.encoding_gate = ry
        elif encoding_gate == 'rz':
            self.encoding_gate = rz
        self.encoding_gate_name = encoding_gate
        
        feature = feature.to(get_float_dtype())
        feature = torch.flatten(feature)
        self.feature = feature
        
        self.gate_info = {
            'gatename': 'AngleEnc',
            'texname': r'$\text{AngleEnc}$',
            'plot_width': 1.5,
        }

    def forward(
            self, state: State = None, invert: bool = False
    ) -> State:
        gate_history = []
        if state is None:
            state = zero_state(self.num_qubits)
        feature = -1 * self.feature if invert else self.feature
        
        for idx, element in enumerate(feature):
            param_matrix = self.encoding_gate(element[0])
            
            state._evolve(param_matrix, [self.qubits_idx[idx]])
            gate_history.append({'gate': self.encoding_gate_name, 'which_qubits': self.qubits_idx[idx], 'theta': element[0]})
        self.gate_history = gate_history
        return state
    
    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")


class IQPEncoding(Operator):
    r"""IQP style encoding gate for encoding input classical data into quantum states.

    Args:
        feature: Vector to be encoded.
        qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        num_repeat: Number of encoding layers. Defaults to ``1``.
    """
    
    __cnot = cnot(torch.complex128)
    __h = h(torch.complex128)
    
    def __init__(
            self, feature: torch.Tensor, qubits_idx: Optional[Iterable[Iterable[int]]] = None,
            num_qubits: Optional[int] = None, num_repeat: Optional[int] = 1,
    ) -> None:
        super().__init__()

        self.num_repeat = num_repeat
        self.num_qubits = num_qubits
        if feature is not None:
            feature = feature.to(get_float_dtype())
            feature = torch.flatten(feature)
            self.feature = feature

        if qubits_idx is None:
            assert num_qubits is not None, \
                    f"Number of qubits must be known to create default patterns: received {num_qubits}"
            qubits_idx = []
            for idx0 in range(num_qubits):
                qubits_idx.extend([idx0, idx1] for idx1 in range(idx0 + 1, num_qubits))
        
        self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits=2)

        self.gate_info = {
            'gatename': 'IQPEnc',
            'texname': r'$\text{IQPEnc}$',
            'plot_width': 1.5,
        }

    def forward(
            self, state: State = None, invert: Optional[bool] = False
    ) -> State:
        gate_history = []
        h, cnot = IQPEncoding.__h.to(self.dtype), IQPEncoding.__cnot.to(self.dtype)
        if state is None:
            state = zero_state(self.num_qubits)
        for _ in range(self.num_repeat):
            if invert:
                for qubits_idx in self.qubits_idx:
                    rz_param = -self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]]
                    state._evolve(cnot, qubits_idx)
                    state._evolve(rz(rz_param), [qubits_idx[1]])
                    state._evolve(cnot, qubits_idx)

                    gate_history.extend([
                        {'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None},
                        {'gate': 'rz', 'which_qubits': qubits_idx[1], 'theta': rz_param},
                        {'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None}])
                    
                for idx in range(self.feature.size):
                    state._evolve(rz(-self.feature[idx]), [idx])
                    gate_history.append({'gate': 'rz', 'which_qubits': idx, 'theta': -self.feature[idx]})

                for idx in range(self.feature.size):
                    state._evolve(h, [idx])
                    gate_history.append({'gate': 'h', 'which_qubits': idx, 'theta': None})
            else:
                for idx in range(self.feature.size):
                    state._evolve(h, [idx])
                    gate_history.append({'gate': 'h', 'which_qubits': idx, 'theta': None})

                for idx in range(self.feature.size):
                    state._evolve(rz(self.feature[idx]), [idx])
                    gate_history.append({'gate': 'rz', 'which_qubits': idx, 'theta': self.feature[idx]})

                for qubits_idx in self.qubits_idx:
                    rz_param = self.feature[qubits_idx[0]] * self.feature[qubits_idx[1]]
                    
                    state._evolve(cnot, qubits_idx)
                    state._evolve(rz(rz_param), [qubits_idx[1]])
                    state._evolve(cnot, qubits_idx)
                    
                    gate_history.extend([
                        {'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None},
                        {'gate': 'rz', 'which_qubits': qubits_idx[1], 'theta': rz_param},
                        {'gate': 'cnot', 'which_qubits': qubits_idx, 'theta': None}])
        
        self.gate_history = gate_history
        return state

    def gate_history_generation(self) -> None:
        if self.gate_history is None:
            raise RuntimeError("you must forward the encoding to receive the gate history")
