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
Source file for transpilation between OperatorInfo and [OpenQASM 2.0](https://github.com/openqasm/openqasm/tree/OpenQASM2.x)
"""

from typing import Dict, Iterable, List, Optional, Union

import torch


class OpenQASM2Transpiler(object):
    r"""A class to handle the translation between OperatorInfo and OpenQASM 2.0 format.
    
    This class provides methods to convert operator information into OpenQASM 2.0 format
    and to handle equivalent translations for specific operators.
    """
    
    __general_case: Dict[str, str] = {
        'h': 'h',
        'x': 'x',
        'y': 'y',
        'z': 'z',
        's': 's',
        'sdg': 'sdg',
        't': 't',
        'tdg': 'tdg',
        'rx': 'rx',
        'ry': 'ry',
        'rz': 'rz',
        'p': 'u1',
        'u3': 'u3',
        'cx': 'cx',
        'cy': 'cy',
        'cz': 'cz',
        'ccx': 'ccx',
        'crz': 'crz',
        'cp': 'cu1',
    }
    r"""A dictionary of equivalent translation between OperatorInfo and OpenQASM 2.0.
    """
    
    __special_case: Dict[str, Dict] = {
        'cswap': {
            'num_qubits': 3,
            'num_param': 0,
            'qasm': lambda gate_idx: [
                f"cx q[{gate_idx[2]}], q[{gate_idx[1]}];",
                f"ccx q[{gate_idx[0]}], q[{gate_idx[1]}], q[{gate_idx[2]}];",
                f"cx q[{gate_idx[2]}], q[{gate_idx[1]}];"
            ]
        },
        'rxx': {
            'num_qubits': 2,
            'num_param': 1,
            'qasm': lambda gate_idx, params: [
                f"h q[{gate_idx[0]}];",
                f"h q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"rz({params[0]:.5f}) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"h q[{gate_idx[0]}];",
                f"h q[{gate_idx[1]}];"
            ]
        },
        'ryy': {
            'num_qubits': 2,
            'num_param': 1,
            'qasm': lambda gate_idx, params: [
                f"rx(pi/2) q[{gate_idx[0]}];",
                f"rx(pi/2) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"rz({params[0]:.5f}) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"rx(-pi/2) q[{gate_idx[0]}];",
                f"rx(-pi/2) q[{gate_idx[1]}];"
            ]
        },
        'rzz': {
            'num_qubits': 2,
            'num_param': 1,
            'qasm': lambda gate_idx, params: [
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"rz({params[0]:.5f}) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];"
            ]
        },
        'ms': {
            'num_qubits': 2,
            'num_param': 0,
            'qasm': lambda gate_idx: [
                f"h q[{gate_idx[0]}];",
                f"h q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"rz(-pi/2) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"h q[{gate_idx[0]}];",
                f"h q[{gate_idx[1]}];"
            ]
        },
        'cu4': {
            'num_qubits': 2,
            'num_param': 4,
            'qasm': lambda gate_idx, params: [
                f"rz({params[3] * 2:.5f}) q[{gate_idx[0]}];",
                f"cu3({params[0]:.5f}, {params[1]:.5f}, {params[2]:.5f}) q[{gate_idx[0]}], q[{gate_idx[1]}];"
            ]
        },
        'swap': {
            'num_qubits': 2,
            'num_param': 0,
            'qasm': lambda gate_idx: [
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"cx q[{gate_idx[1]}], q[{gate_idx[0]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];"
            ]
        },
        'crx': {
            'num_qubits': 2,
            'num_param': 1,
            'qasm': lambda gate_idx, params: [
                f"h q[{gate_idx[1]}];",
                f"crz({params[0]:.5f}) q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"h q[{gate_idx[1]}];"
            ]
        },
        'cry': {
            'num_qubits': 2,
            'num_param': 1,
            'qasm': lambda gate_idx, params: [
                f"ry({params[0] / 2:.5f}) q[{gate_idx[1]}]",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"ry({-params[0] / 2:.5f}) q[{gate_idx[1]}]",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"ry({params[0] / 2:.5f}) q[{gate_idx[1]}]",
            ]
        },
    }
    r"""A dictionary of special cases for translation between OperatorInfo and OpenQASM 2.0.
    """
    
    @staticmethod
    def format_qasm(name: str, system_idx: List[List[int]], param: Optional[torch.Tensor]) -> str:
        r"""Translate OperatorInfo to OpenQASM-like format.

        Args:
            name: The name of the operator.
            system_idx: The indices of the systems the operator acts upon.
            param: The parameters of the operator, if any.

        Returns:
            The OpenQASM-like representation of the operator.
        
        """
        if param is not None:
            param = torch.remainder(param, 2 * torch.pi)
            param = _tensor_to_float(param.squeeze(1))
        
        gate_qasm = []
        for i, idx_group in enumerate(system_idx):
            param_str = ''
            if param is not None:
                param_str = '(' + ", ".join(["{:.5f}".format(p) for p in param[i]]) + ')'
            
            indices = ", ".join([f"q[{idx}]" for idx in idx_group])
            gate_qasm.append(f"{name}{param_str} {indices};")

        return '\n'.join(gate_qasm)
    
    @staticmethod
    def transpile(name: str, system_idx: List[List[int]], param: Optional[torch.Tensor]) -> str:
        r"""Translate OperatorInfo to OpenQASM 2.0 format.

        Args:
            name: The name of the operator.
            system_idx: The indices of the systems the operator acts upon.
            param: The parameters of the operator, if any.

        Returns:
            The OpenQASM 2.0 representation of the operator.
        """
        if name in OpenQASM2Transpiler.__general_case:
            return OpenQASM2Transpiler.format_qasm(OpenQASM2Transpiler.__general_case[name], system_idx, param)
        
        if translation := OpenQASM2Transpiler.__special_case.get(name):
            num_qubits, num_param, qasm = translation['num_qubits'], translation['num_param'], translation['qasm']
            
            if param is not None:
                if param.shape[1] != 1:
                    raise ValueError(f"Expected param to be not batched, got shape {param.shape}")
                assert param.shape[-1] == num_param, \
                    f"Expected {num_param} parameters, got {param.shape[-1]} for {name}."
                param = torch.remainder(param, 2 * torch.pi)
                param = _tensor_to_float(param.squeeze(1))

            gates_qasm = []
            for index, gate_idx in enumerate(system_idx):
                if len(gate_idx) != num_qubits:
                    raise ValueError(f"{name} requires {num_qubits} qubits, got {len(gate_idx)}")

                if param is None:
                    gates_qasm.extend(qasm(gate_idx))
                else:
                    gates_qasm.extend(qasm(gate_idx, param[index]))
            return "\n".join(gates_qasm)
        
        raise NotImplementedError(f"Operator {name} not supported in OpenQASM 2.0")


def _tensor_to_float(arr: torch.Tensor) -> Union[int, Iterable[float]]:
    r"""Convert a tensor to a nested list of float.
    """
    flat_list = list(arr.detach().cpu().numpy().reshape(-1))
    shape = arr.shape
    current = flat_list

    # Group elements into nested lists based on shape (from last dimension to first)
    for dim in reversed(shape):
        current = [current[i:i + dim] for i in range(0, len(current), dim)]

    # Extract the rebuilt nested list structure
    return current[0]
