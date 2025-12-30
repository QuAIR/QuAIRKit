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

import ast
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch


class OpenQASM2Transpiler:
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
        'reset': 'reset'
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
                f"ry({params[0] / 2:.5f}) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
                f"ry({-params[0] / 2:.5f}) q[{gate_idx[1]}];",
                f"cx q[{gate_idx[0]}], q[{gate_idx[1]}];",
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
            param = torch.remainder(param, 4 * torch.pi)
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
                param = torch.remainder(param, 4 * torch.pi)
                param = _tensor_to_float(param.squeeze(1))

            gates_qasm = []
            for index, gate_idx in enumerate(system_idx):
                if len(gate_idx) != num_qubits:
                    raise ValueError(f"{name} requires {num_qubits} qubits, got {len(gate_idx)}")

                if param is None:
                    gates_qasm.extend(qasm(gate_idx))
                else:
                    gates_qasm.extend(qasm(gate_idx, param[index]))
            return '\n'.join(gates_qasm)
        
        raise NotImplementedError(f"Operator {name} not supported in OpenQASM 2.0")
    
    @staticmethod
    def from_qasm2(source: str) -> List[Tuple[str, List[List[int]], Optional[torch.Tensor]]]:
        r"""Parse a complete OpenQASM 2.0 program into a list of (name, system_idx, param).

        Aggregation rule:
            Consecutive gates are combined into a single tuple only if
            - they have the same gate name, and
            - they are placeable in parallel, i.e., they act on disjoint sets of qubits.

        Special handling:
            - measure q -> c: returned as ('measure', [[list_of_all_q_qubits]], None) (not split).
            - reset q: returned as ('reset', [[list_of_all_q_qubits]], None) (not split).
            - Both measure and reset still follow the parallel-combine rule across consecutive statements.

        Args:
            source: Full OpenQASM 2.0 text (statements separated by '\n' and terminated by ';').

        Returns:
            A list of tuples (name, system_idx, param), where:
              - name: internal operator name (note: u1->p, cu1->cp as per __general_case)
              - system_idx: each inner list are the qubit indices for one gate instance
              - param: tensor of shape [len(system_idx), 1, num_params] or None for non-parametric gates

        Raises:
            ValueError: if the program header is missing or does not include "qelib1.inc", or malformed ops.
            NotImplementedError: if the program contains gate/opaque definitions or classically conditioned "if".
        """
        text = _strip_qasm_comments(source or "")

        # Basic header checks
        if not re.search(r'^\s*openqasm\s+2\.0\s*;', text, flags=re.IGNORECASE | re.MULTILINE):
            raise ValueError('Missing "OPENQASM 2.0;" header.')
        if not re.search(r'include\s+["\']qelib1\.inc["\']\s*;', text, flags=re.IGNORECASE):
            raise ValueError('Header must include "qelib1.inc".')

        # Explicitly reject features that are out of scope (measure/reset allowed; barrier ignored)
        if re.search(r'\b(gate|opaque)\b', text, flags=re.IGNORECASE):
            raise NotImplementedError("Custom gate definitions (gate/opaque) are not supported.")
        if re.search(r'\bif\s*\(', text, flags=re.IGNORECASE):
            raise NotImplementedError('Classically-conditioned operations ("if (...)") are not supported.')

        # Prepare name mappings
        inv_general = {v: k for k, v in OpenQASM2Transpiler.__general_case.items()}  # e.g., u1->p, cu1->cp
        # Known qubit arities (internal names)
        num_qubits_map = {
            'h': 1, 'x': 1, 'y': 1, 'z': 1, 's': 1, 'sdg': 1, 't': 1, 'tdg': 1,
            'rx': 1, 'ry': 1, 'rz': 1, 'p': 1, 'u3': 1,
            'cx': 2, 'cy': 2, 'cz': 2, 'ccx': 3, 'crz': 2, 'cp': 2,
            # measure/reset treated as variadic (skip fixed-arity check below)
            'measure': None, 'reset': None,
        }
        for k, v in OpenQASM2Transpiler.__special_case.items():
            num_qubits_map[k] = v['num_qubits']

        # Track quantum registers to flatten indices across multiple qregs (if present)
        qregs: Dict[str, Tuple[int, int]] = {}  # name -> (base_offset, size)
        next_base = 0

        # Tokenize by semicolons (statements)
        statements = [stmt.strip() for stmt in re.split(r';', text) if stmt.strip()]

        results: List[Tuple[str, List[List[int]], Optional[torch.Tensor]]] = []

        # Aggregation state for combining consecutive identical, parallelizable gates
        cur_name: Optional[str] = None
        cur_sys_idx: List[List[int]] = []
        cur_params_list: Optional[List[List[float]]] = None  # None for non-param gates; else NxM list
        cur_param_count: Optional[int] = None
        cur_used_qubits: set[int] = set()

        def flush():
            nonlocal cur_name, cur_sys_idx, cur_params_list, cur_param_count, cur_used_qubits, results
            if cur_name is None:
                return
            if cur_params_list is None or not cur_sys_idx:
                param_tensor = None
            else:
                param_tensor = torch.tensor(
                    cur_params_list, dtype=torch.get_default_dtype()
                ).unsqueeze(1)  # [N, 1, M]
            results.append((cur_name, cur_sys_idx, param_tensor))
            cur_name, cur_sys_idx, cur_params_list, cur_param_count = None, [], None, None
            cur_used_qubits = set()

        def can_place_in_parallel(idx_group: List[int]) -> bool:
            # No shared qubits with what's already in the current group
            return cur_name is not None and set(idx_group).isdisjoint(cur_used_qubits)

        def add_instance(name: str, idx_group: List[int], params: Optional[List[float]]):
            nonlocal cur_name, cur_sys_idx, cur_params_list, cur_param_count, cur_used_qubits
            this_param_count = 0 if (params is None) else len(params)

            if cur_name is None:
                # Start a new aggregation group
                cur_name = name
                cur_sys_idx = [idx_group]
                cur_used_qubits = set(idx_group)
                if this_param_count == 0:
                    cur_params_list = None
                    cur_param_count = 0
                else:
                    cur_params_list = [params]  # type: ignore[list-item]
                    cur_param_count = this_param_count
                return

            # Same gate name, same parameter count, and qubits are disjoint -> can parallelize
            if (name == cur_name) and (this_param_count == (cur_param_count or 0)) and can_place_in_parallel(idx_group):
                cur_sys_idx.append(idx_group)
                cur_used_qubits.update(idx_group)
                if this_param_count > 0 and cur_params_list is not None and params is not None:
                    cur_params_list.append(params)
                return

            # Otherwise, close current group and start a new one
            flush()
            cur_name = name
            cur_sys_idx = [idx_group]
            cur_used_qubits = set(idx_group)
            if this_param_count == 0:
                cur_params_list = None
                cur_param_count = 0
            else:
                cur_params_list = [params]  # type: ignore[list-item]
                cur_param_count = this_param_count

        def resolve_index(reg: str, local_idx: int) -> int:
            # Flatten (reg, local_idx) into a global qubit index using declaration order
            if reg in qregs:
                base, _ = qregs[reg]
                return base + local_idx
            # Fallback: if the reg is unknown (non-standard QASM), treat as global index
            return local_idx

        def operand_to_indices(operand: str) -> List[int]:
            # Accept either "<reg>[<i>]" or "<reg>" (full register)
            m_idx = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*', operand)
            if m_idx:
                reg = m_idx[1]
                loc = int(m_idx[2])
                return [resolve_index(reg, loc)]
            m_reg = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*', operand)
            if m_reg:
                reg = m_reg[1]
                if reg not in qregs:
                    raise ValueError(f"Unknown quantum register '{reg}' used without an index.")
                base, size = qregs[reg]
                return [base + i for i in range(size)]
            raise ValueError(f"Invalid quantum operand: '{operand}'")

        for stmt in statements:
            s = stmt.strip()
            if not s:
                continue

            # Skip/handle non-gate operational statements
            if re.match(r'^\s*openqasm\s+2\.0\s*$', s, flags=re.IGNORECASE):
                continue
            if re.match(r'^\s*include\s+["\'].*?["\']\s*$', s, flags=re.IGNORECASE):
                continue

            # qreg declarations (capture and keep declaration order for flattening)
            m_qreg = re.match(r'^\s*qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$', s, flags=re.IGNORECASE)
            if m_qreg:
                reg_name = m_qreg[1]
                size = int(m_qreg[2])
                if reg_name not in qregs:
                    qregs[reg_name] = (next_base, size)
                    next_base += size
                else:
                    base, old_size = qregs[reg_name]
                    if old_size != size:
                        raise ValueError(f"qreg '{reg_name}' redeclared with different size ({old_size} vs {size}).")
                continue

            # creg declarations are ignored
            if re.match(r'^\s*creg\s+[A-Za-z_]\w*\s*\[\s*\d+\s*\]\s*$', s, flags=re.IGNORECASE):
                continue

            # barrier is ignored and does not break aggregation
            if re.match(r'^\s*barrier\b', s, flags=re.IGNORECASE):
                continue

            # measure: record only quantum side; ignore classical target
            m_measure = re.match(r'^\s*measure\s+(.+?)\s*->\s*(.+?)\s*$', s, flags=re.IGNORECASE)
            if m_measure:
                q_operand = m_measure[1].strip()
                indices = operand_to_indices(q_operand)
                # Do NOT split; treat as single instance with one idx group containing all indices.
                add_instance('measure', indices, None)
                continue

            # reset: supports qubit or entire register; treat full register as single idx group
            m_reset = re.match(r'^\s*reset\s+(.+?)\s*$', s, flags=re.IGNORECASE)
            if m_reset:
                q_operand = m_reset[1].strip()
                indices = operand_to_indices(q_operand)
                add_instance('reset', indices, None)
                continue

            # Parse a gate statement: either name(params) args or name args
            m_param = re.match(r'^\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*(.+)$', s)
            m_nop   = None if m_param else re.match(r'^\s*([A-Za-z_]\w*)\s+(.+)$', s)
            if not m_param and not m_nop:
                raise ValueError(f"Cannot parse statement: {s}")

            raw_name = (m_param or m_nop).group(1).strip().lower()
            args_str = (m_param[3] if m_param else m_nop.group(2)).strip()

            # Extract explicit indexed qubits (q[0], r[1], ...)
            explicit_pairs = re.findall(r'([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]', args_str)
            if not explicit_pairs:
                # Allow register arguments (e.g., "h q", "cx q, r")
                operands = [op.strip() for op in _split_top_level_commas(args_str)]
                if not operands:
                    raise ValueError(f"Expected qubit argument(s) in: {s}")
                op_lists = [operand_to_indices(op) for op in operands]
                max_len = max(len(lst) for lst in op_lists)
                if any(len(lst) not in (1, max_len) for lst in op_lists):
                    raise ValueError(f"Incompatible register sizes in: {s}")
                # Broadcast singletons
                expanded = []
                for i in range(max_len):
                    idx_group = [lst[i] if len(lst) > 1 else lst[0] for lst in op_lists]
                    expanded.append(idx_group)
            else:
                # Only explicit indices present; keep single instance
                idx_group = [resolve_index(reg, int(qidx)) for (reg, qidx) in explicit_pairs]
                expanded = [idx_group]

            # Parameters
            if m_param:
                param_str = m_param[2].strip()
                param_tokens = _split_top_level_commas(param_str) if param_str else []
                params = [_eval_qasm_expr(tok) for tok in param_tokens]
            else:
                params = None

            # Map QASM gate name to internal name + param normalization
            name = raw_name
            if name in inv_general:
                # u1->p, cu1->cp, and identities like x->x
                name = inv_general[name]
            elif name == 'u2':
                # u2(phi, lambda) == u3(pi/2, phi, lambda)
                if params is None or len(params) != 2:
                    raise ValueError(f"u2 expects 2 parameters, got: {params}")
                name = 'u3'
                params = [math.pi / 2.0, params[0], params[1]]
            elif name in ('u',):  # Accept "u" as an alias for u3(theta,phi,lambda)
                if params is None or len(params) != 3:
                    raise ValueError(f"u expects 3 parameters, got: {params}")
                name = 'u3'
            elif name == 'id':
                # Map identity to a zero-angle u3
                name = 'u3'
                params = [0.0, 0.0, 0.0]
            elif name == 'cu3':
                # Map cu3(theta,phi,lambda) to cu4(theta,phi,lambda,0.0)
                if params is None or len(params) != 3:
                    raise ValueError(f"cu3 expects 3 parameters, got: {params}")
                name = 'cu4'
                params = [params[0], params[1], params[2], 0.0]

            # Basic arity check when known (measure/reset are variadic, skip check)
            expected = num_qubits_map.get(name)
            if expected is not None:
                for idx_group in expanded:
                    if len(idx_group) != expected:
                        raise ValueError(f"{name} requires {expected} qubit(s), got {len(idx_group)} in: {s}")

            # Add each expanded instance, applying parallel-aggregation rule
            for idx_group in expanded:
                add_instance(name, idx_group, params)

        # Flush any remaining group
        flush()
        return results


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


def _strip_qasm_comments(s: str) -> str:
    # Remove /* ... */ and // ... comments
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'//.*?$', '', s, flags=re.M)
    return s


def _split_top_level_commas(s: str) -> List[str]:
    # Split a parameter string by commas not enclosed in parentheses
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in s:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == ',' and depth == 0:
            if part := ''.join(buf).strip():
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    if tail := ''.join(buf).strip():
        parts.append(tail)
    return parts


def _eval_qasm_expr(expr: str) -> float:
    # Safe evaluator for QASM parameter expressions (supports +,-,*,/,**, unary +/- and common funcs, with pi)
    allowed_funcs = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'exp': math.exp, 'ln': math.log, 'log': math.log, 'sqrt': math.sqrt,
    }
    allowed_names = {'pi': math.pi, 'e': math.e}

    def _eval(node):
        if isinstance(node, ast.Num):  # Py<3.8
            return float(node.n)
        if isinstance(node, ast.Constant):  # Py>=3.8
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Invalid constant in expression: {expr}")
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return float(allowed_names[node.id])
            raise ValueError(f"Unknown name '{node.id}' in expression: {expr}")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"Unsupported operator in expression: {expr}")
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError(f"Unsupported unary operator in expression: {expr}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"Unsupported call in expression: {expr}")
            fname = node.func.id
            if fname not in allowed_funcs:
                raise ValueError(f"Function '{fname}' not allowed in expression: {expr}")
            args = [_eval(a) for a in node.args]
            if len(node.keywords) != 0:
                raise ValueError(f"Keyword args not allowed in expression: {expr}")
            return float(allowed_funcs[fname](*args))
        raise ValueError(f"Unsupported expression: {expr}")

    node = ast.parse(expr.strip().lower(), mode='eval')
    return float(_eval(node.body))
