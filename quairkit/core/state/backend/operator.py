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
Source file for base class that emulates quantum states.
"""


import copy
import itertools
import math
import re
import warnings
from abc import abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from ... import Hamiltonian, OperatorInfoType, get_float_dtype
from .base import State

try:
    import quairkit._C as _QKIT_C
    _COUNTS_CPP = getattr(_QKIT_C, "counts", None)
except Exception:
    _COUNTS_CPP = None


def _pauli_parity_mask(pauli_str: str) -> int:
    r"""Return a bitmask for parity-based Pauli-Z expectation after basis change.

    The returned mask is compatible with the current Python implementation:
    - measurement outcome indices follow `int(digits, 2)` semantics
    - the first measured classical bit `c[0]` maps to the most-significant bit

    For each measured bit position where the Pauli char is not 'i', the eigenvalue
    flips sign when that bit is 1. This corresponds to:
        sign = (-1)^(popcount(index & mask))
    """
    s = pauli_str.lower()
    mask = 0
    n = len(s)
    for pos, ch in enumerate(s):
        if ch != "i":
            mask |= 1 << (n - 1 - pos)
    return mask


def _normalize_batched_csr_counts(
    counts_result,
    n_circuits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Normalize backend outputs into a batched CSR counts representation on CPU.

    Supported inputs:
    - (offsets, indices, counts) tensors (batched CSR)
    - dense torch.Tensor counts: shape [n_circuits, num_outcomes] or [num_outcomes] (n_circuits=1)
    - list/iterator of:
        - Dict[str,int] (bitstring->count)
        - List[Tuple[int,int]] (index,count) pairs (single circuit or per-circuit list)
        - (indices_tensor, counts_tensor) per circuit
    """
    if (
        isinstance(counts_result, tuple)
        and len(counts_result) == 3
        and all(torch.is_tensor(x) for x in counts_result)
    ):
        offsets, indices, counts = counts_result
        return (
            offsets.to(device="cpu", dtype=torch.int64, copy=False).contiguous(),
            indices.to(device="cpu", dtype=torch.int64, copy=False).contiguous(),
            counts.to(device="cpu", dtype=torch.int64, copy=False).contiguous(),
        )

    if torch.is_tensor(counts_result):
        t = counts_result.to(device="cpu")
        if t.dim() == 1:
            assert n_circuits == 1, "1-D dense counts only supported for a single circuit."
            nz = (t != 0).nonzero(as_tuple=False).flatten()
            idx = nz.to(dtype=torch.int64)
            cnt = t[nz].to(dtype=torch.int64)
            offsets = torch.tensor([0, int(idx.numel())], dtype=torch.int64, device="cpu")
            return offsets, idx, cnt
        if t.dim() == 2:
            assert t.size(0) == n_circuits, "dense counts batch size mismatch."
            offsets_list = [0]
            indices_list = []
            counts_list = []
            for i in range(n_circuits):
                row = t[i]
                nz = (row != 0).nonzero(as_tuple=False).flatten()
                indices_list.append(nz.to(dtype=torch.int64))
                counts_list.append(row[nz].to(dtype=torch.int64))
                offsets_list.append(offsets_list[-1] + int(nz.numel()))
            offsets = torch.tensor(offsets_list, dtype=torch.int64, device="cpu")
            indices = (
                torch.cat(indices_list, dim=0)
                if indices_list
                else torch.empty(0, dtype=torch.int64, device="cpu")
            )
            counts = (
                torch.cat(counts_list, dim=0)
                if counts_list
                else torch.empty(0, dtype=torch.int64, device="cpu")
            )
            return offsets, indices, counts
        raise TypeError(f"Unsupported dense counts tensor rank {t.dim()}.")

    if not isinstance(counts_result, list):
        counts_result = list(counts_result)

    if n_circuits == 1 and counts_result and isinstance(counts_result[0], tuple) and len(counts_result[0]) == 2:
        counts_result = [counts_result]

    assert len(counts_result) == n_circuits, "counts_result length mismatch with n_circuits."

    offsets_list = [0]
    indices_chunks: List[torch.Tensor] = []
    counts_chunks: List[torch.Tensor] = []

    for elem in counts_result:
        if (
            isinstance(elem, tuple)
            and len(elem) == 2
            and all(torch.is_tensor(x) for x in elem)
        ):
            idx_t, cnt_t = elem
            idx_t = idx_t.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            cnt_t = cnt_t.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            indices_chunks.append(idx_t)
            counts_chunks.append(cnt_t)
            offsets_list.append(offsets_list[-1] + int(idx_t.numel()))
            continue

        if isinstance(elem, dict):
            pairs = []
            for digits, cnt in elem.items():
                d = digits if digits != "" else "0"
                pairs.append((int(d, 2), int(cnt)))
            if pairs:
                idx_t = torch.tensor([p[0] for p in pairs], dtype=torch.int64, device="cpu")
                cnt_t = torch.tensor([p[1] for p in pairs], dtype=torch.int64, device="cpu")
            else:
                idx_t = torch.empty(0, dtype=torch.int64, device="cpu")
                cnt_t = torch.empty(0, dtype=torch.int64, device="cpu")
            indices_chunks.append(idx_t)
            counts_chunks.append(cnt_t)
            offsets_list.append(offsets_list[-1] + int(idx_t.numel()))
            continue

        if isinstance(elem, list) or isinstance(elem, tuple):
            if len(elem) == 0:
                idx_t = torch.empty(0, dtype=torch.int64, device="cpu")
                cnt_t = torch.empty(0, dtype=torch.int64, device="cpu")
                indices_chunks.append(idx_t)
                counts_chunks.append(cnt_t)
                offsets_list.append(offsets_list[-1])
                continue

            if isinstance(elem[0], tuple) and len(elem[0]) == 2:
                idx_t = torch.tensor([int(p[0]) for p in elem], dtype=torch.int64, device="cpu")
                cnt_t = torch.tensor([int(p[1]) for p in elem], dtype=torch.int64, device="cpu")
                indices_chunks.append(idx_t)
                counts_chunks.append(cnt_t)
                offsets_list.append(offsets_list[-1] + int(idx_t.numel()))
                continue

        raise TypeError(
            "Unsupported per-circuit counts element type: "
            f"{type(elem)}. Expected dict, list of pairs, or (indices, counts) tensors."
        )

    offsets = torch.tensor(offsets_list, dtype=torch.int64, device="cpu")
    indices = (
        torch.cat(indices_chunks, dim=0)
        if indices_chunks
        else torch.empty(0, dtype=torch.int64, device="cpu")
    )
    counts = (
        torch.cat(counts_chunks, dim=0)
        if counts_chunks
        else torch.empty(0, dtype=torch.int64, device="cpu")
    )
    return offsets, indices, counts


def _convert_basis(qubits_idx: List[int], pauli_str: str) -> Tuple[str, List[int]]:
    r"""Process Pauli string to get operators applied to the state before z-measurement.
    
    Args:
        qubits_idx: the indices of the qubits to be measured.
        pauli_str: the Pauli basis to measure, can be a string composed of 'i', 'x', 'y', or 'z'.
    
    Returns:
        a qasm string that applies the necessary basis change operations before measurement,
        and a list of observable values (+-1) for each measurement outcome.
    """
    obs_vals = [1]
    list_recover_op = []
    for i, char in enumerate(pauli_str.lower()):
        idx = qubits_idx[i]
        if char == "x":
            list_recover_op.append(f"h q[{idx}];")
            obs_vals = [v for x in obs_vals for v in (x, -x)]
        elif char == "y":
            list_recover_op.extend([f"sdg q[{idx}];", f"h q[{idx}];"])
            obs_vals = [v for x in obs_vals for v in (x, -x)]
        elif char in "z":
            obs_vals = [v for x in obs_vals for v in (x, -x)]
        elif char in "i":
            obs_vals = [v for x in obs_vals for v in (x, x)]
        else:
            raise ValueError(f"Unsupported Pauli character {char!r} in {pauli_str!r}.")

    if not list_recover_op:
        return '', obs_vals

    qasm_recover_op = '\n'.join(list_recover_op)
    return qasm_recover_op, obs_vals


def _as_qasm_list(qasm: Union[str, List[str]]) -> List[str]:
    return [qasm] if isinstance(qasm, str) else list(qasm)


def _merge_qasm(qasm1: Union[str, List[str]], qasm2: Union[str, List[str]]) -> Union[str, List[str]]:
    r"""Merge two qasm strings/lists with broadcast semantics."""
    if isinstance(qasm1, str):
        if isinstance(qasm2, str):
            qasm1 += qasm2
        else:
            qasm1 = [qasm1 + item for item in qasm2]
    elif isinstance(qasm2, str):
        qasm1 = [item + qasm2 for item in qasm1]
    else:
        if len(qasm1) != len(qasm2):
            raise ValueError("The circuit has different batch sizes for different gates!")
        for ii in range(len(qasm1)):
            qasm1[ii] += qasm2[ii]
    return qasm1


def _qasm_batch_size(qasm: Union[str, List[str]]) -> int:
    return 1 if isinstance(qasm, str) else len(qasm)


def _empty_measure_probabilities(
    batch_size: int,
    num_outcomes: int,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    r"""Return an empty sparse probability tensor with the public measure shape.

    The tensor is built on CPU to stay consistent with the CSR ingestion contract
    (``_normalize_batched_csr_counts`` always returns CPU tensors) and then moved
    to ``device`` so that callers see an output whose device matches the state's
    active device (mirroring ``DefaultSimulator.measure`` semantics).
    """
    size = (num_outcomes,) if batch_size == 1 else (batch_size, num_outcomes)
    indices = torch.empty((len(size), 0), dtype=torch.int64, device="cpu")
    values = torch.empty((0,), dtype=get_float_dtype(), device="cpu")
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        result = torch.sparse_coo_tensor(
            indices, values, size=size, dtype=get_float_dtype(), device="cpu",
        ).coalesce()
    if device is not None:
        result = result.to(device=device)
    return result


def _csr_counts_to_sparse_probabilities(
    offsets: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    batch_size: int,
    num_outcomes: int,
    shots: int,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    r"""Build sparse empirical probabilities directly from batched CSR counts.

    The full construction runs on CPU because the CSR inputs follow the
    CPU-only contract of ``_normalize_batched_csr_counts``. The returned sparse
    COO tensor is then moved to ``device`` (typically the backing state's
    device) so downstream autograd sees a device-consistent graph.
    """
    if shots == 0 or indices.numel() == 0:
        return _empty_measure_probabilities(batch_size, num_outcomes, device=device)

    values = counts.to(dtype=get_float_dtype()) / float(shots)
    size = (num_outcomes,) if batch_size == 1 else (batch_size, num_outcomes)
    if batch_size == 1:
        sparse_indices = indices.to(dtype=torch.int64, copy=False).reshape(1, -1)
    else:
        row_lengths = offsets[1:] - offsets[:-1]
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int64, device="cpu"),
            row_lengths,
        )
        sparse_indices = torch.stack(
            [row_indices, indices.to(dtype=torch.int64, copy=False)],
            dim=0,
        )
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        result = torch.sparse_coo_tensor(
            sparse_indices,
            values,
            size=size,
            dtype=get_float_dtype(),
            device="cpu",
        ).coalesce()
    if device is not None:
        result = result.to(device=device)
    return result


def _sparse_measure_grad_lookup(
    grad_output: torch.Tensor,
    qasm_batch_size: int,
) -> List[Dict[int, float]]:
    r"""Convert sparse measure gradients into per-batch lookup tables on CPU."""
    grad_sparse = grad_output.detach()
    if grad_sparse.layout != torch.sparse_coo:
        grad_sparse = grad_sparse.to_sparse_coo()
    grad_sparse = grad_sparse.coalesce().to(device="cpu")

    grad_rows: List[Dict[int, float]] = [dict() for _ in range(qasm_batch_size)]
    indices = grad_sparse.indices()
    values = grad_sparse.values()

    if grad_sparse.ndim == 1:
        assert qasm_batch_size == 1, (
            f"1-D sparse gradient is only supported for a single QASM batch, "
            f"received batch size {qasm_batch_size}."
        )
        for p in range(values.numel()):
            grad_rows[0][int(indices[0, p].item())] = float(values[p].item())
        return grad_rows

    if grad_sparse.ndim != 2:
        raise ValueError(
            f"Gradient output for sparse measure should be 1D/2D, "
            f"received shape {list(grad_output.shape)}."
        )

    if grad_sparse.shape[0] != qasm_batch_size:
        raise ValueError(
            f"Gradient batch mismatch in sparse measure backward: gradient batch "
            f"{grad_sparse.shape[0]}, QASM batch {qasm_batch_size}."
        )

    for p in range(values.numel()):
        row = int(indices[0, p].item())
        col = int(indices[1, p].item())
        grad_rows[row][col] = float(values[p].item())
    return grad_rows


def _measure_qasm2(
    base_qasm: Union[str, List[str]], qubits_idx: List[int], num_qubits: int
) -> Union[str, List[str]]:
    r"""Generate the QASM 2.0 string that includes the measurement operation.
    
    Args:
        base_qasm: the base QASM 2.0 string without measurement.
        qubits_idx: the indices of the qubits to be measured.
        num_qubits: the total number of qubits in the system.
        
    Returns:
        A QASM 2.0 string that includes the measurement operation for the specified qubits.
    """
    num_measure = len(qubits_idx)
    qasm_with_measure = f"\n\ncreg c[{num_measure}];\n"
    if num_measure == num_qubits:
        qasm_with_measure += "measure q -> c;\n"
    else:
        for i, idx in enumerate(qubits_idx):
            qasm_with_measure += f"measure q[{idx}] -> c[{i}];\n"
    return _merge_qasm(base_qasm, qasm_with_measure)


def _parse_execution_qasm2(qasm: str) -> Tuple[str, List[int]]:
    r"""Split internal execution QASM into a gate-only circuit and measurement order.

    `StateOperator` backends internally append ``creg`` and ``measure`` statements to a
    gate-only OpenQASM 2.0 program before calling ``_execute``. Public
    ``Circuit.from_qasm2`` now rejects those classical statements, so backends that
    emulate execution locally need a private path that strips the classical suffix and
    recovers the measured qubit order.
    """
    statements = [f"{stmt.strip()};" for stmt in qasm.split(";") if stmt.strip()]
    gate_only_statements: List[str] = []
    qreg_size: Optional[int] = None
    creg_size: Optional[int] = None
    measure_map: Dict[int, int] = {}
    has_full_measure = False

    for statement in statements:
        body = statement[:-1].strip()

        match = re.fullmatch(r"qreg\s+q\s*\[\s*(\d+)\s*\]", body, flags=re.IGNORECASE)
        if match:
            qreg_size = int(match.group(1))
            gate_only_statements.append(statement)
            continue

        match = re.fullmatch(r"creg\s+c\s*\[\s*(\d+)\s*\]", body, flags=re.IGNORECASE)
        if match:
            creg_size = int(match.group(1))
            continue

        if re.fullmatch(r"measure\s+q\s*->\s*c", body, flags=re.IGNORECASE):
            if qreg_size is None:
                raise ValueError("StateOperator execution QASM must declare qreg before measure.")
            measure_map = {idx: idx for idx in range(qreg_size)}
            has_full_measure = True
            continue

        match = re.fullmatch(
            r"measure\s+q\s*\[\s*(\d+)\s*\]\s*->\s*c\s*\[\s*(\d+)\s*\]",
            body,
            flags=re.IGNORECASE,
        )
        if match:
            if has_full_measure:
                raise ValueError("Cannot mix 'measure q -> c' with indexed measure statements.")
            qubit_idx = int(match.group(1))
            classical_idx = int(match.group(2))
            if classical_idx in measure_map and measure_map[classical_idx] != qubit_idx:
                raise ValueError(
                    f"Classical bit c[{classical_idx}] is assigned to multiple qubits "
                    f"({measure_map[classical_idx]} and {qubit_idx})."
                )
            measure_map[classical_idx] = qubit_idx
            continue

        gate_only_statements.append(statement)

    if qreg_size is None:
        raise ValueError("StateOperator execution QASM must declare 'qreg q[...]'.")

    if not measure_map:
        return "\n".join(gate_only_statements), list(range(qreg_size))

    ordered_classical_bits = sorted(measure_map)
    expected_classical_bits = list(range(len(ordered_classical_bits)))
    if ordered_classical_bits != expected_classical_bits:
        raise ValueError(
            "StateOperator execution QASM must measure into contiguous classical bits starting at c[0]."
        )

    measured_qubits = [measure_map[idx] for idx in expected_classical_bits]
    if creg_size is not None and creg_size != len(measured_qubits):
        raise ValueError(
            f"Classical register size mismatch: declared {creg_size}, "
            f"but measured {len(measured_qubits)} qubits."
        )
    if any(idx < 0 or idx >= qreg_size for idx in measured_qubits):
        raise ValueError(
            f"Measured qubit index out of range for qreg q[{qreg_size}]: {measured_qubits}."
        )

    return "\n".join(gate_only_statements), measured_qubits


class StateOperator(State):
    r"""The abstract base class for operating quantum state in QuAIRKit, via emulators or quantum devices.

    Args:
        list_operator: the operators of the circuit that applies to the state. Defaults to an empty list.
        sys_dim: a list of dimensions for each system.

    Note:
        This class is designed to be inherited by specific state emulator classes.
        Such class should emulate quantum computing via OpenQASM 2.0.

    """
    def __init__(self, list_operator: Optional[List[OperatorInfoType]], sys_dim: List[int]) -> None:
        super().__init__(sys_dim)

        self._data: List[OperatorInfoType] = copy.deepcopy(list_operator or [])

    def _execute(self, qasm: str, shots: int) -> Dict[str, int]:
        r"""Execute a quantum circuit with the current operators and return the measurement results.
        
        Args:
            qasm: the QASM 2.0 string that represents the quantum circuit.
            shots: the number of shots to measure.
            
        Returns:
            A dictionary storing the number of each measurement outcomes.
            The keys are the outcome digits in binary string, and the values are the number of occurrences.
        """
        raise NotImplementedError(
            "Unless _multi_execute is overridden, "
            "_execute method should be implemented in the subclass of StateOperator.")
    
    def _multi_execute(
        self, list_qasm: List[str], list_shots: List[int]
    ) -> Union[
        Iterator[Dict[str, int]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        r"""Execute multiple circuits.

        Supported return forms:

        1) Iterator/list of ``Dict[str, int]`` (bitstring -> count), one per circuit.
        2) Batched CSR triplet ``(offsets, indices, counts)``:
           - offsets: int64, length n_circuits+1, slice [offsets[i], offsets[i+1]) is circuit i
           - indices: int32/int64, flattened outcome indices (same as ``int(bitstring, 2)``)
           - counts:  int32/int64, aligned with ``indices``

        Args:
            list_qasm: QASM 2.0 strings.
            list_shots: shots per circuit.

        Returns:
            Iterator/list of dicts (legacy) or a CSR triplet (recommended).
        """
        assert isinstance(list_qasm, list) and all(isinstance(qasm, str) for qasm in list_qasm), \
            f"Expected a list of QASM strings, received {type(list_qasm)}."
        assert len(list_qasm) == len(list_shots), \
            f"The lengths of list_qasm {len(list_qasm)} and list_shots {len(list_shots)} do not match."
        for idx, qasm in enumerate(list_qasm):
            yield self._execute(qasm, list_shots[idx])
    
    def __str__(self) -> str:
        return '\n'.join([op.qasm for op in self._data])
    
    @property
    def param(self) -> List[torch.Tensor]:
        r"""The parameters stored in the operators in StateOperator.
        """
        return [op['param'] for op in self._data if 'param' in op]

    def apply(self, list_operator: List[Union[OperatorInfoType, List[OperatorInfoType]]]) -> None:
        r"""Record a list of operators that will be applied to the device.

        Args:
            list_operator: the list of operators that applies to the state.

        """
        flat_list = []
        for op in list_operator:
            if isinstance(op, List):
                flat_list.extend(op)
            else:
                flat_list.append(op)
        self._data.extend(flat_list)

    @property
    def qasm2(self) -> Union[str, List[str]]:
        r"""The QASM 2.0 string that represents the circuit stored in this state emulator.
        """
        assert self.are_qubits(), \
            "The StateOperator only supports qubit systems. Please use the simulator backend such as default simulator."

        header = 'OPENQASM 2.0;\ninclude "qelib1.inc";'
        qreg = f"qreg q[{self.num_qubits}];"
        qasm: Union[str, List[str]] = '\n'.join([header, qreg])
        for op in self._data:
            qasm = _merge_qasm(qasm, _merge_qasm('\n', op.qasm2))
        return qasm
    
    @State.system_dim.setter
    def system_dim(self, sys_dim: List[int]) -> None:
        raise NotImplementedError(
            "A state operator should not change its system dimension after initialization.")
    
    @abstractmethod
    def clone(self) -> 'StateOperator':
        r"""Return a copy of the quantum state.
        """

    def _measure(self, qubits_idx: List[int], shots: int, pauli_str: str) -> Dict[str, Tuple[int, int]]:
        r"""Measure the quantum state

        Args:
            qubits_idx: the qubit indices to be measured. Defaults to all qubits.
            shots: the number of shots to measure, defaults to 1024.
            pauli_str: the Pauli basis to measure, can be a string composed of 'i', 'x', 'y', or 'z'.
            return_energy: whether the returned dictionary contains the energy (+-1) for each measurement outcome. Defaults to False.

        Returns:
            A dictionary storing the number (and energy) of each measurement outcome.
            The keys are the outcome digits in binary string, and the values are tuples of (number, energy).

        """
        qasm_recover_op, obs_vals = _convert_basis(qubits_idx, pauli_str)

        raw_qasm = self.qasm2
        if qasm_recover_op:
            raw_qasm = _merge_qasm(raw_qasm, '\n' + qasm_recover_op)
        process_qasm = _measure_qasm2(raw_qasm, qubits_idx, self.num_qubits)
        assert isinstance(process_qasm, str), (
            "_measure only supports non-batched QASM input. "
            "Use _measure_probabilities for batched circuits.")
        raw_result = self._execute(process_qasm, shots)

        return {
            digits: (num_occur, obs_vals[int(digits, base=2)])
            for digits, num_occur in raw_result.items()
        }
    
    def _measure_probabilities(self, qubits_idx: List[int], shots: int, pauli_str: str) -> torch.Tensor:
        r"""Measure the quantum state and return sparse empirical probabilities.
        
        Args:
            qubits_idx: the qubit indices to be measured. Defaults to all qubits.
            shots: the number of shots to measure, defaults to 1024.
            pauli_str: the Pauli basis to measure, can be a string composed of 'i', 'x', 'y', or 'z'.

        Returns:
            A sparse tensor storing the empirical probability of each measurement outcome.
            The outcome index follows ``int(bitstring, 2)``.

        """
        num_outcomes = 2 ** len(qubits_idx)
        qasm_recover_op, _ = _convert_basis(qubits_idx, pauli_str)
        raw_qasm = self.qasm2
        if qasm_recover_op:
            raw_qasm = _merge_qasm(raw_qasm, '\n' + qasm_recover_op)
        process_qasm = _measure_qasm2(raw_qasm, qubits_idx, self.num_qubits)
        qasm_list = _as_qasm_list(process_qasm)
        batch_size = len(qasm_list)
        if shots == 0:
            return _empty_measure_probabilities(batch_size, num_outcomes, device=self.device)

        raw_counts = self._multi_execute(qasm_list, [shots] * batch_size)
        offsets, indices, counts = _normalize_batched_csr_counts(raw_counts, batch_size)
        return _csr_counts_to_sparse_probabilities(
            offsets,
            indices,
            counts,
            batch_size,
            num_outcomes,
            shots,
            device=self.device,
        )
    
    def measure(self, system_idx: Optional[Union[int, List[int]]] = None, 
                shots: Optional[int] = None, measure_op: Optional[str] = None) -> torch.Tensor:
        r"""Measure the quantum state
        
        Args:
            system_idx: the system indices to be measured. Defaults to all systems.
            shots: the number of shots to measure, defaults to 1024.
            measure_op: the measurement operator basis to measure. Here we restrict it to be Pauli, 
                which can be a char or a string composed of 'i', 'x', 'y', or 'z'. Defaults to 'z'.
            
        Returns:
            A tensor containing the measurement results. Simulator backends return dense
            probabilities, while ``StateOperator`` backends return sparse empirical
            probabilities over the same outcome space.
        """
        system_idx = self._check_sys_idx(system_idx)

        if shots is None:
            shots = 1024
        else:
            if shots < 0:
                raise ValueError(f"The number of shots should be a non-negative integer, received {shots}.")
            if shots == 0:
                num_outcomes = 2 ** len(system_idx)
                batch_size = _qasm_batch_size(self.qasm2)
                return _empty_measure_probabilities(batch_size, num_outcomes, device=self.device)

        measure_op = 'z' if measure_op is None else measure_op.lower()
        if measure_op in {'x', 'y', 'z', 'i'}:
            measure_op *= len(system_idx)
        elif len(measure_op) != len(system_idx) or any(b not in {'x', 'y', 'z', 'i'} for b in measure_op):
            raise ValueError(
                f"Invalid format: received Pauli basis {measure_op} for the system indices {system_idx}.")

        return _compute_measure(self, system_idx, shots, measure_op, *self.param)
    
    def _expec_val_per_term(self, hamiltonian: Hamiltonian, list_shots: torch.Tensor) -> torch.Tensor:
        r"""The expectation value of each term in the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            list_shots: the number of shots to measure for each term.

        Returns:
            The measured expectation value of each term in the input observable for the quantum state.
        """
        num_terms = hamiltonian.n_terms
        list_qubits_idx = hamiltonian.sites
        list_coef = hamiltonian.coefficients
        list_pauli_str = hamiltonian.pauli_words_r

        raw_qasm, num_qubits = self.qasm2, self.num_qubits
        qasm_batch_size = _qasm_batch_size(raw_qasm)

        list_qasm: List[str] = []
        list_shots_qasm: List[int] = []
        list_masks_qasm: List[int] = []
        term_to_submit_idx: List[List[int]] = [[] for _ in range(num_terms)]

        submit_idx = 0
        for i in range(num_terms):
            qubits_idx, pauli_str = list_qubits_idx[i], list_pauli_str[i]
            if qubits_idx == ['']:
                continue
            qasm_recover_op, _ = _convert_basis(qubits_idx, pauli_str)
            process_qasm = _measure_qasm2(
                _merge_qasm(raw_qasm, '\n' + qasm_recover_op) if qasm_recover_op else raw_qasm,
                qubits_idx,
                num_qubits,
            )
            mask = _pauli_parity_mask(pauli_str)
            qasm_batch = _as_qasm_list(process_qasm)
            assert len(qasm_batch) == qasm_batch_size, (
                f"Batched QASM size mismatch: expect {qasm_batch_size}, received {len(qasm_batch)}.")
            for qasm in qasm_batch:
                list_qasm.append(qasm)
                list_shots_qasm.append(int(list_shots[i].item()))
                list_masks_qasm.append(mask)
                term_to_submit_idx[i].append(submit_idx)
                submit_idx += 1

        if list_qasm:
            raw_counts = self._multi_execute(list_qasm, list_shots_qasm)
            offsets, indices, counts = _normalize_batched_csr_counts(raw_counts, len(list_qasm))
            masks_t = torch.tensor(list_masks_qasm, dtype=torch.int64)
            if _COUNTS_CPP is not None:
                offsets = offsets.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
                indices = indices.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
                counts = counts.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
                masks_t = masks_t.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
                acc_t = _COUNTS_CPP.dot_parity(offsets, indices, counts, masks_t)
            else:
                acc_list: List[int] = []
                for c in range(len(list_qasm)):
                    start = int(offsets[c].item())
                    end = int(offsets[c + 1].item())
                    mask = int(list_masks_qasm[c])
                    a = 0
                    for p in range(start, end):
                        idx_val = int(indices[p].item())
                        cnt_val = int(counts[p].item())
                        if bin(idx_val & mask).count("1") % 2 == 1:
                            a -= cnt_val
                        else:
                            a += cnt_val
                    acc_list.append(a)
                acc_t = torch.tensor(acc_list, dtype=torch.int64)
        else:
            acc_t = torch.empty(0, dtype=torch.int64)

        device = list_shots.device
        coef_t = torch.tensor(list_coef, dtype=get_float_dtype(), device=device)
        sum_per_term = torch.zeros([qasm_batch_size, num_terms], dtype=get_float_dtype(), device=device)
        for i in range(num_terms):
            qubits_idx, coef, shots_per_term = list_qubits_idx[i], coef_t[i], int(list_shots[i].item())
            if qubits_idx == ['']:
                sum_per_term[:, i] = coef * shots_per_term
                continue

            submit_indices = term_to_submit_idx[i]
            assert len(submit_indices) == qasm_batch_size, (
                f"Submission index mismatch for Hamiltonian term {i}: "
                f"expected {qasm_batch_size}, received {len(submit_indices)}.")
            values = acc_t[submit_indices].to(dtype=get_float_dtype(), device=device)
            sum_per_term[:, i] = coef * values

        list_shots_t = list_shots.to(dtype=get_float_dtype(), device=device).reshape([1, -1])
        safe_shots = torch.where(list_shots_t > 0, list_shots_t, torch.ones_like(list_shots_t))
        expec_val_per_term = sum_per_term / safe_shots
        expec_val_per_term = torch.where(
            list_shots_t > 0, expec_val_per_term, torch.zeros_like(expec_val_per_term))
        return expec_val_per_term.squeeze(0) if qasm_batch_size == 1 else expec_val_per_term
    
    def expec_val(self, hamiltonian: Hamiltonian, shots: Optional[int] = None, decompose: bool = False) -> torch.Tensor:
        r"""The expectation value of the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            shots: the total number of shots to measure, defaults to 1024 times the Hamiltonian terms.
            decompose: If decompose is ``True``, it will return the expectation value of each term.

        Returns:
            The measured expectation value (per term) of the input observable for the quantum state.

        """
        if not self.are_qubits():
            raise NotImplementedError(
                f"Currently only support qubit computation in Hamiltonian tasks: received {self._sys_dim}")
        assert hamiltonian.n_qubits <= self.num_qubits, \
                f"The number of qubits in the Hamiltonian {hamiltonian.n_qubits} is greater than that of state qubits {self.num_qubits}"
        
        num_terms = hamiltonian.n_terms
        list_coef = torch.tensor(hamiltonian.coefficients)
        list_abs_coef = torch.abs(list_coef)
        
        shots = shots or 1024 * num_terms
        coef_l1 = float(list_abs_coef.sum().item())
        coef_l2_sq = float(list_abs_coef.square().sum().item())
        effective_terms = coef_l1 * coef_l1 / coef_l2_sq if coef_l2_sq > 0 else 1.0
        min_required_shots = max(10, int(math.ceil(10.0 * effective_terms)))
        if shots < min_required_shots:
            warnings.warn(
                f"Insufficient shots {shots} for Hamiltonian with coefficients {list_coef}: "
                f"recommended at least {min_required_shots} shots for a more reliable result "
                f"under the current weighted-term sampling.",
                UserWarning,
            )
            
        pauli_weight = list_abs_coef / list_abs_coef.sum()
        pauli_sum = list_abs_coef.sum()
        pauli_weight_np = pauli_weight.detach().cpu().numpy()
        list_shots_np = np.random.multinomial(shots, pauli_weight_np)
        list_shots = torch.tensor(list_shots_np, device=self.device, dtype=torch.int64)
        
        expec_val_per_term = _compute_expec_val_per_term(self, hamiltonian, list_shots, *self.param)
        if decompose:
            return expec_val_per_term
        
        sum_per_term = expec_val_per_term * list_shots
        factor_per_term = torch.zeros_like(sum_per_term)
        valid_indices = list_coef != 0
        factor_per_term[..., valid_indices] = pauli_sum / list_abs_coef[valid_indices]
        weighted_sum = sum_per_term * factor_per_term
        return weighted_sum.sum() / shots if weighted_sum.ndim == 1 else weighted_sum.sum(dim=-1) / shots

    def _evaluate_for_grad(
        self, idx: int, param_idx: Union[Tuple[int, int], Tuple[int, int, int]], shift: float,
        eval_func: Callable[['StateOperator'], torch.Tensor]
    ) -> torch.Tensor:
        r"""Helper function to evaluate quantum circuit at shifted parameter values.
        """
        if len(param_idx) == 2:
            i, j = param_idx
            b = 0
        elif len(param_idx) == 3:
            i, b, j = param_idx
        else:
            raise ValueError(f"param_idx should have length 2 or 3, received {param_idx}.")
        with torch.no_grad():
            orig_val = float(self._data[idx]['param'][i, b, j].item())
            self._data[idx]['param'][i, b, j] = orig_val + shift
        result = eval_func(self)
        with torch.no_grad():
            self._data[idx]['param'][i, b, j] = orig_val
        return result


class OperatorMeasure(torch.autograd.Function):
    @staticmethod
    def forward(state: StateOperator, qubits_idx: List[int], shots: int, 
                pauli_str: str, *params: torch.Tensor) -> torch.Tensor:
        return state._measure_probabilities(qubits_idx, shots, pauli_str)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        state, qubits_idx, shots, pauli_str, *params = inputs
        ctx.state = state.clone()
        
        def eval_func(input_state):
            return input_state._measure_probabilities(qubits_idx, shots, pauli_str)
        ctx.eval_func = eval_func
        ctx.qubits_idx = qubits_idx
        ctx.shots = shots
        ctx.pauli_str = pauli_str

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state: StateOperator = ctx.state.clone()
        num_qubits = state.num_qubits
        qubits_idx = ctx.qubits_idx
        shots = int(ctx.shots)
        pauli_str = ctx.pauli_str
        device = grad_output.device

        qasm_recover_op, _ = _convert_basis(qubits_idx, pauli_str)
        qasm_batch_size = _qasm_batch_size(state.qasm2)
        grad_dtype = grad_output.dtype
        grad_cpu: Optional[torch.Tensor] = None
        grad_sparse_rows: Optional[List[Dict[int, float]]] = None
        if grad_output.layout == torch.strided:
            grad_cpu = grad_output.detach().to(device="cpu").contiguous()
            if grad_cpu.ndim == 1:
                grad_cpu = grad_cpu.unsqueeze(0)
            assert grad_cpu.ndim == 2, (
                f"Gradient output for measure should be 1D/2D, received shape {list(grad_output.shape)}.")
            assert grad_cpu.shape[0] == qasm_batch_size, (
                f"Gradient batch mismatch in measure backward: gradient batch {grad_cpu.shape[0]}, "
                f"QASM batch {qasm_batch_size}.")
            grad_dtype = grad_cpu.dtype
        else:
            grad_sparse_rows = _sparse_measure_grad_lookup(grad_output, qasm_batch_size)

        list_qasm_total: List[str] = []
        list_shots_total: List[int] = []

        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue
            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_batch_size = int(param_tensor.shape[1])
            assert param_batch_size in {1, qasm_batch_size}, (
                f"Unsupported parameter batch size {param_batch_size} for gate {op_info['name']}: "
                f"expected 1 or {qasm_batch_size}.")
            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, _coeffs = get_shift_rule(op_info['name'])
                with torch.no_grad():
                    if param_batch_size == 1:
                        orig_val = state._data[idx]['param'][i, 0, j].clone()
                    else:
                        orig_val = state._data[idx]['param'][i, :, j].clone()

                for shift in shifts:
                    with torch.no_grad():
                        if param_batch_size == 1:
                            state._data[idx]['param'][i, 0, j] = orig_val + shift
                        else:
                            state._data[idx]['param'][i, :, j] = orig_val + shift

                    raw_qasm = state.qasm2
                    if qasm_recover_op:
                        raw_qasm = _merge_qasm(raw_qasm, '\n' + qasm_recover_op)
                    process_qasm = _measure_qasm2(raw_qasm, qubits_idx, num_qubits)
                    process_qasm = _as_qasm_list(process_qasm)
                    assert len(process_qasm) == qasm_batch_size, (
                        f"QASM batch mismatch in measure backward: expect {qasm_batch_size}, "
                        f"received {len(process_qasm)}.")

                    list_qasm_total.extend(process_qasm)
                    list_shots_total.extend([shots] * qasm_batch_size)

                with torch.no_grad():
                    if param_batch_size == 1:
                        state._data[idx]['param'][i, 0, j] = orig_val
                    else:
                        state._data[idx]['param'][i, :, j] = orig_val

        if len(list_qasm_total) == 0:
            num_param_inputs = len(state.param)
            return (None, None, None, None, *([None] * num_param_inputs))

        list_param_grad: List[Optional[torch.Tensor]] = []
        raw_counts = state._multi_execute(list_qasm_total, list_shots_total)
        offsets, indices, counts = _normalize_batched_csr_counts(raw_counts, len(list_qasm_total))

        shots_t = torch.tensor(shots, dtype=torch.int64)
        if grad_cpu is not None and _COUNTS_CPP is not None and qasm_batch_size == 1:
            dots = _COUNTS_CPP.dot_prob_grad(offsets, indices, counts, grad_cpu.squeeze(0), shots_t)
        else:
            dots_list: List[float] = []
            for c in range(len(list_qasm_total)):
                start = int(offsets[c].item())
                end = int(offsets[c + 1].item())
                if shots == 0:
                    dots_list.append(0.0)
                    continue
                if grad_cpu is not None:
                    idx_c = indices[start:end]
                    cnt_c = counts[start:end].to(dtype=grad_dtype)
                    grad_row = grad_cpu[c % qasm_batch_size]
                    val = float((cnt_c * grad_row[idx_c]).sum().item()) / float(shots)
                else:
                    assert grad_sparse_rows is not None
                    grad_row = grad_sparse_rows[c % qasm_batch_size]
                    acc = 0.0
                    for p in range(start, end):
                        idx_val = int(indices[p].item())
                        grad_val = grad_row.get(idx_val)
                        if grad_val is None:
                            continue
                        acc += float(counts[p].item()) * grad_val
                    val = acc / float(shots)
                dots_list.append(val)
            dots = torch.tensor(dots_list, dtype=grad_dtype)

        dot_ptr = 0
        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                list_param_grad.append(None)
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_batch_size = int(param_tensor.shape[1])
            param_grad = torch.zeros_like(param_tensor)

            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, coeffs = get_shift_rule(op_info['name'])
                if param_batch_size == 1:
                    acc = 0.0
                    for s in range(len(shifts)):
                        dot_batch = dots[dot_ptr:dot_ptr + qasm_batch_size]
                        dot_ptr += qasm_batch_size
                        acc += float(coeffs[s]) * float(dot_batch.sum().item())
                    param_grad[i, 0, j] = torch.tensor(acc, dtype=param_tensor.dtype, device=device)
                else:
                    acc = torch.zeros(param_batch_size, dtype=grad_dtype)
                    for s in range(len(shifts)):
                        dot_batch = dots[dot_ptr:dot_ptr + qasm_batch_size]
                        dot_ptr += qasm_batch_size
                        acc += float(coeffs[s]) * dot_batch.to(dtype=grad_dtype)
                    param_grad[i, :, j] = acc.to(dtype=param_tensor.dtype, device=device)

            list_param_grad.append(param_grad)

        return None, None, None, None, *list_param_grad


class OperatorExpecVal(torch.autograd.Function):
    @staticmethod
    def forward(state: StateOperator, hamiltonian: Hamiltonian, 
                list_shots: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        return state._expec_val_per_term(hamiltonian, list_shots)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        state, hamiltonian, list_shots, *params = inputs
        ctx.state = state.clone()

        def eval_func(input_state):
            return input_state._expec_val_per_term(hamiltonian, list_shots)

        ctx.eval_func = eval_func
        ctx.hamiltonian = hamiltonian
        ctx.list_shots = list_shots

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        state: StateOperator = ctx.state.clone()
        hamiltonian: Hamiltonian = ctx.hamiltonian
        num_qubits = state.num_qubits
        list_shots: torch.Tensor = ctx.list_shots
        device = grad_output.device

        num_terms = hamiltonian.n_terms
        list_qubits_idx = hamiltonian.sites
        list_coef = torch.tensor(hamiltonian.coefficients, dtype=get_float_dtype(), device=device)
        qasm_batch_size = _qasm_batch_size(state.qasm2)

        qasm_recover_ops_per_term: List[str] = []
        parity_mask_per_term: List[Optional[int]] = []
        for t in range(num_terms):
            qubits_idx, pauli_str = list_qubits_idx[t], hamiltonian.pauli_words_r[t]
            if qubits_idx == ['']:
                qasm_recover_ops_per_term.append('')
                parity_mask_per_term.append(None)
            else:
                qasm_recover_op, _obs_vals = _convert_basis(qubits_idx, pauli_str)
                qasm_recover_ops_per_term.append(qasm_recover_op)
                parity_mask_per_term.append(_pauli_parity_mask(pauli_str))

        list_qasm_total: List[str] = []
        list_shots_total: List[int] = []
        list_masks_total: List[int] = []

        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_batch_size = int(param_tensor.shape[1])
            assert param_batch_size in {1, qasm_batch_size}, (
                f"Unsupported parameter batch size {param_batch_size} for gate {op_info['name']}: "
                f"expected 1 or {qasm_batch_size}.")
            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, _coeffs = get_shift_rule(op_info['name'])
                with torch.no_grad():
                    if param_batch_size == 1:
                        orig_val = state._data[idx]['param'][i, 0, j].clone()
                    else:
                        orig_val = state._data[idx]['param'][i, :, j].clone()

                for shift in shifts:
                    with torch.no_grad():
                        if param_batch_size == 1:
                            state._data[idx]['param'][i, 0, j] = orig_val + shift
                        else:
                            state._data[idx]['param'][i, :, j] = orig_val + shift

                    raw_qasm = state.qasm2
                    for t in range(num_terms):
                        if list_qubits_idx[t] == ['']:
                            continue
                        qasm_recover_op = qasm_recover_ops_per_term[t]
                        mask_t = parity_mask_per_term[t]
                        assert mask_t is not None
                        process_qasm = _measure_qasm2(
                            _merge_qasm(raw_qasm, '\n' + qasm_recover_op) if qasm_recover_op else raw_qasm,
                            list_qubits_idx[t],
                            num_qubits,
                        )
                        process_qasm = _as_qasm_list(process_qasm)
                        assert len(process_qasm) == qasm_batch_size, (
                            f"QASM batch mismatch in expec backward: expect {qasm_batch_size}, "
                            f"received {len(process_qasm)}.")
                        list_qasm_total.extend(process_qasm)
                        list_shots_total.extend([int(list_shots[t].item())] * qasm_batch_size)
                        list_masks_total.extend([int(mask_t)] * qasm_batch_size)

                with torch.no_grad():
                    if param_batch_size == 1:
                        state._data[idx]['param'][i, 0, j] = orig_val
                    else:
                        state._data[idx]['param'][i, :, j] = orig_val

        if len(list_qasm_total) == 0:
            num_param_inputs = len(state.param)
            return (None, None, None, *([None] * num_param_inputs))

        list_param_grad: List[Optional[torch.Tensor]] = []
        dtype = get_float_dtype()

        raw_counts = state._multi_execute(list_qasm_total, list_shots_total)
        offsets, indices, counts = _normalize_batched_csr_counts(raw_counts, len(list_qasm_total))
        masks_t = torch.tensor(list_masks_total, dtype=torch.int64)
        if _COUNTS_CPP is not None:
            offsets = offsets.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            indices = indices.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            counts = counts.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            masks_t = masks_t.to(device="cpu", dtype=torch.int64, copy=False).contiguous()
            acc_t = _COUNTS_CPP.dot_parity(offsets, indices, counts, masks_t)
        else:
            acc_list: List[int] = []
            for c in range(len(list_qasm_total)):
                start = int(offsets[c].item())
                end = int(offsets[c + 1].item())
                mask = int(list_masks_total[c])
                a = 0
                for p in range(start, end):
                    idx_val = int(indices[p].item())
                    cnt_val = int(counts[p].item())
                    if bin(idx_val & mask).count("1") % 2 == 1:
                        a -= cnt_val
                    else:
                        a += cnt_val
                acc_list.append(a)
            acc_t = torch.tensor(acc_list, dtype=torch.int64)

        grad_cpu = grad_output.detach().to(device="cpu", dtype=dtype).contiguous()
        if grad_cpu.ndim == 1:
            grad_cpu = grad_cpu.unsqueeze(0)
        assert grad_cpu.ndim == 2, (
            f"Gradient output for expectation should be 1D/2D, received {list(grad_output.shape)}.")
        assert grad_cpu.shape[0] == qasm_batch_size, (
            f"Gradient batch mismatch in expectation backward: gradient batch {grad_cpu.shape[0]}, "
            f"QASM batch {qasm_batch_size}.")
        assert grad_cpu.shape[-1] == num_terms, (
            f"Gradient term mismatch in expectation backward: gradient terms {grad_cpu.shape[-1]}, "
            f"Hamiltonian terms {num_terms}.")
        coef_cpu = torch.tensor(hamiltonian.coefficients, dtype=dtype)
        shots_cpu = list_shots.detach().to(device="cpu", dtype=torch.int64)

        acc_ptr = 0
        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                list_param_grad.append(None)
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_batch_size = int(param_tensor.shape[1])
            param_grad = torch.zeros_like(param_tensor)

            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, coeffs = get_shift_rule(op_info['name'])
                if param_batch_size == 1:
                    deriv = 0.0
                    for s in range(len(shifts)):
                        shift_dot = 0.0
                        for t in range(num_terms):
                            if list_qubits_idx[t] == ['']:
                                shift_dot += float(coef_cpu[t].item()) * float(grad_cpu[:, t].sum().item())
                                continue

                            acc_batch = acc_t[acc_ptr:acc_ptr + qasm_batch_size].to(dtype=dtype)
                            acc_ptr += qasm_batch_size
                            shots_t = int(shots_cpu[t].item())
                            if shots_t == 0:
                                continue
                            shift_dot += float(coef_cpu[t].item()) * float(
                                ((acc_batch / float(shots_t)) * grad_cpu[:, t]).sum().item()
                            )
                        deriv += float(coeffs[s]) * shift_dot

                    param_grad[i, 0, j] = torch.tensor(deriv, dtype=param_tensor.dtype, device=device)
                else:
                    deriv = torch.zeros(qasm_batch_size, dtype=dtype)
                    for s in range(len(shifts)):
                        shift_dot = torch.zeros(qasm_batch_size, dtype=dtype)
                        for t in range(num_terms):
                            if list_qubits_idx[t] == ['']:
                                shift_dot += float(coef_cpu[t].item()) * grad_cpu[:, t]
                                continue

                            acc_batch = acc_t[acc_ptr:acc_ptr + qasm_batch_size].to(dtype=dtype)
                            acc_ptr += qasm_batch_size
                            shots_t = int(shots_cpu[t].item())
                            if shots_t == 0:
                                continue
                            shift_dot += float(coef_cpu[t].item()) * (
                                (acc_batch / float(shots_t)) * grad_cpu[:, t]
                            )
                        deriv += float(coeffs[s]) * shift_dot

                    param_grad[i, :, j] = deriv.to(dtype=param_tensor.dtype, device=device)

            list_param_grad.append(param_grad)

        return None, None, None, *list_param_grad


SHIFT_RULES = {
    'two_term': {
        'gates': ['rx', 'ry', 'rz', 'u3', 'p', 'cp'],
        'shifts': [math.pi/2, -math.pi/2],
        'coeffs': [0.5, -0.5]
    },
    'four_term': {
        'gates': ['crz', 'cu3'],
        'shifts': [math.pi/2, -math.pi/2, 3*math.pi/2, -3*math.pi/2],
        'coeffs': [
            (math.sqrt(2) + 1) / (4 * math.sqrt(2)),
            -(math.sqrt(2) + 1) / (4 * math.sqrt(2)),
            -(math.sqrt(2) - 1) / (4 * math.sqrt(2)),
            (math.sqrt(2) - 1) / (4 * math.sqrt(2))
        ]
    }
}
DELTA = 1e-2


def get_shift_rule(gate_name: str) -> Optional[Tuple[List[float], List[float]]]:
    r"""Retrieve shift rule for a given quantum gate.
    """
    gate_name = gate_name.lower()
    return next(
        (
            (rule['shifts'], rule['coeffs'])
            for rule in SHIFT_RULES.values()
            if gate_name in rule['gates']
        ),
        ([DELTA, -DELTA], [0.5 / DELTA, -0.5 / DELTA]),
    )


_compute_measure: Callable[[StateOperator, List[int], int, str, Tuple[torch.Tensor, ...]], torch.Tensor] = OperatorMeasure.apply
_compute_expec_val_per_term: Callable[[StateOperator, Hamiltonian, torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor] = OperatorExpecVal.apply


class SimpleStateOperator(StateOperator):
    r"""A minimal backend built atop the default simulator for quick experimentation.

    It executes QASM locally (via the default backend) and returns sparse counts dict.
    This is primarily for users to try `StateOperator` without integrating a real device.
    """

    backend = "test-simple"

    def __init__(self, list_operator, sys_dim):
        super().__init__(list_operator, sys_dim)

    def _execute(self, qasm: str, shots: int = None):
        import quairkit as qkit
        from quairkit import Circuit

        shots = 1024 if shots is None else shots
        gate_only_qasm, measured_qubits = _parse_execution_qasm2(qasm)
        qkit.set_backend("default")
        try:
            probs = Circuit.from_qasm2(gate_only_qasm)().measure(system_idx=measured_qubits)
        finally:
            qkit.set_backend(self.backend)

        probs = probs.reshape(-1)
        if shots == 0:
            return {}
        probs /= torch.sum(probs)
        samples = torch.multinomial(probs, shots, replacement=True)
        counts = torch.bincount(samples, minlength=len(probs))
        nz = (counts != 0).nonzero(as_tuple=False).flatten()
        return {bin(int(i.item()))[2:]: int(counts[i].item()) for i in nz}

    def clone(self):
        new_ops = []
        new_ops.extend(copy.deepcopy(op) for op in self._data)
        return SimpleStateOperator(new_ops, self._sys_dim)

