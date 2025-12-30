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
from abc import abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from ... import Hamiltonian, OperatorInfoType, get_float_dtype
from .base import State


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


def _measure_qasm2(base_qasm: str, qubits_idx: List[int], num_qubits: int) -> str:
    r"""Generate the QASM 2.0 string that includes the measurement operation.
    
    Args:
        base_qasm: the base QASM 2.0 string without measurement.
        qubits_idx: the indices of the qubits to be measured.
        num_qubits: the total number of qubits in the system.
        
    Returns:
        A QASM 2.0 string that includes the measurement operation for the specified qubits.
    """
    num_measure = len(qubits_idx)
    qasm_with_measure = f"{base_qasm}\n\ncreg c[{num_measure}];\n"
    if num_measure == num_qubits:
        qasm_with_measure += "measure q -> c;\n"
    else:
        for i, idx in enumerate(qubits_idx):
            qasm_with_measure += f"measure q[{idx}] -> c[{i}];\n"
    return qasm_with_measure


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

    @abstractmethod
    def _execute(self, qasm: str, shots: int) -> Dict[str, int]:
        r"""Execute a quantum circuit with the current operators and return the measurement results.
        
        Args:
            qasm: the QASM 2.0 string that represents the quantum circuit.
            shots: the number of shots to measure.
            
        Returns:
            A dictionary storing the number of each measurement outcomes.
            The keys are the outcome digits in binary string, and the values are the number of occurrences.
        """
    
    def _multi_execute(self, list_qasm: List[str], list_shots: List[int]) -> Iterator[Dict[str, int]]:
        r"""Execute multiple quantum circuits with the current (batched) operators.

        Args:
            list_qasm: a list of QASM 2.0 strings that represent the quantum circuits.
            list_shots: a list of shots to measure for each circuit.

        Yields:
            For each circuit (in order), a dictionary mapping outcome bitstrings (binary)
            to their counts.
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
    def qasm2(self) -> str:
        r"""The QASM 2.0 string that represents the circuit stored in this state emulator.
        """
        assert self.are_qubits(), \
            "The StateOperator only supports qubit systems. Please use the simulator backend such as default simulator."

        qasm = [op.qasm2 for op in self._data]

        header = 'OPENQASM 2.0;\ninclude "qelib1.inc";'
        qreg = f"qreg q[{self.num_qubits}];\n"
        return '\n'.join([header, qreg] + qasm)
    
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
            raw_qasm += '\n' + qasm_recover_op
        process_qasm = _measure_qasm2(raw_qasm, qubits_idx, self.num_qubits)
        raw_result = self._execute(process_qasm, shots)

        return {
            digits: (num_occur, obs_vals[int(digits, base=2)])
            for digits, num_occur in raw_result.items()
        }
    
    def _measure_probabilities(self, qubits_idx: List[int], shots: int, pauli_str: str) -> torch.Tensor:
        r"""Measure the quantum state and return the probabilities of each measurement outcome.
        
        Args:
            qubits_idx: the qubit indices to be measured. Defaults to all qubits.
            shots: the number of shots to measure, defaults to 1024.
            pauli_str: the Pauli basis to measure, can be a string composed of 'i', 'x', 'y', or 'z'.

        Returns:
            A dictionary storing the probability of each measurement outcome.
            The keys are the outcome digits in binary string, and the values are probabilities.

        """
        num_outcomes = 2 ** len(qubits_idx)
        probs = torch.zeros(num_outcomes, dtype=get_float_dtype())

        if shots == 0:
            return probs

        measurement_result = self._measure(qubits_idx, shots, pauli_str)
        for digits, (count, _) in measurement_result.items():
            idx = int(digits, 2)
            probs[idx] = count / shots
        return probs
    
    def measure(self, system_idx: Optional[Union[int, List[int]]] = None, 
                shots: Optional[int] = None, measure_op: Optional[str] = None) -> torch.Tensor:
        r"""Measure the quantum state
        
        Args:
            system_idx: the system indices to be measured. Defaults to all systems.
            shots: the number of shots to measure, defaults to 1024.
            measure_op: the measurement operator basis to measure. Here we restrict it to be Pauli, 
                which can be a char or a string composed of 'i', 'x', 'y', or 'z'. Defaults to 'z'.
            
        Returns:
            A tensor containing the measurement results, where each element corresponds to empirical probabilities of the measurement outcomes.
        """
        if shots is None:
            shots = 1024
        else:
            if shots < 0:
                raise ValueError(f"The number of shots should be a non-negative integer, received {shots}.")

            system_idx = self._check_sys_idx(system_idx)
            if shots == 0:
                return torch.zeros(2 ** len(system_idx), dtype=get_float_dtype())

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

        # Build batched QASM only for non-identity terms (sites == [''] is treated as identity)
        list_qasm: List[str] = []
        list_shots_qasm: List[int] = []
        obs_vals_per_term: List[Optional[List[int]]] = [None] * num_terms

        for i in range(num_terms):
            qubits_idx, pauli_str = list_qubits_idx[i], list_pauli_str[i]
            if qubits_idx == ['']:
                continue  # no submission; identity term contributes deterministically
            qasm_recover_op, obs_vals = _convert_basis(qubits_idx, pauli_str)
            obs_vals_per_term[i] = obs_vals
            process_qasm = _measure_qasm2(
                raw_qasm + '\n' + qasm_recover_op if qasm_recover_op else raw_qasm,
                qubits_idx,
                num_qubits,
            )
            list_qasm.append(process_qasm)
            list_shots_qasm.append(int(list_shots[i].item()))

        meas_iter = self._multi_execute(list_qasm, list_shots_qasm) if list_qasm else iter(())

        sum_per_term: List[float] = []
        for i in range(num_terms):
            qubits_idx, coef, shots_per_term = list_qubits_idx[i], list_coef[i], int(list_shots[i].item())
            if qubits_idx == ['']:
                sum_per_term.append(float(coef) * shots_per_term)
                continue

            measurement_result = next(meas_iter)
            obs_vals = obs_vals_per_term[i]
            assert obs_vals is not None

            val = 0.0
            for digits, num_occur in measurement_result.items():
                val += num_occur * obs_vals[int(digits, 2)]

            sum_per_term.append(float(coef) * val)

        sum_per_term_t = torch.tensor(sum_per_term, dtype=get_float_dtype())
        list_shots_t = list_shots.to(dtype=get_float_dtype())

        expec_val_per_term = torch.zeros_like(sum_per_term_t)
        valid_indices = list_shots_t > 0
        expec_val_per_term[valid_indices] = sum_per_term_t[valid_indices] / list_shots_t[valid_indices]
        return expec_val_per_term
    
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
        min_required_shots = 1 / list_abs_coef[list_abs_coef != 0].min() * 10
        if shots < min_required_shots:
            raise ValueError(
                f"Insufficient shots {shots} for Hamiltonian with coefficients {list_coef}.")
            
        pauli_weight = list_abs_coef / list_abs_coef.sum()
        pauli_sum = list_abs_coef.sum()
        list_shots = torch.from_numpy(np.random.multinomial(shots, pauli_weight))
        
        expec_val_per_term = _compute_expec_val_per_term(self, hamiltonian, list_shots, *self.param)
        if decompose:
            return expec_val_per_term
        
        sum_per_term = expec_val_per_term * list_shots
        factor_per_term = torch.zeros_like(sum_per_term)
        valid_indices = list_coef != 0
        factor_per_term[valid_indices] = pauli_sum / list_abs_coef[valid_indices]
        return (sum_per_term * factor_per_term).sum() / shots

    def _evaluate_for_grad(
        self, idx: int, param_idx: Tuple[int, int], shift: float,
        eval_func: Callable[['StateOperator'], torch.Tensor]
    ) -> torch.Tensor:
        r"""Helper function to evaluate quantum circuit at shifted parameter values.
        """
        i, j = param_idx
        with torch.no_grad():
            orig_val = float(self._data[idx]['param'][i, 0, j].item())
            self._data[idx]['param'][i, 0, j] = orig_val + shift
        result = eval_func(self)
        with torch.no_grad():
            self._data[idx]['param'][i, 0, j] = orig_val
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

        qasm_recover_op, _ = _convert_basis(qubits_idx, pauli_str)

        # Build the full submission list (multi-submit first)
        list_qasm_total: List[str] = []
        list_shots_total: List[int] = []

        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue
            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, _coeffs = get_shift_rule(op_info['name'])
                with torch.no_grad():
                    orig_val = float(state._data[idx]['param'][i, 0, j].item())

                for shift in shifts:
                    with torch.no_grad():
                        state._data[idx]['param'][i, 0, j] = orig_val + shift

                    raw_qasm = state.qasm2
                    if qasm_recover_op:
                        raw_qasm += '\n' + qasm_recover_op
                    process_qasm = _measure_qasm2(raw_qasm, qubits_idx, num_qubits)

                    list_qasm_total.append(process_qasm)
                    list_shots_total.append(shots)

                with torch.no_grad():
                    state._data[idx]['param'][i, 0, j] = orig_val  # restore

        # If nothing to evaluate, return Nones for all inputs
        if len(list_qasm_total) == 0:
            num_param_inputs = len(state.param)  # matches *params passed to .apply(...)
            return (None, None, None, None, *([None] * num_param_inputs))

        # Fetch results as they are yielded (in order of submission)
        meas_iter = state._multi_execute(list_qasm_total, list_shots_total)

        list_param_grad: List[Optional[torch.Tensor]] = []
        num_outcomes = 2 ** len(qubits_idx)
        dtype = get_float_dtype()

        # Second pass in the same nested-loop order, consuming the iterator on the fly
        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                list_param_grad.append(None)
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_grad = torch.zeros_like(param_tensor)

            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, coeffs = get_shift_rule(op_info['name'])
                expec_val_derivative = torch.zeros(num_outcomes, dtype=dtype)

                for s in range(len(shifts)):
                    meas_dict = next(meas_iter)
                    probs = torch.zeros(num_outcomes, dtype=dtype)
                    if shots != 0:
                        for digits, count in meas_dict.items():
                            probs[int(digits, 2)] = count / shots
                    expec_val_derivative += coeffs[s] * probs

                param_grad[i, 0, j] = (expec_val_derivative * grad_output).sum()

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

        num_terms = hamiltonian.n_terms
        list_qubits_idx = hamiltonian.sites
        list_coef = torch.tensor(hamiltonian.coefficients, dtype=get_float_dtype())

        # Precompute basis-recovery ops and observable value maps per term
        qasm_recover_ops_per_term: List[str] = []
        list_obs_vals_per_term: List[Optional[List[int]]] = []
        for t in range(num_terms):
            qubits_idx, pauli_str = list_qubits_idx[t], hamiltonian.pauli_words_r[t]
            if qubits_idx == ['']:
                qasm_recover_ops_per_term.append('')
                list_obs_vals_per_term.append(None)
            else:
                qasm_recover_op, obs_vals = _convert_basis(qubits_idx, pauli_str)
                qasm_recover_ops_per_term.append(qasm_recover_op)
                list_obs_vals_per_term.append(obs_vals)

        # Build full submission list: for each (param element, shift) enqueue all non-identity terms
        list_qasm_total: List[str] = []
        list_shots_total: List[int] = []

        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, _coeffs = get_shift_rule(op_info['name'])
                with torch.no_grad():
                    orig_val = float(state._data[idx]['param'][i, 0, j].item())

                for shift in shifts:
                    with torch.no_grad():
                        state._data[idx]['param'][i, 0, j] = orig_val + shift

                    raw_qasm = state.qasm2
                    for t in range(num_terms):
                        if list_qubits_idx[t] == ['']:
                            continue  # identity term: no submission
                        qasm_recover_op = qasm_recover_ops_per_term[t]
                        process_qasm = _measure_qasm2(
                            raw_qasm + '\n' + qasm_recover_op if qasm_recover_op else raw_qasm,
                            list_qubits_idx[t],
                            num_qubits,
                        )
                        list_qasm_total.append(process_qasm)
                        list_shots_total.append(int(list_shots[t].item()))

                with torch.no_grad():
                    state._data[idx]['param'][i, 0, j] = orig_val  # restore

        if len(list_qasm_total) == 0:
            num_param_inputs = len(state.param)
            return (None, None, None, *([None] * num_param_inputs))

        meas_iter = state._multi_execute(list_qasm_total, list_shots_total)

        list_param_grad: List[Optional[torch.Tensor]] = []
        dtype = get_float_dtype()

        # Consume results in the exact same order we enqueued them
        for idx, op_info in enumerate(state._data):
            if 'param' not in op_info:
                continue

            param_tensor = op_info['param']
            if not param_tensor.requires_grad:
                list_param_grad.append(None)
                continue

            num_op, num_acted_param = param_tensor.shape[0], param_tensor.shape[-1]
            param_grad = torch.zeros_like(param_tensor)

            for i, j in itertools.product(range(num_op), range(num_acted_param)):
                shifts, coeffs = get_shift_rule(op_info['name'])
                expec_vals_per_shift: List[torch.Tensor] = []

                for s in range(len(shifts)):
                    vals: List[float] = []
                    for t in range(num_terms):
                        if list_qubits_idx[t] == ['']:
                            vals.append(float(list_coef[t].item()))
                            continue

                        meas_dict = next(meas_iter)
                        obs_vals = list_obs_vals_per_term[t]
                        assert obs_vals is not None

                        acc = 0.0
                        for digits, count in meas_dict.items():
                            acc += count * obs_vals[int(digits, 2)]

                        shots_t = float(list_shots[t].item())
                        if shots_t == 0.0:
                            vals.append(0.0)
                        else:
                            vals.append(float(list_coef[t].item()) * (acc / shots_t))

                    expec_vals_per_shift.append(torch.tensor(vals, dtype=dtype))

                weighted = sum(coeffs[k] * expec_vals_per_shift[k] for k in range(len(shifts)))
                param_grad[i, 0, j] = (weighted * grad_output).sum()

            list_param_grad.append(param_grad)

        return None, None, None, *list_param_grad


# Initialize shift rules for OpenQASM 2.0. gates, see [Quantum, 2022, 6: 677]
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
DELTA = 1e-2  # Default shift value for finite difference approximation


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
        
