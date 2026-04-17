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
Built-in Gate matrices.
"""

import importlib
from typing import Iterable, List, Union

import torch

from .. import utils


def _require_cpp_submodule(submodule: str):
    """Return a required C++ extension submodule.

    This module no longer supports a Python fallback implementation for kernels
    that have C++ equivalents. If the C++ extension is not available, raise an
    ImportError with actionable guidance.
    """
    try:
        mod = importlib.import_module("quairkit._C")
    except Exception as e:
        raise ImportError(
            "QuAIRKit requires the compiled C++ extension (quairkit._C). "
            "Please build it first (e.g. `python setup.py build_ext --inplace`)."
        ) from e
    cpp_mod = getattr(mod, submodule, None)
    if cpp_mod is None:
        raise ImportError(
            f"quairkit._C is available but missing submodule '{submodule}'. "
            "Please rebuild the C++ extension (e.g. `python setup.py build_ext --inplace`)."
        )
    return cpp_mod


_CPP_MATRIX = _require_cpp_submodule("matrix")


def __get_complex_dtype(float_dtype: torch.dtype) -> torch.dtype:
    if float_dtype == torch.float32:
        complex_dtype = torch.complex64
    elif float_dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(
            f"The dtype should be torch.float32 or torch.float64: received {float_dtype}")
    return complex_dtype


def _zero(dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    return torch.tensor([0], dtype=dtype, device=device)


def _one(dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    return torch.tensor([1], dtype=dtype, device=device)


def _phase(dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.phase(dim, dtype)


def _shift(dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.shift(dim, dtype)


def _grover(oracle: torch.Tensor) -> torch.Tensor:
    complex_dtype = oracle.dtype
    dimension = oracle.shape[0]
    ket_zero = torch.eye(dimension, 1).to(complex_dtype)

    diffusion_op = (2 + 0j) * ket_zero @ ket_zero.T - torch.eye(dimension)
    reflection_op = torch.kron(_z(), torch.eye(dimension // 2)).to(complex_dtype)
    
    return oracle @ diffusion_op @ oracle.conj().T @ reflection_op


def _qft(N: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.qft(N, dtype)


def _h(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.h(dtype)


def _s(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.s(dtype)


def _sdg(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.sdg(dtype)


def _t(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.t(dtype)


def _tdg(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.tdg(dtype)


def _eye(dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.eye(dim, dtype)


def _x(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.x(dtype)


def _y(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.y(dtype)


def _z(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.z(dtype)


def _p(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.p(theta)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.rx(theta)


def _ry(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.ry(theta)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.rz(theta)


def _u3(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError(f"theta must have shape [B, 3], got {list(theta.shape)}")
    return _CPP_MATRIX.u3(theta)







def _cnot(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.cnot(dtype)


def _cy(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.cy(dtype)


def _cz(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.cz(dtype)


def _swap(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.swap(dtype)



def _cp(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.cp(theta)


def _crx(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.crx(theta)


def _cry(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.cry(theta)


def _crz(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.crz(theta)


def _cu(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 4:
        raise ValueError(f"theta must have shape [B, 4], got {list(theta.shape)}")
    return _CPP_MATRIX.cu(theta)


def _rxx(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.rxx(theta)


def _ryy(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.ryy(theta)


def _rzz(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError(f"theta must have shape [B, 1], got {list(theta.shape)}")
    return _CPP_MATRIX.rzz(theta)


def _ms(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.ms(dtype)


def _cswap(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.cswap(dtype)


def _toffoli(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return _CPP_MATRIX.toffoli(dtype)


def _universal2(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 15:
        raise ValueError(f"theta must have shape [B, 15], got {list(theta.shape)}")
    complex_dtype = __get_complex_dtype(theta.dtype)
    I2 = _eye(2, complex_dtype)
    cnot_01 = _cnot(complex_dtype)
    swap = _swap(complex_dtype)
    cnot_10 = swap @ cnot_01 @ swap

    gates = [
        utils.linalg._kron(_u3(theta[:, [0, 1, 2]]), I2),
        utils.linalg._kron(I2, _u3(theta[:, [3, 4, 5]])),
        cnot_10,
        utils.linalg._kron(_rz(theta[:, [6]]), I2),
        utils.linalg._kron(I2, _ry(theta[:, [7]])),
        cnot_01,
        utils.linalg._kron(I2, _ry(theta[:, [8]])),
        cnot_10,
        utils.linalg._kron(_u3(theta[:, [9, 10, 11]]), I2),
        utils.linalg._kron(I2, _u3(theta[:, [12, 13, 14]])),
    ]

    unitary = gates[0]
    for gate in gates[1:]:
        unitary = torch.matmul(gate, unitary)

    return unitary


def _universal3(theta: torch.Tensor) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        raise TypeError(f"theta must be torch.Tensor, got {type(theta)}")
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    if theta.ndim != 2 or theta.shape[1] != 81:
        raise ValueError(f"theta must have shape [B, 81], got {list(theta.shape)}")
    complex_dtype = __get_complex_dtype(theta.dtype)
    I2, I4 = _eye(2, complex_dtype), _eye(4, complex_dtype)
    h, s = _h(complex_dtype), _s(complex_dtype)
    cnot = _cnot(complex_dtype)

    num_qubits = 3

    def _embed_gate(gate: torch.Tensor, qubit_idx: Union[List[int], int]) -> torch.Tensor:
        if not isinstance(qubit_idx, Iterable) or isinstance(qubit_idx, str):
            qubit_idx = [qubit_idx]
        perm = list(qubit_idx) + [q for q in range(num_qubits) if q not in qubit_idx]
        perm_mat = _permutation(perm, [2] * num_qubits).to(gate.dtype).to(gate.device)
        rest_dim = 2 ** (num_qubits - len(qubit_idx))
        eye_rest = torch.eye(rest_dim, dtype=gate.dtype, device=gate.device)
        full_gate = utils.linalg._kron(gate, eye_rest)
        return perm_mat.T @ full_gate @ perm_mat

    cnot01_3q = _embed_gate(cnot, [0, 1])
    cnot12_3q = _embed_gate(cnot, [1, 2])
    cnot10_3q = _embed_gate(cnot, [1, 0])
    cnot02_3q = _embed_gate(cnot, [0, 2])
    cnot20_3q = _embed_gate(cnot, [2, 0])
    cnot21_3q = _embed_gate(cnot, [2, 1])

    psi = torch.reshape(theta[:, :60], shape=[-1, 4, 15])
    phi = torch.reshape(theta[:, 60:], shape=[-1, 7, 3])

    def __block_u(_unitary, _theta):
        gates = [
            cnot12_3q,
            _embed_gate(_ry(_theta[:, 0]), 1),
            cnot01_3q,
            _embed_gate(_ry(_theta[:, 1]), 1),
            cnot01_3q,
            cnot12_3q,
            _embed_gate(h, 2),
            cnot10_3q,
            cnot02_3q,
            cnot12_3q,
            _embed_gate(_rz(_theta[:, 2]), 2),
            cnot12_3q,
            cnot02_3q,
        ]
        for gate in gates:
            _unitary = torch.matmul(gate, _unitary)
        return _unitary

    def __block_v(_unitary, _theta):
        gates = [
            cnot20_3q,
            cnot12_3q,
            cnot21_3q,
            _embed_gate(_ry(_theta[:, 0]), 2),
            cnot12_3q,
            _embed_gate(_ry(_theta[:, 1]), 2),
            cnot12_3q,
            _embed_gate(s, 2),
            cnot20_3q,
            cnot01_3q,
            cnot10_3q,
            _embed_gate(h, 2),
            cnot02_3q,
            _embed_gate(_rz(_theta[:, 2]), 2),
            cnot02_3q,
        ]
        for gate in gates:
            _unitary = torch.matmul(gate, _unitary)
        return _unitary

    gates = [
        _embed_gate(_universal2(psi[:, 0]), [0, 1]),
        _embed_gate(_u3(phi[:, 0, 0:3]), 2),
    ]
    unitary = gates[0]
    for gate in gates[1:]:
        unitary = torch.matmul(gate, unitary)
    unitary = __block_u(unitary, phi[:, 1])

    gates = [
        _embed_gate(_universal2(psi[:, 1]), [0, 1]),
        _embed_gate(_u3(phi[:, 2, 0:3]), 2),
    ]
    for gate in gates:
        unitary = torch.matmul(gate, unitary)
    unitary = __block_v(unitary, phi[:, 3])

    gates = [
        _embed_gate(_universal2(psi[:, 2]), [0, 1]),
        _embed_gate(_u3(phi[:, 4, 0:3]), 2),
    ]
    for gate in gates:
        unitary = torch.matmul(gate, unitary)
    unitary = __block_u(unitary, phi[:, 5])

    gates = [
        _embed_gate(_universal2(psi[:, 3]), [0, 1]),
        _embed_gate(_u3(phi[:, 6, 0:3]), 2),
    ]
    for gate in gates:
        unitary = torch.matmul(gate, unitary)

    return unitary


def _permutation(perm: List[int], system_dim: List[int]) -> torch.Tensor:
    return _CPP_MATRIX.permutation(perm, system_dim)


def _param_generator(theta: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    if theta.ndim == 1:
        theta = theta.reshape([1, -1])
    return _CPP_MATRIX.param_generator(theta, generator)
