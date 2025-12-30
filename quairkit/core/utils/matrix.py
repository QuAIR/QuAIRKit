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

import math
from typing import List, Union

import numpy as np
import torch

from .. import utils


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
    w = np.exp(2 * math.pi * 1j / dim)
    return torch.from_numpy(np.diag([w**i for i in range(dim)])).to(dtype)


def _shift(dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return torch.roll(torch.eye(dim), shifts=1, dims=0).to(dtype)


def _grover(oracle: torch.Tensor) -> torch.Tensor:
    complex_dtype = oracle.dtype
    dimension = oracle.shape[0]
    ket_zero = torch.eye(dimension, 1).to(complex_dtype)

    diffusion_op = (2 + 0j) * ket_zero @ ket_zero.T - torch.eye(dimension)
    reflection_op = torch.kron(_z(), torch.eye(dimension // 2)).to(complex_dtype)
    
    return oracle @ diffusion_op @ oracle.conj().T @ reflection_op


def _qft(num_qubits: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    N = 2**num_qubits
    omega_N = np.exp(1j * 2 * math.pi / N)

    qft_mat = np.ones([N, N]).astype("complex128")
    for i in range(1, N):
        for j in range(1, N):
            qft_mat[i, j] = omega_N ** ((i * j) % N)

    return torch.tensor(qft_mat / math.sqrt(N)).to(dtype)


def _h(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    element = math.sqrt(2) / 2
    gate_matrix = [
        [element, element],
        [element, -element],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _s(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0],
        [0, 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _sdg(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0],
        [0, -1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _t(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 + math.sqrt(2) / 2 * 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _tdg(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0],
        [0, math.sqrt(2) / 2 - math.sqrt(2) / 2 * 1j],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _eye(dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    return torch.eye(dim, dtype=dtype)


def _x(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [0, 1],
        [1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _y(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [0, -1j],
        [1j, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _z(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0],
        [0, -1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _p(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0, _1 = torch.zeros_like(theta), torch.ones_like(theta)
    gate_matrix = [
        _1, _0,
        _0, torch.cos(theta) + 1j * torch.sin(theta),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 2, 2])


def _rx(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    gate_matrix = [
        torch.cos(theta / 2), -1j * torch.sin(theta / 2),
        -1j * torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 2, 2])


def _ry(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    gate_matrix = [
        torch.cos(theta / 2), -torch.sin(theta / 2),
        torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 2, 2]) + 0j


def _rz(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0 = torch.zeros_like(theta)
    gate_matrix = [
        torch.exp(-1j * theta / 2), _0,
        _0, torch.exp(1j * theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 2, 2])


def _u3(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 3, 1])
    theta, phi, lam = theta[:, 0], theta[:, 1], theta[:, 2]
    gate_matrix = [
        torch.cos(theta / 2), -torch.exp(1j * lam) * torch.sin(theta / 2),
        torch.exp(1j * phi) * torch.sin(theta / 2), torch.exp(1j * (phi + lam)) * torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 2, 2])



# ------------------------------------------------- Split line -------------------------------------------------

    #Belows are multi-qubit matrices.



def _cnot(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _cy(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _cz(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _swap(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)



def _cp(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0, _1 = torch.zeros_like(theta), torch.ones_like(theta)
    gate_matrix = [
        _1, _0, _0, _0,
        _0, _1, _0, _0,
        _0, _0, _1, _0,
        _0, _0, _0, torch.cos(theta) + 1j * torch.sin(theta),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _crx(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0, _1 = torch.zeros_like(theta), torch.ones_like(theta)
    gate_matrix = [
        _1, _0, _0, _0, 
        _0, _1, _0, _0,
        _0, _0, torch.cos(theta / 2), -1j * torch.sin(theta / 2),
        _0, _0, -1j * torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _cry(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0, _1 = torch.zeros_like(theta), torch.ones_like(theta)
    gate_matrix = [
        _1, _0, _0, _0,
        _0, _1, _0, _0,
        _0, _0, torch.cos(theta / 2), -torch.sin(theta / 2),
        _0, _0, torch.sin(theta / 2), torch.cos(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4]) + 0j


def _crz(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0, _1 = torch.zeros_like(theta), torch.ones_like(theta)
    gate_matrix = [
        _1, _0, _0, _0,
        _0, _1, _0, _0,
        _0, _0, torch.cos(theta / 2) - 1j * torch.sin(theta / 2), _0,
        _0, _0, _0, torch.cos(theta / 2) + 1j * torch.sin(theta / 2),
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _cu(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 4, 1])
    _0, _1 = torch.zeros_like(theta[:, -1]), torch.ones_like(theta[:, -1])

    entry22 = torch.cos(theta[:, 0] / 2) * (torch.cos(theta[:, 3]) + 1j * torch.sin(theta[:, 3]))
    entry23 = -torch.sin(theta[:, 0] / 2) * (torch.cos(theta[:, 2] + theta[:, 3]) + 1j * torch.sin(theta[:, 2] + theta[:, 3]))
    entry32 = torch.sin(theta[:, 0] / 2) * (torch.cos(theta[:, 1] + theta[:, 3]) + 1j * torch.sin(theta[:, 1] + theta[:, 3])) 
    entry33 = torch.cos(theta[:, 0] / 2) * (
        torch.cos(theta[:, 1] + theta[:, 2] + theta[:, 3])
        + 1j * torch.sin(theta[:, 1] + theta[:, 2] + theta[:, 3])
    )
    
    gate_matrix = [
        _1, _0, _0, _0,
        _0, _1, _0, _0,
        _0, _0, entry22, entry23,
        _0, _0, entry32, entry33,
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _rxx(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0 = torch.zeros_like(theta)
    _cos = torch.cos(theta / 2)
    _sin = -1j * torch.sin(theta / 2)
    
    gate_matrix = [
        _cos, _0, _0, _sin,
        _0, _cos, _sin, _0,
        _0, _sin, _cos, _0,
        _sin, _0, _0, _cos,
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _ryy(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0 = torch.zeros_like(theta)
    
    param1 = torch.cos(theta / 2)
    param2 = -1j * torch.sin(theta / 2)
    param3 = 1j * torch.sin(theta / 2)
    
    gate_matrix = [
        param1, _0, _0, param3,
        _0, param1, param2, _0,
        _0, param2, param1, _0,
        param3, _0, _0, param1,
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _rzz(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view([-1, 1])
    _0 = torch.zeros_like(theta)
    
    param1 = torch.cos(theta / 2) - 1j * torch.sin(theta / 2)
    param2 = torch.cos(theta / 2) + 1j * torch.sin(theta / 2)
    
    gate_matrix = [
        param1, _0, _0, _0,
        _0, param2, _0, _0,
        _0, _0, param2, _0,
        _0, _0, _0, param1,
    ]
    return torch.cat(gate_matrix, dim=-1).view([-1, 4, 4])


def _ms(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    val1 = math.sqrt(2) / 2
    val2 = 1j / math.sqrt(2)
    gate_matrix = [
        [val1, 0, 0, val2],
        [0, val1, val2, 0],
        [0, val2, val1, 0],
        [val2, 0, 0, val1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _cswap(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _toffoli(dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    gate_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    return torch.tensor(gate_matrix, dtype=dtype)


def _universal2(theta: torch.Tensor) -> torch.Tensor:
    theta, complex_dtype = theta.view([-1, 15]), __get_complex_dtype(theta.dtype)
    unitary = _eye(4, complex_dtype)
    _cnot_gate = _cnot(complex_dtype)

    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(theta[:, [0, 1, 2]]), qubit_idx=0, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(theta[:, [3, 4, 5]]), qubit_idx=1, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2
    )

    unitary = utils.linalg._unitary_transformation(
        unitary, _rz(theta[:, [6]]), qubit_idx=0, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _ry(theta[:, [7]]), qubit_idx=1, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _cnot_gate, qubit_idx=[0, 1], num_qubits=2
    )

    unitary = utils.linalg._unitary_transformation(
        unitary, _ry(theta[:, [8]]), qubit_idx=1, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _cnot_gate, qubit_idx=[1, 0], num_qubits=2
    )

    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(theta[:, [9, 10, 11]]), qubit_idx=0, num_qubits=2
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(theta[:, [12, 13, 14]]), qubit_idx=1, num_qubits=2
    )

    return unitary


def _universal3(theta: torch.Tensor) -> torch.Tensor:
    theta, complex_dtype = theta.view([-1, 81]), __get_complex_dtype(theta.dtype)
    unitary = _eye(8, complex_dtype)
    __h, __s, __cnot = _h(complex_dtype), _s(complex_dtype), _cnot(complex_dtype)

    psi = torch.reshape(theta[:, :60], shape=[-1, 4, 15])
    phi = torch.reshape(theta[:, 60:], shape=[-1, 7, 3])

    def __block_u(_unitary, _theta):
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, _ry(_theta[:, 0]), qubit_idx=1, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 1], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, _ry(_theta[:, 1]), qubit_idx=1, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 1], num_qubits=3
        )

        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(_unitary, __h, qubit_idx=2, num_qubits=3)
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 0], num_qubits=3
        )

        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, _rz(_theta[:, 2]), qubit_idx=2, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 2], num_qubits=3
        )
        return _unitary

    def __block_v(_unitary, _theta):
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[2, 0], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[2, 1], num_qubits=3
        )

        _unitary = utils.linalg._unitary_transformation(
            _unitary, _ry(_theta[:, 0]), qubit_idx=2, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, _ry(_theta[:, 1]), qubit_idx=2, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 2], num_qubits=3
        )

        _unitary = utils.linalg._unitary_transformation(_unitary, __s, qubit_idx=2, num_qubits=3)
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[2, 0], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 1], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[1, 0], num_qubits=3
        )

        _unitary = utils.linalg._unitary_transformation(_unitary, __h, qubit_idx=2, num_qubits=3)
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 2], num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, _rz(_theta[:, 2]), qubit_idx=2, num_qubits=3
        )
        _unitary = utils.linalg._unitary_transformation(
            _unitary, __cnot, qubit_idx=[0, 2], num_qubits=3
        )
        return _unitary

    unitary = utils.linalg._unitary_transformation(
        unitary, _universal2(psi[:, 0]), qubit_idx=[0, 1], num_qubits=3
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(phi[:, 0, 0:3]), qubit_idx=2, num_qubits=3
    )
    unitary = __block_u(unitary, phi[:, 1])

    unitary = utils.linalg._unitary_transformation(
        unitary, _universal2(psi[:, 1]), qubit_idx=[0, 1], num_qubits=3
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(phi[:, 2, 0:3]), qubit_idx=2, num_qubits=3
    )
    unitary = __block_v(unitary, phi[:, 3])

    unitary = utils.linalg._unitary_transformation(
        unitary, _universal2(psi[:, 2]), qubit_idx=[0, 1], num_qubits=3
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(phi[:, 4, 0:3]), qubit_idx=2, num_qubits=3
    )
    unitary = __block_u(unitary, phi[:, 5])

    unitary = utils.linalg._unitary_transformation(
        unitary, _universal2(psi[:, 3]), qubit_idx=[0, 1], num_qubits=3
    )
    unitary = utils.linalg._unitary_transformation(
        unitary, _u3(phi[:, 6, 0:3]), qubit_idx=2, num_qubits=3
    )
    return unitary


def _permutation(perm: List[int], system_dim: List[int]) -> torch.Tensor:
    num_system, dimension = len(perm), np.prod(system_dim)
    mat = torch.eye(dimension).view(2 * system_dim)
    idx = perm + list(range(num_system, 2 * num_system))
    return torch.permute(mat, idx).reshape([dimension, dimension])


def _param_generator(theta: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    r"""Generate a unitary with the given parameters and generators.
    Such unitary is universal when generator forms a basis of the unitary group.
    """
    num_param = generator.shape[0]
    theta = theta.view([-1, num_param, 1, 1])
    hamiltonian = torch.sum(torch.mul(theta, generator), dim=-3)
    return torch.matrix_exp(1j * hamiltonian)
