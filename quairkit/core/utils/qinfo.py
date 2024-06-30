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
The library of functions in quantum information.
"""


import itertools
import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from .. import utils


def _choi_to_kraus(choi_repr: torch.Tensor, tol: float) -> List[torch.Tensor]:
    r"""Transform the Choi representation to the Kraus representation
    """
    ndim = int(math.sqrt(choi_repr.shape[0]))
    w, v = torch.linalg.eigh(choi_repr)

    # add abs to make eigvals safe
    w = torch.abs(w)
    l_cut = 0
    for l in range(len(w) - 1, -1, -1):
        if torch.sum(torch.abs(w[l:])) / torch.sum(torch.abs(w)) > 1 - tol:
            l_cut = l
            break
    return torch.stack([(v * torch.sqrt(w))[:, l].reshape([ndim, ndim]).T for l in range(l_cut, ndim**2)])


def _kraus_to_stinespring(kraus_repr: List[torch.Tensor]) -> torch.Tensor:
    r"""Transform the Kraus representation to the Stinespring representation
    """
    # TODO simplify logic
    j_dim = kraus_repr.shape[0]
    i_dim = kraus_repr.shape[1]
    stinespring_repr = kraus_repr.permute([1, 0, 2])
    return stinespring_repr.reshape([i_dim * j_dim, i_dim])


def _choi_to_stinespring(choi_repr: torch.Tensor, tol: float) -> List[torch.Tensor]:
    r"""Transform the Choi representation to the Stinespring representation
    """
    # TODO: need a more straightforward transformation
    return _kraus_to_stinespring(_choi_to_kraus(choi_repr, tol))


def _kraus_to_choi(kraus_repr: List[torch.Tensor]) -> torch.Tensor:
    r"""Transform the Kraus representation to the Choi representation
    """
    ndim = kraus_repr[0].shape[0]
    y = kraus_repr[0]
    kraus_oper_tensor = torch.reshape(
        torch.cat([torch.kron(x, x.conj().T.contiguous()) for x in kraus_repr]),
        shape=[len(kraus_repr), ndim, -1]
    )
    choi_repr = torch.sum(kraus_oper_tensor, axis=0).reshape([ndim for _ in range(4)]).permute([2, 1, 0, 3])
    return choi_repr.permute([0, 2, 1, 3]).reshape([ndim * ndim, ndim * ndim])


def _stinespring_to_kraus(stinespring_repr: torch.Tensor) -> torch.Tensor:
    r"""Transform the Stinespring representation to the Kraus representation
    """
    i_dim = stinespring_repr.shape[1]
    j_dim = stinespring_repr.shape[0] // i_dim
    return stinespring_repr.reshape([i_dim, j_dim, i_dim]).permute([1, 0, 2])


def _stinespring_to_choi(stinespring_repr: torch.Tensor) -> torch.Tensor:
    r"""Transform the Stinespring representation to the Choi representation
    """
    # TODO: need a more straightforward transformation
    return _kraus_to_choi(_stinespring_to_kraus(stinespring_repr))


def _trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    batch_dim = list(rho.shape[:-2]) or list(sigma.shape[:-2])
    list_eigval = torch.linalg.eigvalsh(rho - sigma)
    dist = 0.5 * torch.sum(torch.abs(list_eigval), axis=-1)
    return dist.view(rho.shape[:-2] or sigma.shape[:-2])


def _state_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sqrt_rho = utils.linalg._sqrtm(rho)
    fidelity = utils.linalg._trace(utils.linalg._sqrtm(sqrt_rho @ sigma @ sqrt_rho)).real
    return fidelity.view(rho.shape[:-2] or sigma.shape[:-2])


def _gate_fidelity(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    fidelity = torch.abs(utils.linalg._trace(U @ V.mH)) / U.shape[-1]
    return fidelity.view(U.shape[:-2] or V.shape[:-2])

def _purity(rho: torch.Tensor) -> torch.Tensor:
    product = torch.matmul(rho, rho)
    diagonal_elements = product.diagonal(dim1=-2, dim2=-1)
    purity = diagonal_elements.sum(dim=-1, keepdim=True).real
    return purity.view(rho.shape[:-2] + (1,))


def _von_neumann_entropy(rho: torch.Tensor, base: Optional[int] = 2) -> torch.Tensor:
    entropy =  -1 * math.log(math.e, 2) * utils.linalg._trace(rho @ utils.linalg._logm(rho)).real
    return entropy.view(rho.shape[:-2])


def _relative_entropy(rho: torch.Tensor, sig: torch.Tensor, base: Optional[int] = 2) -> torch.Tensor:
    entropy = math.log(math.e, base) * utils.linalg._trace(rho @ utils.linalg._logm(rho) - rho @ utils.linalg._logm(sig)).real
    return entropy.view(rho.shape[:-2] or sig.shape[:-2])


def _partial_transpose_2(
    density_op: torch.Tensor, sub_system: Optional[int] = 2
) -> torch.Tensor:
    n_qubits = int(math.log2(density_op.shape[-1]))

    if sub_system == 1:
        n = n_qubits // 2
    else:
        n = n_qubits - n_qubits // 2

    transposed_density_op = utils.linalg._partial_transpose(density_op, n)
    if sub_system == 2:
        transposed_density_op = transposed_density_op.transpose(-2, -1)
        
    return transposed_density_op


def _negativity(density_op: torch.Tensor) -> torch.Tensor:

    num_half_qubits = int(math.log2(density_op.shape[-1])) // 2
    density_op_transposed = utils.linalg._partial_transpose(density_op, num_half_qubits)
    # Calculate eigenvalues for the entire batch
    eigen_vals = torch.linalg.eigvalsh(density_op_transposed)
    # Calculate negativity for each density operator in the batch
    negativities = torch.sum(torch.where(eigen_vals <= -1e-16 * density_op.shape[-1], eigen_vals, 0.), dim=-1)
    return torch.abs(negativities).unsqueeze(-1).view(density_op.shape[:-2])


def _logarithmic_negativity(density_op: torch.Tensor) -> torch.Tensor:
    return torch.log2(2 * _negativity(density_op) + 1)


# TODO: double value & insufficient precision in matlab
def _diamond_norm(choi_matrix: torch.Tensor, dim_io: Optional[Union[int, Tuple[int, int]]] = None, **kwargs) -> float:
    import cvxpy

    if dim_io is None:    # Default to dim_in == dim_out
        dim_in = dim_out = int(math.sqrt(choi_matrix.shape[0]))
    elif isinstance(dim_io, tuple):
        dim_in = int(dim_io[0])
        dim_out = int(dim_io[1])
    elif isinstance(dim_io, int):
        dim_in = dim_io
        dim_out = dim_io
    else:
        raise TypeError('"dim_io" should be "int" or "tuple".')
    kron_size = dim_in * dim_out

    # Cost function : Trace( \Omega @ Choi_matrix )
    rho = cvxpy.Variable(shape=(dim_in, dim_in), complex=True)
    omega = cvxpy.Variable(shape=(kron_size, kron_size), complex=True)
    identity = np.eye(dim_out)

    # \rho \otimes 1 \geq \Omega
    cons_matrix = cvxpy.kron(rho, identity) - omega
    cons = [
        rho >> 0,
        rho.H == rho,
        cvxpy.trace(rho) == 1,

        omega >> 0,
        omega.H == omega,

        cons_matrix >> 0
    ]

    obj = cvxpy.Maximize(2 * cvxpy.real((cvxpy.trace(omega @ choi_matrix))))
    prob = cvxpy.Problem(obj, cons)

    return prob.solve(**kwargs)


def _create_choi_repr(
    linear_map: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    Choi_matrix = 0

    for j in range(input_dim):
        for i in range(input_dim):
            E_ij = torch.zeros(
                input_dim,
                input_dim,
                dtype=input_dtype,
            )
            E_ij[i, j] = 1
            Choi_matrix += E_ij.kron(linear_map(E_ij))
    return Choi_matrix


def _decomp_1qubit(
    unitary: torch.Tensor, return_global: bool = False
) -> Tuple[torch.Tensor, ...]:
    a00 = unitary[..., 0, 0]
    a10 = unitary[..., 1, 0]
    a11 = unitary[..., 1, 1]

    alpha = (torch.angle(a00) + torch.angle(a11)) / 2
    beta = torch.angle(a10) - torch.angle(a00)
    delta = torch.angle(a11) - torch.angle(a10)
    gamma = 2 * torch.acos(a00 / torch.exp(1j * (alpha - beta / 2 - delta / 2))).real

    if return_global:
        return alpha, beta, gamma, delta
    else:
        return beta, gamma, delta


def _decomp_ctrl_1qubit(
    unitary: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    alpha, beta, gamma, delta = _decomp_1qubit(unitary, return_global=True)

    ry = lambda theta: torch.tensor(
        [
            [torch.cos(theta / 2), -torch.sin(theta / 2)],
            [torch.sin(theta / 2), torch.cos(theta / 2)],
        ],
        dtype=unitary.dtype,
    )

    rz = lambda theta: torch.tensor(
        [[torch.exp(-1j * theta / 2), 0], [0, torch.exp(1j * theta / 2)]]
    )
    A = rz(beta) @ ry(gamma / 2)
    B = ry(-gamma / 2) @ rz(-(delta + beta) / 2)
    C = rz((delta - beta) / 2)

    return alpha, A, B, C
