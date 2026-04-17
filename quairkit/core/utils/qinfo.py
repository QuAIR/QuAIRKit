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


import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import importlib

from .. import utils


def _as_batch_square(mat: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mat, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(mat)}")
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)
    if mat.ndim != 3:
        raise ValueError(f"{name} must have shape [B, d, d] (or [d, d]), got {list(mat.shape)}")
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f"{name} must be square on last 2 dims, got {list(mat.shape)}")
    return mat


def _as_batch_ket(ket: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(ket, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(ket)}")
    if ket.ndim == 2:
        ket = ket.unsqueeze(0)
    if ket.ndim != 3:
        raise ValueError(f"{name} must have shape [B, d, 1] (or [d, 1]), got {list(ket.shape)}")
    if ket.shape[-1] != 1:
        raise ValueError(f"{name} must have last dim 1, got {list(ket.shape)}")
    return ket


def _as_batch_unitary_2x2(unitary: torch.Tensor) -> torch.Tensor:
    unitary = _as_batch_square(unitary, "unitary")
    if unitary.shape[-2:] != (2, 2):
        raise ValueError(f"unitary must have last dims (2,2), got {list(unitary.shape)}")
    return unitary


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


_CPP_QINFO = _require_cpp_submodule("qinfo")


def _choi_to_kraus(choi_repr: torch.Tensor, tol: float) -> List[torch.Tensor]:
    if not isinstance(choi_repr, torch.Tensor):
        raise TypeError(f"choi_repr must be torch.Tensor, got {type(choi_repr)}")
    if choi_repr.ndim < 2 or choi_repr.shape[-1] != choi_repr.shape[-2]:
        raise ValueError(f"choi_repr must have shape [..., d^2, d^2], got {list(choi_repr.shape)}")
    return _CPP_QINFO.choi_to_kraus(choi_repr, tol)


def _kraus_to_stinespring(kraus_repr: List[torch.Tensor]) -> torch.Tensor:
    r"""Transform the Kraus representation to the Stinespring representation."""
    if isinstance(kraus_repr, (list, tuple)):
        kraus_repr = torch.stack(kraus_repr)
    if kraus_repr.ndim < 3:
        raise ValueError(f"kraus_repr must have shape [..., r, d, d], got {list(kraus_repr.shape)}")
    return _CPP_QINFO.kraus_to_stinespring(kraus_repr)


def _choi_to_stinespring(choi_repr: torch.Tensor, tol: float) -> List[torch.Tensor]:
    return _kraus_to_stinespring(_choi_to_kraus(choi_repr, tol))


def _kraus_to_choi(kraus_repr: List[torch.Tensor]) -> torch.Tensor:
    r"""Transform the Kraus representation to the Choi representation."""
    if isinstance(kraus_repr, (list, tuple)):
        kraus_repr = torch.stack(kraus_repr)
    if kraus_repr.ndim < 3:
        raise ValueError(f"kraus_repr must have shape [..., r, d, d], got {list(kraus_repr.shape)}")
    return _CPP_QINFO.kraus_to_choi(kraus_repr)


def _stinespring_to_kraus(stinespring_repr: torch.Tensor) -> torch.Tensor:
    r"""Transform the Stinespring representation to the Kraus representation."""
    if not isinstance(stinespring_repr, torch.Tensor):
        raise TypeError(f"stinespring_repr must be torch.Tensor, got {type(stinespring_repr)}")
    if stinespring_repr.ndim < 2:
        raise ValueError(f"stinespring_repr must have shape [..., r*d_out, d_in], got {list(stinespring_repr.shape)}")
    return _CPP_QINFO.stinespring_to_kraus(stinespring_repr)


def _stinespring_to_choi(stinespring_repr: torch.Tensor) -> torch.Tensor:
    return _kraus_to_choi(_stinespring_to_kraus(stinespring_repr))


def _trace_distance_pp(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    psi = _as_batch_ket(psi, "psi")
    phi = _as_batch_ket(phi, "phi")
    return _CPP_QINFO.trace_distance_pp(psi, phi)


def _trace_distance_pm(psi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    psi = _as_batch_ket(psi, "psi")
    rho = _as_batch_square(rho, "rho")
    return _CPP_QINFO.trace_distance_pm(psi, rho)


def _trace_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    rho = _as_batch_square(rho, "rho")
    sigma = _as_batch_square(sigma, "sigma")
    return _CPP_QINFO.trace_distance(rho, sigma)


def _state_fidelity_pp(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    psi = _as_batch_ket(psi, "psi")
    phi = _as_batch_ket(phi, "phi")
    return _CPP_QINFO.state_fidelity_pp(psi, phi)


def _state_fidelity_pm(psi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    psi = _as_batch_ket(psi, "psi")
    rho = _as_batch_square(rho, "rho")
    return _CPP_QINFO.state_fidelity_pm(psi, rho)


def _state_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    rho = _as_batch_square(rho, "rho")
    sigma = _as_batch_square(sigma, "sigma")
    return _CPP_QINFO.state_fidelity(rho, sigma)


def _gate_fidelity(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    U = _as_batch_square(U, "U")
    V = _as_batch_square(V, "V")
    if U.shape != V.shape:
        raise ValueError(f"U and V must have the same shape, got {list(U.shape)} and {list(V.shape)}")
    return _CPP_QINFO.gate_fidelity(U, V)

def _purity(rho: torch.Tensor) -> torch.Tensor:
    rho = _as_batch_square(rho, "rho")
    return _CPP_QINFO.purity(rho)


def _von_neumann_entropy(rho: torch.Tensor, base: Optional[Union[int, float]] = 2) -> torch.Tensor:
    rho = _as_batch_square(rho, "rho")
    entropy = -1 * math.log(math.e, base) * utils.linalg._trace(rho @ utils.linalg._logm(rho)).real
    return entropy


def _relative_entropy(rho: torch.Tensor, sig: torch.Tensor, base: Optional[Union[int, float]] = 2) -> torch.Tensor:
    rho = _as_batch_square(rho, "rho")
    sig = _as_batch_square(sig, "sig")
    if rho.shape != sig.shape:
        raise ValueError(f"rho and sig must have the same shape, got {list(rho.shape)} and {list(sig.shape)}")
    entropy = math.log(math.e, base) * utils.linalg._trace(rho @ utils.linalg._logm(rho) - rho @ utils.linalg._logm(sig)).real
    return entropy


def _negativity(density_op: torch.Tensor) -> torch.Tensor:
    density_op = _as_batch_square(density_op, "density_op")
    return _CPP_QINFO.negativity(density_op)


def _logarithmic_negativity(density_op: torch.Tensor) -> torch.Tensor:
    density_op = _as_batch_square(density_op, "density_op")
    return _CPP_QINFO.logarithmic_negativity(density_op)


def _diamond_norm(choi_matrix: torch.Tensor, dim_io: Optional[Union[int, Tuple[int, int]]] = None, **kwargs) -> float:
    import cvxpy

    if dim_io is None:
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

    rho = cvxpy.Variable(shape=(dim_in, dim_in), complex=True)
    omega = cvxpy.Variable(shape=(kron_size, kron_size), complex=True)
    identity = np.eye(dim_out)

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
    unitary = _as_batch_unitary_2x2(unitary)
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
    unitary = _as_batch_unitary_2x2(unitary)
    alpha, beta, gamma, delta = _decomp_1qubit(unitary, return_global=True)

    def _ry_batch(theta: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        row0 = torch.stack([c, -s], dim=-1)
        row1 = torch.stack([s, c], dim=-1)
        return torch.stack([row0, row1], dim=-2).to(dtype=unitary.dtype)

    def _rz_batch(theta: torch.Tensor) -> torch.Tensor:
        e1 = torch.exp(-1j * theta / 2)
        e2 = torch.exp(1j * theta / 2)
        z0 = torch.zeros_like(e1)
        row0 = torch.stack([e1, z0], dim=-1)
        row1 = torch.stack([z0, e2], dim=-1)
        return torch.stack([row0, row1], dim=-2).to(dtype=unitary.dtype)

    A = torch.matmul(_rz_batch(beta), _ry_batch(gamma / 2))
    B = torch.matmul(_ry_batch(-gamma / 2), _rz_batch(-(delta + beta) / 2))
    C = _rz_batch((delta - beta) / 2)

    return alpha, A, B, C


def _stab_renyi(
    density: torch.Tensor,
    alpha: torch.Tensor,
    num: int,
    indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    density = density.to(torch.complex128)


    Chi_func = (
        utils.linalg._trace(
            density.unsqueeze(1) @ pauli.unsqueeze(0), axis1=-2, axis2=-1
        ).real
    ) ** 2 / (
        2**n
    )

    def _compute_p_norm_vector(vec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        p_norm = vec.abs().pow(p).sum(dim=1).pow(1 / p)
        return p_norm

    alpha_norm = _compute_p_norm_vector(Chi_func, alpha)

    result_counts = torch.full((num,), float("nan"), dtype=torch.float)

    result_counts[indices] = alpha_norm.to(result_counts.dtype)

    return alpha / (1 - alpha) * torch.log2(result_counts) - math.log2(2**n)


def _stab_nullity(
    unitary: torch.Tensor,
    num_unitary: int,
    unitary_indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    unitary_expanded = unitary.unsqueeze(1).unsqueeze(
        2
    )
    unitary_d_expanded = (
        utils.linalg._dagger(unitary).unsqueeze(1).unsqueeze(2)
    )
    pauli_expanded_i = pauli.unsqueeze(0).unsqueeze(2)
    pauli_expanded_k = pauli.unsqueeze(0).unsqueeze(1)

    paulifunc = utils.linalg._trace(
        pauli_expanded_i @ unitary_expanded @ pauli_expanded_k @ unitary_d_expanded,
        axis1=-2,
        axis2=-1,
    ).real / (
        2**n
    )

    tolerance = 1e-6
    mask = (torch.abs(paulifunc - 1) < tolerance) | (
        torch.abs(paulifunc + 1) < tolerance
    )

    counts = torch.sum(mask, dim=(-2, -1), dtype=torch.float)

    result_counts = torch.full((num_unitary,), float("nan"), dtype=torch.float)

    result_counts[unitary_indices] = counts

    return 2 * n - torch.log2(result_counts)


def _mana_state(state: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    state = _as_batch_square(state, "state").to(torch.complex128)
    W = 1 / dim * utils.linalg._trace(state.unsqueeze(1) @ A.unsqueeze(0)).real

    return torch.log2((torch.abs(W).sum(dim=-1)))


def _mana_channel(
    channel: torch.Tensor,
    A_a: torch.Tensor,
    A_b: torch.Tensor,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    channel = _as_batch_square(channel, "channel").to(torch.complex128)
    A_kron = torch.einsum("aij,bkl->abikjl", A_a.transpose(1, 2), A_b).reshape(
        in_dim**2, out_dim**2, in_dim * out_dim, in_dim * out_dim
    )
    W = (
        1
        / out_dim
        * utils.linalg._trace(
            channel.unsqueeze(1).unsqueeze(2) @ A_kron.unsqueeze(0)
        ).real
    )
    return torch.log2(torch.max(torch.sum(torch.abs(W), dim=-1), dim=-1).values)


def _stab_renyi(
    density: torch.Tensor,
    alpha: torch.Tensor,
    num: int,
    indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    density = density.to(torch.complex128)


    Chi_func = (
        utils.linalg._trace(
            density.unsqueeze(1) @ pauli.unsqueeze(0), axis1=-2, axis2=-1
        ).real
    ) ** 2 / (
        2**n
    )

    def _compute_p_norm_vector(vec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        p_norm = vec.abs().pow(p).sum(dim=1).pow(1 / p)
        return p_norm

    alpha_norm = _compute_p_norm_vector(Chi_func, alpha)

    result_counts = torch.full((num,), float("nan"), dtype=torch.float)

    result_counts[indices] = alpha_norm.to(result_counts.dtype)

    return alpha / (1 - alpha) * torch.log2(result_counts) - math.log2(2**n)


def _stab_nullity(
    unitary: torch.Tensor,
    num_unitary: int,
    unitary_indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    unitary_expanded = unitary.unsqueeze(1).unsqueeze(
        2
    )
    unitary_d_expanded = (
        utils.linalg._dagger(unitary).unsqueeze(1).unsqueeze(2)
    )
    pauli_expanded_i = pauli.unsqueeze(0).unsqueeze(2)
    pauli_expanded_k = pauli.unsqueeze(0).unsqueeze(1)

    paulifunc = utils.linalg._trace(
        pauli_expanded_i @ unitary_expanded @ pauli_expanded_k @ unitary_d_expanded,
        axis1=-2,
        axis2=-1,
    ).real / (
        2**n
    )

    tolerance = 1e-6
    mask = (torch.abs(paulifunc - 1) < tolerance) | (
        torch.abs(paulifunc + 1) < tolerance
    )

    counts = torch.sum(mask, dim=(-2, -1), dtype=torch.float)

    result_counts = torch.full((num_unitary,), float("nan"), dtype=torch.float)

    result_counts[unitary_indices] = counts

    return 2 * n - torch.log2(result_counts)


def _mana_state(state: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    state = state.to(torch.complex128)
    state = state.unsqueeze(0) if state.dim() == 2 else state
    W = 1 / dim * utils.linalg._trace(state.unsqueeze(1) @ A.unsqueeze(0)).real

    return torch.log2((torch.abs(W).sum(dim=-1)))


def _mana_channel(
    channel: torch.Tensor,
    A_a: torch.Tensor,
    A_b: torch.Tensor,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    channel = channel.to(torch.complex128)
    A_kron = torch.einsum("aij,bkl->abikjl", A_a.transpose(1, 2), A_b).reshape(
        in_dim**2, out_dim**2, in_dim * out_dim, in_dim * out_dim
    )
    channel = channel.unsqueeze(0) if channel.dim() == 2 else channel
    W = (
        1
        / out_dim
        * utils.linalg._trace(
            channel.unsqueeze(1).unsqueeze(2) @ A_kron.unsqueeze(0)
        ).real
    )
    return torch.log2(torch.max(torch.sum(torch.abs(W), dim=-1), dim=-1).values)

def _general_state_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    trace_norm_term = _state_fidelity(rho, sigma)
    general_fidelity = (trace_norm_term + torch.sqrt(
        (1 - utils.linalg._trace(rho)) * (1 - utils.linalg._trace(sigma))
    )).real
    return general_fidelity.view(rho.shape[:-2] or sigma.shape[:-2])

def _mutual_information(rho: torch.Tensor, dim_A: int, dim_B: int) -> torch.Tensor:
    system_dim = [dim_A, dim_B]
    rho_A = utils.linalg._partial_trace(rho, [1], system_dim)
    rho_B = utils.linalg._partial_trace(rho, [0], system_dim)
    return _von_neumann_entropy(rho_A) + _von_neumann_entropy(rho_B) - _von_neumann_entropy(rho)


def _link(
        JE: Tuple[torch.Tensor, str, List[int], List[int]],
        JF: Tuple[torch.Tensor, str, List[int], List[int]]
        ) -> Tuple[torch.Tensor, str, List[int], List[int]]:
    JE_matrix, JE_entry_exit, JE_input_dims, JE_output_dims = JE
    JF_matrix, JF_entry_exit, JF_input_dims, JF_output_dims = JF

    JE_entry, JE_exit = JE_entry_exit.split('->')
    JF_entry, JF_exit = JF_entry_exit.split('->')

    overlap_subsystem = set(JE_exit).intersection(set(JF_entry))

    new_index = list(range(len(JE_entry) + len(JE_exit)))
    overlap_indices = [JE_exit.index(x) + len(JE_entry) for x in overlap_subsystem]
    exchange_indices = list(range(len(overlap_subsystem)))

    for old, new in zip(overlap_indices, exchange_indices):
        new_index[new], new_index[old] = new_index[old], new_index[new]

    JE_choi_permuted = utils.linalg._permute_systems(
        JE_matrix, 
        new_index, 
        dim_list=JE_input_dims + JE_output_dims
    )

    JE_transposed = utils.linalg._partial_transpose(
        JE_choi_permuted, [0],
        [2 ** len(overlap_subsystem), JE_choi_permuted.shape[0] // (2 ** len(overlap_subsystem))]
    )

    JE_choi_transposed = utils.linalg._permute_systems(
        JE_transposed, 
        new_index, 
        JE_input_dims + JE_output_dims
    )

    JE_pairs = list(zip(JE_entry, JE_input_dims)) + list(zip(JE_exit, JE_output_dims))
    JF_pairs = list(zip(JF_entry, JF_input_dims)) + list(zip(JF_exit, JF_output_dims))

    JE_dim_dict = dict(JE_pairs)
    JF_dim_dict = dict(JF_pairs)

    for letter in overlap_subsystem:
        JE_dim_dict.pop(letter, None)
        JF_dim_dict.pop(letter, None)

    non_overlap_dims_E = list(JE_dim_dict.values())
    non_overlap_dims_F = list(JF_dim_dict.values())

    multiplication = (
        torch.kron(JE_choi_transposed.contiguous(), torch.eye(np.prod(non_overlap_dims_F))) 
        @ torch.kron(torch.eye(np.prod(non_overlap_dims_E)), JF_matrix.contiguous())
    )

    total_str = (
        JE_entry 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + JF_entry 
        + JF_exit
    )
    
    index_dict = {char: idx for idx, char in enumerate(total_str)}
    for letter in overlap_subsystem:
        index_dict.pop(letter, None)

    result_matrix = utils.linalg._partial_trace_discontiguous(
        multiplication, 
        list(index_dict.values())
    )

    combined_str = (
        JE_entry 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + ''.join([c for c in JF_entry if c not in overlap_subsystem]) 
        + JF_exit
    )
    index_nonoverlap_dict = {char: idx for idx, char in enumerate(combined_str)}

    permute_str = (
        JE_entry 
        + ''.join([c for c in JF_entry if c not in overlap_subsystem]) 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + JF_exit
    )
    permute_list = [index_nonoverlap_dict[char] for char in permute_str]

    all_dims = non_overlap_dims_E + non_overlap_dims_F
    permute_dims = [all_dims[i] for i in permute_list]

    result_matrix = utils.linalg._permute_systems(result_matrix, permute_list, permute_dims)

    entry_exit = (
        JE_entry 
        + ''.join([c for c in JF_entry if c not in overlap_subsystem]) 
        + '->' 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + JF_exit
    )

    entry, exit = entry_exit.split('->')

    entry_index = [index_nonoverlap_dict[char] for char in entry]
    input_dims = [all_dims[i] for i in entry_index]

    exit_index = [index_nonoverlap_dict[char] for char in exit]
    output_dims = [all_dims[i] for i in exit_index]

    return result_matrix, entry_exit, input_dims, output_dims
