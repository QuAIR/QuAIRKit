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
    return purity.view(rho.shape[:-2])


def _von_neumann_entropy(rho: torch.Tensor, base: Optional[Union[int, float]] = 2) -> torch.Tensor:
    entropy = -1 * math.log(math.e, base) * utils.linalg._trace(rho @ utils.linalg._logm(rho)).real
    return entropy.view(rho.shape[:-2])


def _relative_entropy(rho: torch.Tensor, sig: torch.Tensor, base: Optional[Union[int, float]] = 2) -> torch.Tensor:
    entropy = math.log(math.e, base) * utils.linalg._trace(rho @ utils.linalg._logm(rho) - rho @ utils.linalg._logm(sig)).real
    return entropy.view(rho.shape[:-2] or sig.shape[:-2])


def _negativity(density_op: torch.Tensor) -> torch.Tensor:
    half_dim = math.isqrt(density_op.shape[-1])
    density_op_transposed = utils.linalg._transpose_1(density_op, half_dim)
    
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


def _stab_renyi(
    density: torch.Tensor,
    alpha: torch.Tensor,
    num: int,
    indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    density = density.to(torch.complex128)

    # Add new dimensions for broadcasting
    # density.unsqueeze(1)   Shape: (m, 1, 2^n, 2^n)
    # pauli.unsqueeze(0)   Shape: (1, 4^n, 2^n, 2^n)

    Chi_func = (
        utils.linalg._trace(
            density.unsqueeze(1) @ pauli.unsqueeze(0), axis1=-2, axis2=-1
        ).real
    ) ** 2 / (
        2**n
    )  # shape (m,4^n)

    def _compute_p_norm_vector(vec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)  # Convert to (1, P)
        # Compute the 1/2-norm
        p_norm = vec.abs().pow(p).sum(dim=1).pow(1 / p)
        return p_norm

    alpha_norm = _compute_p_norm_vector(Chi_func, alpha)

    # Initialize a result tensor with NaN values
    result_counts = torch.full((num,), float("nan"), dtype=torch.float)

    # Fill in the counts for density matrices
    result_counts[indices] = alpha_norm.to(result_counts.dtype)

    # Compute the alpha-stabilizer renyi entropy
    return alpha / (1 - alpha) * torch.log2(result_counts) - math.log2(2**n)


def _stab_nullity(
    unitary: torch.Tensor,
    num_unitary: int,
    unitary_indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    # Expand dimensions for broadcasting
    unitary_expanded = unitary.unsqueeze(1).unsqueeze(
        2
    )  # Shape (batchsize, 1, 1, 2^n, 2^n)
    unitary_d_expanded = (
        utils.linalg._dagger(unitary).unsqueeze(1).unsqueeze(2)
    )  # Shape (batchsize, 1, 1, 2^n, 2^n)
    pauli_expanded_i = pauli.unsqueeze(0).unsqueeze(2)  # Shape (1, 4^n, 1, 2^n, 2^n)
    pauli_expanded_k = pauli.unsqueeze(0).unsqueeze(1)  # Shape (1, 1, 4^n, 2^n, 2^n)

    # Third compute the Pauli function of unitary varing Pauli
    paulifunc = utils.linalg._trace(
        pauli_expanded_i @ unitary_expanded @ pauli_expanded_k @ unitary_d_expanded,
        axis1=-2,
        axis2=-1,
    ).real / (
        2**n
    )  # Shape (m, 4^n, 4^n, 2^n, 2^n) to Shape (m, 4^n, 4^n)

    # Count the number of +1 and -1 of paulifunc for every unitary to get the s(U)
    # Define a small tolerance
    tolerance = 1e-6
    # Create the mask with tolerance
    mask = (torch.abs(paulifunc - 1) < tolerance) | (
        torch.abs(paulifunc + 1) < tolerance
    )

    # Count the occurrences of True values in the mask
    counts = torch.sum(mask, dim=(-2, -1), dtype=torch.float)

    # Initialize a result tensor with NaN values
    result_counts = torch.full((num_unitary,), float("nan"), dtype=torch.float)

    # Fill in the counts for unitary matrices
    result_counts[unitary_indices] = counts

    # Compute the unitary-stabilizer nullity
    return 2 * n - torch.log2(result_counts)


def _mana_state(state: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    state = state.to(torch.complex128)
    # Call trace function to get Wigner function
    # If state has shape (d, d), expand its dimensions to (1, d, d)
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
    # Compute the Kronecker product
    A_kron = torch.einsum("aij,bkl->abikjl", A_a.transpose(1, 2), A_b).reshape(
        in_dim**2, out_dim**2, in_dim * out_dim, in_dim * out_dim
    )
    # compute the wigner function of a quantum channel
    channel = channel.unsqueeze(0) if channel.dim() == 2 else channel
    W = (
        1
        / out_dim
        * utils.linalg._trace(
            channel.unsqueeze(1).unsqueeze(2) @ A_kron.unsqueeze(0)
        ).real
    )
    # Compute the mana of channels
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

    # Add new dimensions for broadcasting
    # density.unsqueeze(1)   Shape: (m, 1, 2^n, 2^n)
    # pauli.unsqueeze(0)   Shape: (1, 4^n, 2^n, 2^n)

    Chi_func = (
        utils.linalg._trace(
            density.unsqueeze(1) @ pauli.unsqueeze(0), axis1=-2, axis2=-1
        ).real
    ) ** 2 / (
        2**n
    )  # shape (m,4^n)

    def _compute_p_norm_vector(vec: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)  # Convert to (1, P)
        # Compute the 1/2-norm
        p_norm = vec.abs().pow(p).sum(dim=1).pow(1 / p)
        return p_norm

    alpha_norm = _compute_p_norm_vector(Chi_func, alpha)

    # Initialize a result tensor with NaN values
    result_counts = torch.full((num,), float("nan"), dtype=torch.float)

    # Fill in the counts for density matrices
    result_counts[indices] = alpha_norm.to(result_counts.dtype)

    # Compute the alpha-stabilizer renyi entropy
    return alpha / (1 - alpha) * torch.log2(result_counts) - math.log2(2**n)


def _stab_nullity(
    unitary: torch.Tensor,
    num_unitary: int,
    unitary_indices: torch.Tensor,
    pauli: torch.Tensor,
    n: int,
) -> torch.Tensor:

    # Expand dimensions for broadcasting
    unitary_expanded = unitary.unsqueeze(1).unsqueeze(
        2
    )  # Shape (batchsize, 1, 1, 2^n, 2^n)
    unitary_d_expanded = (
        utils.linalg._dagger(unitary).unsqueeze(1).unsqueeze(2)
    )  # Shape (batchsize, 1, 1, 2^n, 2^n)
    pauli_expanded_i = pauli.unsqueeze(0).unsqueeze(2)  # Shape (1, 4^n, 1, 2^n, 2^n)
    pauli_expanded_k = pauli.unsqueeze(0).unsqueeze(1)  # Shape (1, 1, 4^n, 2^n, 2^n)

    # Third compute the Pauli function of unitary varing Pauli
    paulifunc = utils.linalg._trace(
        pauli_expanded_i @ unitary_expanded @ pauli_expanded_k @ unitary_d_expanded,
        axis1=-2,
        axis2=-1,
    ).real / (
        2**n
    )  # Shape (m, 4^n, 4^n, 2^n, 2^n) to Shape (m, 4^n, 4^n)

    # Count the number of +1 and -1 of paulifunc for every unitary to get the s(U)
    # Define a small tolerance
    tolerance = 1e-6
    # Create the mask with tolerance
    mask = (torch.abs(paulifunc - 1) < tolerance) | (
        torch.abs(paulifunc + 1) < tolerance
    )

    # Count the occurrences of True values in the mask
    counts = torch.sum(mask, dim=(-2, -1), dtype=torch.float)

    # Initialize a result tensor with NaN values
    result_counts = torch.full((num_unitary,), float("nan"), dtype=torch.float)

    # Fill in the counts for unitary matrices
    result_counts[unitary_indices] = counts

    # Compute the unitary-stabilizer nullity
    return 2 * n - torch.log2(result_counts)


def _mana_state(state: torch.Tensor, A: torch.Tensor, dim: int) -> torch.Tensor:
    state = state.to(torch.complex128)
    # Call trace function to get Wigner function
    # If state has shape (d, d), expand its dimensions to (1, d, d)
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
    # Compute the Kronecker product
    A_kron = torch.einsum("aij,bkl->abikjl", A_a.transpose(1, 2), A_b).reshape(
        in_dim**2, out_dim**2, in_dim * out_dim, in_dim * out_dim
    )
    # compute the wigner function of a quantum channel
    channel = channel.unsqueeze(0) if channel.dim() == 2 else channel
    W = (
        1
        / out_dim
        * utils.linalg._trace(
            channel.unsqueeze(1).unsqueeze(2) @ A_kron.unsqueeze(0)
        ).real
    )
    # Compute the mana of channels
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
    # FIXME: does not work for comb case, i.e., when PI -> OF and O -> I
    # Unpack variables
    JE_matrix, JE_entry_exit, JE_input_dims, JE_output_dims = JE
    JF_matrix, JF_entry_exit, JF_input_dims, JF_output_dims = JF

    JE_entry, JE_exit = JE_entry_exit.split('->')
    JF_entry, JF_exit = JF_entry_exit.split('->')

    # Calculate overlap
    overlap_subsystem = set(JE_exit).intersection(set(JF_entry))

    # Generate index based on overlap for subsequent permutation
    new_index = list(range(len(JE_entry) + len(JE_exit)))
    overlap_indices = [JE_exit.index(x) + len(JE_entry) for x in overlap_subsystem]
    exchange_indices = list(range(len(overlap_subsystem)))

    for old, new in zip(overlap_indices, exchange_indices):
        new_index[new], new_index[old] = new_index[old], new_index[new]

    # Permute, partial transpose, and permute back
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

    # Generate dictionaries for each system and its corresponding dimension
    JE_pairs = list(zip(JE_entry, JE_input_dims)) + list(zip(JE_exit, JE_output_dims))
    JF_pairs = list(zip(JF_entry, JF_input_dims)) + list(zip(JF_exit, JF_output_dims))

    JE_dim_dict = dict(JE_pairs)
    JF_dim_dict = dict(JF_pairs)

    # Remove overlap subsystem from dictionaries
    for letter in overlap_subsystem:
        JE_dim_dict.pop(letter, None)
        JF_dim_dict.pop(letter, None)

    # Get the non-overlapping dimensions
    non_overlap_dims_E = list(JE_dim_dict.values())
    non_overlap_dims_F = list(JF_dim_dict.values())

    # Multiply the two Choi matrices
    multiplication = (
        torch.kron(JE_choi_transposed.contiguous(), torch.eye(np.prod(non_overlap_dims_F))) 
        @ torch.kron(torch.eye(np.prod(non_overlap_dims_E)), JF_matrix.contiguous())
    )

    # Generate index for subsequent partial trace
    total_str = (
        JE_entry 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + JF_entry 
        + JF_exit
    )
    
    index_dict = {char: idx for idx, char in enumerate(total_str)}
    for letter in overlap_subsystem:
        index_dict.pop(letter, None)

    # Calculate the partial trace over the joint system
    result_matrix = utils.linalg._partial_trace_discontiguous(
        multiplication, 
        list(index_dict.values())
    )

    # Permute the result matrix to the correct order
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

    # Generate the entry and exit string for the final Choi matrix
    entry_exit = (
        JE_entry 
        + ''.join([c for c in JF_entry if c not in overlap_subsystem]) 
        + '->' 
        + ''.join([c for c in JE_exit if c not in overlap_subsystem]) 
        + JF_exit
    )

    # Extract the input and output dimensions
    entry, exit = entry_exit.split('->')

    entry_index = [index_nonoverlap_dict[char] for char in entry]
    input_dims = [all_dims[i] for i in entry_index]

    exit_index = [index_nonoverlap_dict[char] for char in exit]
    output_dims = [all_dims[i] for i in exit_index]

    return result_matrix, entry_exit, input_dims, output_dims
