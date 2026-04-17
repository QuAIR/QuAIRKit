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
Linear algebra functions in QuAIRKit.
"""

import math
import os
from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
import importlib

_WINDOWS_DLL_DIR_HANDLES: List[object] = []


def _prepare_windows_dll_search_path() -> None:
    if os.name != "nt":
        return
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib_dir):
        return
    try:
        _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(torch_lib_dir))
    except (AttributeError, FileNotFoundError, OSError):
        pass


def _require_cpp_submodule(submodule: str):
    """Return a required C++ extension submodule.

    This module no longer supports a Python fallback implementation for
    performance-critical kernels. If the C++ extension is not available, raise an
    ImportError with actionable guidance.
    """
    _prepare_windows_dll_search_path()
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


_CPP_LINALG = _require_cpp_submodule("linalg")


def _abs_norm(mat: torch.Tensor) -> float: 
    norms = torch.norm(torch.abs(mat), dim=(-2, -1))
    return norms.item() if mat.ndim == 2 else norms.tolist()


def _p_norm_herm(mat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    eigval = torch.linalg.eigvalsh(mat)
    norm = torch.abs(eigval).pow(p).sum(dim = len(list(mat.shape[:-2]))).pow(1 / p)
    return norm.view(mat.shape[:-2])


def _p_norm(mat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.svdvals(mat)
    norm = s.pow(p).sum(dim = len(list(mat.shape[:-2]))).pow(1 / p) 
    return norm.view(mat.shape[:-2])


def _dagger(mat: torch.Tensor) -> torch.Tensor: 
    return _CPP_LINALG.dagger(mat)


def _block_enc_herm(   
    mat: torch.Tensor, num_block_qubits: Optional[int] = 1
) -> torch.Tensor:  
    device = mat.device
    H = mat.detach().cpu().numpy()
    complex_dtype = mat.dtype

    num_qubits = int(math.log2(mat.shape[0]))
    H_complement = scipy.linalg.sqrtm(np.eye(2**num_qubits) - H @ H)
    block_enc = np.block([[H, 1j * H_complement], [1j * H_complement, H]])
    block_enc = torch.from_numpy(block_enc.astype("complex128")).to(dtype=complex_dtype, device=device)

    if num_block_qubits > 1:
        block_enc = _direct_sum(
            block_enc,
            torch.eye(
                2 ** (num_block_qubits + num_qubits) - 2 ** (num_qubits + 1),
                dtype=complex_dtype,
                device=device,
            ),
        )

    return block_enc



def _direct_sum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor: 
    
    batch_dim = list(A.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    
    A = A.reshape(batch_size + list(A.shape[-2:]))
    B = B.reshape(batch_size + list(B.shape[-2:]))
    
    A_rows, A_cols = A.shape[-2], A.shape[-1]
    B_rows, B_cols = B.shape[-2], B.shape[-1]

    batch_size = int(np.prod(batch_dim))
    device = A.device
    zero_AB = torch.zeros((batch_size, A_rows, B_cols), dtype=A.dtype, device=device)
    zero_BA = torch.zeros((batch_size, B_rows, A_cols), dtype=B.dtype, device=device)

    mat_upper = torch.cat((A, zero_AB), dim=-1)
    mat_lower = torch.cat((zero_BA, B), dim=-1)
    mat = torch.cat((mat_upper, mat_lower), dim=-2)

    return mat.squeeze()


def _herm_transform( 
    fcn: Callable[[float], float],
    mat: torch.Tensor,
    ignore_zero: Optional[bool] = False,
) -> torch.Tensor:
    eigval, eigvec = torch.linalg.eigh(mat)
    eigval = eigval.tolist()
    eigvec = eigvec.T

    mat = torch.zeros(mat.shape).to(mat.dtype)
    for i in range(len(eigval)):
        vec = eigvec[i].reshape([mat.shape[0], 1])

        if np.abs(eigval[i]) < 1e-5 and ignore_zero:
            continue
        mat += (fcn(eigval[i]) + 0j) * vec @ torch.conj(vec.T)

    return mat


def _subsystem_decomposition(
    mat: torch.Tensor,
    first_basis: List[torch.Tensor],
    second_basis: List[torch.Tensor],
    inner_prod: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if mat.ndim == 3:
        batch_size, d1, d2 = mat.shape
    else:
        d1, d2 = mat.shape
        batch_size = 1
        mat = mat.unsqueeze(0)

    assert (
        (d1, d2) == torch.kron(first_basis[0], second_basis[0]).shape
    ), f"The shape does not agree: received {mat.shape, first_basis[0].shape, second_basis[0].shape}"

    first_dim, second_dim = len(first_basis), len(second_basis)

    kron_products = torch.stack([
        torch.kron(first_basis[i], second_basis[j])
        for i, j in product(range(first_dim), range(second_dim))
    ])

    kron_products = kron_products.unsqueeze(0).expand(batch_size, -1, -1, -1)

    mat = mat.unsqueeze(1).expand(-1, first_dim * second_dim, -1, -1)
    kron_products = kron_products.reshape(batch_size * first_dim * second_dim, d1, d2)
    mat = mat.reshape(batch_size * first_dim * second_dim, d1, d2)

    coefs = inner_prod(kron_products, mat).unsqueeze(0) 

    coefs = coefs.view(batch_size, first_dim, second_dim)

    return coefs.squeeze(0) if batch_size == 1 else coefs


def _perm_to_left(mat: torch.Tensor, perm_system_idx: List[int], system_dim: List[int]) -> torch.Tensor:
    r"""Permute the given systems of input matrix to the left.
    
    Args:
        mat: input matrix.
        perm_system_idx: the indices of the systems to be permuted.
        system_dim: the dimensions of all systems.
    
    """
    num_systems = len(system_dim)
    if perm_system_idx == list(range(num_systems)):
        return mat
    target_idx = perm_system_idx + [x for x in list(range(num_systems)) if x not in perm_system_idx]
    return _permute_systems(mat, target_idx, system_dim)


def _trace_1(mat: torch.Tensor, dim1: int) -> torch.Tensor:
    r"""Tracing out the first system of the matrix
    
    Args:
        dim1: the dimension of the first system.
    
    """
    return _CPP_LINALG.trace_1(mat, dim1)


def _ptrace_1(xy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Tracing out the first system of the product unit vector xy
    
    Args:
        x: the unit vector of the first system.
    
    """
    dim_x = x.shape[-1]
    batch_dims, dim_y = list(xy.shape[:-2]), int(xy.shape[-1] // dim_x)
    
    xy, x = xy.view(batch_dims + [-1, dim_x, dim_y]), x.view([-1, dim_x, 1])
    y = (xy * x.conj()).sum(dim=-2)
    return y.squeeze(-2)


def _transpose_1(mat: torch.Tensor, dim1: int) -> torch.Tensor:
    r"""Transpose the first system of the matrix
    
    Args:
        dim1: the dimension of the first system.
    
    """
    return _CPP_LINALG.transpose_1(mat, dim1)


def _partial_trace(
    state: torch.Tensor, trace_idx: List[int], system_dim: List[int]
) -> torch.Tensor:
    state = _perm_to_left(state, trace_idx, system_dim)
    
    traced_dim = int(np.prod([system_dim[i] for i in trace_idx]))
    return _trace_1(state, traced_dim)


def _partial_trace_discontiguous(  
    rho: torch.Tensor, preserve_qubits: List[int]
) -> torch.Tensor:
    if preserve_qubits is None:
        return rho

    n = int(math.log2(rho.shape[-1]))
    system_dim, traced_qubits = [2] * n, [i for i in range(n) if i not in preserve_qubits]
    return _partial_trace(rho, traced_qubits, system_dim)


def _density_to_vector(rho: torch.Tensor) -> torch.Tensor: 

    batch_dim = list(rho.shape[:-2])
    batch_size = [int(np.prod(batch_dim))]

    rho_size = [rho.shape[-1]]
    eigval, eigvec = torch.linalg.eigh(rho)
    eigvec = eigvec.reshape(batch_size + 2*rho_size)
    eigval = eigval.reshape(batch_size + rho_size)

    max_eigval_values, max_eigval_indices = torch.max(eigval, dim=1)

    eig_test = torch.abs(max_eigval_values - 1) > 1e-6
    if torch.any(eig_test):
        raise ValueError(
            f"The output state may not be a pure state, maximum eigenvalue distance: {torch.abs(max_eigval_values - 1)}"
        )

    return eigvec[torch.arange(len(max_eigval_indices)), : ,max_eigval_indices].squeeze().unsqueeze(-1)


def _trace(mat: torch.Tensor, axis1: int = -2, axis2: int =- 1) -> torch.Tensor:
    return _CPP_LINALG.trace(mat, axis1, axis2)


def _kron(matrix_A: torch.Tensor, matrix_B: torch.Tensor) -> torch.Tensor:
    r"""(batched) Kronecker product
    
    Args:
        matrix_A: input (batched) matrix
        matrix_B: input (batched) matrix
    
    Returns:
        The Kronecker product of the two (batched) matrices
    
    Note:
        See https://discuss.pytorch.org/t/kronecker-product/3919/11
    """
    return _CPP_LINALG.kron(matrix_A, matrix_B)


def _nkron( 
    matrix_1st: torch.Tensor, *args: torch.Tensor
) -> torch.Tensor:
    return _CPP_LINALG.nkron([matrix_1st] + list(args))


def _cartesian_expand_batch(*tensors: torch.Tensor) -> List[torch.Tensor]:
    r"""Expand multiple tensors' batch dimensions into Cartesian product form.

    Given tensors with batch sizes [d0], [d1], ..., [dn], expand each tensor so
    that all have the same batch size ``d0 * d1 * ... * dn``, corresponding to
    the Cartesian product of the original batch indices.

    Args:
        tensors: Input tensors, each assumed to have a batch dimension at dim 0.

    Returns:
        A list of expanded tensors with aligned batch dimensions.
    """
    if not tensors:
        return []
    if len(tensors) == 1:
        return list(tensors)

    batch_dims = [int(t.shape[0]) for t in tensors]
    expanded: List[torch.Tensor] = []

    for i, tensor in enumerate(tensors):
        interleave = math.prod(batch_dims[i + 1 :]) if i < len(batch_dims) - 1 else 1
        tile = math.prod(batch_dims[:i]) if i > 0 else 1

        out = tensor.repeat_interleave(interleave, dim=0) if interleave > 1 else tensor
        if tile > 1:
            out = out.repeat([tile] + [1] * (out.ndim - 1))
        expanded.append(out)

    return expanded


def _partial_transpose(state: torch.Tensor, transpose_idx: List[int], system_dim: List[int]) -> torch.Tensor: 
    state = _perm_to_left(state, transpose_idx, system_dim)
    transpose_dim = int(np.prod([system_dim[i] for i in transpose_idx]))
    state = _transpose_1(state, transpose_dim)

    original_seq = list(range(len(system_dim)))
    current_seq = transpose_idx + [i for i in original_seq if i not in transpose_idx]
    current_system_dim = [system_dim[x] for x in current_seq]
    map_original = _perm_of_list(current_seq, original_seq)
    return _permute_systems(state, map_original, current_system_dim)


def _permute_systems(mat: torch.Tensor, perm_list: List[int], dim_list: List[int]) -> torch.Tensor:
    if perm_list == list(range(len(dim_list))):
        return mat
    return _CPP_LINALG.permute_systems(mat, perm_list, dim_list)


def _vector_to_prod_sum(
    vec: torch.Tensor,
    system_dim: List[int],
    tol: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""Factorize a (batched) state vector into subgroup-level product-sum factors.

    Args:
        vec: Input vector with shape ``[..., prod(system_dim)]`` or ``[..., prod(system_dim), 1]``.
        system_dim: Local dimensions for each subgroup.
        tol: Singular-value truncation threshold (absolute).

    Returns:
        A tuple ``(factors, coeffs)`` where:
            - ``factors[i]`` has shape ``[..., r_{i-1}, d_i, r_i]``.
            - ``coeffs[i]`` has shape ``[..., r_i]``.
    """
    return _CPP_LINALG.vector_to_prod_sum(vec, [int(d) for d in system_dim], float(tol))


def _matrix_to_prod_sum(
    matrix: torch.Tensor,
    system_dim: List[int],
    tol: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    r"""Factorize a (batched) operator into subgroup-level product-sum factors.

    Args:
        matrix: Input matrix with shape ``[..., prod(system_dim), prod(system_dim)]``.
        system_dim: Local dimensions for each subgroup.
        tol: Singular-value truncation threshold (absolute).

    Returns:
        A tuple ``(factors, coeffs)`` where:
            - ``factors[i]`` has shape ``[..., r_{i-1}, d_i, d_i, r_i]``.
            - ``coeffs[i]`` has shape ``[..., r_i]``.
    """
    return _CPP_LINALG.matrix_to_prod_sum(matrix, [int(d) for d in system_dim], float(tol))


def _schmidt_decompose(
    psi: torch.Tensor, sys_A: Optional[List[int]] = None  
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    psi = psi.reshape(-1, 1)
    psi_len = psi.shape[-2]
    num_qubits = int(math.log2(psi_len))

    sys_A = sys_A if sys_A is not None else list(range(num_qubits // 2))
    
    sys_B = [i for i in range(num_qubits) if i not in sys_A]


    psi = psi.reshape([-1] + [2] * num_qubits).permute([0] + [x + 1 for x in sys_A + sys_B])

    amp_vec = psi.reshape([-1, 2 ** len(sys_A), 2 ** len(sys_B)]).reshape([-1, psi_len])

    factors, coeffs = _vector_to_prod_sum(
        amp_vec,
        [2 ** len(sys_A), 2 ** len(sys_B)],
        tol=psi_len * 1e-13,
    )

    c = coeffs[0]
    u = factors[0].squeeze(-3).transpose(-1, -2).reshape([c.size(0), -1, 1])
    v = factors[1].squeeze(-1).reshape([c.size(0), -1, 1])
    return c, u, v



def _adjoint(A, E, f) -> torch.Tensor:
    r"""Calculate gradient of the function f
    Args:
        A: Input matrix.
        E: Undetermined matrix. Usually choose identity
        f: A differentiable function
    Returns:
        The numerical gradient of the function f given input A: \delta f(A)
    """

    batch_dim = list(A.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    A = A.reshape(batch_size + list(A.shape[-2:]))
        
    batch_dims = A.shape[0]
    A_H = A.mH.to(E.dtype)
    n = A.size(1)
    M = torch.zeros(batch_dims, 2 * n, 2 * n, dtype=E.dtype, device=E.device)
    M[:, :n, :n] = A_H
    M[:, n:, n:] = A_H
    M[:, :n, n:] = E
    fM = f(M)
    fM = fM.reshape(batch_size + list(fM.shape[-2:]))
    return fM[:, :n, n:].to(A.dtype).squeeze()



def _logm_scipy(A: torch.Tensor) -> torch.Tensor:
    r"""Calculate logarithm of a matrix
    Args:
        A: Input matrix.
    Returns:
        The matrix of natural base logarithms
    """

    batch_dim = list(A.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    
    A = A.reshape(batch_size + list(A.shape[-2:]))

    batch_dims = A.shape[0]
    logmA = torch.zeros(A.shape, dtype=torch.complex128)
    for i in range(batch_dims):
        logmA[i,:,:] = torch.from_numpy(
            scipy.linalg.logm(A[i,:,:].cpu(), disp=False)[0].astype("complex128"))

    return logmA.squeeze().to(A.device, dtype=A.dtype)



def _perm_of_list(origin: List[int], target: List[int]) -> List[int]:
    r"""Find the permutation mapping the original list to the target list
    """
    perm_map = {val: index for index, val in enumerate(origin)}
    return [perm_map[val] for val in target]


def _permute_sv(state: torch.Tensor, perm: Union[List, Tuple],
                    system_dim: List[int]) -> torch.Tensor:
    r"""speed-up logic using using np.transpose + torch.gather.

    Args:
        state: input state data.
        perm: permutation of system sequence.
        system_dim: list of dimensions for all systems

    Returns:
        permuted state.

    """
    return _CPP_LINALG.permute_sv(state, perm, system_dim)


def _permute_dm(state: torch.Tensor, perm: Union[List, Tuple],
                           system_dim: List[int]) -> torch.Tensor:
    r"""speed-up logic using using np.transpose + torch.gather.

    Args:
        state: input state data.
        perm: permutation of system sequence.
        system_dim: list of dimensions for all systems

    Returns:
        permuted state.
    """
    return _CPP_LINALG.permute_dm(state, perm, system_dim)

class __Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.size(-1) == A.size(-2)
        assert A.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(A)
        return _logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return _adjoint(A, G, _logm_scipy)


_logm = __Logm.apply


def _sqrtm(A: torch.Tensor) -> torch.Tensor:
    return _CPP_LINALG.sqrtm(A)


def _create_matrix(
    linear_map: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    identity_matrix = torch.eye(input_dim, dtype=input_dtype)

    mapped_vectors = [
        linear_map(identity_matrix[:, i].unsqueeze(1)) for i in range(input_dim)
    ]

    return torch.stack(mapped_vectors, dim=1).squeeze()


def _unitary_transformation(
        U: torch.Tensor, V: torch.Tensor, qubit_idx: Union[List[int], int], num_qubits: int
) -> torch.Tensor:
    r"""Compute :math:`VU`, where :math:`U` is a unitary matrix acting on all qubits.

    Args:
        U: unitary matrix of the Circuit
        V: The gate that acts on the circuit
        qubit_idx: The indices of the qubits on which the gate is acted.
        num_qubits: The number of the qubits in the input quantum state.

    Returns:
        The transformed quantum state.
    """
    batch_dims = list(U.shape[:-2])
    num_batch_dims = len(batch_dims)
    
    if not isinstance(qubit_idx, Iterable):
        qubit_idx = [qubit_idx]
        
    num_acted_qubits = len(qubit_idx)
    origin_seq = list(range(num_qubits))
    seq_for_acted = qubit_idx + [x for x in origin_seq if x not in qubit_idx]
    swapped = [False] * num_qubits
    swap_ops = []
    for idx in range(num_qubits):
        if not swapped[idx]:
            next_idx = idx
            swapped[next_idx] = True
            while not swapped[seq_for_acted[next_idx]]:
                swapped[seq_for_acted[next_idx]] = True
                if next_idx < seq_for_acted[next_idx]:
                    swap_ops.append((next_idx, seq_for_acted[next_idx]))
                else:
                    swap_ops.append((seq_for_acted[next_idx], next_idx))
                next_idx = seq_for_acted[next_idx]

    for swap_op in swap_ops:
        shape = batch_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * num_qubits - last_idx - 1))
        U = torch.reshape(U, shape)
        U = torch.permute(
            U, tuple(range(num_batch_dims)) + tuple(item + num_batch_dims for item in [0, 3, 2, 1, 4])
        )
    U = torch.reshape(
        U, batch_dims.copy() + [2 ** num_acted_qubits, 2 ** (2 * num_qubits - num_acted_qubits)]
    )
    U = torch.matmul(V, U)
    swap_ops.reverse()
    for swap_op in swap_ops:
        shape = batch_dims.copy()
        last_idx = -1
        for idx in swap_op:
            shape.append(2 ** (idx - last_idx - 1))
            shape.append(2)
            last_idx = idx
        shape.append(2 ** (2 * num_qubits - last_idx - 1))
        U = torch.reshape(U, shape)
        U = torch.permute(
            U, tuple(range(num_batch_dims)) + tuple(item + num_batch_dims for item in [0, 3, 2, 1, 4])
        )
    U = torch.reshape(U, batch_dims.copy() + [2 ** num_qubits, 2 ** num_qubits])
    return U


def _hessian(loss_function: Callable[[torch.Tensor], torch.Tensor], var: torch.Tensor) -> torch.Tensor:
    n, m = var.shape
    mat = torch.empty(m, n, n)
    for i in range(m):
        mat[i, :, :] = torch.autograd.functional.hessian(loss_function, var[:,i])
    return mat


def _gradient(loss_function: Callable[[torch.Tensor], torch.Tensor], var: torch.Tensor, n: int) -> torch.Tensor:
    m = var.shape[0]
    mat = torch.empty(m,1)
    for j in range(m):
        func_var = loss_function(var)
        for _ in range(n):
            grads = torch.autograd.grad(func_var, var, create_graph=True)[0]
            func_var = grads[j]
        mat[j] = func_var
        
    return mat


def _prob_sample(distributions: torch.Tensor, shots: int = 1024, 
                 binary: bool = True, proportional: bool = False) -> Dict[str, torch.Tensor]:
    batch_dim =list(distributions.shape[:-1])
    distributions = distributions.view([-1, distributions.shape[-1]])
    
    num_elements = distributions.size(-1)
    num_bits = num_elements.bit_length() - 1
    
    sampled_indices = torch.multinomial(distributions, shots, replacement=True)
    
    counts = torch.stack([(sampled_indices == i).sum(dim = -1) for i in range(num_elements)], dim = -1)
    
    if proportional:
        counts = counts / shots
    
    keys = [f'{i:0{num_bits}b}' if binary else str(i) for i in range(num_elements)]
    results = OrderedDict((key, counts[:, i].view(batch_dim)) 
                          for i, key in enumerate(keys))
    return dict(results)


def _get_swap_indices(pos1: int, pos2: int, system_indices: List[List[int]], 
                      system_dim: List[int], device: torch.device) -> torch.Tensor:
    r"""Get the swapped indices given swap_indices, to rearrange the elements
    
    Args:
        pos1: the 1st position to be swapped
        pos2: the 2nd position to be swapped
        system_indices: list of system indices for these swap operations,
        system_dim: dimensions of subsystems
    
    Returns:
        a list of swapped indices
    
    """
    total_dim = np.prod(system_dim)
    n = len(system_dim)

    weights = [1] * n
    for i in range(n - 2, -1, -1):
        weights[i] = weights[i + 1] * system_dim[i + 1]

    indices = torch.arange(total_dim, device=device)
    
    for qubits_idx in reversed(system_indices):
        i, j = qubits_idx
        dim_i, dim_j = system_dim[i], system_dim[j]
        weight_i, weight_j = weights[i], weights[j]

        state_i = (indices // weight_i) % dim_i
        state_j = (indices // weight_j) % dim_j
        joint_state = state_i * dim_j + state_j

        remainder = indices - state_i * weight_i - state_j * weight_j

        new_joint_state = torch.where(
            joint_state == pos1, pos2,
            torch.where(joint_state == pos2, pos1, joint_state)
        )

        new_state_i = new_joint_state // dim_j
        new_state_j = new_joint_state % dim_j

        indices = remainder + new_state_i * weight_i + new_state_j * weight_j

    return indices
