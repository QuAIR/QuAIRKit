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
from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch


def _abs_norm(mat: torch.Tensor) -> float: 
    norms = torch.norm(torch.abs(mat), dim=(-2, -1))
    return norms.item() if mat.ndim == 2 else norms.tolist()

def _p_norm_herm(mat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    eigval = torch.linalg.eigvalsh(mat)
    norm = torch.abs(eigval).pow(p).sum(dim = len(list(mat.shape[:-2]))).pow(1 / p).view(-1)
    return norm


def _p_norm(mat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.svdvals(mat)
    norm = s.pow(p).sum(dim = len(list(mat.shape[:-2]))).pow(1 / p).view(-1)    
    return norm


def _dagger(mat: torch.Tensor) -> torch.Tensor: 
    return mat.mH.contiguous()


def _block_enc_herm(   
    mat: torch.Tensor, num_block_qubits: Optional[int] = 1
) -> torch.Tensor:  
    H = mat.detach().numpy()
    complex_dtype = mat.dtype

    num_qubits = int(math.log2(mat.shape[0]))
    H_complement = scipy.linalg.sqrtm(np.eye(2**num_qubits) - H @ H)
    block_enc = np.block([[H, 1j * H_complement], [1j * H_complement, H]])
    block_enc = torch.from_numpy(block_enc.astype("complex128")).to(complex_dtype)

    if num_block_qubits > 1:
        block_enc = _direct_sum(
            block_enc,
            torch.eye(2 ** (num_block_qubits + num_qubits) - 2 ** (num_qubits + 1)).to(
                complex_dtype
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
    # Create zero matrices with appropriate shapes
    zero_AB = torch.zeros((batch_size, A_rows, B_cols), dtype=A.dtype)
    zero_BA = torch.zeros((batch_size, B_rows, A_cols), dtype=B.dtype)

    # Stack matrices to form the direct sum
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

# No reference
def _subsystem_decomposition(
    mat: torch.Tensor,
    first_basis: List[torch.Tensor],
    second_basis: List[torch.Tensor],
    inner_prod: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    # Check if the input has a batch dimension
    if len(mat.shape) == 3:
        batch_size, d1, d2 = mat.shape
    else:
        d1, d2 = mat.shape
        batch_size = 1
        mat = mat.unsqueeze(0)  # Add a batch dimension

    assert (
        (d1, d2) == torch.kron(first_basis[0], second_basis[0]).shape
    ), f"The shape does not agree: received {mat.shape, first_basis[0].shape, second_basis[0].shape}"

    first_dim, second_dim = len(first_basis), len(second_basis)

    # Create Kronecker products for all pairs in the basis sets
    kron_products = torch.stack([
        torch.kron(first_basis[i], second_basis[j])
        for i, j in product(range(first_dim), range(second_dim))
    ])

    # Expand dimensions to match the batch size
    kron_products = kron_products.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Reshape for batch matrix multiplication
    mat = mat.unsqueeze(1).expand(-1, first_dim * second_dim, -1, -1)
    kron_products = kron_products.reshape(batch_size * first_dim * second_dim, d1, d2)
    mat = mat.reshape(batch_size * first_dim * second_dim, d1, d2)

    # Compute inner product in a batched way
    coefs = inner_prod(kron_products, mat).unsqueeze(0) 

    # Reshape the result to [batch_size, first_dim, second_dim]
    coefs = coefs.view(batch_size, first_dim, second_dim)

    return coefs.squeeze(0) if batch_size == 1 else coefs


def _partial_trace( 
    state: torch.Tensor, dim1: int, dim2: int, A_or_B: int
) -> torch.Tensor:

    batch_dims = list(state.shape[:-2])

    new_state = _trace(
        torch.reshape(
            state,
            batch_dims + [dim1, dim2, dim1, dim2],
        ),
        axis1=-1 + A_or_B + len(batch_dims),
        axis2=1 + A_or_B + len(batch_dims),
    )

    return new_state


def _partial_trace_discontiguous(  
    rho: torch.Tensor, preserve_qubits: Optional[list] = None
) -> torch.Tensor:
    if preserve_qubits is None:
        return rho

    n = int(math.log2(rho.shape[-1]))

    def new_partial_trace_singleOne(rho: torch.Tensor, at: int) -> torch.Tensor:
        n_qubits = int(math.log2(rho.shape[-1]))
        batch_dims = list(rho.shape[:-2])
        rho = _trace(
            torch.reshape(
                rho,
                batch_dims.copy()
                + [
                    2**at,
                    2,
                    2 ** (n_qubits - at - 1),
                    2**at,
                    2,
                    2 ** (n_qubits - at - 1),
                ],
            ),
            axis1=1 + len(batch_dims),
            axis2=4 + len(batch_dims),
        )
        return torch.reshape(
            rho, batch_dims.copy() + [2 ** (n_qubits - 1), 2 ** (n_qubits - 1)]
        )

    for i, at in enumerate(x for x in range(n) if x not in preserve_qubits):
        rho = new_partial_trace_singleOne(rho, at - i)

    return rho


def _zero(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    dtype = torch.float64 if dtype is None else dtype
    return torch.tensor([0], dtype=dtype)


def _one(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    dtype = torch.float64 if dtype is None else dtype
    return torch.tensor([1], dtype=dtype)



def _density_to_vector(rho: torch.Tensor) -> torch.Tensor: 

    # Handle a batch of density matrices
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

    return eigvec[torch.arange(len(max_eigval_indices)), : ,max_eigval_indices].squeeze()



def _trace(mat: torch.Tensor, axis1: Optional[int]=-2, axis2: Optional[int]=-1) -> torch.Tensor: #No change
    dia_elements = torch.diagonal(mat, offset=0 ,dim1=axis1, dim2=axis2)
    return torch.sum(dia_elements, dim=-1, keepdim=False)



def _nkron( 
    matrix_A: torch.Tensor, matrix_B: torch.Tensor, *args: torch.Tensor
) -> torch.Tensor:
    batch_dim = list(matrix_A.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    
    matrix_A = matrix_A.reshape(batch_size + list(matrix_A.shape[-2:]))
    matrix_B = matrix_B.reshape(batch_size + list(matrix_B.shape[-2:]))
    
    def batch_kron(a, b):
        siz1 = torch.Size([a.shape[-2] * b.shape[-2], a.shape[-1] * b.shape[-1]])
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)
        
    initial_kron = torch.stack([torch.kron(matrix_A[i], matrix_B[i]) for i in range(matrix_A.size(0))])
    return reduce(batch_kron, args, initial_kron).squeeze()


def _partial_transpose(density_op: torch.Tensor, n: int) -> torch.Tensor: 
    n_qubits = int(math.log2(density_op.shape[-1]))
    batch_dims = list(density_op.shape[:-2])
    batch_size = [int(np.prod(batch_dims))] 
    
    density_op = torch.reshape(
        density_op, batch_size + [2**n, 2 ** (n_qubits - n), 2**n, 2 ** (n_qubits - n)]
    )
    
    density_op = torch.permute(density_op, [0, 3, 2, 1, 4])
        
    density_op = torch.reshape(density_op, batch_dims.copy() + [2**n_qubits, 2**n_qubits])

    return density_op



def _permute_systems(mat: torch.Tensor, perm_list: List[int], dim_list: List[int]) -> torch.Tensor:

    # generalize from _base_transpose_for_dm in intrinsic
    batch_dim = list(mat.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    
    mat = mat.reshape(batch_size + list(mat.shape[-2:]))
    dim_tensor = torch.tensor(dim_list)
    num_aran = torch.prod(dim_tensor)
    
    # Using the logic changing the order of each component in a 2**n array
    base_idx = torch.arange(num_aran).view(dim_list)

    base_idx = torch.permute(base_idx, dims=perm_list).contiguous()
    
    # left permute
    mat = mat.gather(1, index=base_idx.view([1, -1, 1]).expand(mat.shape))
    
    # right permute
    mat = mat.gather(2, index=base_idx.view([1, 1, -1]).expand(mat.shape))

    return mat.squeeze()


def _schmidt_decompose(  #Done
    psi: torch.Tensor, sys_A: Optional[List[int]] = None  
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    psi = psi.reshape(-1, 1)
    psi_len = psi.shape[-2]
    # Determine the number of qubits
    num_qubits = int(math.log2(psi_len))

    # If sys_A is not specified, use the first half of qubits
    sys_A = sys_A if sys_A is not None else list(range(num_qubits // 2))
    
    # Determine sys_B as the complement of sys_A
    sys_B = [i for i in range(num_qubits) if i not in sys_A]


    # Permute qubit indices
    psi = psi.reshape([-1] + [2] * num_qubits).permute([0] + [x + 1 for x in sys_A + sys_B])

    # Construct amplitude matrix
    amp_mtr = psi.reshape([-1, 2 ** len(sys_A), 2 ** len(sys_B)])

    # Standard process to obtain Schmidt decomposition
    u, c, v = torch.svd(amp_mtr)

    # Count non-zero singular values
    k = torch.count_nonzero(c > psi_len*1e-13, dim=-1)

    # Select top-k singular values and reshape singular vectors
    c = c[:, :k.max()]
    u = u[:, :, :k.max()].transpose(1, 2).reshape([c.size(0), -1, 1])
    v = v[:, :, :k.max()].transpose(1, 2).reshape([c.size(0), -1, 1])

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


def _sqrtm_scipy(A: torch.Tensor) -> torch.Tensor:
    r"""Calculate square root of a matrix
    Args:
        A: Input matrix.
    Returns:
        The square root of the matrix
    """
    batch_dim = list(A.shape[:-2])
    batch_size = [int(np.prod(batch_dim))] 
    
    A = A.reshape(batch_size + list(A.shape[-2:]))
    
    batch_dims = A.shape[0]
    sqrtmA = torch.zeros(A.shape, dtype=torch.complex128)
    for i in range(batch_dims):
        sqrtmA[i,:,:] = torch.from_numpy(
            scipy.linalg.sqrtm(A[i,:,:].cpu(), disp=False)[0].astype("complex128")
        )
    
    return sqrtmA.squeeze().to(A.device, dtype=A.dtype)



def _perm_of_list(orig_list: List[int], targ_list: List[int]) -> List[int]:
    r"""Find the permutation mapping the original list to the target list
    """
    perm_map = {val: index for index, val in enumerate(orig_list)}
    return [perm_map[val] for val in targ_list]


def _base_transpose(state: torch.Tensor, perm: Union[List, Tuple]) -> torch.Tensor:
    r"""speed-up logic using using np.transpose + torch.gather.

    Args:
        state: input state data.
        perm: permutation of qubit sequence.

    Returns:
        torch.Tensor: permuted state.

    """
    num_qubits = len(perm)
    # Using the logic changing the order of each component in a 2**n array
    base_idx = torch.arange(2 ** num_qubits).view([2] * num_qubits)
    base_idx = torch.permute(base_idx, dims=perm)
    base_idx = base_idx.reshape([1, -1]).expand(state.shape)
    
    return state.gather(1, index=base_idx)


def _base_transpose_for_dm(state: torch.Tensor, perm: Union[List, Tuple]) -> torch.Tensor:
    r"""speed-up logic using using np.transpose + torch.gather.

    Args:
        state: input state data.
        perm: permutation of qubit sequence.

    Returns:
        torch.Tensor: permuted state.
    """
    num_qubits = len(perm)
    # Using the logic changing the order of each component in a 2**n array
    base_idx = torch.arange(2 ** num_qubits).view([2] * num_qubits)
    base_idx = torch.permute(base_idx, dims=perm).contiguous()
    
    # left permute
    state = state.gather(1, index=base_idx.view([1, -1, 1]).expand(state.shape))
    
    # right permute
    state = state.gather(2, index=base_idx.view([1, 1, -1]).expand(state.shape))

    return state

class __Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.size(-1) == A.size(-2)  # Square matrix
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




class __Sqrtm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.size(-1) == A.size(-2)  # Square matrix
        assert A.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(A)
        return _sqrtm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return _adjoint(A, G, _sqrtm_scipy)



_logm = __Logm.apply
_sqrtm = __Sqrtm.apply


def _create_matrix(
    linear_map: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    # Create an identity matrix representing all basis vectors
    identity_matrix = torch.eye(input_dim, dtype=input_dtype)

    # Apply the linear map to each column of the identity matrix
    mapped_vectors = [
        linear_map(identity_matrix[:, i].unsqueeze(1)) for i in range(input_dim)
    ]

    # Stack all mapped vectors to form the final matrix
    return torch.stack(mapped_vectors, dim=1).squeeze()


# TODO: this is a temporal solution for programs that use this function, will be depreciated in the future
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
    # The order of the tensor in paddle is less than 10.
    batch_dims = list(U.shape[:-2])
    num_batch_dims = len(batch_dims)
    
    if not isinstance(qubit_idx, Iterable):
        qubit_idx = [qubit_idx]
        
    # generate swap_list
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
    num_elements = distributions.size(-1)
    num_bits = num_elements.bit_length() - 1
    
    sampled_indices = torch.multinomial(distributions, shots, replacement=True)
    
    counts = torch.stack([(sampled_indices == i).sum(dim = -1) for i in range(num_elements)], dim = -1)
    
    if proportional:
        counts = counts / shots
    
    keys = [f'{i:0{num_bits}b}' if binary else str(i) for i in range(num_elements)]
    
    results = dict(OrderedDict((key, counts[:, i]) for i, key in enumerate(keys)))
    
    return results