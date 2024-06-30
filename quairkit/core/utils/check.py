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
Check functions in QuAIRKit.
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from .. import utils


def _is_hermitian(mat: torch.Tensor, eps: Optional[float] = 1e-6, sys_dim: Union[int, List[int]] = 2) -> Union[bool, List[bool]]:
    # TODO depreciate num_qubits checkmat = mat.to(torch.complex128)
    shape = mat.shape

    num_sys = len(sys_dim) if isinstance(sys_dim, list) else int(math.log(shape[-1], sys_dim))
    expected_dimension = math.prod(sys_dim) if isinstance(sys_dim, list) else sys_dim ** num_sys

    if shape[-1] != shape[-2] or shape[-1] != expected_dimension:
        # not a square mat / shape is not in form sys_dim^num_qubits x sys_dim^num_qubits
        return False
    return (torch.norm(mat - mat.mH, dim=(-2, -1)) < eps * expected_dimension).tolist()


def _is_positive(mat: torch.Tensor, eps: Optional[float] = 1e-6, sys_dim: Union[int, List[int]] = 2) -> Union[bool, List[bool]]:
    # TODO depreciate num_qubits check
    mat = mat.to(torch.complex128)
    if _is_hermitian(mat, eps, sys_dim):
        return (torch.min(torch.linalg.eigvalsh(mat), dim = -1).values >= -eps).tolist()
    return False


def _is_state_vector(vec: torch.Tensor, eps: Optional[float] = None, sys_dim: Union[int, List[int]] = 2,\
                      is_batch: Optional[bool] = False) -> Union[Tuple[bool, int],Tuple[List[bool], List[int]]]:
    vec = vec.to(torch.complex128)
    vec = torch.squeeze(vec)
    
    shape = vec.shape
    if (not is_batch and len(vec.shape) != 1)\
        or (is_batch and len(vec.shape) != 2):
        # not a vector / not a batch of vectors
        return False, -3
    
    num_sys = len(sys_dim) if isinstance(sys_dim, list) else int(math.log(shape[-1], sys_dim))
    expected_dimension = math.prod(sys_dim) if isinstance(sys_dim, list) else sys_dim ** num_sys

    if expected_dimension != shape[-1]:
        # not a vector of expected dimension
        return False, -2

    if eps is None: # check unnormalized state vector
        return True, num_sys
    
    vec = vec.reshape([*shape, 1])
    vec_bra = torch.conj(vec.transpose(-2, -1))
    eps = min(eps * shape[-1], 1e-2)
    result = torch.abs(vec_bra @ vec - (1 + 0j)) < eps
    result = result.view(1,len(result)).squeeze()
    return result.tolist(), torch.where(result, torch.tensor(num_sys), torch.tensor(-1)).tolist()

# TODO: repeated ``-4`` error in _is_positive
def _is_density_matrix(rho: torch.Tensor, eps: Optional[float] = None, sys_dim: Union[int, List[int]] = 2,\
                        is_batch: Optional[bool] = False) -> Union[Tuple[bool, int],Tuple[List[bool], List[int]]]:
    rho = rho.to(torch.complex128)
    shape = rho.shape

    if (not is_batch and len(rho.shape)!= 2)\
        or (is_batch and len(rho.shape) != 3)\
              or shape[-1] != shape[-2]:
        # not a square mat / not a batch of square mat / not a square mat
        return False, -4

    num_sys = len(sys_dim) if isinstance(sys_dim, list) else int(math.log(shape[-1], sys_dim))
    expected_dimension = math.prod(sys_dim) if isinstance(sys_dim, list) else sys_dim ** num_sys

    if expected_dimension != shape[-1]:
        # not a mat of expected dimension
        return False, -3

    if eps is None:
        # check unnormalized density matrix
        return True, num_sys

    eps = min(eps * shape[-1], 1e-2)
    result_int = 1
    if is_batch:
        result_bool = torch.abs(torch.einsum('bii->b', rho) - (1 + 0j)) < eps # trace condition
        result_int = torch.where(result_bool, torch.tensor(num_sys), torch.tensor(-2)).tolist()
    else:
        if torch.abs(torch.trace(rho) - (1 + 0j)).item() > eps:
            return False, -2

    result_bool = _is_positive(rho, eps, sys_dim) # positive-semidefinite condition
    result_int = torch.Tensor([result_int, torch.where(torch.tensor(result_bool), torch.tensor(num_sys), torch.tensor(-1)).tolist()])
    return (torch.min(result_int, dim = 0).values > 0).tolist(), torch.min(result_int, dim = 0).values.int().tolist()


def _is_projector(mat: torch.Tensor, eps: Optional[float] = 1e-6) -> Union[bool, List[bool]]:
    # TODO depreciate num_qubits check
    mat = mat.to(torch.complex128)
    shape = mat.shape
    if shape[-1] != shape[-2] or math.log2(shape[-1]) != math.ceil(math.log2(shape[-1])):
        # not a mat / not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return (torch.norm(torch.abs(mat @ mat - mat), dim=(-2, -1)) < eps).tolist()


def _is_unitary(mat: torch.Tensor, eps: Optional[float] = 1e-4) -> Union[bool, List[bool]]:
    # TODO depreciate num_qubits check
    mat = mat.to(torch.complex128)
    shape = mat.shape
    eps = min(eps * shape[-1], 1e-2)
    if shape[-1] != shape[-2] or math.log2(shape[-1]) != math.ceil(math.log2(shape[-1])):
        # not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return (torch.norm(torch.abs(mat @ mat.mH - torch.eye(shape[-1]).to(mat.dtype)), dim=(-2, -1)) < eps).tolist()


def _is_ppt(density_op: torch.Tensor, eps: Optional[float] = 1e-6) -> Union[bool, List[bool]]:
    density_op = density_op.to(torch.complex128)
    result = utils.qinfo._negativity(density_op) <= eps
    return result.tolist() if result.numel() > 1 else result.item()


def _is_choi(op: torch.Tensor) -> Union[bool, List[bool]]:
    op = op.to(torch.complex128)
    n = int(math.log2(op.shape[-1]))
    sys_dim = 2 ** (n // 2)
    
    # CP condition and Trace non-increasing condition
    is_pos = utils.check._is_positive(op)
    
    partial_op = utils.linalg._partial_trace(op, sys_dim, sys_dim, 2)
    is_trace_non_inc = utils.check._is_positive(torch.eye(sys_dim).expand_as(partial_op) - partial_op)
    
    result = np.logical_and(is_pos, is_trace_non_inc)
    return result.tolist() if isinstance(is_pos, np.ndarray) else result


def _is_linear(
    func: Callable[[torch.Tensor], torch.Tensor],
    generator: Union[List[int], Callable[[], torch.Tensor]],
    input_dtype: torch.dtype,
    eps: Optional[float] = 1e-5,
) -> bool:
    list_err = []
    for _ in range(5):
        A = generator()
        B = generator()
        k = torch.rand(1, dtype=input_dtype)

        list_err.append(
            torch.norm(torch.abs(func(k * A + B) - k * func(A) - func(B))).view([1])
        )

    return torch.mean(torch.concat(list_err)).item() < eps
