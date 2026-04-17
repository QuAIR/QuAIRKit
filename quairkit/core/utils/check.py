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

from . import linalg, qinfo


def _as_batch_square(mat: torch.Tensor, name: str) -> Tuple[torch.Tensor, List[int]]:
    if not isinstance(mat, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(mat)}")
    if mat.ndim < 2:
        raise ValueError(f"{name} must have shape [d, d] or [B, d, d], got {list(mat.shape)}")
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(f"{name} must be square on last two dims, got {list(mat.shape)}")
    batch_shape: List[int] = [] if mat.ndim == 2 else list(mat.shape[:-2])
    mat = mat if mat.ndim > 2 else mat.unsqueeze(0)
    return mat, batch_shape


def _as_batch_vec(vec: torch.Tensor, name: str) -> Tuple[torch.Tensor, List[int]]:
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(vec)}")
    if vec.ndim < 2:
        raise ValueError(f"{name} must have shape [d, 1] or [B, d, 1], got {list(vec.shape)}")
    if vec.shape[-1] != 1:
        raise ValueError(f"{name} must have last dim 1, got {list(vec.shape)}")
    batch_shape: List[int] = [] if vec.ndim == 2 else list(vec.shape[:-2])
    vec = vec if vec.ndim > 2 else vec.unsqueeze(0)
    return vec, batch_shape


def _as_batch_set(set_op: torch.Tensor, name: str) -> Tuple[torch.Tensor, List[int]]:
    if not isinstance(set_op, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(set_op)}")
    if set_op.ndim < 3:
        raise ValueError(f"{name} must have shape [K, d, d] or [B, K, d, d], got {list(set_op.shape)}")
    if set_op.shape[-1] != set_op.shape[-2]:
        raise ValueError(f"{name} must be square on last two dims, got {list(set_op.shape)}")
    batch_shape: List[int] = [] if set_op.ndim == 3 else list(set_op.shape[:-3])
    set_op = set_op if set_op.ndim > 3 else set_op.unsqueeze(0)
    return set_op, batch_shape


def _is_square(mat: torch.Tensor) -> bool:
    return mat.ndim >= 2 and mat.shape[-1] == mat.shape[-2]


def _is_hermitian(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat, batch_shape = _as_batch_square(mat, "mat")
    mat = mat.to(torch.complex128)
    out = (mat - linalg._dagger(mat)).norm(dim=(-2, -1)) < eps
    return out if batch_shape else out.squeeze(0)


def _is_positive(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat, batch_shape = _as_batch_square(mat, "mat")
    mat = mat.to(torch.complex128)
    herm_check = _is_hermitian(mat, eps)
    pos_check = torch.min(torch.linalg.eigvalsh(mat), dim=-1).values >= -eps
    out = herm_check & pos_check
    return out if batch_shape else out.squeeze(0)


def _is_state_vector(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    vec, batch_shape = _as_batch_vec(vec, "vec")
    vec = vec.to(torch.complex128)
    eps = min(eps * vec.shape[-2], 1e-4)
    vec_bra = linalg._dagger(vec)
    out = (vec_bra @ vec - (1 + 0j)).norm(dim=(-2, -1)) < eps
    return out if batch_shape else out.squeeze(0)


def _is_density_matrix(rho: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho, batch_shape = _as_batch_square(rho, "rho")
    rho = rho.to(torch.complex128)
    eps = min(eps * rho.shape[-1], 1e-4)
    is_trace_one = torch.abs(linalg._trace(rho, -2, -1) - 1) < eps
    is_pos = _is_positive(rho, eps)
    out = is_trace_one & is_pos
    return out if batch_shape else out.squeeze(0)


def _is_projector(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat, batch_shape = _as_batch_square(mat, "mat")
    mat = mat.to(torch.complex128)
    out = (mat @ mat - mat).norm(dim=(-2, -1)) < eps
    return out if batch_shape else out.squeeze(0)


def _is_isometry(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat, batch_shape = _as_batch_square(mat, "mat")
    mat = mat.to(torch.complex128)
    dim = mat.shape[-1]
    eps = min(eps * dim, 1e-2)
    
    identity = torch.eye(dim, device=mat.device).expand(mat.shape[0], dim, dim)
    out = (linalg._dagger(mat) @ mat - identity).norm(dim=(-2, -1)) < eps
    return out if batch_shape else out.squeeze(0)


def _is_unitary(mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    iso = _is_isometry(mat, eps)
    herm = _is_hermitian(linalg._dagger(mat) @ mat, eps)
    return iso & herm


def _is_ppt(density_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    density_op, batch_shape = _as_batch_square(density_op, "density_op")
    density_op = density_op.to(torch.complex128)
    out = qinfo._negativity(density_op) <= eps
    return out if batch_shape else out.squeeze(0)


def _is_choi(op: torch.Tensor, trace_preserving: bool = True, eps: float = 1e-6) -> torch.Tensor:
    op, batch_shape = _as_batch_square(op, "op")
    op = op.to(torch.complex128)
    sys_dim = math.isqrt(op.shape[-1])
    
    is_pos = _is_positive(op)
    
    partial_op = linalg._partial_trace(op, [1], [sys_dim, sys_dim])
    identity = torch.eye(sys_dim).expand_as(partial_op)
    
    if trace_preserving:
        is_trace = (identity - partial_op).norm(dim=(-2, -1)) < eps
    else:
        is_trace = _is_positive(identity - partial_op, eps)
    
    out = is_pos & is_trace
    return out if batch_shape else out.squeeze(0)


def _is_linear(
    func: Callable[[torch.Tensor], torch.Tensor],
    generator: Union[List[int], Callable[[], torch.Tensor]],
    input_dtype: torch.dtype,
    eps: float = 1e-5,
) -> torch.Tensor:
    list_err = []
    for _ in range(5):
        A = generator()
        B = generator()
        k = torch.rand(1, dtype=input_dtype)

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B)
        
        kA_plus_B = k * A + B
        
        f_kA_B = func(kA_plus_B)
        if isinstance(f_kA_B, np.ndarray):
            f_kA_B = torch.from_numpy(f_kA_B)
        
        f_A = func(A)
        if isinstance(f_A, np.ndarray):
            f_A = torch.from_numpy(f_A)
        
        f_B = func(B)
        if isinstance(f_B, np.ndarray):
            f_B = torch.from_numpy(f_B)
        
        diff = f_kA_B - k * f_A - f_B
        list_err.append(
            torch.norm(diff).view([1])
        )

    return torch.mean(torch.concat(list_err)) < eps


def _is_povm(set_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    set_op, batch_shape = _as_batch_set(set_op, "set_op")
    set_op = set_op.to(torch.complex128)
    B, K, d, _ = set_op.shape
    pos_check = _is_positive(set_op.reshape(B * K, d, d), eps).view(B, K)
    pos_check = torch.all(pos_check, dim=-1)

    oper_sum = set_op.sum(dim=1)
    identity = torch.eye(d, device=oper_sum.device).expand(B, d, d)
    complete_check = (identity - oper_sum).norm(dim=(-2, -1)) < eps
    out = pos_check & complete_check
    if not batch_shape:
        return out.squeeze(0)
    return out.reshape(batch_shape)


def _is_pvm(set_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    set_op, batch_shape = _as_batch_set(set_op, "set_op")
    set_op = set_op.to(torch.complex128)
    B, K, d, _ = set_op.shape
    povm_check = _is_povm(set_op, eps)
    if batch_shape:
        povm_check = povm_check.reshape(B)

    proj_check = _is_projector(set_op.reshape(B * K, d, d), eps).view(B, K)
    proj_check = torch.all(proj_check, dim=-1)

    set_isometry = set_op.reshape(B, K, -1)
    cross_product = torch.matmul(set_isometry, linalg._dagger(set_isometry))
    cross_product.diagonal(dim1=-1, dim2=-2).zero_()
    isometry_check = cross_product.norm(dim=(-2, -1)) < eps

    out = povm_check & proj_check & isometry_check
    if not batch_shape:
        return out.squeeze(0)
    return out.reshape(batch_shape)
