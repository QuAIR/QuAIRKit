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


def _is_square(mat: torch.Tensor) -> bool:
    return mat.ndim >= 2 and mat.shape[-1] == mat.shape[-2]


def _is_hermitian(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = mat.to(torch.complex128)
    return (mat - utils.linalg._dagger(mat)).norm(dim=(-2, -1)) < eps


def _is_positive(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = mat.to(torch.complex128)
    
    herm_check = _is_hermitian(mat, eps)
    pos_check = torch.min(torch.linalg.eigvalsh(mat), dim=-1).values >= -eps
    
    return herm_check & pos_check


def _is_state_vector(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    vec = vec.to(torch.complex128)
    eps = min(eps * vec.shape[-2], 1e-4)
    
    vec_bra = utils.linalg._dagger(vec)
    return (vec_bra @ vec - (1 + 0j)).norm(dim=(-2, -1)) < eps


def _is_density_matrix(rho: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rho = rho.to(torch.complex128)
    eps = min(eps * rho.shape[-1], 1e-4)
    
    is_trace_one = torch.abs(utils.linalg._trace(rho, -2, -1) - 1) < eps
    is_pos = _is_positive(rho, eps)
    return is_trace_one & is_pos


def _is_projector(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = mat.to(torch.complex128)
    return (mat @ mat - mat).norm(dim=(-2, -1)) < eps


def _is_isometry(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mat = mat.to(torch.complex128)
    dim = mat.shape[-1]
    eps = min(eps * dim, 1e-2)
    
    identity = torch.eye(dim, device=mat.device)
    return (utils.linalg._dagger(mat) @ mat - identity).norm(dim=(-2, -1)) < eps


def _is_unitary(mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return _is_isometry(mat, eps) & _is_hermitian(utils.linalg._dagger(mat) @ mat, eps)


def _is_ppt(density_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    density_op = density_op.to(torch.complex128)
    return utils.qinfo._negativity(density_op) <= eps


def _is_choi(op: torch.Tensor, trace_preserving: bool = True, eps: float = 1e-6) -> torch.Tensor:
    op = op.to(torch.complex128)
    sys_dim = math.isqrt(op.shape[-1])
    
    is_pos = _is_positive(op)
    
    partial_op = utils.linalg._partial_trace(op, [1], [sys_dim, sys_dim])
    identity = torch.eye(sys_dim).expand_as(partial_op)
    
    if trace_preserving:
        is_trace = (identity - partial_op).norm(dim=(-2, -1)) < eps
    else:
        is_trace = _is_positive(identity - partial_op, eps)
    
    return is_pos & is_trace


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

        list_err.append(
            torch.norm(func(k * A + B) - k * func(A) - func(B)).view([1])
        )

    return torch.mean(torch.concat(list_err)) < eps


def _is_povm(set_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dim, batch_shape, set_op = set_op.shape[-1], list(set_op.shape[:-3]), set_op.view([-1] + list(set_op.shape[-3:]))
    
    pos_check = _is_positive(set_op.reshape([-1, dim, dim]), eps)
    pos_check = torch.all(pos_check.view(set_op.shape[:-2]), dim=-1)
    
    oper_sum = set_op.sum(dim=-3)
    identity = torch.eye(dim, device=oper_sum.device)
    complete_check = (identity - oper_sum).norm(dim=(-2, -1)) < eps
    return (pos_check & complete_check).view(batch_shape)


def _is_pvm(set_op: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    set_op = set_op.to(torch.complex128)
    
    povm_check = _is_povm(set_op, eps)

    proj_check = _is_projector(set_op, eps)
    proj_check = torch.all(proj_check, dim=-1)
    
    set_isometry = set_op.view(list(set_op.shape[:-2]) + [-1])
    cross_product = set_isometry @ utils.linalg._dagger(set_isometry)
    cross_product.diagonal(dim1=-1, dim2=-2).zero_()
    isometry_check = cross_product.norm(dim=(-2, -1)) < eps

    return povm_check & proj_check & isometry_check
