# !/usr/bin/env python3
# Copyright (c) 2024 QuAIR team. All Rights Reserved.
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


from typing import Callable, List, Optional, Union

import numpy as np
import torch

from quairkit.core import utils
from quairkit.core.intrinsic import _is_sample_linear

from ..core import State, utils
from ..core.intrinsic import _type_transform

__all__ = [
    "is_choi",
    "is_density_matrix",
    "is_hermitian",
    "is_linear",
    "is_positive",
    "is_povm",
    "is_projector",
    "is_ppt",
    "is_pvm",
    "is_state_vector",
    "is_unitary",
]


def is_choi(op: Union[np.ndarray, torch.Tensor]) -> Union[bool, List[bool]]:
    r"""Check if the input op is a Choi operator of a physical operation.
        Support batch input.

    Args:
        op: matrix form of the linear operation.

    Returns:
        Whether the input op is a valid quantum operation Choi operator. 
        For batch input, return a boolean array with the same batch dimensions as input.

    Note:
        The operation op is (default) applied to the second system.
    """
    op = _type_transform(op, "tensor")

    assert utils.check._is_square(op), \
        f"The input matrix is not a square matrix: received shape {op.shape}"

    return utils.check._is_choi(op).tolist()


def is_density_matrix(
    rho: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``rho`` is a legal quantum density matrix
        Support batch input

    Args:
        rho: density matrix candidate
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness

    Returns:
        determine whether ``rho`` is a PSD matrix with trace 1

    """
    rho = _type_transform(rho, "tensor")

    assert utils.check._is_square(rho), \
        f"The input matrix is not a square matrix: received shape {rho.shape}"

    return utils.check._is_density_matrix(rho, eps).tolist()


def is_hermitian(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is Hermitian.
        Support batch input.

    Args:
        mat: hermitian candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`P - P^\dagger = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")

    assert utils.check._is_square(mat), \
        f"The input matrix is not a square matrix: received shape {mat.shape}"

    return utils.check._is_hermitian(mat, eps).tolist()


def is_linear(
    func: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
    info: Union[List[int], Callable[[], np.ndarray], Callable[[], torch.Tensor]],
    input_dtype: torch.dtype = None,
    eps: Optional[float] = 1e-5,
) -> bool:
    """Check if the provided function 'func' is linear.

    Args:
        func: A callable function to be tested. This function should accept and return either a torch.Tensor or a numpy.ndarray.
        info: A parameter specifying the shape of the input for 'func'. It can be a list of two integers (defining the shape of a tensor),
              a callable that returns a numpy.ndarray, or a callable that returns a torch.Tensor.
        eps: An optional tolerance value used to determine if the function's behavior is close enough to linear. Default value is 1e-6.

    Returns:
        bool: True if 'func' behaves as a linear function within the specified tolerance 'eps'; False otherwise.

    Raises:
        TypeError: If 'func' is not callable, does not accept a torch.Tensor or numpy.ndarray as input, or does not return a torch.Tensor or numpy.ndarray.
                   If 'info' is not a valid type (not a list of integers or a callable returning a torch.Tensor or numpy.ndarray).
    """
    func, generator, input_dtype, _ = _is_sample_linear(func, info, input_dtype)
    return utils.check._is_linear(
        func=func, generator=generator, input_dtype=input_dtype, eps=eps
    )


def is_positive(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a positive semi-definite matrix.
        Support batch input.

    Args:
        mat: positive operator candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`P` is Hermitian and eigenvalues are non-negative
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")

    assert utils.check._is_square(mat), \
        f"The input matrix is not a square matrix: received shape {mat.shape}"

    return utils.check._is_positive(mat, eps).tolist()


def is_povm(
    set_op: Union[torch.Tensor, np.ndarray],
    eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Check if a set of operators forms a positive operator-valued measure (POVM).

    Args:
        set_op: A set of operators indexed by the first dimension
        eps: An optional tolerance value. Default value is 1e-6.

    Returns:
        whether the operators form a POVM.

    """
    set_op = _type_transform(set_op, "tensor")
    return utils.check._is_povm(set_op, eps)


def is_projector(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a projector.
        Support batch input.

    Args:
        mat: projector candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP - P = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")

    assert utils.check._is_square(mat), \
        f"The input matrix is not a square matrix: received shape {mat.shape}"

    return utils.check._is_projector(mat, eps).tolist()


def is_ppt(density_op: Union[np.ndarray, torch.Tensor, State]) -> Union[bool, List[bool]]:
    r"""Check if the input quantum state is PPT.
        Support batch input.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        Whether the input quantum state is PPT. 
        For batch input, return a boolean array with the same batch dimensions as input.
    """
    density_op = _type_transform(density_op, "state").density_matrix
    return utils.check._is_ppt(density_op).tolist()


def is_pvm(
    set_op: Union[torch.Tensor, np.ndarray],
    eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Check if a set of operators forms a projection-valued measure (PVM).

    Args:
        set_op: A set of operators indexed by the first dimension
        eps: An optional tolerance value. Default value is 1e-6.

    Returns:
        bool or List[bool]: True if the operators form a PVM; otherwise, False.

    """
    set_op = _type_transform(set_op, "tensor")
    return utils.check._is_pvm(set_op, eps)


def is_state_vector(
    vec: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-6
) -> Union[bool, List[bool]]:
    r"""Verify whether ``vec`` is a legal quantum state vector.
        Support batch input.

    Args:
        vec: state vector candidate :math:`x`
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness

    Returns:
        determine whether :math:`x^\dagger x = 1`

    """
    vec = _type_transform(vec, "tensor")

    if vec.ndim == 1:
        vec = vec.view([-1, 1])
    else:
        vec = vec.view(list(vec.shape[:-2]) + [-1, 1])

    return utils.check._is_state_vector(vec, eps).tolist()


def is_unitary(
    mat: Union[np.ndarray, torch.Tensor], eps: Optional[float] = 1e-4
) -> Union[bool, List[bool]]:
    r"""Verify whether ``mat`` is a unitary.
        Support batch input.

    Args:
        mat: unitary candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP^\dagger - I = 0`
        For batch input, return a boolean array with the same batch dimensions as input.

    """
    mat = _type_transform(mat, "tensor")

    assert utils.check._is_square(mat), \
        f"The input matrix is not a square matrix: received shape {mat.shape}"

    return utils.check._is_unitary(mat, eps).tolist()
