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

import math
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..core import utils
from ..core.intrinsic import (
    _ArrayLike,
    _is_sample_linear,
    _SingleParamLike,
    _StateLike,
    _ensure_mat_3d,
    _type_fetch,
    _type_transform,
    _unflatten_batch,
)
from ..database import pauli_basis, phase_space_point
from ..operator import Channel

__all__ = [
    "channel_repr_convert",
    "create_choi_repr",
    "decomp_1qubit",
    "decomp_ctrl_1qubit",
    "diamond_norm",
    "gate_fidelity",
    "general_state_fidelity",
    "link",
    "logarithmic_negativity",
    "mana",
    "mutual_information",
    "negativity",
    "pauli_str_convertor",
    "purity",
    "relative_entropy",
    "stab_nullity",
    "stab_renyi",
    "state_fidelity",
    "trace_distance",
    "von_neumann_entropy",
]


def channel_repr_convert(
    representation: Union[_ArrayLike, List[torch.Tensor], List[np.ndarray]],
    source: str,
    target: str,
    tol: float = 1e-6,
) -> _ArrayLike:
    r"""Convert the given representation of a channel to the target implementation.

    Args:
        representation: Input representation.
        source: Input form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``.
        target: Target form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``.
        tol: Error tolerance for the conversion from Choi, :math:`10^{-6}` by default.

    Raises:
        ValueError: Unsupported channel representation: require Choi, Kraus or Stinespring.

    Returns:
        Quantum channel by the target implementation.

    Examples:
        .. code-block:: python

            # Convert Choi to Kraus representation
            bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex64) / torch.sqrt(torch.tensor(2))
            choi = torch.outer(bell, bell.conj())
            kraus_ops = channel_repr_convert(choi, 'choi', 'kraus')
            print(f'The Kraus operators are:\n{kraus_ops}')

        ::

            The Kraus operators are:
            tensor([[-0.7071+0.j,  0.0000+0.j],
                    [ 0.0000+0.j, -0.7071+0.j]])

    Examples:
        .. code-block:: python

            # Convert Kraus to Stinespring representation
            p = 0.1
            k0 = torch.tensor([[1, 0], [0, torch.sqrt(torch.tensor(1 - p))]], dtype=torch.complex64)
            k1 = torch.tensor([[0, torch.sqrt(torch.tensor(p))], [0, 0]], dtype=torch.complex64)
            stinespring = channel_repr_convert([k0, k1], 'kraus', 'stinespring')
            print(f'The Stinespring representation is:\n{stinespring}')

        ::

            The Stinespring representation is:
            tensor([[1.0000+0.j, 0.0000+0.j],
                    [0.0000+0.j, 0.3162+0.j],
                    [0.0000+0.j, 0.9487+0.j],
                    [0.0000+0.j, 0.0000+0.j]])

    Examples:
        .. code-block:: python

            # Convert Stinespring to Choi representation
            stine = torch.zeros((4, 2), dtype=torch.complex64)
            stine[:2, :2] = torch.eye(2)  # Valid isometry (V†V = I)
            choi = channel_repr_convert(stine, 'stinespring', 'choi')
            print(f'The Choi representation is:\n{choi}')

        ::

            The Choi representation is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    Note:
        choi -> kraus currently has the error of order 1e-6 caused by eigh.

    Raises:
        NotImplementedError: Does not support the conversion of input data type.
    """
    source, target = source.lower(), target.lower()
    if target not in ["choi", "kraus", "stinespring"]:
        raise ValueError(
            f"Unsupported channel representation: require 'choi', 'kraus' or 'stinespring', not {target}"
        )
    if source == target:
        return representation
    if source not in ["choi", "kraus", "stinespring"]:
        raise ValueError(
            f"Unsupported channel representation: require 'choi', 'Kraus' or 'stinespring', not {source}"
        )
    if isinstance(representation, List):
        if isinstance(representation[0], torch.Tensor):
            representation = torch.stack(representation)
        else:
            representation = np.stack(representation)
    
    if len(representation.shape) > 2:
        assert (
            source == "kraus"
        ), f"Unsupported data input: expected Kraus representation, received {source}"
    type_str = _type_fetch(representation)
    representation = _type_transform(representation, "tensor")

    oper = Channel(source, representation)
    if source == "choi":
        if target == "kraus":
            representation = oper.kraus_repr

        representation = oper.stinespring_repr

    elif source == "kraus":
        representation = oper.choi_repr if target == "choi" else oper.stinespring_repr

    else:
        if target == "kraus":
            representation = oper.kraus_repr

        representation = oper.choi_repr

    return _type_transform(representation, type_str)


def create_choi_repr(
    linear_map: Callable[[_ArrayLike], _ArrayLike],
    input_dim: int,
    input_dtype: Optional[torch.dtype] = None,
) -> _ArrayLike:
    r"""Create the Choi representation of a linear map with input checks.

    This function verifies if the map is linear and if the output is a square matrix.

    Args:
        linear_map: A function representing the linear map, which takes and returns a square matrix.
        input_dim: The dimension of the space in which the linear map operates.
        input_dtype: The dtype of the input. Defaults to None.

    Returns:
        torch.Tensor: The Choi matrix of the linear map.

    Examples:
        .. code-block:: python

            def identity_map(matrix: torch.Tensor) -> torch.Tensor:
                return matrix

            choi = create_choi_repr(identity_map, input_dim=2)
            print(f'The Choi representation of identity is:\n{choi}')

        ::

            The Choi representation of identity is:
            tensor([[1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [1.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])
    
    Raises:
        RuntimeWarning: If `linear_map` is not linear or the output is not a square matrix.
    """
    linear_map, generator, input_dtype, type_str = _is_sample_linear(
        linear_map, [input_dim, input_dim], input_dtype
    )

    if not utils.check._is_linear(linear_map, generator, input_dtype):
        warnings.warn("linear_map is not linear", RuntimeWarning)

    sample = linear_map(torch.randn(input_dim, input_dim, dtype=input_dtype))
    if sample.shape[-2] != sample.shape[-1]:
        warnings.warn(
            f"The output of this linear map is not a square matrix: received {sample.shape}",
            RuntimeWarning,
        )

    return _type_transform(
        utils.qinfo._create_choi_repr(
            linear_map=linear_map, input_dim=input_dim, input_dtype=input_dtype
        ), type_str)


def decomp_1qubit(
    unitary: _ArrayLike, return_global: bool = False
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a single-qubit unitary operator into Z-Y-Z rotation angles.

    Decomposes a single-qubit unitary :math:`U` as :math:`U = e^{i\alpha} R_Z(\beta) R_Y(\gamma) R_Z(\delta)`.

    Args:
        unitary: A single 2x2 unitary matrix or a batch of 2x2 unitary matrices representing single-qubit gates,
                 as either a numpy ndarray or a torch Tensor. Shape should be ``(2, 2)`` for single or ``(m, 2, 2)`` for batch.
        return_global: If set to True, the global phase angle :math:`\alpha` is also returned. Defaults to False.

    Returns:
        A tuple containing the angles ``(beta, gamma, delta)`` or ``(alpha, beta, gamma, delta)`` if ``return_global`` is True.
        The type of the tuple elements matches the input type. For batch input, each angle is a tensor with shape ``(m,)``.

    Note:
        This function supports batch operations. When input is a batch of unitaries, all angles are computed in parallel.

    Examples:
        .. code-block:: python

            from quairkit.database import h
            from quairkit.qinfo import decomp_1qubit

            # Decompose Hadamard gate
            angles = decomp_1qubit(h())
            print(f'Z-Y-Z angles for H gate: {angles}')

        ::

            Z-Y-Z angles for H gate: (tensor(0.), tensor(1.5708), tensor(3.1416))

        .. code-block:: python

            # Decompose with global phase
            angles_with_phase = decomp_1qubit(h(), return_global=True)
            print(f'With global phase: {angles_with_phase}')

        ::

            With global phase: (tensor(0.), tensor(0.), tensor(1.5708), tensor(3.1416))
    
    Raises:
        ValueError: If the input matrix is not a 2x2 unitary matrix or batch of 2x2 unitary matrices.
    """
    type_str = _type_fetch(unitary)
    unitary = _type_transform(unitary, "tensor")
    unitary3, batch_shape = _ensure_mat_3d(unitary)

    if unitary3.shape[-2:] != (2, 2):
        raise ValueError(
            f"Input must be 2x2 unitary matrices with shape (..., 2, 2), but got shape {unitary.shape}"
        )

    result_b = utils.qinfo._decomp_1qubit(unitary3, return_global)
    result = tuple(_unflatten_batch(t, batch_shape) for t in result_b)
    return tuple(_type_transform(t, type_str) for t in result)


def decomp_ctrl_1qubit(
    unitary: _ArrayLike
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a controlled single-qubit unitary operator into its components.

    Args:
        unitary: A 2x2 unitary matrix representing the single-qubit gate, as either a numpy ndarray or a torch Tensor.

    Returns:
        A tuple containing the global phase `alpha` and the matrices `A, B, C`, which are components of the decomposition.
        The type of the tuple elements matches the input type.

    Examples:
        .. code-block:: python

            # Decompose X gate
            alpha, A, B, C = decomp_ctrl_1qubit(x())
            print(f'The matrices are:\n{alpha, A, B, C }')

        ::

            The matrices are:
            (tensor(0.), tensor([[ 0.7071+0.j, -0.7071+0.j],
                    [ 0.7071+0.j,  0.7071+0.j]]),
             tensor([[ 0.7071+0.j,  0.7071+0.j],
                    [-0.7071+0.j,  0.7071+0.j]]),
             tensor([[1.-0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j]]))

    Examples:
        .. code-block:: python

            # Decompose random unitary
            alpha, A, B, C = decomp_ctrl_1qubit(random_unitary(1))
            print(f'The matrices are:\n{alpha, A, B, C }')

        ::

            The matrices are:
            (tensor(-0.3577),
             tensor([[ 0.1966+0.9432j, -0.0546-0.2620j],
                     [ 0.0546-0.2620j,  0.1966-0.9432j]]),
             tensor([[ 0.3154-0.9104j,  0.0876+0.2529j],
                     [-0.0876+0.2529j,  0.3154+0.9104j]]),
             tensor([[0.9918-0.1277j, 0.0000+0.0000j],
                     [0.0000+0.0000j, 0.9918+0.1277j]]))
    
    Raises:
        ValueError: Raises a ValueError if the input matrix is not a 2x2 unitary matrix.
    """
    type_str = _type_fetch(unitary)
    unitary = _type_transform(unitary, "tensor")
    unitary3, batch_shape = _ensure_mat_3d(unitary)

    if unitary3.shape[-2:] != (2, 2) or not torch.all(utils.check._is_unitary(unitary3)).item():
        raise ValueError(
            f"Input must be 2x2 unitary matrices with shape (..., 2, 2), but got shape {unitary.shape}"
        )

    result_b = utils.qinfo._decomp_ctrl_1qubit(unitary3)
    result = tuple(_unflatten_batch(t, batch_shape) for t in result_b)
    return tuple(_type_transform(t, type_str) for t in result)


def diamond_norm(
    channel_repr: Union[Channel, torch.Tensor],
    dim_io: Union[int, Tuple[int, int]] = None,
    **kwargs,
) -> float:
    r"""Calculate the diamond norm of input.

    Args:
        channel_repr: A ``Channel`` or a ``torch.Tensor`` instance.
        dim_io: The input and output dimensions.
        kwargs: Parameters to set cvx.

    Raises:
        RuntimeError: `channel_repr` must be `Channel` or `torch.Tensor`.
        TypeError: "dim_io" should be "int" or "tuple".

    Warning:
        `channel_repr` is not in Choi representation, and is converted into `ChoiRepr`.

    Returns:
        Its diamond norm.

    Examples:
        .. code-block:: python

            def depolarizing_choi(p: float) -> torch.Tensor:
                bell = bell_state(2).density_matrix
                return (1 - p) * bell + p / 3 * (torch.kron(x(), x()) + torch.kron(y(), y()) + torch.kron(z(), z())) / 2

            choi = depolarizing_choi(0.5)
            dn = diamond_norm(choi, dim_io=2)
            print(f'The diamond norm of this channel is:\n{dn}')

        ::

            The diamond norm of this channel is:
            0.7500035113999476
    """
    if isinstance(channel_repr, Channel):
        choi_matrix = channel_repr.choi_repr

    elif isinstance(channel_repr, torch.Tensor):
        choi_matrix = channel_repr

    else:
        raise RuntimeError(
            "`channel_repr` must be `ChoiRepr`or `KrausRepr` or `StinespringRepr` or `torch.Tensor`."
        )

    return utils.qinfo._diamond_norm(choi_matrix, dim_io, **kwargs)


def gate_fidelity(
    U: _ArrayLike, V: _ArrayLike
) -> _ArrayLike:
    r"""Calculate the fidelity between two quantum gates.

    .. math::

        F(U, V) = \frac{|\text{tr}(UV^\dagger)|}{2^n}

    where :math:`U` and :math:`V` are :math:`2^n\times 2^n` unitary gates.

    Args:
        U: First quantum gate in matrix form (can be a Tensor, ndarray, or State).
        V: Second quantum gate in matrix form (can be a Tensor, ndarray, or State).

    Returns:
        Fidelity between the two gates. Returns a scalar for single gates,
        or a tensor for batched gates.

    Note:
        This function supports batch operations. When input gates have batch dimensions,
        the fidelity is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import x, y
            from quairkit.qinfo import gate_fidelity

            # Calculate fidelity between X and Y gates
            fid = gate_fidelity(x(), y())
            print(f'Fidelity between X and Y gates: {fid}')

        ::

            Fidelity between X and Y gates: 0.0

        .. code-block:: python

            # Batch gate fidelity example
            from quairkit.database import random_unitary

            batch_U = random_unitary(1, size=3)  # Batch of 3 single-qubit unitaries
            batch_V = random_unitary(1, size=3)  # Batch of 3 single-qubit unitaries
            batch_fid = gate_fidelity(batch_U, batch_V)
            print(f'Batch fidelity shape: {batch_fid.shape}')

        ::

            Batch fidelity shape: torch.Size([3])
    """
    type_u, type_v = _type_fetch(U), _type_fetch(V)
    U = _type_transform(U, "tensor")
    V = _type_transform(V, "tensor")

    U3, batch_u = _ensure_mat_3d(U)
    V3, batch_v = _ensure_mat_3d(V)
    if U3.shape[-2:] != V3.shape[-2:]:
        raise ValueError(f"Two matrices mismatch: received {list(U.shape)} and {list(V.shape)}")

    B = max(U3.shape[0], V3.shape[0])
    if U3.shape[0] not in (1, B) or V3.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: U {U3.shape[0]} vs V {V3.shape[0]}")
    if U3.shape[0] == 1 and B > 1:
        U3 = U3.expand(B, -1, -1)
    if V3.shape[0] == 1 and B > 1:
        V3 = V3.expand(B, -1, -1)

    batch_shape = batch_u if batch_u else batch_v
    fidelity_b = utils.qinfo._gate_fidelity(U3, V3.to(dtype=U3.dtype))
    fidelity = fidelity_b.reshape(batch_shape) if batch_shape else fidelity_b.squeeze(0)

    if type_u == "numpy" and type_v == "numpy":
        return fidelity.detach().numpy()
    return fidelity
    

def general_state_fidelity(rho: _StateLike, sigma: _StateLike) -> _ArrayLike:
    r"""Calculate the fidelity measure of two general (possibly subnormalized) quantum states.
    
    .. math::

        F_*(\rho, \sigma) = F(\rho, \sigma) + \sqrt{(1 - \text{tr}[\rho])(1 - \text{tr}[\sigma])}
    
    where :math:`F(\rho, \sigma)` is the standard state fidelity without square.
    
    This measure is useful for subnormalized states where :math:`\text{tr}[\rho] \leq 1`.
    
    Args:
        rho: First quantum state (can be a State, Tensor, or ndarray). May be subnormalized.
        sigma: Second quantum state (can be a State, Tensor, or ndarray). May be subnormalized.
    
    Returns:
        The general state fidelity of the input states. Returns a scalar for single states,
        or a tensor for batched states.

    Note:
        This function supports batch operations. When input states have batch dimensions,
        the fidelity is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import bell_state, zero_state
            from quairkit.qinfo import general_state_fidelity

            # Calculate general fidelity between Bell state and zero state
            fid = general_state_fidelity(bell_state(2), zero_state(2))
            print(f'General fidelity: {fid}')

        ::

            General fidelity: 0.70710688829422

        .. code-block:: python

            # Batch general fidelity example
            from quairkit.database import random_state

            batch_rho = random_state(1, size=3)  # Batch of 3 states
            batch_sigma = random_state(1, size=3)  # Batch of 3 states
            batch_fid = general_state_fidelity(batch_rho, batch_sigma)
            print(f'Batch general fidelity shape: {batch_fid.shape}')

        ::

            Batch general fidelity shape: torch.Size([3])
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix

    rho3, batch_r = _ensure_mat_3d(rho)
    sig3, batch_s = _ensure_mat_3d(sigma)
    if rho3.shape[-2:] != sig3.shape[-2:]:
        raise ValueError(f"Two states mismatch: received {list(rho.shape)} and {list(sigma.shape)}")

    B = max(rho3.shape[0], sig3.shape[0])
    if rho3.shape[0] not in (1, B) or sig3.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: rho {rho3.shape[0]} vs sigma {sig3.shape[0]}")
    if rho3.shape[0] == 1 and B > 1:
        rho3 = rho3.expand(B, -1, -1)
    if sig3.shape[0] == 1 and B > 1:
        sig3 = sig3.expand(B, -1, -1)

    batch_shape = batch_r if batch_r else batch_s
    fidelity_b = utils.qinfo._general_state_fidelity(rho3, sig3)
    fidelity = fidelity_b.reshape(batch_shape) if batch_shape else fidelity_b.squeeze(0)

    if type_rho == "numpy" and type_sigma == "numpy":
        return fidelity.detach().numpy()
    return fidelity


def link(
    JE: Tuple[_ArrayLike, str, Union[List[int], int], Union[List[int], int]],
    JF: Tuple[_ArrayLike, str, Union[List[int], int], Union[List[int], int]],
) -> Tuple[_ArrayLike, str, List[int], List[int]]:
    r"""Calculate the link product of two Choi matrices of quantum channels.
    
    Args:
        JE: Tuple containing the Choi representation of channel E, its label, input dimensions, and output dimensions.
        JF: Tuple containing the Choi representation of channel F, its label, input dimensions, and output dimensions.
    
    Returns:
        The resulting Choi matrix after the link product, its label, and input/output dimensions.
    
    Note:
        The identification convention for input label is exemplified by "AB->CD", where the same letter in different cases
        (uppercase vs lowercase) is recognized as the same system, and an apostrophe indicates a different system.
        When input and output dimensions are specified as an int, it implies that each system has the same dimensionality.

    Examples:
        .. code-block:: python

            def identity_choi(dim: int) -> torch.Tensor:
                bell = eye(dim).flatten().to(torch.complex64)
                return torch.outer(bell, bell.conj()) / dim

            JE = (identity_choi(2), "A->B", [2], [2])
            JF = (identity_choi(2), "B->C", [2], [2])
            J_result, label, dim_in, dim_out = link(JE, JF)
            print(f'The resulting Choi matrix after the link product is {J_result}.\n'
                  f'Its label is {label}.\n'
                  f'Input/output dimensions are:{dim_in} and {dim_out}.')

        ::

            The resulting Choi matrix after the link product is tensor([[0.2500+0.j, 0.0000+0.j, 0.0000+0.j, 0.2500+0.j],
                                                                       [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                                                                       [0.0000+0.j, 0.0000+0.j, 0.0000+0.j, 0.0000+0.j],
                                                                       [0.2500+0.j, 0.0000+0.j, 0.0000+0.j, 0.2500+0.j]]).
            Its label is A->C.
            Input/output dimensions are: [2] and [2].
    """
    JE_matrix_type = _type_fetch(JE[0])
    JE_matrix = _type_transform(JE[0], "tensor")

    JF_matrix_type = _type_fetch(JF[0])
    JF_matrix = _type_transform(JF[0], "tensor")

    def map_to_lower_if_apostrophe(s: str) -> str:
        result = ""
        i = 0
        while i < len(s):
            if i + 1 < len(s) and s[i + 1] == "'":
                result += s[i].lower()
                i += 1
            else:
                result += s[i].upper()
            i += 1
        return result

    JE_entry_exit = map_to_lower_if_apostrophe(JE[1])
    JF_entry_exit = map_to_lower_if_apostrophe(JF[1])

    JE_entry, JE_exit = JE_entry_exit.split("->")
    JF_entry, JF_exit = JF_entry_exit.split("->")

    JE_dims = (
        [JE[2]] * len(JE_entry) if isinstance(JE[2], int) else JE[2],
        [JE[3]] * len(JE_exit) if isinstance(JE[3], int) else JE[3],
    )
    JF_dims = (
        [JF[2]] * len(JF_entry) if isinstance(JF[2], int) else JF[2],
        [JF[3]] * len(JF_exit) if isinstance(JF[3], int) else JF[3],
    )

    expected_JE_shape = (np.prod(JE_dims[0] + JE_dims[1]), np.prod(JE_dims[0] + JE_dims[1]))
    expected_JF_shape = (np.prod(JF_dims[0] + JF_dims[1]), np.prod(JF_dims[0] + JF_dims[1]))

    assert (
        JE_matrix.shape == expected_JE_shape
    ), f"JE_matrix shape mismatch: expected {expected_JE_shape}, got {JE_matrix.shape}"
    assert (
        JF_matrix.shape == expected_JF_shape
    ), f"JF_matrix shape mismatch: expected {expected_JF_shape}, got {JF_matrix.shape}"

    overlap_subsystem = set(JE_exit).intersection(set(JF_entry))
    for subsystem in overlap_subsystem:
        JE_subsystem_index = JE_exit.index(subsystem)
        JF_subsystem_index = JF_entry.index(subsystem)

        JE_subsystem_dim = JE[3][JE_subsystem_index]
        JF_subsystem_dim = JF[2][JF_subsystem_index]

        assert (
            JE_subsystem_dim == JF_subsystem_dim
        ), f"JE and JF overlap system '{subsystem}' dimension mismatch: JE has dimension {JE_subsystem_dim}, JF has dimension {JF_subsystem_dim}."

    JE = (JE_matrix, JE_entry_exit) + JE_dims
    JF = (JF_matrix, JF_entry_exit) + JF_dims

    result = utils.qinfo._link(JE, JF)

    if JE_matrix_type == "numpy" and JF_matrix_type == "numpy":
        result = (_type_transform(result[0], "numpy"),) + result[1:]

    return result


def logarithmic_negativity(
    density_op: _StateLike
) -> _ArrayLike:
    r"""Calculate the Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Logarithmic Negativity of the input quantum state.

    Examples:
        .. code-block:: python

            log_neg = logarithmic_negativity(bell_state(2))
            print(f'The logarithmic negativity of Bell state is:\n{log_neg}')

        ::

            The logarithmic negativity of Bell state is:
            1.0
    """
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "state").density_matrix
    rho3, batch_shape = _ensure_mat_3d(density_op)

    log_neg_b = utils.qinfo._logarithmic_negativity(rho3)
    log_neg = log_neg_b.reshape(batch_shape) if batch_shape else log_neg_b.squeeze(0)
    return log_neg.detach().numpy() if type_str == "numpy" else log_neg


def mana(matrix: _StateLike, input_str: str,
         out_dim: Optional[int]=None) -> _StateLike:
    r"""Compute the mana of states or channels.

    Args:
        matrix: Quantum state or channel, when "channel", it should be the choi matrix of channel.
        input_str: "state" or "channel".
        out_dim: Output system dimension, only need to compute mana of channel.
    
    Returns:
        The output mana.

    Examples:
        .. code-block:: python

            bell_mana = mana(bell_state(2), input_str="state")
            print(f'The mana of Bell state is:\n{bell_mana}')

        ::

            The mana of Bell state is:
            tensor([0.6813], dtype=torch.float64)

    Examples:
        .. code-block:: python

            choi_matrix = eye(4) / 2
            mana_chan = mana(choi_matrix, input_str="channel", out_dim=2)
            print(f'The mana of a Choi channel is:\n{mana_chan}')

        ::

            The mana of a Choi channel is:
            tensor([0.], dtype=torch.float64)
    """
    type_str = _type_fetch(matrix)
    if input_str == "channel":
        mat = _type_transform(matrix, "tensor")
        mat3, batch_shape = _ensure_mat_3d(mat)
        in_dim = mat3.shape[-1] // out_dim
        A_a = phase_space_point(in_dim)
        A_b = phase_space_point(out_dim)
        mana_b = utils.qinfo._mana_channel(mat3, A_a, A_b, out_dim, in_dim)
        mana_out = mana_b.reshape(batch_shape) if batch_shape else mana_b.squeeze(0)
        return mana_out.detach().numpy() if type_str == "numpy" else mana_out
    elif input_str == "state":
        state = _type_transform(matrix, "state").density_matrix
        rho3, batch_shape = _ensure_mat_3d(state)
        d = rho3.shape[-1]
        A = phase_space_point(d)
        mana_b = utils.qinfo._mana_state(rho3, A=A, dim=d)
        mana_out = mana_b.reshape(batch_shape) if batch_shape else mana_b.squeeze(0)
        return mana_out.detach().numpy() if type_str == "numpy" else mana_out

    raise ValueError("Invalid input. Please enter 'state' or 'channel'.")


def mutual_information(state: _StateLike, dim_A: int, dim_B: int) -> _ArrayLike:
    r"""Compute the mutual information of a bipartite state.
    
    Args:
        state: Input bipartite quantum state with system AB.
        dim_A: Dimension of system A.
        dim_B: Dimension of system B.
        
    Returns:
        The mutual information of the input quantum state.

    Examples:
        .. code-block:: python

            mi = mutual_information(bell_state(2), dim_A=2, dim_B=2)
            print(f'The mutual information of Bell state is:\n{mi}')

        ::

            The mutual information of Bell state is:
            2.0
    """
    type_str = _type_fetch(state)
    rho = _type_transform(state, "state").density_matrix
    rho3, batch_shape = _ensure_mat_3d(rho)

    if rho3.shape[-1] != dim_A * dim_B:
        raise ValueError(
            f"State dimension mismatch: got {rho3.shape[-1]} but dim_A*dim_B={dim_A * dim_B}"
        )

    mi_b = utils.qinfo._mutual_information(rho3, dim_A, dim_B)
    mi = mi_b.reshape(batch_shape) if batch_shape else mi_b.squeeze(0)
    return mi.detach().numpy() if type_str == "numpy" else mi


def negativity(density_op: _StateLike) -> _ArrayLike:
    r"""Compute the Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Negativity of the input quantum state.

    Examples:
        .. code-block:: python

            neg = negativity(bell_state(2))
            print(f'The negativity of Bell state is:\n{neg}')

        ::

            The negativity of Bell state is:
            0.4999999701976776
    """
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "state").density_matrix
    rho3, batch_shape = _ensure_mat_3d(density_op)

    neg_b = utils.qinfo._negativity(rho3)
    neg = neg_b.reshape(batch_shape) if batch_shape else neg_b.squeeze(0)
    return neg.detach().numpy() if type_str == "numpy" else neg


def pauli_str_convertor(observable: List) -> List:
    r"""Concatenate the input observable with coefficient 1.

    For example, if the input ``observable`` is ``[['z0,x1'], ['z1']]``,
    then this function returns the observable ``[[1, 'z0,x1'], [1, 'z1']]``.

    Args:
        observable: The observable to be concatenated with coefficient 1.

    Returns:
        The observable with coefficient 1.

    Examples:
        .. code-block:: python

            pauli_terms = [['Z0,X1'], ['Y2']]
            converted = pauli_str_convertor(pauli_terms)
            print(f'The converted result is:\n{converted}')

        ::

            The converted result is:
            [[1, 'Z0,X1'], [1, 'Y2']]
    """
    for i in range(len(observable)):
        assert len(observable[i]) == 1, "Each term should only contain one string"

    return [[1, term] for term in observable]


def purity(rho: _StateLike) -> _ArrayLike:
    r"""Calculate the purity of a quantum state.

    .. math::

        P = \text{tr}(\rho^2)

    The purity ranges from :math:`1/d` (maximally mixed) to 1 (pure state), where :math:`d` is the dimension.

    Args:
        rho: Density matrix form of the quantum state (can be a State, Tensor, or ndarray).

    Returns:
        The purity of the input quantum state. Returns a scalar for single states,
        or a tensor for batched states.

    Note:
        This function supports batch operations. When input states have batch dimensions,
        the purity is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import eye, bell_state
            from quairkit.qinfo import purity

            # Purity of maximally mixed state
            max_mixed_purity = purity(eye(2) / 2)
            print(f'Purity of 1-qubit maximally mixed state: {max_mixed_purity}')

        ::

            Purity of 1-qubit maximally mixed state: 0.5

        .. code-block:: python

            # Purity of pure state (Bell state)
            bell_purity = purity(bell_state(2))
            print(f'Purity of Bell state: {bell_purity}')

        ::

            Purity of Bell state: 1.0
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "state").density_matrix
    rho3, batch_shape = _ensure_mat_3d(rho)

    gamma_b = utils.qinfo._purity(rho3)
    gamma = gamma_b.reshape(batch_shape) if batch_shape else gamma_b.squeeze(0)
    return gamma.detach().numpy() if type_rho == "numpy" else gamma


def relative_entropy(
    rho: _StateLike,
    sigma: _StateLike,
    base: Optional[int] = 2,
) -> _ArrayLike:
    r"""Calculate the relative entropy of two quantum states.

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

    Args:
        rho: Density matrix form of the quantum state.
        sigma: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        Relative entropy between input quantum states.

    Examples:
        .. code-block:: python

            rel_ent = relative_entropy(bell_state(2), eye(4) / 4)
            print(f'The relative entropy between 2-qubit maximal mixed state and Bell state is:\n{rel_ent}')

        ::

            The relative entropy between 2-qubit maximal mixed state and Bell state is:
            1.999999761581421
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix

    rho3, batch_r = _ensure_mat_3d(rho)
    sig3, batch_s = _ensure_mat_3d(sigma)
    if rho3.shape[-2:] != sig3.shape[-2:]:
        raise ValueError(f"Two quantum states mismatch: received {list(rho.shape)} and {list(sigma.shape)}")

    B = max(rho3.shape[0], sig3.shape[0])
    if rho3.shape[0] not in (1, B) or sig3.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: rho {rho3.shape[0]} vs sigma {sig3.shape[0]}")
    if rho3.shape[0] == 1 and B > 1:
        rho3 = rho3.expand(B, -1, -1)
    if sig3.shape[0] == 1 and B > 1:
        sig3 = sig3.expand(B, -1, -1)

    batch_shape = batch_r if batch_r else batch_s
    entropy_b = utils.qinfo._relative_entropy(rho3, sig3.to(rho3.dtype), base)
    entropy = entropy_b.reshape(batch_shape) if batch_shape else entropy_b.squeeze(0)

    if type_rho == "numpy" and type_sigma == "numpy":
        return entropy.detach().numpy()
    return entropy


def stab_nullity(unitary: _ArrayLike) -> _ArrayLike:
    r"""Tool for calculation of unitary-stabilizer nullity.

    Args:
        unitary: A batch of unitary matrices.

    Returns:
        Unitary-stabilizer nullity for each unitary matrix.

    Examples:
        .. code-block:: python

            unitary_stabilizer_nullity = stab_nullity(h().to(torch.complex128))
            print(f'The unitary-stabilizer nullity for Hadamard gate is:\n{unitary_stabilizer_nullity}')

        ::

            The unitary-stabilizer nullity for Hadamard gate is:
            tensor([0.])
    """
    type_str = _type_fetch(unitary)
    unitary = _type_transform(unitary, "tensor")
    unitary3, batch_shape = _ensure_mat_3d(unitary)

    is_unitary_matrices = utils.check._is_unitary(unitary3)
    is_unitary_list = is_unitary_matrices.tolist()
    unitary_indices = torch.tensor([i for i, ok in enumerate(is_unitary_list) if ok], dtype=torch.long)

    num_unitary = unitary3.shape[0]
    if len(unitary_indices) == 0:
        raise ValueError("All matrices are not unitary. Exiting the program.")
    if len(unitary_indices) != len(is_unitary_list):
        non_unitary_indices = [i for i, ok in enumerate(is_unitary_list) if not ok]
        print(f"The following matrices are not unitary and will be ignored: {non_unitary_indices}")

    unitary_valid = unitary3 if len(unitary_indices) == num_unitary else unitary3[unitary_indices]
    n = int(math.log2(unitary_valid.shape[-1]))
    pauli = (pauli_basis(n) * math.sqrt(2)**n).to(torch.complex128)
    
    out = utils.qinfo._stab_nullity(
        unitary=unitary_valid,
        num_unitary=num_unitary,
        unitary_indices=unitary_indices if len(unitary_indices) != num_unitary else 0,
        pauli=pauli,
        n=n,
    )
    if batch_shape:
        out = out.reshape(batch_shape)
    return _type_transform(out, type_str)


def stab_renyi(density: _StateLike, alpha: _SingleParamLike) -> _ArrayLike:
    r"""Tool for calculation of stabilizer Renyi entropy.

    Args:
        density: A batch of density matrices.
        alpha: The Renyi entropy exponent.

    Returns:
        Stabilizer Renyi entropy for each density matrix.

    Examples:
        .. code-block:: python

            stabilizer_renyi_entropy = stab_renyi(zero_state(1), alpha=2)
            print(f'The stabilizer Renyi entropy for zero state with alpha=2 is:\n{stabilizer_renyi_entropy}')

        ::

            The stabilizer Renyi entropy for zero state with alpha=2 is:
            (tensor([2.3842e-07]),)
    """
    type_str = _type_fetch(density)
    alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
    density = _type_transform(density, "state").density_matrix
    density3, batch_shape = _ensure_mat_3d(density)
    
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    
    is_density = utils.check._is_density_matrix(density3, eps=1e-4)
    is_density_list = is_density.tolist()
    density_indices = torch.tensor([i for i, ok in enumerate(is_density_list) if ok], dtype=torch.long)

    num_density = density3.shape[0]
    if len(density_indices) == 0:
        raise ValueError("All matrices are not density matrices. Exiting the program.")
    if len(density_indices) != len(is_density_list):
        non_density_indices = [i for i, ok in enumerate(is_density_list) if not ok]
        print(f"The following matrices are not density matrices and will be ignored: {non_density_indices}")

    density_valid = density3 if len(density_indices) == num_density else density3[density_indices]
    
    n = int(math.log2(density_valid.shape[-1]))
    pauli = (pauli_basis(n) * math.sqrt(2)**n).to(torch.complex128)

    renyi = utils.qinfo._stab_renyi(
        density=density_valid,
        alpha=alpha,
        num=num_density,
        indices=density_indices if len(density_indices) != num_density else 0,
        pauli=pauli,
        n=n,
    )
    if batch_shape:
        renyi = renyi.reshape(batch_shape)
    return renyi.detach().numpy() if type_str == "numpy" else renyi


def state_fidelity(rho: _StateLike, sigma: _StateLike) -> _ArrayLike:
    r"""Calculate the fidelity of two quantum states, no extra square is taken.

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    For pure states, this simplifies to :math:`|\langle\psi|\phi\rangle|`.

    Args:
        rho: First quantum state (can be a State, Tensor, or ndarray).
        sigma: Second quantum state (can be a State, Tensor, or ndarray).

    Returns:
        The fidelity between the input quantum states. Returns a scalar for single states,
        or a tensor for batched states.
    
    Note:
        The fidelity equation is based on Equation (9.53) in Quantum Computation & Quantum Information, 10th edition.
        This function supports batch operations. When input states have batch dimensions,
        the fidelity is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import bell_state, zero_state, eye
            from quairkit.qinfo import state_fidelity

            # Single state fidelity
            fidelity = state_fidelity(bell_state(2), eye(4)/4)
            print(f'Fidelity between Bell state and maximal mixed state: {fidelity}')

        ::

            Fidelity between Bell state and maximal mixed state: 0.5000725984573364

        .. code-block:: python

            # Batch fidelity example
            from quairkit.database import random_state

            batch_rho = random_state(1, size=3)  # Batch of 3 states
            batch_sigma = random_state(1, size=3)  # Batch of 3 states
            batch_fid = state_fidelity(batch_rho, batch_sigma)
            print(f'Batch fidelity shape: {batch_fid.shape}')

        ::

            Batch fidelity shape: torch.Size([3])
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix

    rho3, batch_r = _ensure_mat_3d(rho)
    sig3, batch_s = _ensure_mat_3d(sigma)
    if rho3.shape[-2:] != sig3.shape[-2:]:
        raise ValueError(f"Two quantum states mismatch: received {rho3.shape[-2:]} and {sig3.shape[-2:]}")

    B = max(rho3.shape[0], sig3.shape[0])
    if rho3.shape[0] not in (1, B) or sig3.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: rho {rho3.shape[0]} vs sigma {sig3.shape[0]}")
    if rho3.shape[0] == 1 and B > 1:
        rho3 = rho3.expand(B, -1, -1)
    if sig3.shape[0] == 1 and B > 1:
        sig3 = sig3.expand(B, -1, -1)

    batch_shape = batch_r if batch_r else batch_s
    fidelity_b = utils.qinfo._state_fidelity(rho3, sig3.to(rho3.dtype))
    fidelity = fidelity_b.reshape(batch_shape) if batch_shape else fidelity_b.squeeze(0)

    if type_rho == "numpy" and type_sigma == "numpy":
        return fidelity.detach().numpy()
    return fidelity


def trace_distance(rho: _StateLike, sigma: _StateLike) -> _ArrayLike:
    r"""Calculate the trace distance of two quantum states.

    .. math::
        D(\rho, \sigma) = \frac{1}{2}\text{tr}|\rho-\sigma|

    Args:
        rho: First quantum state (can be a State, Tensor, or ndarray).
        sigma: Second quantum state (can be a State, Tensor, or ndarray).

    Returns:
        The trace distance between the input quantum states. Returns a scalar for single states,
        or a tensor for batched states.

    Note:
        This function supports batch operations. When input states have batch dimensions,
        the trace distance is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import bell_state, eye
            from quairkit.qinfo import trace_distance

            # Single state trace distance
            tr_dist = trace_distance(bell_state(2), eye(4)/4)
            print(f'Trace distance between Bell state and maximal mixed state: {tr_dist}')

        ::

            Trace distance between Bell state and maximal mixed state: 0.75

        .. code-block:: python

            # Batch trace distance example
            from quairkit.database import random_state

            batch_rho = random_state(1, size=3)  # Batch of 3 states
            batch_sigma = random_state(1, size=3)  # Batch of 3 states
            batch_dist = trace_distance(batch_rho, batch_sigma)
            print(f'Batch trace distance shape: {batch_dist.shape}')

        ::

            Batch trace distance shape: torch.Size([3])
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix

    rho3, batch_r = _ensure_mat_3d(rho)
    sig3, batch_s = _ensure_mat_3d(sigma)
    if rho3.shape[-2:] != sig3.shape[-2:]:
        raise ValueError(f"Two quantum states mismatch: received {rho3.shape[-2:]} and {sig3.shape[-2:]}")

    B = max(rho3.shape[0], sig3.shape[0])
    if rho3.shape[0] not in (1, B) or sig3.shape[0] not in (1, B):
        raise ValueError(f"Incompatible batch sizes: rho {rho3.shape[0]} vs sigma {sig3.shape[0]}")
    if rho3.shape[0] == 1 and B > 1:
        rho3 = rho3.expand(B, -1, -1)
    if sig3.shape[0] == 1 and B > 1:
        sig3 = sig3.expand(B, -1, -1)

    batch_shape = batch_r if batch_r else batch_s
    dist_b = utils.qinfo._trace_distance(rho3, sig3.to(rho3.dtype))
    dist = dist_b.reshape(batch_shape) if batch_shape else dist_b.squeeze(0)
    return dist.detach().numpy() if type_rho == "numpy" and type_sigma == "numpy" else dist


def von_neumann_entropy(rho: _StateLike, base: Optional[Union[_SingleParamLike, int]] = 2) -> _ArrayLike:
    r"""Calculate the von Neumann entropy of a quantum state.

    .. math::
        S = -\text{tr}(\rho \log_\text{base}(\rho))

    The von Neumann entropy measures the amount of quantum information in a state.
    For a :math:`d`-dimensional system, it ranges from 0 (pure state) to :math:`\log_d(\text{base})` (maximally mixed).

    Args:
        rho: Density matrix form of the quantum state (can be a State, Tensor, or ndarray).
        base: The base of logarithm. Defaults to 2 (bits). Use :math:`e` for nats.

    Returns:
        The von Neumann entropy of the input quantum state. Returns a scalar for single states,
        or a tensor for batched states.

    Note:
        This function supports batch operations. When input states have batch dimensions,
        the entropy is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import completely_mixed_computational, bell_state
            from quairkit.qinfo import von_neumann_entropy

            # Entropy of maximally mixed state
            ent = von_neumann_entropy(completely_mixed_computational(1))
            print(f'Von Neumann entropy of 1-qubit maximally mixed state: {ent}')

        ::

            Von Neumann entropy of 1-qubit maximally mixed state: 1.0

        .. code-block:: python

            # Entropy of pure state (should be 0)
            pure_ent = von_neumann_entropy(bell_state(2))
            print(f'Von Neumann entropy of Bell state: {pure_ent}')

        ::

            Von Neumann entropy of Bell state: 0.0
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "state").density_matrix
    rho3, batch_shape = _ensure_mat_3d(rho)

    entropy_b = utils.qinfo._von_neumann_entropy(rho3, base)
    entropy = entropy_b.reshape(batch_shape) if batch_shape else entropy_b.squeeze(0)
    return entropy.detach().numpy() if type_rho == "numpy" else entropy
