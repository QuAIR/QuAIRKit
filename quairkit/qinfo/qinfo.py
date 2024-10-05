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

from ..core import State, utils
from ..core.intrinsic import _is_sample_linear, _type_fetch, _type_transform
from ..database import pauli_basis, phase_space_point
from ..operator import Channel

__all__ = [
    "channel_repr_convert",
    "create_choi_repr",
    "decomp_1qubit",
    "decomp_ctrl_1qubit",
    "diamond_norm",
    "gate_fidelity",
    "logarithmic_negativity",
    "mana",
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
    representation: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
    source: str,
    target: str,
    tol: float = 1e-6,
) -> Union[torch.Tensor, np.ndarray]:
    r"""convert the given representation of a channel to the target implementation

    Args:
        representation: input representation
        source: input form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``
        target: target form, should be ``'choi'``, ``'kraus'`` or ``'stinespring'``
        tol: error tolerance for the conversion from Choi, :math:`10^{-6}` by default

    Raises:
        ValueError: Unsupported channel representation: require Choi, Kraus or Stinespring.

    Returns:
        quantum channel by the target implementation

    Note:
        choi -> kraus currently has the error of order 1e-6 caused by eigh

    Raises:
        NotImplementedError: does not support the conversion of input data type

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

        # stinespring repr
        representation = oper.stinespring_repr

    elif source == "kraus":
        representation = oper.choi_repr if target == "choi" else oper.stinespring_repr

    else:  # if source == 'stinespring'
        if target == "kraus":
            representation = oper.kraus_repr

        # choi repr
        representation = oper.choi_repr

    return _type_transform(representation, type_str)


def create_choi_repr(
    linear_map: Callable[
        [Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]
    ],
    input_dim: int,
    input_dtype: torch.dtype = None,
) -> torch.Tensor:
    r"""Create the Choi representation of a linear map with input checks.

    This function verifies if the map is linear and if the output is a square matrix.

    Args:
        linear_map: A function representing the linear map, which takes and returns a square matrix.
        input_dim: The dimension of the space in which the linear map operates.
        input_dtype: The dtype of the input. Defaults to None.


    Returns:
        torch.Tensor: The Choi matrix of the linear map.

    Raises:
        RuntimeWarning: If `linear_map` is not linear or the output is not a square matrix.
    
    """
    linear_map, generator, input_dtype, type_str = _is_sample_linear(
        linear_map, [input_dim, input_dim], input_dtype
    )

    # Check if the linear_map is linear and issue a warning if not
    if not utils.check._is_linear(linear_map, generator, input_dtype):
        warnings.warn("linear_map is not linear", RuntimeWarning)

    # Check if the output is a square matrix
    sample = linear_map(torch.randn(input_dim, input_dim, dtype=input_dtype))
    if sample.shape[0] != sample.shape[1]:
        warnings.warn(
            f"The output of this linear map is not a square matrix: received {sample.shape}",
            RuntimeWarning,
        )

    # Compute and return the Choi representation
    return _type_transform(
        utils.qinfo._create_choi_repr(
            linear_map=linear_map, input_dim=input_dim, input_dtype=input_dtype
        ),
        type_str,
    )


def decomp_1qubit(
    unitary: Union[np.ndarray, torch.Tensor], return_global: bool = False
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a single-qubit unitary operator into Z-Y-Z rotation angles.

    Args:
        unitary: A batch of 2x2 unitary matrices representing single-qubit gates,
                 as either a numpy ndarray or a torch Tensor. The shape should be (m, 2, 2).
        return_global: If set to True, the global phase angle `alpha` is also returned.

    Returns:
        A tuple containing the angles `(beta, gamma, delta)` or `(alpha, beta, gamma, delta)` if `return_global` is True.
        The type of the tuple elements matches the input type.

    Raises:
        ValueError: Raises a ValueError if the input matrix is not a batch of 2x2 unitary matrices.
    """
    type_str = _type_fetch(unitary)

    unitary = _type_transform(unitary, "tensor")

    if unitary.shape[-2:] != (2, 2):
        raise ValueError(
            f"Input must be a batch of 2x2 unitary matrices with shape (m, 2, 2), but got shape {unitary.shape}"
        )
    result = utils.qinfo._decomp_1qubit(unitary, return_global)
    # Transform each tensor in the result tuple individually
    return tuple(_type_transform(tensor, type_str) for tensor in result)


def decomp_ctrl_1qubit(
    unitary: Union[np.ndarray, torch.Tensor]
) -> Union[Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...]]:
    r"""Decompose a controlled single-qubit unitary operator into its components.

    Args:
        unitary: A 2x2 unitary matrix representing the single-qubit gate, as either a numpy ndarray or a torch Tensor.

    Returns:
        A tuple containing the global phase `alpha` and the matrices `A, B, C`, which are components of the decomposition.
        The type of the tuple elements matches the input type.

    Raises:
        ValueError: Raises a ValueError if the input matrix is not a 2x2 unitary matrix.

    Reference:
        Nielsen, M. A., & Chuang, I. L. (2000). Quantum Computation and Quantum Information. 
        Corollary 4.2 for the decomposition of a single-qubit unitary operation.
    """
    type_str = _type_fetch(unitary)

    unitary = _type_transform(unitary, "tensor")

    if unitary.shape[-2:] != (2, 2) or not utils.check._is_unitary(unitary):
        raise ValueError(f"Input must be a batch of 2x2 unitary matrices with shape (m, 2, 2), but got shape {unitary.shape}")

    result = utils.qinfo._decomp_ctrl_1qubit(unitary)
    # Transform each tensor in the result tuple individually
    return tuple(_type_transform(tensor, type_str) for tensor in result)


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

    Reference:
        Khatri, Sumeet, and Mark M. Wilde. "Principles of quantum communication theory: A modern approach."
        arXiv preprint arXiv:2011.04672 (2020).
        Watrous, J. . "Semidefinite Programs for Completely Bounded Norms."
        Theory of Computing 5.1(2009):217-238.
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
    U: Union[np.ndarray, torch.Tensor], V: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    r"""calculate the fidelity between gates

    .. math::

        F(U, V) = |\text{tr}(UV^\dagger)|/2^n

    :math:`U` is a :math:`2^n\times 2^n` unitary gate

    Args:
        U: quantum gate :math:`U`  in matrix form
        V: quantum gate :math:`V`  in matrix form

    Returns:
        fidelity between gates

    """
    type_u, type_v = _type_fetch(U), _type_fetch(V)
    U, V = _type_transform(U, "tensor"), _type_transform(V, "tensor")
    assert U.shape == V.shape, "The shape of two matrices are different"

    fidelity = utils.qinfo._gate_fidelity(U, V.to(dtype=U.dtype))

    return (
        fidelity.detach().numpy()
        if type_u == "numpy" and type_v == "numpy"
        else fidelity
    )


def logarithmic_negativity(
    density_op: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Logarithmic Negativity of the input quantum state.
    """
    # Calculate the negativity
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "state").density_matrix

    log_neg = utils.qinfo._logarithmic_negativity(density_op)

    return log_neg.detach().numpy() if type_str == "numpy" else log_neg


def mana(matrix: Union[np.ndarray, torch.Tensor, State], input_str: str,
         out_dim: Optional[int]=None) -> Union[np.ndarray, torch.Tensor, State]:
    r"""Compute the mana of states or channels

    Args:
        matrix: quantum state or channel, when "channel", it should be the choi matrix of channel.
        input_str: "state" or "channel"
        out_dim: output system dimension, only need to compute mana of channel.
    
    Returns:
        the output mana
    
    """
    type_str = _type_fetch(matrix)
    if input_str == "channel":
        in_dim = matrix.shape[-1] // out_dim
        A_a = phase_space_point(in_dim)
        A_b = phase_space_point(out_dim)
        mana_chan = utils.qinfo._mana_channel(matrix, A_a, A_b, out_dim, in_dim)
        return mana_chan.detach().numpy() if type_str == "numpy" else mana_chan
    elif input_str == "state":
        state = _type_transform(matrix, "state").density_matrix
        d = state.shape[-1]
        # generate the phase operator
        A = phase_space_point(d)
        mana_state = utils.qinfo._mana_state(state, A=A, dim=d)
        return mana_state.detach().numpy() if type_str == "numpy" else mana_state

    raise ValueError("Invalid input. Please enter 'state' or 'channel'.")


def negativity( #Todo
    density_op: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Compute the Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Negativity of the input quantum state.
    """
    # Implement the partial transpose
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "state").density_matrix

    neg = utils.qinfo._negativity(density_op)

    return neg.detach().numpy() if type_str == "numpy" else neg


# TODO
def pauli_str_convertor(observable: List) -> List:
    r"""Concatenate the input observable with coefficient 1.

    For example, if the input ``observable`` is ``[['z0,x1'], ['z1']]``,
    then this function returns the observable ``[[1, 'z0,x1'], [1, 'z1']]``.

    Args:
        observable: The observable to be concatenated with coefficient 1.

    Returns:
        The observable with coefficient 1
    """

    for i in range(len(observable)):
        assert len(observable[i]) == 1, "Each term should only contain one string"

    return [[1, term] for term in observable]


def purity(
    rho: Union[np.ndarray, torch.Tensor, State]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the purity of a quantum state.

    .. math::

        P = \text{tr}(\rho^2)

    Args:
        rho: Density matrix form of the quantum state.

    Returns:
        The purity of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "state").density_matrix

    gamma = utils.qinfo._purity(rho)

    return gamma.detach().numpy() if type_rho == "numpy" else gamma


def relative_entropy(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
    base: Optional[int] = 2,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the relative entropy of two quantum states.

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

    Args:
        rho: Density matrix form of the quantum state.
        sigma: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        Relative entropy between input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix
    assert rho.shape == sigma.shape, "The shape of two quantum states are different"

    entropy = utils.qinfo._relative_entropy(rho, sigma.to(rho.dtype), base)

    return (
        entropy.detach().numpy()
        if type_rho == "numpy" and type_sigma == "numpy"
        else entropy
    )


def stab_nullity(unitary: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    r"""Tool for calculation of unitary-stabilizer nullity.

    Args:
        unitary: A batch of unitary matrices.

    Returns:
        Unitary-stabilizer nullity for each unitary matrix.
    """
    type_str = _type_fetch(unitary)
    unitary = _type_transform(unitary, "tensor")
    
    # Check if the input is a single matrix or a batch of matrices
    is_single_matrix = unitary.dim() == 2
    # Check if the matrix/matrices are unitary
    is_unitary_matrices = utils.check._is_unitary(unitary)
    
    if is_single_matrix:
        unitary = unitary.unsqueeze(0)  # Shape (1, 2^n, 2^n), expand dimensions for broadcasting
        unitary_indices = 0
        num_unitary = unitary.shape[0]   
        if not is_unitary_matrices:
          raise ValueError("The input matrix is not unitary.")
    else:
        is_unitary_list = is_unitary_matrices.tolist() if torch.is_tensor(is_unitary_matrices) else is_unitary_matrices
        unitary_indices = torch.tensor([i for i, unitary in enumerate(is_unitary_list) if unitary], dtype=torch.long)

        if len(unitary_indices) == 0:
            raise ValueError("All matrices are not unitary. Exiting the program.")
        
        if len(unitary_indices) != len(is_unitary_list):
            non_unitary_indices = [i for i, unitary in enumerate(is_unitary_list) if not unitary]
            print(f"The following matrices are not unitary and will be ignored: {non_unitary_indices}")
     # Store the number of unitary and filter out non-unitary matrices
        num_unitary = unitary.shape[0]    
        unitary = unitary[unitary_indices]
    # Second identify the unitary matrix dimension to get n (n-qubit)
    n=int(math.log2(unitary.shape[-1]))
    #Pauli basis*(sqrt(2)^n) to get Pauli group
    pauli=(pauli_basis(n) * math.sqrt(2)**n).to(torch.complex128)
    
    return _type_transform(
           utils.qinfo._stab_nullity(
            unitary=unitary,num_unitary = num_unitary, unitary_indices = unitary_indices,pauli=pauli,n=n
        ),
        type_str,
     )


def stab_renyi(density: Union[np.ndarray, torch.Tensor,State],alpha: Union[np.ndarray, torch.Tensor, float]) -> Union[np.ndarray, torch.Tensor]:
    r"""Tool for calculation of stabilizer renyi entropy.

    Args:
        density: A batch of density matrices.

    Returns:
        Stabilizer renyi entropy for each density matrix
    """
    type_str = _type_fetch(density)
    alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
    density = _type_transform(density, "state").density_matrix
    
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    
   
    # Check if the input is a single matrix or a batch of matrices
    is_single_matrix = (density.dim() == 2)

    #First check if it is density matrices
    is_density = utils.check._is_density_matrix(density,eps=1e-4)
    if is_single_matrix:
        density = density.unsqueeze(0)  # Shape (1, 2^n, 2^n), expand dimensions for broadcasting
        density_indices = 0
        num_density = density.shape[0]   
        if not is_density:
            raise ValueError("The input matrix is not density matrix.")
    else:
        # Ensure is_density_list is a list of individual boolean values
        is_density_list = is_density.tolist() if torch.is_tensor(is_density) else is_density
        density_indices = torch.tensor([i for i, density in enumerate(is_density_list) if density], dtype=torch.long)

        if len(density_indices) == 0:
            raise ValueError("All matrices are not density matrices. Exiting the program.")
        
        if len(density_indices) != len(is_density_list):
            non_density_indices = [i for i, density in enumerate(is_density_list) if not density]
            print(f"The following matrices are not density matrices and will be ignored: {non_density_indices}")
     # Store the number of density and filter out non-density matrices
        num_density = density.shape[0]   
        density = density[density_indices] 
    
    # Generate pauli 
    # Pauli basis*(sqrt(2)^n) to get Pauli str
    n = int(math.log2(density.shape[-1])) # n-qubit
    pauli = (pauli_basis(n) * math.sqrt(2)**n).to(torch.complex128)

    renyi = utils.qinfo._stab_renyi(
            density=density,alpha=alpha, num = num_density,indices = density_indices,pauli=pauli,n=n
        ),
   
    return renyi.detach().numpy() if type_str == "numpy" else renyi 


def state_fidelity(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the fidelity of two quantum states.

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The fidelity between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix

    fidelity = utils.qinfo._state_fidelity(rho, sigma.to(rho.dtype))

    return (
        fidelity.detach().numpy()
        if type_rho == "numpy" and type_sigma == "numpy"
        else fidelity
    )


def trace_distance(
    rho: Union[np.ndarray, torch.Tensor, State],
    sigma: Union[np.ndarray, torch.Tensor, State],
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the trace distance of two quantum states.

    .. math::
        D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The trace distance between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "state").density_matrix
    sigma = _type_transform(sigma, "state").density_matrix
    assert rho.shape == sigma.shape, "The shape of two quantum states are different"

    dist = utils.qinfo._trace_distance(rho, sigma.to(rho.dtype))
    return (
        dist.detach().numpy() if type_rho == "numpy" and type_sigma == "numpy" else dist
    )


def von_neumann_entropy(
    rho: Union[np.ndarray, torch.Tensor, State], base: Optional[int] = 2
) -> Union[np.ndarray, torch.Tensor]:
    r"""Calculate the von Neumann entropy of a quantum state.

    .. math::

        S = -\text{tr}(\rho \log(\rho))

    Args:
        rho: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        The von Neumann entropy of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "state").density_matrix

    entropy = utils.qinfo._von_neumann_entropy(rho, base)

    return entropy.detach().numpy() if type_rho == "numpy" else entropy
