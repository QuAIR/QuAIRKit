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
Common functions for the State class.
"""

import warnings
from functools import reduce
from typing import List, Optional, Type, Union

import numpy as np
import torch

from ..base import get_device, get_dtype
from ..operator import OperatorInfo, OperatorInfoType
from .backend import (DefaultSimulator, State, StateOperator, StateSimulator,
                      get_backend)


def __to_simulator_state(
    initializer: Type[StateSimulator],
    data: Union[torch.Tensor, np.ndarray, State],
    system_dim: Union[int, List[int]],
    eps: Optional[float],
    prob: Optional[List[torch.Tensor]],
) -> Union[DefaultSimulator, StateSimulator]:
    r"""Convert the input data to a state simulator."""
    if issubclass(initializer, StateOperator):
        raise NotImplementedError(
            f"Cannot convert numerical data to a state when the backend {initializer.backend} is not a simulator. "
            "Please set the backend to a simulator first, e.g., `set_backend('default')`."
        )

    dtype, device = get_dtype(), get_device()
    data = (
        data.to(dtype=dtype, device=device)
        if isinstance(data, torch.Tensor)
        else torch.tensor(data, dtype=dtype, device=device)
    )

    num_systems = initializer.check(data, system_dim, eps)
    system_dim = (
        [system_dim] * num_systems if isinstance(system_dim, int) else system_dim
    )
    return initializer(data, system_dim, list(range(num_systems)), prob)


def __to_operator_state(
    initializer: Type[StateOperator],
    data: List[OperatorInfoType],
    system_dim: Union[int, List[int]],
    eps: Optional[float],
    prob: Optional[List[torch.Tensor]],
) -> StateOperator:
    r"""Convert the input data to a state operator."""
    if eps != 1e-4:
        warnings.warn(
            "Error tolerance `eps` cannot used for the state operator that inputs a list of operators. "
        )

    if prob:
        warnings.warn(
            "Probability distribution `prob` cannot used for the state operator that inputs a list of operators. "
        )

    if issubclass(initializer, StateSimulator):
        raise NotImplementedError(
            f"Cannot convert a state to a operator when the backend {initializer.backend} is not a operator. "
        )

    if data is None:
        return initializer(data, system_dim)
    
    if isinstance(system_dim, int):
        raise NotImplementedError(
            "For state operator, `system_dim` should be a list of integers, but got an integer.")

    assert isinstance(data, List) and all(
        isinstance(op, OperatorInfo) for op in data
    ), f"Data should be a list of OperatorInfo, but got {type(data)}."

    return initializer(data, system_dim)


def to_state(
    data: Union[torch.Tensor, np.ndarray, State, Optional[List[OperatorInfoType]]],
    system_dim: Union[int, List[int]] = 2,
    eps: Optional[float] = 1e-4,
    backend: Optional[str] = None,
    prob: Optional[List[torch.Tensor]] = None,
) -> Union[DefaultSimulator, State]:
    r"""The function to generate a specified state instance.

    Args:
        data: a representation of the quantum state in allowable backend, or an instance of the State class.
        system_dim: (list of) dimension(s) of the systems, can be a list of integers or an integer.
         For example, ``system_dim = 2`` means the systems are qubits (default setup); ``system_dim = [2, 3]`` means
         the first system is a qubit and the second is a qutrit.
        eps: The tolerance for checking the validity of the input state. Can be adjusted to ``None`` to disable the check.
        backend: The name of the backend to use. If not specified, the default backend will be used.
        prob: The (list of) probability distribution of the state. The length of the list denotes the number of distributions.

    Returns:
        The generated quantum state.

    """
    if isinstance(data, State):
        if backend and data.backend != backend:
            raise NotImplementedError(
                f"Cannot convert a state to another backend. The current backend is {data.backend}, "
                f"but the target backend is {backend}. Please set the backend to {data.backend} first."
            )

        new_state = data.clone()
        system_dim = (
            [system_dim] * new_state.num_systems
            if isinstance(system_dim, int)
            else system_dim
        )
        new_state.system_dim = system_dim
        return new_state

    initializer = get_backend(backend)
    if isinstance(data, (torch.Tensor, np.ndarray)):
        return __to_simulator_state(initializer, data, system_dim, eps, prob)
    return __to_operator_state(initializer, data, system_dim, eps, prob)


def tensor_state(state_1st: StateSimulator, *args: StateSimulator) -> StateSimulator:
    r"""calculate tensor product (kronecker product) between at least two state. This function automatically returns State instance

    Args:
        state_1st: the first state
        args: other states

    Returns:
        tensor product state of input states

    Note:
        Need to be careful with the backend of states; Support broadcasting for batch states.
        Use ``quairkit.linalg.nkron`` if the input datatype is ``torch.Tensor`` or ``numpy.ndarray``.

    """
    __kron_state = lambda state_1st, state_2nd: state_1st.kron(state_2nd)
    return reduce(__kron_state, args, state_1st) if args else state_1st
