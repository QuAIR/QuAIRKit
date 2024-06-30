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

from typing import Dict, List, Optional, Type, Union

import numpy as np
import torch

from .. import get_backend, get_dtype, utils
from . import MixedState, PureState, State

__state_dict: Dict[str, State] = {'state_vector': PureState, 'density_matrix': MixedState}


# TODO: should be default instead of choosing state_vector
def __state_convert(input_state: State, backend: str) -> State:
    r"""Copy and convert the input state to the expected state.
    
    Args:
        state: the input state.
        backend: the backend of the input state.
    
    Returns:
        state in the specified backend.
    
    """
    if backend == input_state.backend:
        return input_state.clone()
    
    initializer = __state_dict[backend]
    return initializer(input_state.fit(backend), input_state.system_dim, input_state.system_seq)


def __fetch_default_init(list_dim: List[int]) -> Type[State]:
    r"""Determine whether the input dimension should be a (batch of) pure/mixed state in ``'default'`` backend.
    
    Note:
        The convention is given by the table in ``to_state``.
    """
    if len(list_dim) == 1:
        return PureState

    if len(list_dim) == 2:
        return MixedState if list_dim[0] == list_dim[1] else PureState
    
    # batch case
    return MixedState if list_dim[-1] > 1 else PureState


def to_state(
        data: Union[torch.Tensor, np.ndarray, State], 
        sys_dim: Optional[Union[int, List[int]]] = 2,
        dtype: Optional[str] = None,
        state_backend: Optional[str] = None
) -> State:
    r"""The function to generate a specified state instance.

    Args:
        data: a representation of the quantum state in allowable backend, or an instance of the State class.
        sys_dim: (list of) dimension(s) of the systems, can be a list of integers or an integer.
        For example, ``sys_dim = 2`` means the systems are qubits (default setup); ``sys_dim = [2, 3]`` means 
        the first system is a qubit and the second is a qutrit.
        dtype: Used to specify the data dtype of the data. Defaults to the global setup.
        state_backend: The backend of the state. Specified only when the input data is an instance of the State class.

    Returns:
        The generated quantum state.
    
    Note:
        When the ``backend`` is set as ``'default'``, the backend of this state is determined by the table
        
        +----------------+---------------------+---------------------+
        |                | single              | batch               |
        +================+=====================+=====================+
        | state_vector   | [d], [1, d], [d, 1] | [d1, ..., dn, d, 1] |
        +----------------+---------------------+---------------------+
        | density_matrix | [d, d]              | [d1, ..., dn, d, d] |
        +----------------+---------------------+---------------------+
    
    """
    if isinstance(data, State):
        assert state_backend is not None, \
            "The backend that the input state converts to must be specified."
        return __state_convert(data, state_backend)
    
    dtype = get_dtype() if dtype is None else dtype
    data = data.to(dtype=dtype) if isinstance(data, torch.Tensor) else \
        torch.from_numpy(data).to(dtype=dtype)
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    list_dim = data.shape
    
    if get_backend() == 'default':
        initializer = __fetch_default_init(list_dim)
    else:
        raise NotImplementedError(
            f"the backend is not recognized or implemented: receive {get_backend()}")
    
    num_systems = initializer.check(data, sys_dim)
    sys_dim = [sys_dim] * num_systems if isinstance(sys_dim, int) else sys_dim
    return initializer(data, sys_dim)


def image_to_density_matrix(image_filepath: str) -> State:
    r"""Encode image to density matrix

    Args:
        image_filepath: Path to the image file.

    Returns:
        The density matrix obtained by encoding
    """
    import matplotlib
    image_matrix = matplotlib.image.imread(image_filepath)

    # Converting images to grayscale
    image_matrix = image_matrix.mean(axis=2)

    # Fill the matrix so that it becomes a matrix whose shape is [2**n,2**n]
    length = int(2 ** np.ceil(np.log2(np.max(image_matrix.shape))))
    image_matrix = np.pad(
        image_matrix,
        ((0, length - image_matrix.shape[0]), (0, length - image_matrix.shape[1])),
        "constant",
    )
    # Density matrix whose trace  is 1
    rho = image_matrix @ image_matrix.T
    rho = rho / np.trace(rho)
    return to_state(rho)



def tensor_state(state_a: State, state_b: State, *args: State) -> State:
    r"""calculate tensor product (kronecker product) between at least two state. This function automatically returns State instance

    Args:
        state_a: State
        state_b: State
        *args: other states

    Returns:
        tensor product state of input states

    Note:
        Need to be careful with the backend of states; 
        Use ``quairkit.linalg.NKron`` if the input datatype is ``torch.Tensor`` or ``numpy.ndarray``.

    """
    return to_state(utils.qinfo._nkron(state_a.density_matrix, state_b.density_matrix, [st.density_matrix for st in args]))
