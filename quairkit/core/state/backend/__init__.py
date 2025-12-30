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
The module that implements various backends of the state.
"""

from typing import Dict, Optional, Type, Union

from .base import State
from .default import DefaultSimulator, MixedState, PureState
from .operator import StateOperator
from .simulator import StateSimulator

r"""Backend classifier in the QuAIRKit.
""" 
BACKEND_LIST: Dict[str, Type[State]] = {DefaultSimulator.backend: DefaultSimulator}

r"""Default simulator for the state backend.
"""
DEFAULT_BACKEND: str = DefaultSimulator.backend


def set_backend(backend: Union[str, State]) -> None:
    r"""Set the backend implementation of QuAIRKit state.

    Args:
        backend: The name of the backend.
    """
    global BACKEND_LIST
    list_backend = list(BACKEND_LIST.keys())

    if isinstance(backend, str):
        assert backend in BACKEND_LIST, \
            f"Backend '{backend}' is not founded in the backend list: {list_backend}."

        global DEFAULT_BACKEND
        DEFAULT_BACKEND = backend
        return

    assert issubclass(
        backend, (StateOperator, StateSimulator)
    ), f"New backend should be a subclass of `StateOperator` or 'StateSimulator', but got {type(backend)}"
    
    backend_str = backend.backend
    assert backend_str is not None, \
        (f"New backend should have a non-trivial `backend` string attribute, but got {backend_str}" +
         "Please add a class attribute `backend` to the new backend class.")
    if backend_str in list_backend:
        assert backend.__name__ == (search_backend := BACKEND_LIST[backend_str].__name__), \
            (f"'{backend_str}' is already registered as {search_backend} in the backend list, "
             f"which is not the same as the new one {backend.__name__}.")
    
    BACKEND_LIST[backend_str] = backend
    DEFAULT_BACKEND = backend_str


def get_backend(backend: Optional[str] = None) -> Union[Type[StateOperator], Type[StateSimulator]]:
    r"""Get the current backend of QuAIRKit.

    Args:
        backend: The name of the backend.
    
    Returns:
        The state initializer of currently used backend.
    """
    backend = backend or DEFAULT_BACKEND
    assert backend in BACKEND_LIST, \
            f"Backend '{backend}' is not founded in the backend list: {list(BACKEND_LIST.keys())}."
    return BACKEND_LIST[DEFAULT_BACKEND]
