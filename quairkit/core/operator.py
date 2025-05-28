# !/usr/bin/env python3
# Copyright (c) 2025 QuAIR team. All Rights Reserved.
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
Basic operator.
"""


from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import torch

from .base import Backend, get_backend, get_device, get_dtype


class Operator(torch.nn.Module):
    r"""The basic class to implement the operation in QuAIRKit.

    Args:
        backend: The backend implementation of the operator.
            Defaults to ``None``, which means to use the default backend implementation.
        dtype: The data type of the operator.
            Defaults to ``None``, which means to use the default data type.

    """

    def __init__(self, backend: Optional[Backend] = None, dtype: Optional[str] = None):
        if dtype is None:
            self.dtype = get_dtype()
            super().__init__()
        else:
            self.dtype = dtype
            super().__init__(dtype=dtype)
        self.device = get_device()
        self.backend = backend if backend is not None else get_backend()

        self._info: OperatorInfoType = OperatorInfo()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Operator):
            if value.backend is None:
                value.backend = get_backend() if self.backend is None else self.backend
            if value.dtype is None:
                value.dtype = get_dtype() if self.dtype is None else self.dtype

    @property
    def info(self) -> 'OperatorInfoType':
        r"""Information of this operator."""
        return self._info

    def to(self, backend: Optional[Backend] = None, dtype: Optional[str] = None):
        if backend is not None:
            self.backend = backend
            for sub_layer in self.children():
                sub_layer.backend = backend

        if dtype is not None:
            self.dtype = dtype
            for sub_layer in self.children():
                sub_layer.dtype = dtype

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class OperatorInfoType(TypedDict):
    r"""A dictionary storing basic information for Operator.

    Keys:
        - name: Operator name.
        - type: Operator type, can be one of:
            - 'general': General operator.
            - 'gate': Quantum gate.
            - 'channel': Quantum channel.
            - 'locc': One-way local operation and classical communication.
        - tex: LaTeX representation (without '$').
        - system_idx: Physical system index or indices the operator acts upon. Shape:
            ``[# of operators, num_acted_system]``
        - num_ctrl_system: Number of indices that control the operator.
        - label: Labels for control systems, collapse events (measurement), or measurement outcomes.
        - param: Parameter tensor for parameterized operators. Shape:
            ``[1 or len(system_idx), batch_size, num_acted_param]``
        - param_sharing: Boolean indicating if parameters are shared. Defaults to ``False``.
        - matrix: Matrix representation (if type is 'gate').
        - api: Circuit API method that invokes the operator in a circuit.
        - permute: Index representation for permutation operators.
        - kwargs: Additional keyword arguments for circuit APIs.
        - plot_width: Plot width (optional, default: ``None``).
    """
    
    name: str
    type: str
    tex: str
    system_idx: List[List[int]]
    num_ctrl_system: int
    label: Union[List[str], List[int], str]
    param: Union[Tuple[float], torch.Tensor]
    param_sharing: bool
    matrix: torch.Tensor
    api: str
    permute: List[int]
    kwargs: Dict[str, Any]

    plot_width: Optional[float]  # TODO depreciated after new matplotlib version


class OperatorInfo(dict):
    __slots__ = ()
    __allowed_keys = {
        "name",
        "type",
        "tex",
        "system_idx",
        "num_ctrl_system",
        "label",
        "param",
        "param_sharing",
        "matrix",
        "api",
        "permute",
        "kwargs",
        "plot_width",  # TODO depreciated after new matplotlib version
    }
    __open_qasm: Dict[str, Optional[str]] = {
        'x': 'x',
        'ctrl-x': 'cx',
        'rx': 'rx',
        'oracle': None,
    }
    r"""A dictionary of translating QuAIRKit operators to 
    [QpenQASM 3 operators](https://openqasm.com/language/standard_library.html#standard-library)
    
    Here value None means that the operator is not supported in OpenQASM 3.
    
    TODO: Add more operators.
    """

    def __init__(self, name: str = "unidentified", type: str = "general"):
        super().__init__(
            name=name,
            type=type,
        )

    def __setitem__(self, key, value):
        if key not in self.__allowed_keys:
            raise KeyError(
                f"Invalid key '{key}' for OperatorInfo, accept one of {self.__allowed_keys}."
            )
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @property
    def qasm(self) -> str:
        r"""Translate to open qasm format
        
        TODO: incorporate with __open_qasm
        """
        name = self['name']
        list_system_idx, idx_str = self['system_idx'], ''
        for system_idx in list_system_idx:
            idx_str += f"{name} q[{system_idx[0]}]"
            for idx in system_idx[1:]:
                idx_str += f", q[{idx}]"
            idx_str += ';\n'
        return idx_str
