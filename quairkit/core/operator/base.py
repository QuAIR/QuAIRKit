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
Source file for basic operator.
"""


import copy
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch

from ..base import get_device, get_dtype
from .transpile import OpenQASM2Transpiler


class OperatorInfoType(TypedDict):
    r"""A dictionary storing basic information for Operator.

    Keys:
        - uid: Unique identifier for the operator.
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
    
    Note:
        This class is used as a type hint for `OperatorInfo`, and should not be instantiated directly.
        It is designed to provide a structured way to store and access operator information in QuAIRKit.
    
    """

    uid: str
    name: str
    type: str
    tex: str
    system_idx: List[List[int]]
    num_ctrl_system: int
    label: Union[List[str], List[int], str]
    param: torch.Tensor
    param_sharing: bool
    matrix: torch.Tensor
    api: str
    permute: List[int]
    kwargs: Dict[str, Any]
    plot_width: Optional[float]  # TODO depreciated in the next version
    
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "OperatorInfoType is used as a type hint for OperatorInfo, and hence should not be instantiated.")
    
    @property
    def qasm(self) -> Union[str, List[str]]:
        r"""Display in OpenQASM-like format
        """
        pass
    
    @property
    def qasm2(self) -> Union[str, List[str]]:
        r"""Display in OpenQASM 2.0 format
        """
        pass     


class OperatorInfo(dict):
    __slots__ = ()
    __allowed_keys = {
        "uid",
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
        "plot_width",  # TODO depreciated in the next version
    }

    def __init__(self, name: str = "unidentified", type: str = "general", **kwargs) -> None:
        uid = str(uuid.uuid4())
        super().__init__(uid=uid, name=name, type=type, **kwargs)

    def __setitem__(self, key, value):
        if key not in self.__allowed_keys:
            raise KeyError(
                f"Invalid key '{key}' for OperatorInfo, accept one of {self.__allowed_keys}."
            )

        elif key == "uid" and self.get("uid"):
            raise KeyError(
                "uid is automatically generated and should not be set manually."
            )

        elif key == "type":
            if value not in {"general", "gate", "channel", "locc"}:
                raise ValueError(
                    f"Invalid type '{value}' for OperatorInfo, accept one of "
                    "{'general', 'gate', 'channel', 'locc'}."
                )

        elif key == "system_idx":
            if not isinstance(value, List) or not all(isinstance(idx, List) for idx in value):
                raise TypeError(f"system_idx must be a list of lists: received {value}")
            if not all(isinstance(idx, int) for sublist in value for idx in sublist):
                raise TypeError(f"All indices in system_idx must be integers: received {value}")
            value = copy.deepcopy(value)

        elif key == "param":
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"param must be a torch.Tensor: received type {type(value)}")
            if value.ndim != 3:
                raise ValueError(f"param tensor must be 3-dimensional: received shape {value.shape}")
            if (original_param := self.get('param')) is not None and original_param.shape != value.shape:
                raise ValueError(
                    f"Changing param shape from {original_param.shape} to {value.shape} is not allowed."
                )

        elif key == "matrix":
            if self['type'] != "gate":
                raise ValueError(f"matrix is only valid for gate type operators, not {self['type']}.")
            if not (isinstance(value, torch.Tensor) or (value is None)):
                raise TypeError("matrix must be a torch.Tensor or None")

        super().__setitem__(key, value)
        
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        
        new_instance = OperatorInfo(name=self['name'], type=self['type'])
        memo[id(self)] = new_instance
        
        for key in self.keys():
            if key == "uid":
                continue
            value = self[key]
            
            if isinstance(value, torch.Tensor):
                # FIXME Sometimes requires_grad is not set correctly without manually setting requires_grad_(True)
                new_instance[key] = value.clone().requires_grad_(True) if value.requires_grad else value.clone()
            else:
                new_instance[key] = copy.deepcopy(value, memo)
                
        return new_instance

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @property
    def qasm(self) -> Union[str, List[str]]:
        r"""Display in OpenQASM-like format
        """
        if self.get('param', None) is not None and self['param'].shape[1] > 1:
            split = torch.split(self['param'], 1, dim=1)
            return [OpenQASM2Transpiler.format_qasm(self['name'], self['system_idx'], param) for param in split]

        return OpenQASM2Transpiler.format_qasm(self['name'], self['system_idx'], self.get('param', None))

    @property
    def qasm2(self) -> Union[str, List[str]]:
        r"""Display in OpenQASM 2.0 format
        """
        if self.get('param', None) is not None and self['param'].shape[1] > 1:
            split = torch.split(self['param'], 1, dim=1)
            return [OpenQASM2Transpiler.transpile(self['name'], self['system_idx'], param) for param in split]

        return OpenQASM2Transpiler.transpile(self['name'], self['system_idx'], self.get('param', None))


def qasm2_to_info(qasm2: str) -> List[OperatorInfoType]:
    r"""Convert an OpenQASM 2.0 string to a list of OperatorInfo instances.

    Args:
        qasm2: A complete OpenQASM 2.0 string representing a circuit.

    Returns:
        A list of OperatorInfo instances corresponding to the provided OpenQASM 2.0 string.
    """
    list_info = []
    for name, system_idx, param in OpenQASM2Transpiler.from_qasm2(qasm2):
        kwargs = {'name': name,
                  'system_idx': system_idx,
                  'api': name if name != 'cx' else 'cnot'}
        if name == 'reset':
            kwargs['type'] = 'channel'
            kwargs['tex'] = r'\ket{0}'
            kwargs['kwargs'] = {'replace_dm': None}
        elif name == 'measure':
            kwargs['type'] = 'channel'
            kwargs['kwargs'] = {'if_print': False,
                                'measure_basis': None}
        else:
            kwargs['type'] = 'gate'
        
        if param is not None:
            kwargs['param'] = param
            kwargs['param_sharing'] = False
        list_info.append(OperatorInfo(**kwargs))
    return list_info


class Operator(torch.nn.Module):
    r"""The basic class to implement the operation in QuAIRKit.

    Args:
        dtype: The data type of the operator.
            Defaults to ``None``, which means to use the default data type.

    """

    def __init__(self, dtype: Optional[str] = None):
        if dtype is None:
            self.dtype = get_dtype()
            super().__init__()
        else:
            self.dtype = dtype
            super().__init__(dtype=dtype)
        self.device = get_device()

        self._info: OperatorInfoType = OperatorInfo()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Operator) and value.dtype is None:
            value.dtype = self.dtype

    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this operator."""
        return copy.deepcopy(self._info)

    def to(self, dtype: Optional[str] = None) -> None:
        if dtype is not None:
            self.dtype = dtype
            for sub_layer in self.children():
                sub_layer.dtype = dtype

    def forward(self, *inputs, **kwargs) -> None:
        raise NotImplementedError

    def dagger(self) -> None:
        raise NotImplementedError(
            "General operations are not physically reversible. Please check whether the circuit "
            "contains non-unitary operations such as general channels or measurement.")
