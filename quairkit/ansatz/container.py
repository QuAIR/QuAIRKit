# !/usr/bin/env python3
# Copyright (c) 2023 Paddle Quantum Authors. All Rights Reserved.
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
The source file of the Sequential class.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from ..core import Operator, get_backend, get_dtype
from ..operator import ParamGate


class OperatorList(torch.nn.Sequential):
    r"""Sequential container.

    Args:
        *operators: initial operators ready to be a sequential

    Note:
        Sublayers will be added to this container in the order of argument in the constructor.
        The argument passed to the constructor can be iterable Layers or iterable name Layer pairs.
    """
    def __init__(self, *operators: Operator) -> None:
        super().__init__(*operators)
        self.backend = get_backend()
        self.dtype = get_dtype()
        
        for operator in self:
            operator.to(backend=self.backend, dtype=self.dtype)
    
    def append(self, op: Operator) -> 'OperatorList':
        r"""Appends an given operator to the end.

        Args:
            module (nn.Module): module to append
        """
        if not isinstance(op, Operator):
            warnings.warn(
                f"OperatorList shall only append Operator: received {type(op)}", UserWarning)
        return super().append(op)
        
    def extend(self, sequential: 'OperatorList') -> 'OperatorList':
        r"""Appends an given operator to the end.

        Args:
            module (nn.Module): module to append
        """
        if not isinstance(sequential, OperatorList):
            warnings.warn(
                f"OperatorList shall only extend lists of Operator: received {type(sequential)}", UserWarning)
        return super().extend(sequential)
    
    def to(self, dtype: Optional[str] = None):
        if dtype is not None:
            self.dtype = dtype
        
        for operator in self:
            operator.to(backend=self.backend, dtype=self.dtype)
    
    @property
    def oper_history(self) -> List[Dict[str, Union[str, List[int], torch.Tensor]]]:
        r"""Return the operator history of this Sequential
        """
        oper_history = []
        for op in self.sublayers():
            if isinstance(op, ParamGate):
                oper_history.append({
                    'name': op.__class__.__name__,
                    'qubits_idx': op.qubits_idx,
                    'depth': op.depth,
                    'param': op.theta,
                    'param_sharing': op.param_sharing
                })
            elif hasattr(op, 'qubits_idx'):
                oper_history.append({
                    'name': op.__class__.__name__,
                    'qubits_idx': op.qubits_idx,
                    'depth': op.depth if hasattr(op, 'depth') else 1
                })
            else:
                warnings.warn(
                    f"Cannot recognize the operator: expected an operator with attribute qubits_idx, received {type(op)}.", UserWarning)
                oper_history.append(None)
        return oper_history
