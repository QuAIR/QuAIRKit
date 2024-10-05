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

import numpy as np
import torch
from torch.nn.parameter import Parameter

from ..core import Operator, get_backend, get_dtype, get_float_dtype
from ..operator import ParamGate


class OperatorList(torch.nn.Sequential):
    r"""Sequential container.

    Args:
        operators: initial operators ready to be a sequential

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
            op: module to append
        """
        if not isinstance(op, Operator):
            warnings.warn(
                f"OperatorList shall only append Operator: received {type(op)}", UserWarning)
        return super().append(op)

    def extend(self, sequential: 'OperatorList') -> 'OperatorList':
        r"""Appends an given operator to the end.

        Args:
            sequential: module to append
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
                    'system_idx': op.system_idx,
                    'depth': op.depth,
                    'param': op.theta,
                    'param_sharing': op.param_sharing
                })
            elif hasattr(op, 'system_idx'):
                oper_history.append({
                    'name': op.__class__.__name__,
                    'system_idx': op.system_idx,
                    'depth': op.depth if hasattr(op, 'depth') else 1
                })
            else:
                warnings.warn(
                    f"Cannot recognize the operator: expected an operator with attribute system_idx, received {type(op)}.", UserWarning)
                oper_history.append(None)
        return oper_history

    @property
    def param(self) -> torch.Tensor:
        r"""Flattened parameters in this list.
        """
        assert self._modules, \
                "The operator list is empty, please add some operators first."
        if flattened_params := [
            torch.flatten(param.clone()) for param in self.parameters()
        ]:
            concatenated_params = torch.cat(flattened_params).detach()
        else:
            concatenated_params = torch.tensor([])
        return concatenated_params

    @property
    def grad(self) -> np.ndarray:
        r"""Gradients with respect to the flattened parameters.
        """
        assert self._modules, \
            "The operator list is empty, please add some operators first."
        grad_list = []
        for param in self.parameters():
            assert param.grad is not None, (
                'The gradient is None, run the backward first before calling this property, '
                'otherwise check where the gradient chain is broken.')
            grad_list.append(param.grad.detach().numpy().flatten())
        return np.concatenate(grad_list) if grad_list != [] else grad_list

    def update_param(self, theta: Union[torch.Tensor, np.ndarray, float], 
                     idx: Optional[Union[int, None]] = None) -> None:
        r"""Replace parameters of all/one layer(s) by ``theta``.

        Args:
            theta: New parameters
            idx: Index of replacement. Defaults to None, referring to all layers.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        theta = torch.flatten(theta).to(dtype=get_float_dtype())

        if idx is None:
            assert self.param.shape == theta.shape, \
                f"the shape of input parameters is not correct: expect {self.param.shape}, received {theta.shape}"
            for layer in self:
                for name, param in layer.named_parameters():
                    num_param = int(torch.numel(param))
                    layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))

                    if num_param == theta.shape[0]:
                        return
                    theta = theta[num_param:]
        elif isinstance(idx, int):
            assert idx < len(self), f"the index is out of range, expect below {len(self)}"

            layer = self[idx]
            assert theta.shape == torch.cat([torch.flatten(param) for param in layer.parameters()]).shape, (
                "The shape of input parameters is not correct.")

            for name, param in layer.named_parameters():
                num_param = int(torch.numel(param))
                layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))

                if num_param == theta.shape[0]:
                    return
                theta = theta[num_param:]
        else:
            raise ValueError("idx must be an integer or None")

    def transfer_static(self) -> None:
        r"""
        set ``stop_gradient`` of all parameters of the list as ``True``
        """
        for layer in self:
            for name, param in layer.named_parameters():
                param.requires_grad = False
                layer.register_parameter(name, param)

    def randomize_param(self, arg0: float = 0, arg1: float = 2 * np.pi, 
                        method: str = 'Uniform') -> None:
        r"""Randomize parameters of the list based on the initializer.  
        Current we only support Uniform and Normal initializer. 

        Args:
            arg0: first argument of the initializer. Defaults to 0.
            arg1: first argument of the initializer. Defaults to 2 pi.
            method: The sampling method. Defaults to 'Uniform'.
        """
        assert method in {
            "uniform",
            "normal",
        }, "The initializer should be Uniform or Normal."

        for layer in self:
            for name, param in layer.named_parameters():

                if method == "normal":
                    new_param = Parameter(
                        torch.normal(
                            mean=arg0,
                            std=arg1,
                            size=param.shape,
                            dtype=param.dtype,
                            device=param.device,
                        )
                    )
                elif method == "uniform":
                    new_param = Parameter(
                        torch.rand(param.shape, dtype=param.dtype, device=param.device)
                        * (arg1 - arg0) + arg0
                    )
                else:
                    raise NotImplementedError

                layer.register_parameter(name, new_param)
