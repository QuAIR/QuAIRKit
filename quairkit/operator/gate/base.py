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
The source file of the basic class for the quantum gates.
"""

from typing import Callable, Iterable, List, Union

import matplotlib
import torch

from ...core import (OperatorInfoType, StateSimulator, get_float_dtype,
                     intrinsic, utils)
from ..channel import Channel


class Gate(Channel):
    r"""Base class for quantum gates.

    Args:
        matrix: the matrix of this gate. Defaults to ``None`` i.e. not specified.
        system_idx: indices of the systems that this gate acts on. Defaults to ``None``. i.e. list(range(# of acted systems)).
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        check_legality: whether check the completeness of the matrix if provided. Defaults to ``True``.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
            Defaults to ``None``.
    """

    def __init__(
            self, matrix: torch.Tensor = None, system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            acted_system_dim: Union[List[int], int] = 2, check_legality: bool = True, gate_info: dict = None
    ):
        assert not (matrix is None and isinstance(acted_system_dim, int)), (
            "Received None for matrix and integer for acted_system_dim: "
            "either one of them must be specified to initialize a Gate instance.")
        
        super().__init__('gate', matrix, system_idx, acted_system_dim, check_legality, gate_info)
        
        self.__matrix = matrix if matrix is None else matrix.to(dtype=self.dtype, device=self.device).contiguous()
        self._is_dagger: bool = False
        self._is_hermitian: bool = False
        
    def __call__(self, state: StateSimulator) -> StateSimulator:
        return self.forward(state)

    @property
    def matrix(self) -> torch.Tensor:
        r"""Unitary matrix of this gate

        Raises:
            ValueError: Need to specify the matrix form in this Gate instance.

        """
        if self.__matrix is None:
            raise ValueError(
                "Need to specify the matrix form in this Gate instance.")
        return utils.linalg._dagger(self.__matrix) if self._is_dagger else self.__matrix
    
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this gate.
        """
        info = super().info
        if self._is_dagger and (tex := info.get('tex', None)):
            info['tex'] = r'{' + tex + r'}^\dagger'
        return info

    def forward(self, state: StateSimulator) -> StateSimulator:
        state._evolve_many(self.matrix, self.system_idx, on_batch=True)
        return state

    def dagger(self) -> None:
        r"""Change the dagger and tex info and system_idx of this gate.
            No changes will be made for some hermitian gates.
        """
        self.system_idx = list(reversed(self.system_idx))
        if self._is_dagger:
            self._is_dagger = False
            return
        
        if self._is_hermitian:
            return
        self._is_dagger = True


class ParamGate(Gate):
    r"""Base class for quantum parameterized gates.

    Args:
        generator: function that generates the unitary matrix of this gate.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        system_idx: indices of the qubits on which this gate acts on. Defaults to ``None``.
            i.e. list(range(num_acted_qubits)).
        num_acted_param: the number of parameters required for a single operation.
        param_sharing: whether all operations are shared by the same parameter set.
        acted_system_dim: dimension of systems that this gate acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
        check_legality: whether check the completeness of the matrix if provided. Defaults to ``True``.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit.
            Defaults to ``None``.
        support_batch: whether generator support batch inputs. Defaults to ``True``.
    """

    def __init__(
        self, generator: Callable[[torch.Tensor], torch.Tensor],
        param: Union[torch.Tensor, float, List[float]] = None,
        num_acted_param: int = 1, param_sharing: bool = False,
        system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        acted_system_dim: Union[List[int], int] = 2, check_legality: bool = True, gate_info: dict = None,
        support_batch: bool = True
    ):
        if isinstance(acted_system_dim, int):
            ex_param = torch.randn([1, num_acted_param], dtype=get_float_dtype())
            ex_matrix = generator(ex_param)
            ex_matrix = ex_matrix[0] if isinstance(ex_matrix, torch.Tensor) and ex_matrix.ndim == 3 else ex_matrix
        else:
            ex_matrix = None

        super().__init__(ex_matrix, system_idx, acted_system_dim, check_legality, gate_info)
        
        intrinsic._theta_generation(self, param, self.system_idx, num_acted_param, param_sharing)
        self.param_sharing = param_sharing
        self.__generator = generator
        self.__support_batch = support_batch
    
    def __call__(self, state: StateSimulator) -> StateSimulator:
        return self.forward(state)

    @property
    def matrix(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        theta = self.theta.repeat(len(self.system_idx), 1, 1) if self.param_sharing else self.theta
        if self._is_dagger:
            theta = torch.flip(theta, [0])
        
        if self.__support_batch:
            theta2 = theta.reshape([-1, theta.shape[-1]])
            mat = self.__generator(theta2).to(device=self.device)
            return utils.linalg._dagger(mat) if self._is_dagger else mat
        
        matrices_list = [
            torch.stack([self.__generator(param) for param in theta[param_idx]])
            for param_idx in range(len(self.system_idx))
        ]
        matrices_list = torch.stack(matrices_list).squeeze().to(device=self.device)
        return utils.linalg._dagger(matrices_list) if self._is_dagger else matrices_list

    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this gate.
        """
        theta = self.theta.repeat(len(self.system_idx), 1, 1) if self.param_sharing else self.theta
        if self._is_dagger:
            theta = torch.flip(theta, [0])
        
        info = super().info
        info.update({
            'param': theta,
            'matrix': None if self.theta.shape[1] > 1 else self.matrix,
        })
        return info

    def forward(self, state: StateSimulator) -> StateSimulator:
        dim = self.dim
        matrices_list = self.matrix.view([-1, self.theta.shape[-2], dim, dim]).squeeze(-3)
        state._evolve_many_batched_groups([matrices_list], [self.system_idx], on_batch=True)
        return state
