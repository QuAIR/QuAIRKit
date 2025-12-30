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
from .visual import _base_gate_display, _base_param_gate_display


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

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float) -> float:
        r"""The display function called by circuit instance when plotting.

        Args:
            ax: the ``matplotlib.axes.Axes`` instance
            x: the start horizontal position

        Returns:
            the total width occupied

        Note:
            Users could overload this function for custom display.
        """
        return _base_gate_display(self, ax, x)

    def _single_qubit_combine_with_threshold6(self, state: StateSimulator, matrices: List[torch.Tensor]) -> None:
        r"""
        Combines single-qubit gates in a circuit with 6-qubit threshold and updates the state accordingly.

        Args:
            state: The current state, represented as a `State` object.
            matrices: A list of `torch.Tensor` objects representing the single-qubit gates to be combined.

        Returns:
            The updated state after the single-qubit gates have been combined, represented as a `State` object.
        """
        threshold_dim = 2 ** 6
        threshold_systems = threshold_dim // self.dim
        tensor_times, tensor_left = divmod(len(self.system_idx), threshold_systems)
        for threshold_idx in range(tensor_times):
            idx_beg = threshold_idx * threshold_systems
            idx_end = (threshold_idx + 1) * threshold_systems
            
            matrix = utils.linalg._nkron(*matrices[idx_beg:idx_end])
            state._evolve(matrix, sum(self.system_idx[idx_beg:idx_end], []))
        
        if tensor_left > 0:
            idx_beg = tensor_times * threshold_systems
            
            matrix = utils.linalg._nkron(*matrices[idx_beg:]) if tensor_left > 1 else matrices[idx_beg]
            state._evolve(matrix, sum(self.system_idx[idx_beg:], []))
        
        return state

    def forward(self, state: StateSimulator) -> StateSimulator:
        if self.num_acted_system == 1 and self.dim == 2:
            state = self._single_qubit_combine_with_threshold6(
                state=state, matrices=[self.matrix for _ in self.system_idx])
        else:
            for system_idx in self.system_idx:
                state._evolve(self.matrix, system_idx)

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
        # if # of acted system is unknown, generate an example matrix to run Gate.__init__
        ex_matrix = generator(torch.randn(
            [num_acted_param], dtype=get_float_dtype())) if isinstance(acted_system_dim, int) else None

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
            mat = self.__generator(theta).squeeze().to(device=self.device)
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

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        r"""The display function called by circuit instance when plotting.

        Args:
            ax: the ``matplotlib.axes.Axes`` instance
            x: the start horizontal position

        Returns:
            the total width occupied

        Note:
            Users could overload this function for custom display.
        """
        return _base_param_gate_display(self, ax, x)

    def forward(self, state: StateSimulator) -> StateSimulator:
        dim = self.dim
        matrices_list = self.matrix.view([-1, self.theta.shape[-2], dim, dim]).squeeze(-3)
        
        if self.num_acted_system == 1 and self.dim == 2:
            state = self._single_qubit_combine_with_threshold6(
                state=state, matrices=matrices_list)
        else:
            for param_idx, system_idx in enumerate(self.system_idx):
                state._evolve(matrices_list[param_idx], system_idx)
        
        return state
