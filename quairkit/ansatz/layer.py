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
The source file of the class for quantum circuit templates.
"""

import math
from typing import List, Optional

import torch

from ..operator import CNOT, RY, RZ, U3, H, ParamGate, S
from .container import Layer

__all__ = ['LinearEntangledLayer', 'RealEntangledLayer', 'ComplexEntangledLayer', 'ComplexBlockLayer', 'RealBlockLayer', 'Universal2', 'Universal3']


def _format_param_with_flat_batch(
        param: Optional[torch.Tensor], total_param_num: int, layer_name: str
) -> torch.Tensor:
    r"""Format layer parameters to ``[batch_size, total_param_num]``.

    The accepted input formats are:
    - ``None``: random initialization with shape ``[1, total_param_num]``.
    - single-sample tensor with ``numel == total_param_num``.
    - batched tensor with shape ``[batch_size, total_param_num]``.
    Legacy class-specific batched layouts are not accepted.
    """
    assert total_param_num >= 0, (
        f"Invalid total parameter number in {layer_name}: received {total_param_num}.")

    if param is None:
        return torch.rand([1, total_param_num]) * 2 * torch.pi

    assert isinstance(param, torch.Tensor), (
        f"Parameter in {layer_name} should be a torch.Tensor, received {type(param)}.")

    if total_param_num == 0:
        assert param.numel() == 0, (
            f"Number of parameters does not match in {layer_name}: "
            f"received {list(param.shape)}, expected 0.")
        return param.reshape([1, 0])

    if param.ndim == 2 and int(param.shape[1]) == total_param_num:
        return param.reshape([-1, total_param_num])

    if param.numel() == total_param_num:
        return param.reshape([1, total_param_num])

    assert False, (
        f"Batched parameter shape mismatch in {layer_name}: "
        f"received {list(param.shape)}, expected [batch_size, {total_param_num}] "
        f"or single-sample numel {total_param_num}.")


class LinearEntangledLayer(Layer):
    r"""Linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        depth: Number of layers.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, 2 * depth * len(qubits_idx)]``. Single-sample input
            with ``numel == 2 * depth * len(qubits_idx)`` is also accepted.
    
    """
    def __init__(self, qubits_idx: List[int], depth: int,
                 param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, depth, 'Linear Entangled Layer')
        self._assert_qubits()

        num_qubits = len(qubits_idx)
        flat_param = _format_param_with_flat_batch(
            param, depth * 2 * num_qubits, layer_name=self.name)
        param = flat_param.reshape([flat_param.shape[0], depth, 2, num_qubits]).permute(1, 2, 3, 0)
        self.__add_layer(param, self.system_idx, self.num_systems)
    
    def __add_layer(self, list_param: torch.Tensor, qubits_idx: List[int], num_qubits: int) -> None:
        acted_list = [(qubits_idx[idx], qubits_idx[idx + 1])
                      for idx in range(num_qubits - 1)]
        
        for param in list_param:
            self.append(RY(qubits_idx, param=torch.nn.Parameter(param[0].unsqueeze(-1))))
            self.append(CNOT(qubits_idx=acted_list))
            self.append(RZ(qubits_idx, param=torch.nn.Parameter(param[1].unsqueeze(-1))))
            self.append(CNOT(qubits_idx=acted_list))


class RealEntangledLayer(Layer):
    r"""Strongly entangled layers consisting of Ry gates and CNOT gates.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        depth: Number of layers. Defaults to ``1``.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, depth * len(qubits_idx)]``. Single-sample input
            with ``numel == depth * len(qubits_idx)`` is also accepted.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/pdf/1905.10876.pdf.
    """
    def __init__(self, qubits_idx: List[int], depth: int,
                 param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, depth, 'Real Entangled Layer')
        self._assert_qubits()

        num_qubits = len(qubits_idx)
        flat_param = _format_param_with_flat_batch(
            param, depth * num_qubits, layer_name=self.name)
        param = flat_param.reshape([flat_param.shape[0], depth, num_qubits]).permute(1, 2, 0)
        self.__add_layer(param, self.system_idx, self.num_systems)
    
    def __add_layer(self, list_param: torch.Tensor, qubits_idx: List[int], num_qubits: int) -> None:
        acted_list = [(qubits_idx[idx], qubits_idx[(idx + 1) % num_qubits])
                      for idx in range(num_qubits)]

        for param in list_param:
            self.append(RY(qubits_idx, param=torch.nn.Parameter(param.unsqueeze(-1))))
            self.append(CNOT(qubits_idx=acted_list))
            

class ComplexEntangledLayer(Layer):
    r"""Strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        depth: Number of layers. Defaults to ``1``.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, 3 * depth * len(qubits_idx)]``. Single-sample input
            with ``numel == 3 * depth * len(qubits_idx)`` is also accepted.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.
        This ansatz is from the following paper: https://arxiv.org/abs/1804.00633.
    """
    def __init__(self, qubits_idx: List[int], depth: int,
                 param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, depth, 'Complex Entangled Layer')
        self._assert_qubits()

        num_qubits = len(qubits_idx)
        flat_param = _format_param_with_flat_batch(
            param, depth * num_qubits * 3, layer_name=self.name)
        param = flat_param.reshape([flat_param.shape[0], depth, num_qubits, 3]).permute(1, 2, 0, 3)
        self.__add_layer(param, self.system_idx, self.num_systems)
    
    def __add_layer(self, list_param: torch.Tensor, qubits_idx: List[int], num_qubits: int) -> None:
        acted_list = [(qubits_idx[idx], qubits_idx[(idx + 1) % num_qubits])
                      for idx in range(num_qubits)]

        for param in list_param:
            self.append(U3(qubits_idx, param=torch.nn.Parameter(param)))
            self.append(CNOT(qubits_idx=acted_list))


class RealBlockLayer(Layer):
    r"""Weakly entangled layers consisting of CNOT gates surrounded by RY gates

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        depth: Number of layers. Defaults to ``1``.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, 2 * depth * (len(qubits_idx) - 1) + len(qubits_idx)]``.
            Single-sample input with matching ``numel`` is also accepted.

    Note:
        The mathematical representation of this layer of quantum gates is a real unitary matrix.
    
    """
    def __init__(self, qubits_idx: List[int], depth: int,
                 param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, depth, 'Real Block Layer')
        self._assert_qubits()
        
        num_qubits = self.num_systems
        num_param = 2 * depth * (num_qubits - 1) + num_qubits
        flat_param = _format_param_with_flat_batch(
            param, num_param, layer_name=self.name)
        param = flat_param.transpose(0, 1)
        self.__add_layer(param, self.system_idx, num_qubits, depth)
    
    def __add_layer(self, list_param: torch.Tensor, qubits_idx: List[int], num_qubits: int, depth: int) -> None:
        batch_size = list_param.shape[1]
        list_param, end_param = list_param[:-num_qubits], list_param[-num_qubits:]
        for param in list_param.view([depth, -1, batch_size]):
            param_0, param_1 = param[:num_qubits], param[num_qubits:]
            self.__add_ry_layer(param_0, qubits_idx, [0, num_qubits - 1])
            
            if num_qubits % 2 == 0:
                self.__add_cnot_layer(qubits_idx, [0, num_qubits - 1])
                if num_qubits > 2:
                    self.__add_ry_layer(param_1, qubits_idx, [1, num_qubits - 2])
                    self.__add_cnot_layer(qubits_idx, [1, num_qubits - 2])
            else:
                self.__add_cnot_layer(qubits_idx, [0, num_qubits - 2])
                self.__add_ry_layer(param_1, qubits_idx, [1, num_qubits - 2])
                self.__add_cnot_layer(qubits_idx, [1, num_qubits - 1])
        
        self.__add_ry_layer(end_param, qubits_idx, [0, num_qubits - 1])

    def __add_cnot_layer(self, qubits_idx: List[int], position: List[int]) -> None:
        cnot_acted_list = [[qubits_idx[i], qubits_idx[i+1]] for i in range(position[0], position[1], 2)]
        self.append(CNOT(cnot_acted_list))

    def __add_ry_layer(self, param: torch.Tensor, qubits_idx: List[int], position: List[int]):
        ry_acted_list = [[qubits_idx[i]] for i in range(position[0], position[1] + 1)]
        self.append(RY(ry_acted_list, param=torch.nn.Parameter(param.unsqueeze(-1))))


class ComplexBlockLayer(Layer):
    r"""Weakly entangled layers consisting of CNOT gates surrounded by U3 gates

    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        depth: Number of layers. Defaults to ``1``.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, (2 * depth * (len(qubits_idx) - 1) + len(qubits_idx)) * 3]``.
            Single-sample input with matching ``numel`` is also accepted.

    Note:
        The mathematical representation of this layer of quantum gates is a complex unitary matrix.
    
    """
    def __init__(self, qubits_idx: List[int], depth: int,
                 param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, depth, 'Complex Block Layer')
        self._assert_qubits()
        
        num_qubits = self.num_systems
        num_u3 = 2 * depth * (num_qubits - 1) + num_qubits
        flat_param = _format_param_with_flat_batch(
            param, num_u3 * 3, layer_name=self.name)
        param = flat_param.reshape([flat_param.shape[0], num_u3, 3]).permute(1, 0, 2)
        self.__add_layer(param, self.system_idx, num_qubits, depth)
    
    def __add_layer(self, list_param: torch.Tensor, qubits_idx: List[int], num_qubits: int, depth: int) -> None:
        batch_size = list_param.shape[1]
        list_param, end_param = list_param[:-num_qubits], list_param[-num_qubits:]
        for param in list_param.view([depth, -1, batch_size, 3]):
            param_0, param_1 = param[:num_qubits], param[num_qubits:]
            self.__add_u3_layer(param_0, qubits_idx, [0, num_qubits - 1])
            
            if num_qubits % 2 == 0:
                self.__add_cnot_layer(qubits_idx, [0, num_qubits - 1])
                if num_qubits > 2:
                    self.__add_u3_layer(param_1, qubits_idx, [1, num_qubits - 2])
                    self.__add_cnot_layer(qubits_idx, [1, num_qubits - 2])
            else:
                self.__add_cnot_layer(qubits_idx, [0, num_qubits - 2])
                self.__add_u3_layer(param_1, qubits_idx, [1, num_qubits - 2])
                self.__add_cnot_layer(qubits_idx, [1, num_qubits - 1])
        
        self.__add_u3_layer(end_param, qubits_idx, [0, num_qubits - 1])

    def __add_cnot_layer(self, qubits_idx: List[int], position: List[int]) -> None:
        cnot_acted_list = [[qubits_idx[i], qubits_idx[i+1]] for i in range(position[0], position[1], 2)]
        self.append(CNOT(cnot_acted_list))

    def __add_u3_layer(self, param: torch.Tensor, qubits_idx: List[int], position: List[int]):
        u3_acted_list = [[qubits_idx[i]] for i in range(position[0], position[1] + 1)]
        self.append(U3(u3_acted_list, param=torch.nn.Parameter(param)))


class Universal2(Layer):
    r"""A circuit layer representing universal two-qubit gates. One of such a layer requires 15 parameters.
    
    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, 15]``. Single-sample input with ``numel == 15`` is
            also accepted.
    
    """
    def __init__(self, qubits_idx: List[int], param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, 1, 'Universal 2-qubit Layer')
        self._assert_qubits()
        
        assert len(qubits_idx) == 2, \
            f"The width of the layer should be 2: received indices {qubits_idx}."
        
        param = _format_param_with_flat_batch(param, 15, layer_name=self.name).transpose(0, 1)
        self.__add_layer(param, self.system_idx)
    
    def __add_layer(self, theta: torch.Tensor, qubits_idx: List[int]) -> None:
        batch_size = theta.shape[1]
        u3_param_1 = theta[:6].view([2, 3, batch_size]).permute(0, 2, 1)
        u3_param_2 = theta[9:].view([2, 3, batch_size]).permute(0, 2, 1)
        
        self.append(U3([qubits_idx[0], qubits_idx[1]], param=torch.nn.Parameter(u3_param_1)))
        self.append(CNOT(qubits_idx=[[qubits_idx[1], qubits_idx[0]]]))
        
        self.append(RZ(qubits_idx[0], param=torch.nn.Parameter(theta[6].view([1, batch_size, 1]))))
        self.append(RY(qubits_idx[1], param=torch.nn.Parameter(theta[7].view([1, batch_size, 1]))))
        self.append(CNOT(qubits_idx=[[qubits_idx[0], qubits_idx[1]]]))
        
        self.append(RY(qubits_idx[1], param=torch.nn.Parameter(theta[8].view([1, batch_size, 1]))))
        self.append(CNOT(qubits_idx=[[qubits_idx[1], qubits_idx[0]]]))
        self.append(U3([qubits_idx[0], qubits_idx[1]], param=torch.nn.Parameter(u3_param_2)))


class Universal3(Layer):
    r"""A circuit layer representing universal three-qubit gates. One of such a layer requires 81 parameters.
    
    Args:
        qubits_idx: Indices of the qubits on which the layer is applied.
        param: Initial parameters for the layer. Batched input should have shape
            ``[batch_size, 81]``. Single-sample input with ``numel == 81`` is
            also accepted.
    
    """
    def __init__(self, qubits_idx: List[int], param: Optional[torch.Tensor] = None) -> None:
        super().__init__(qubits_idx, 1, 'Universal 3-qubit Layer')
        self._assert_qubits()
        
        assert len(qubits_idx) == 3, \
            f"The width of the layer should be 3: received indices {qubits_idx}."
        
        param = _format_param_with_flat_batch(param, 81, layer_name=self.name).transpose(0, 1)
        
        self.__add_layer(param, self.system_idx)
        
    @property
    def param(self) -> torch.Tensor:
        template = next(
            (op.theta for op in self.children() if isinstance(op, ParamGate)),
            None,
        )
        if template is None:
            return torch.tensor([])

        batch_size = int(template.shape[1])
        psi, phi = [], []
        switch_to_psi = True
        current_psi_count, current_phi_count = 0, 0
        psi_chunk, phi_chunk = 15 * batch_size, 6 * batch_size
        for op in self.children():
            if isinstance(op, ParamGate):
                param = torch.flatten(op.theta)
                
                if switch_to_psi:
                    current_psi_count += len(param)
                    psi.append(param)
                    
                    if current_psi_count % psi_chunk == 0:
                        switch_to_psi = False
                        continue
                
                else:
                    current_phi_count += len(param)
                    phi.append(param)
                    
                    if current_phi_count % phi_chunk == 0:
                        switch_to_psi = True
                        continue
        if not psi and not phi:
            return torch.empty(0, dtype=template.dtype, device=template.device)
        return torch.cat(psi + phi)
        
    def __add_layer(self, theta: torch.Tensor, qubits_idx: List[int]) -> None:
        batch_size = theta.shape[1]
        psi = theta[:60].view([4, 15, batch_size])
        phi = theta[60:].view([7, 3, batch_size])
        
        self.extend(Universal2([qubits_idx[0], qubits_idx[1]], param=psi[0].transpose(0, 1)))
        self.append(U3(qubits_idx[2], param=torch.nn.Parameter(phi[0].transpose(0, 1).view([1, batch_size, 3]))))
        
        self.__block_u(phi[1], qubits_idx)
        
        self.extend(Universal2([qubits_idx[0], qubits_idx[1]], param=psi[1].transpose(0, 1)))
        self.append(U3(qubits_idx[2], param=torch.nn.Parameter(phi[2].transpose(0, 1).view([1, batch_size, 3]))))
        
        self.__block_v(phi[3], qubits_idx)
        
        self.extend(Universal2([qubits_idx[0], qubits_idx[1]], param=psi[2].transpose(0, 1)))
        self.append(U3(qubits_idx[2], param=torch.nn.Parameter(phi[4].transpose(0, 1).view([1, batch_size, 3]))))
        
        self.__block_u(phi[5], qubits_idx)
        
        self.extend(Universal2([qubits_idx[0], qubits_idx[1]], param=psi[3].transpose(0, 1)))
        self.append(U3(qubits_idx[2], param=torch.nn.Parameter(phi[6].transpose(0, 1).view([1, batch_size, 3]))))
        
    def __block_u(self, phi: torch.Tensor, qubits_idx: List[int]) -> None:
        batch_size = phi.shape[1]
        self.append(CNOT([[qubits_idx[1], qubits_idx[2]]]))
        self.append(RY(qubits_idx[1], param=torch.nn.Parameter(phi[0].view([1, batch_size, 1]))))
        
        self.append(CNOT([[qubits_idx[0], qubits_idx[1]]]))
        self.append(RY(qubits_idx[1], param=torch.nn.Parameter(phi[1].view([1, batch_size, 1]))))
        
        self.append(CNOT([[qubits_idx[0], qubits_idx[1]],
                          [qubits_idx[1], qubits_idx[2]]]))
        self.append(H(qubits_idx[2]))
        
        self.append(CNOT([[qubits_idx[1], qubits_idx[0]],
                          [qubits_idx[0], qubits_idx[2]],
                          [qubits_idx[1], qubits_idx[2]]]))
        self.append(RZ(qubits_idx[2], param=torch.nn.Parameter(phi[2].view([1, batch_size, 1]))))
        self.append(CNOT([[qubits_idx[1], qubits_idx[2]],
                          [qubits_idx[0], qubits_idx[2]]]))
        
    def __block_v(self, phi: torch.Tensor, qubits_idx: List[int]) -> None:
        batch_size = phi.shape[1]
        self.append(CNOT([[qubits_idx[2], qubits_idx[0]],
                          [qubits_idx[1], qubits_idx[2]],
                          [qubits_idx[2], qubits_idx[1]]]))
        
        self.append(RY(qubits_idx[2], param=torch.nn.Parameter(phi[0].view([1, batch_size, 1]))))
        self.append(CNOT([[qubits_idx[1], qubits_idx[2]]]))
        self.append(RY(qubits_idx[2], param=torch.nn.Parameter(phi[1].view([1, batch_size, 1]))))
        self.append(CNOT([[qubits_idx[1], qubits_idx[2]]]))
        
        self.append(S(qubits_idx[2]))
        self.append(CNOT([[qubits_idx[2], qubits_idx[0]],
                          [qubits_idx[0], qubits_idx[1]],
                          [qubits_idx[1], qubits_idx[0]]]))
        
        self.append(H(qubits_idx[2]))
        self.append(CNOT([[qubits_idx[0], qubits_idx[2]]]))
        self.append(RZ(qubits_idx[2], param=torch.nn.Parameter(phi[2].view([1, batch_size, 1]))))
        self.append(CNOT([[qubits_idx[0], qubits_idx[2]]]))
