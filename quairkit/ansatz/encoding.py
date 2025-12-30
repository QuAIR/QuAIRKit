#!/usr/bin/env python3
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
The source file of encoding layers for classical data, including
- Basic encoding
- Amplitude encoding
- Angle encoding
- IQP encoding
"""

import warnings
from typing import List, Type, Union

import torch

from ..operator import RX, RY, RZ, RZZ, U3, H, Oracle, X
from .container import Layer

__all__ = ['BasisEncoding', 'AmplitudeEncoding', 'AngleEncoding', 'IQPEncoding']


class BasisEncoding(Layer):
    r"""Basis encoding for classical data.

    Encodes an integer into a computational basis state:

    .. math::
        U_{\text{basic}}|0\rangle^{\otimes n} = |x\rangle

    where :math:`x` is the integer to be encoded.

    Args:
        qubits_idx: Indices of the qubits on which the encoding is applied.
        data: Integer to be encoded (must be in [0, 2**n_qubits - 1]).
              If batched, must be List[int], torch.Tensor.
    """

    def __init__(self, data: Union[int, List[int], torch.Tensor], qubits_idx: List[int]) -> None:
        super().__init__(qubits_idx, 1, 'Basis Encoding')
        self._assert_qubits()
        n_qubits = len(qubits_idx)
        max_val = 2 ** n_qubits - 1
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.tensor(data, dtype=torch.int64)
        else:
            data_tensor = data.to(torch.int64)
        if torch.any(data_tensor < 0) or torch.any(data_tensor > max_val):
            raise ValueError(f"Data must be all in [0, {max_val}], got {data}")
        
        data_tensor = data_tensor.view(-1)
        if data_tensor.numel() == 1:
            self.__add_layer(data_tensor)
        else:
            self.__add_batched_layer(data_tensor)
    
    def __add_layer(self, data: int) -> None:
        bin_str = bin(data)[2:].zfill(self.num_systems)
        if flip_qubits := [
            self.system_idx[i] for i, bit in enumerate(bin_str) if bit == '1'
        ]:
            self.append(X(flip_qubits))
            
    def __add_batched_layer(self, data: torch.Tensor) -> None:
        bits_columns = [torch.tensor([int(item) for item in bin(int(data[i].item()))[2:].zfill(self.num_systems)],
                                     dtype=torch.bool).unsqueeze(1) for i in range(data.numel())]

        bits_columns = torch.cat(bits_columns, dim=1).unsqueeze(-1).repeat(1, 1, 3)

        mask = torch.zeros_like(bits_columns, dtype=torch.bool)
        mask[:, :, 0], mask[:, :, 2] = True, True
        bits_columns = torch.where(
            mask,
            torch.where(bits_columns, torch.pi, 0.0),
            0.0
        )
        self.append(U3(self.system_idx, param=bits_columns))


class AmplitudeEncoding(Layer):
    r"""Amplitude encoding for classical data.

    Encodes a vector into quantum state amplitudes:

    .. math::
        U_{\text{amp}}|0\rangle^{\otimes n} = \sum_{i=0}^{d-1} x_i |i\rangle

    where :math:`\mathbf{x}` is the normalized input vector.

    Args:
        vector: Input vector to be encoded. If batched, size must be 2^n_qubits * batch_size
        qubits_idx: Indices of the qubits on which the encoding is applied.
    
    """

    def __init__(self, vector: torch.Tensor, qubits_idx: List[int]) -> None:
        super().__init__(qubits_idx, 1, 'Amplitude Encoding')
        self._assert_qubits()
        
        vector = vector.to(dtype=self.dtype, device=self.device)
        if vector.ndim == 1:
            vector = vector.reshape(-1, 1)
        norm = torch.norm(vector, dim=0, keepdim=True)
        if torch.any(norm < 1e-6):
            raise ValueError(f"Input vector cannot be the zero vector: received norm {norm.tolist():.6f}")
        if torch.any(torch.abs(norm - 1) > 1e-4):
            warnings.warn(f"Input vector is not normalized: received norm {norm.tolist():.4f}. Auto-normalizing.")
            vector = vector / norm

        self.__add_layer(vector)

    def __add_layer(self, vector: torch.Tensor) -> None:
        oracle = self._householder_reflection(vector)
        acted_system_dim = [self.system_dim[idx] for idx in self.system_idx]

        gate_info = {
            'name': 'Amplitude Encoding',
            'tex': r'\text{' + 'Amplitude Encoding' + r'}'
        }
        self.append(Oracle(oracle, self.system_idx, acted_system_dim, gate_info))

    def _householder_reflection(self, vector: torch.Tensor) -> torch.Tensor:
        dim = 2 ** self.num_systems
        
        # Given a vector v, return a unitary U that U e0 = v
        state_vector = torch.nn.functional.pad(vector.H, (0, dim - vector.shape[0]))
        batch_size, _ = state_vector.shape
        e0 = torch.zeros(dim, dtype=vector.dtype, device=vector.device)
        e0[0] = 1
        
        unitary = e0.unsqueeze(0) - state_vector  # e0 - v_i for each row
        denom = 1 - state_vector[:, 0].conj()
        c = torch.where(torch.abs(denom) > 1e-10, 1.0 / denom, torch.zeros_like(denom))
        outer = unitary.unsqueeze(2) * unitary.conj().unsqueeze(1)
        
        identity = torch.eye(dim, dtype=vector.dtype,
                             device=vector.device).unsqueeze(0).expand(batch_size, dim, dim)
        oracle = identity - c.view(batch_size, 1, 1) * outer
        mask = torch.abs(denom) <= 1e-10
        if mask.any():
            oracle[mask] = identity[mask]
        return oracle


class AngleEncoding(Layer):
    r"""Angle encoding for classical data.

    Encodes classical data into rotation angles:

    .. math::
        |x\rangle = U_{\text{angle}} |0\rangle^{\otimes n}
        = \bigotimes_{i=1}^{n} R_{\alpha}(\theta_i)|0\rangle^{\otimes n}

    where :math:`\alpha \in \{X, Y, Z\}` and :math:`\theta_i` are input angles.

    Args:
        angles: Input vector of angles. If batched, size must be num_qubits * batch_size
        qubits_idx: Indices of the qubits on which the encoding is applied.
        rotation: Type of rotation gate ('RY', 'RZ', or 'RX').
    """

    def __init__(self, angles: torch.Tensor, qubits_idx: List[int], rotation: str = 'RY') -> None:
        super().__init__(qubits_idx, 1, 'Angle Encoding')
        self._assert_qubits()
        if angles.shape[0] != len(qubits_idx):
            raise ValueError("Length of angles must match number of qubits")

        rotation = rotation.upper()
        assert rotation in {
            'RY',
            'RZ',
            'RX',
        }, f"Invalid rotation type: {rotation}. Must be one of 'RY', 'RZ', or 'RX'."

        self.rotation = rotation
        self.angles = angles.to(dtype=self.dtype, device=self.device)
        self.__add_layer(angles)

    def __add_layer(self, angles: torch.Tensor) -> None:
        rotation_gate: Union[Type[RY], Type[RZ], Type[RX]] = {
            'RY': RY,
            'RZ': RZ,
            'RX': RX
        }[self.rotation]

        self.append(rotation_gate(self.system_idx, param=angles))


class IQPEncoding(Layer):
    r"""Instantaneous Quantum Polynomial (IQP) encoding.

    Implements the encoding:

    .. math::
        U_{\text{IQP}} = \left( \prod_{(i,j)\in S} R_{z_i z_j}(x_ix_j) \bigotimes_{k=1}^{n} R_z(x_k)  H^{\otimes n}\right)^r

    where :math:`x` is the integer to be encoded, :math:`S` is the set containing all pairs of qubits
    to be entangled using :math:`R_{zz}` gates, and :math:`r` is the depth of the circuit.

    Args:
        features: Input vector for encoding. If batched, size must be num_qubits * batch_size
        set_entanglement: the set containing all pairs of qubits to be entangled using RZZ gates
        qubits_idx: Indices of the qubits on which the encoding is applied.
        depth: Number of depth
    """

    def __init__(self, features: torch.Tensor, set_entanglement: List[List[int]],
                 qubits_idx: List[int], depth: int) -> None:
        super().__init__(qubits_idx, depth, 'IQP Encoding')
        self._assert_qubits()

        if features.shape[0] != len(qubits_idx):
            raise ValueError("Length of features must match number of qubits")

        self.features = features.to(dtype=self.dtype, device=self.device)
        self.__add_layer(features, set_entanglement, depth)

    def __add_layer(self, features: torch.Tensor, set_entanglement: List[List[int]], depth: int) -> None:
        for _ in range(depth):
            self.append(H(self.system_idx))
            self.append(RZ(self.system_idx, features))
            for pair in set_entanglement:
                if features.ndim == 1:
                    self.append(RZZ(pair, param=features[pair[0]] * features[pair[1]]))
                else:
                    self.append(RZZ(pair, param=features[pair[0], :] * features[pair[1], :]))
