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

import numpy as np
import torch

from ..operator import RX, RY, RZ, RZZ, H, Oracle, X
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
        data: Integer to be encoded (must be in [0, 2**n - 1] where n is number of qubits).
    """

    def __init__(self, data: int, qubits_idx: List[int]) -> None:
        super().__init__(qubits_idx, 1, 'Basis Encoding')
        self._assert_qubits()
        n_qubits = len(qubits_idx)
        max_val = 2 ** n_qubits - 1

        if data < 0 or data > max_val:
            raise ValueError(f"Data must be in [0, {max_val}], got {data}")
        self.__add_layer(data)

    def __add_layer(self, data: int) -> None:
        bin_str = bin(data)[2:].zfill(self.num_systems)
        flip_qubits = []
        flip_qubits.extend(
            self.system_idx[i] for i, bit in enumerate(bin_str) if bit == '1'
        )
        if flip_qubits:
            self.append(X(flip_qubits))


class AmplitudeEncoding(Layer):
    r"""Amplitude encoding for classical data.

    Encodes a vector into quantum state amplitudes:

    .. math::
        U_{\text{amp}}|0\rangle^{\otimes n} = \sum_{i=0}^{d-1} x_i |i\rangle

    where :math:`\mathbf{x}` is the normalized input vector.

    Args:
        vector: Input vector to be encoded.
        qubits_idx: Indices of the qubits on which the encoding is applied.
    
    """

    def __init__(self, vector: torch.Tensor, qubits_idx: List[int]) -> None:
        super().__init__(qubits_idx, 1, 'Amplitude Encoding')
        self._assert_qubits()
        
        vector = vector.to(dtype=self.dtype, device=self.device)
        norm = torch.norm(vector).item()
        if norm < 1e-6:
            raise ValueError(f"Input vector cannot be the zero vector: received norm {norm:.6f}")
        if np.abs(norm - 1) > 1e-4:
            warnings.warn("Input vector is not normalized: received norm {norm:.4f}. Auto-normalizing.")
            vector = vector / norm

        self.__add_layer(vector)

    def __add_layer(self, vector: torch.Tensor) -> None:
        state_vector = torch.nn.functional.pad(vector, (0, 2 ** self.num_systems - vector.shape[0]))

        oracle = self._householder_reflection(state_vector)
        acted_system_dim = [self.system_dim[idx] for idx in self.system_idx]

        gate_info = {
            'name': 'Amplitude Encoding',
            'tex': r'\text{' + 'Amplitude Encoding' + r'}'
        }
        self.append(Oracle(oracle, self.system_idx, acted_system_dim, gate_info))

    def _householder_reflection(self, vector) -> torch.Tensor:
        # Given a vector v, return a unitary U that U e0 = v
        length = len(vector)
        e0 = torch.zeros(length, dtype=self.dtype)
        e0[0] = 1
        if torch.allclose(vector, e0):
            return torch.eye(length)
        unitary = e0 - vector
        if torch.abs(1 - torch.dot(vector.conj(), e0)) < 1e-10:
            return torch.eye(length)
        c = 1 / (1 - torch.dot(vector.conj(), e0))
        return torch.eye(length) - c * torch.outer(unitary, unitary.conj())


class AngleEncoding(Layer):
    r"""Angle encoding for classical data.

    Encodes classical data into rotation angles:

    .. math::
        |x\rangle = U_{\text{angle}} |0\rangle^{\otimes n}
        = \bigotimes_{i=1}^{n} R_{\alpha}(\theta_i)|0\rangle^{\otimes n}

    where :math:`\alpha \in \{X, Y, Z\}` and :math:`\theta_i` are input angles.

    Args:
        angles: Input vector of angles.
        qubits_idx: Indices of the qubits on which the encoding is applied.
        rotation: Type of rotation gate ('RY', 'RZ', or 'RX').
    """
    def __init__(self, angles: torch.Tensor, qubits_idx: List[int], rotation: str = 'RY') -> None:
        super().__init__(qubits_idx, 1, 'Angle Encoding')
        self._assert_qubits()

        if len(angles) != len(qubits_idx):
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
        features: Input vector for encoding.
        set_entanglement: the set containing all pairs of qubits to be entangled using RZZ gates
        qubits_idx: Indices of the qubits on which the encoding is applied.
        depth: Number of depth
    """

    def __init__(self, features: torch.Tensor, set_entanglement: List[List[int]], 
                 qubits_idx: List[int], depth: int) -> None:
        super().__init__(qubits_idx, depth, 'IQP Encoding')
        self._assert_qubits()

        if len(features) != len(qubits_idx):
            raise ValueError("Length of features must match number of qubits")

        self.features = features.to(dtype=self.dtype, device=self.device)
        self.__add_layer(features, set_entanglement, depth)

    def __add_layer(self, features: torch.Tensor, set_entanglement: List[List[int]], depth: int) -> None:
        for _ in range(depth):
            self.append(H(self.system_idx))
            self.append(RZ(self.system_idx, features))

            for pair in set_entanglement:
                self.append(RZZ(pair, param=features[pair[0]] * features[pair[1]]))
