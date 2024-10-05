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
The source file of the classes for custom quantum channels.
"""

from typing import Iterable, List, Union

import torch

from ...core.intrinsic import _alias
from .base import Channel


class ChoiRepr(Channel):
    r"""A custom channel in Choi representation.

    Args:
        choi_repr: Choi operator of this channel.
        system_idx: Indices of the systems on which this channel acts. Defaults to ``None``.
        acted_system_dim: dimension of systems that this channel acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self,
        choi_repr: torch.Tensor,
        system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        acted_system_dim: Union[List[int], int] = 2
    ):
        super().__init__('choi', choi_repr, system_idx, acted_system_dim)


class KrausRepr(Channel):
    r"""A custom channel in Kraus representation.

    Args:
        kraus_repr: list of Kraus operators of this channel.
        system_idx: Indices of the systems on which this channel acts. Defaults to ``None``.
        acted_system_dim: dimension of systems that this channel acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self, kraus_repr: Union[torch.Tensor, List[torch.Tensor]], 
        system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        acted_system_dim: Union[List[int], int] = 2
    ):
        super().__init__('kraus', kraus_repr, system_idx, acted_system_dim)


class StinespringRepr(Channel):
    r"""A custom channel in Stinespring representation.

    Args:
        stinespring_mat: Stinespring matrix that represents this channel.
        system_idx: Indices of the systems on which this channel acts. Defaults to ``None``.
        acted_system_dim: dimension of systems that this channel acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to be qubit case.
    
    """
    @_alias({'system_idx': 'qubits_idx'})
    def __init__(
        self,
        stinespring_mat: torch.Tensor,
        system_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
        acted_system_dim: Union[List[int], int] = 2
    ):
        super().__init__('stinespring', stinespring_mat, system_idx, acted_system_dim)
