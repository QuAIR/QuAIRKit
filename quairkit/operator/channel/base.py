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
The source file of the basic class for the quantum channels.
"""

import copy
import math
import warnings
from typing import Iterable, List, Optional, Union

import torch

from ...core import Operator, OperatorInfoType, StateSimulator, utils
from ...core.intrinsic import _format_operator_idx
from ...core.utils.qinfo import (_choi_to_kraus, _choi_to_stinespring,
                                 _kraus_to_choi, _kraus_to_stinespring,
                                 _stinespring_to_choi, _stinespring_to_kraus)


class Channel(Operator):
    r"""Basic class for quantum channels.

    Args:
        type_repr: type of a representation, should be ``'choi'``, ``'kraus'``, ``'stinespring'`` or ``'gate'``.
        representation: the representation of this channel. Defaults to ``None`` i.e. not specified.
        system_idx: indices of the system on which this channel acts on. Defaults to ``None``.
            i.e. list(range(number of acted systems)).
        acted_system_dim: dimension of systems that this channel acts on. Can be a list of system dimensions 
            or an int representing the dimension of all systems. Defaults to ``None``.
        check_legality: whether check the completeness of the representation if provided. Defaults to ``True``.
        channel_info: additional information of this channel. Defaults to ``None``.

    Raises:
        ValueError: Unsupported channel representation for ``type_repr``.
        NotImplementedError: The noisy channel can only run in density matrix mode.
        TypeError: Unexpected data type for Channel representation.
        
    Note:
        Assume the input dimension is the same as the output.
    
    """
    def __init__(
        self, type_repr: str, representation: Optional[Union[torch.Tensor]] = None,
        system_idx: Optional[Union[Iterable[Iterable[int]], Iterable[int], int]] = None,
        acted_system_dim: Optional[Union[List[int], int]] = None, check_legality: Optional[bool] = True,
        channel_info: dict = None
    ) -> None:
        type_repr = type_repr.lower()
        if type_repr not in ['choi', 'kraus', 'gate', 'stinespring']:
            raise ValueError(
                "Unsupported channel representation:"
                f"require 'choi', 'kraus', 'gate', or 'stinespring', not {type_repr}")
        self.type_repr = type_repr
        super().__init__()
        self._info.update(channel_info or {})
        
        if representation is None:
            assert not isinstance(acted_system_dim, int), \
                f"Need to specify all system dimensions to get # of acted systems: received {acted_system_dim}"
            self.system_dim, num_acted_system = acted_system_dim, len(acted_system_dim)
        else:
            representation = representation.to(dtype=self.dtype, device=self.device)
            num_acted_system = getattr(self, f'_Channel__{type_repr}_init')(representation, acted_system_dim, check_legality)
        
        if system_idx is None:
            self.system_idx = [list(range(num_acted_system))]
        else:
            self.system_idx = _format_operator_idx(system_idx, num_acted_system)
        self.num_acted_system = num_acted_system
    
    def __call__(self, state: StateSimulator) -> StateSimulator:
        return self.forward(state)
    
    def __register_dim(self, input_dim: int, acted_system_dim: Union[List[int], int, None]) -> int:
        if acted_system_dim is None:
            self.system_dim, num_acted_system = [input_dim], 1
        elif isinstance(acted_system_dim, int):
            num_acted_system = int(math.log(input_dim, acted_system_dim))
            
            assert input_dim == (acted_system_dim ** num_acted_system), \
                f"The input dimension of {self.type_repr} representation does not match with acted system dimension: received {acted_system_dim}."
            
            self.system_dim = [acted_system_dim] * num_acted_system
        else:
            assert input_dim == (expec_dim := math.prod(acted_system_dim)), \
                (f"The input dimension of {self.type_repr} representation should match with system dimension {acted_system_dim}:" + 
                 f"input {input_dim}, expect {expec_dim}.")
            self.system_dim, num_acted_system = acted_system_dim, len(acted_system_dim)
        return num_acted_system

    def __choi_init(self, choi_repr: torch.Tensor, 
                    acted_system_dim: Union[List[int], int, None], 
                    check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'choi'``
        """
        input_dim = math.isqrt(choi_repr.shape[-1])
        num_acted_system = self.__register_dim(input_dim, acted_system_dim)

        if check_legality and not torch.all(utils.check._is_choi(choi_repr)):
            warnings.warn(
                "The input data may not be a Choi representation of a channel.", UserWarning)
        
        self.__choi_repr = choi_repr.contiguous()
        return num_acted_system

    def __kraus_init(self, kraus_repr: torch.Tensor, 
                     acted_system_dim: Union[List[int], int, None], 
                     check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'kraus'``
        """
        if len(kraus_repr.shape) == 2:
            kraus_repr = kraus_repr.unsqueeze(0)

        input_dim = kraus_repr.shape[-1]
        num_acted_system = self.__register_dim(input_dim, acted_system_dim)

        # sanity check
        if check_legality:
            oper_sum = (utils.linalg._dagger(kraus_repr) @ kraus_repr).sum(dim=-3)
            identity = torch.eye(input_dim).to(device=oper_sum.device)
            err = torch.norm(torch.abs(oper_sum - identity)).item()
            if err > min(1e-6 * input_dim * len(kraus_repr), 0.01):
                warnings.warn(
                    f"The input data may not be a Kraus representation of a channel: norm(sum(E * E^d) - I) = {err}.",
                    UserWarning)

        self.__kraus_repr = kraus_repr.contiguous()
        return num_acted_system
    
    def __gate_init(self, mat: torch.Tensor, 
                    acted_system_dim: Union[List[int], int, None], 
                    check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'gate'``
        """
        input_dim = mat.shape[-1]
        num_acted_system = self.__register_dim(input_dim, acted_system_dim)

        if check_legality:
            identity = torch.eye(input_dim).to(device=mat.device).expand_as(mat)
            err = torch.norm(torch.abs(utils.linalg._dagger(mat) @ mat - identity)).item()
            if err > min(1e-5 * input_dim, 0.01):
                warnings.warn(
                    f"\nThe input gate matrix may not be a unitary: norm(U * U^d - I) = {err}.", UserWarning)
        
        self.__kraus_repr = mat.unsqueeze(-3).contiguous()
        return num_acted_system

    def __stinespring_init(self, stinespring_repr: torch.Tensor, 
                           acted_system_dim: Union[List[int], int, None], 
                           check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Stinespring'``
        """
        input_dim = stinespring_repr.shape[-1]
        num_acted_system = self.__register_dim(input_dim, acted_system_dim)

        # sanity check
        if check_legality:
            # TODO: need to add more sanity check for stinespring
            out_dim = stinespring_repr.shape[-2]
            assert out_dim % input_dim == 0, \
                'The width of stinespring matrix should be the factor of its height'

        self.__stinespring_repr = stinespring_repr.contiguous()
        return num_acted_system

    @property
    def dim(self) -> int:
        r"""Dimension of the input/output system of this channel
        """
        return math.prod(self.system_dim)

    @property
    def choi_repr(self) -> torch.Tensor:
        r"""Choi representation of a channel

        Returns:
            a tensor with shape :math:`[d_\text{out}^2, d_\text{in}^2]`, where :math:`d_\text{in/out}` is
            the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Choi representation in this Channel instance.
        """
        type_repr = self.type_repr
        if type_repr == 'choi':
            if self.__choi_repr is None:
                raise ValueError(
                    "Need to specify the Choi representation in this Channel instance.")
            return self.__choi_repr
        elif type_repr == 'kraus':
            return _kraus_to_choi(self.kraus_repr)
        elif type_repr == 'gate':
            return _kraus_to_choi(self.matrix.unsqueeze(-3))
        else:
            return _stinespring_to_choi(self.stinespring_repr)

    @property
    def kraus_repr(self) -> List[torch.Tensor]:
        r"""Kraus representation of a channel

        Returns:
            a list of tensors with shape :math:`[d_\text{out}, d_\text{in}]`, where :math:`d_\text{in/out}` is
            the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Kraus representation in this Channel instance.
        """
        type_repr = self.type_repr

        if type_repr == 'choi':
            return _choi_to_kraus(self.choi_repr, tol=1e-6)
        elif type_repr == 'gate':
            return self.matrix.unsqueeze(-3)
        elif type_repr == 'kraus':
            if self.__kraus_repr is None:
                raise ValueError(
                    "Need to specify the Kraus representation in this Channel instance.")
            return self.__kraus_repr
        else:
            return _stinespring_to_kraus(self.stinespring_repr)

    @property
    def stinespring_repr(self) -> torch.Tensor:
        r"""Stinespring representation of a channel

        Returns:
            a tensor with shape :math:`[r * d_\text{out}, d_\text{in}]`, where :math:`r` is the rank of this channel and
            :math:`d_\text{in/out}` is the input/output dimension of this channel

        Raises:
            ValueError: Need to specify the Stinespring representation in this Channel instance.
        """
        type_repr = self.type_repr

        if type_repr == 'choi':
            return _choi_to_stinespring(self.choi_repr, tol=1e-6)
        elif type_repr == 'stinespring':
            if self.__stinespring_repr is None:
                raise ValueError(
                    "Need to specify the Stinespring representation in this Channel instance.")
            return self.__stinespring_repr
        else:
            return _kraus_to_stinespring(self.kraus_repr)
        
    @property
    def info(self) -> OperatorInfoType:
        r"""Information of this channel
        """
        info = super().info
        info.update({
            'type': 'gate' if self.type_repr == 'gate' else 'channel',
            'system_idx': copy.deepcopy(self.system_idx),
        })
        return info

    def forward(self, state: StateSimulator) -> StateSimulator:
        # TODO support auto transform
        if state.backend == 'default-pure':
            state = state.to_mixed() 
        
        for system_idx in self.system_idx:
            if self.type_repr == 'choi':
                state._transform(self.choi_repr, system_idx, 'choi')
            else:
                state._transform(self.kraus_repr, system_idx, 'kraus')

        return state
