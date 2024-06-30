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

import math
import warnings
from typing import Iterable, List, Optional, Union

import torch

from quairkit.core.utils.qinfo import (_choi_to_kraus, _choi_to_stinespring,
                                       _kraus_to_choi, _kraus_to_stinespring,
                                       _stinespring_to_choi,
                                       _stinespring_to_kraus)

from ...core import MixedState, Operator, State, utils
from ...core.intrinsic import _format_qubits_idx


class Channel(Operator):
    r"""Basic class for quantum channels.

    Args:
        type_repr: type of a representation. should be ``'choi'``, ``'kraus'``, ``'stinespring'``.
        representation: the representation of this channel. Defaults to ``None`` i.e. not specified.
        qubits_idx: indices of the qubits on which this channel acts on. Defaults to ``None``.
            i.e. list(range(num_acted_qubits)).
        num_qubits: total number of qubits. Defaults to ``None``.
        check_legality: whether check the completeness of the representation if provided. Defaults to ``True``.
        num_acted_qubits: the number of qubits that this channel acts on.  Defaults to ``None``.

    Raises:
        ValueError: Unsupported channel representation for ``type_repr``.
        NotImplementedError: The noisy channel can only run in density matrix mode.
        TypeError: Unexpected data type for Channel representation.

    Note:
        If ``representation`` is given, then ``num_acted_qubits`` will be determined by ``representation``, no matter
        ``num_acted_qubits`` is ``None`` or not.
    """
    def __init__(
            self, type_repr: str, representation: torch.Tensor = None,
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            num_qubits: int = None, check_legality: bool = True, num_acted_qubits: int = None,
    ) -> None:
        super().__init__()
        assert representation is not None or num_acted_qubits is not None, (
            "Received None for representation and num_acted_qubits: "
            "either one of them must be specified to initialize a Channel instance.")
        type_repr = type_repr.lower()
        if type_repr not in ['choi', 'kraus', 'gate', 'stinespring']:
            raise ValueError(
                "Unsupported channel representation:"
                f"require 'choi', 'kraus', 'gate', or 'stinespring', not {type_repr}")

        if representation is None:
            assert num_acted_qubits is not None, (
                "Received None for representation and num_acted_qubits: "
                "either one of them must be specified to initialize a Channel instance.")
        else:
            num_acted_qubits = getattr(self, f'_Channel__{type_repr}_init')(representation, check_legality)

        if qubits_idx is None:
            self.qubits_idx = [[0] if num_acted_qubits == 1 else list(range(num_acted_qubits))]
        else:
            self.qubits_idx = _format_qubits_idx(qubits_idx, num_qubits, num_acted_qubits)
        self.type_repr = type_repr
        self.num_acted_qubits = num_acted_qubits
        self.num_qubits = num_qubits
    
    def __call__(self, state: State) -> State:
        return self.forward(state)

    def __choi_init(self, choi_repr: torch.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'choi'``
        """
        #TODO add batch support

        if not isinstance(choi_repr, torch.Tensor):
            raise TypeError(
                f"Unexpected data type for Choi representation: expected torch.Tensor, received {type(choi_repr)}")
        choi_repr = choi_repr.to(self.dtype)

        num_acted_qubits = int(math.log2(choi_repr.shape[-1]) / 2)
        if check_legality:
            # TODO: need to add more sanity check for choi
            assert 2 ** (2 * num_acted_qubits) == choi_repr.shape[-1], \
                "The shape of Choi representation should be the integer power of 4: check your inputs."

        self.__choi_repr = choi_repr
        return num_acted_qubits

    def __kraus_init(self, kraus_repr: torch.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'kraus'``
        """
        #TODO add batch support

        if len(kraus_repr.shape) == 2:
            kraus_repr = kraus_repr.unsqueeze(0)
        kraus_repr = kraus_repr.to(self.dtype)

        dimension = kraus_repr.shape[-1]
        num_acted_qubits = int(math.log2(dimension))
        assert 2 ** num_acted_qubits == dimension, \
                "The length of oracle should be integer power of 2."

        # sanity check
        if check_legality:
            oper_sum = (utils.linalg._dagger(kraus_repr) @ kraus_repr).sum(dim=-3)
            identity = torch.eye(dimension).to(device=oper_sum.device).expand_as(oper_sum)
            err = torch.norm(torch.abs(oper_sum - identity)).item()
            if err > min(1e-6 * dimension * len(kraus_repr), 0.01):
                warnings.warn(
                    f"\nThe input data may not be a Kraus representation of a channel: norm(sum(E * E^d) - I) = {err}.",
                    UserWarning)

        self.__kraus_repr = kraus_repr
        return num_acted_qubits
    
    def __gate_init(self, gate_matrix: torch.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'gate'``
        """
        complex_dtype = self.dtype
        gate_matrix = gate_matrix.to(complex_dtype)

        if not isinstance(gate_matrix, torch.Tensor):
            raise TypeError(
                f"Unexpected data type for quantum gate: expected torch.Tensor, received {type(gate_matrix)}")

        dimension = gate_matrix.shape[-1]

        if check_legality:
            identity = torch.eye(dimension).to(device=gate_matrix.device).expand_as(gate_matrix)
            err = torch.norm(
                torch.abs(utils.linalg._dagger(gate_matrix) @ gate_matrix - identity)
            ).item()
            if err > min(1e-6 * dimension, 0.01):
                warnings.warn(
                    f"\nThe input gate matrix may not be a unitary: norm(U * U^d - I) = {err}.", UserWarning)
        num_acted_qubits = int(math.log2(dimension))

        self.__kraus_repr = gate_matrix.unsqueeze(-3)
        self.__matrix = gate_matrix
        return num_acted_qubits

    def __stinespring_init(self, stinespring_repr: torch.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Stinespring'``
        """

        stinespring_repr = stinespring_repr.to(self.dtype)
        num_acted_qubits = int(math.log2(stinespring_repr.shape[-1]))

        # sanity check
        if check_legality:
            # TODO: need to add more sanity check for stinespring
            dim_ancilla = stinespring_repr.shape[-2] // stinespring_repr.shape[-1]
            dim_act = stinespring_repr.shape[-1]
            assert dim_act * dim_ancilla == stinespring_repr.shape[-2], \
                'The width of stinespring matrix should be the factor of its height'

        self.__stinespring_repr = stinespring_repr
        return num_acted_qubits

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
            return _kraus_to_choi(self.__matrix.unsqueeze(-3))
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
            return self.__matrix.unsqueeze(-3)
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

    def forward(self, state: State) -> State:
        if state.backend == 'state_vector':
            state = MixedState(state.fit('density_matrix'), state.system_dim, state.system_seq)
        else:
            state = state.clone()
        
        for qubits_idx in self.qubits_idx:
            if self.type_repr == 'choi':
                state._transform(self.choi_repr, qubits_idx, 'choi')
            else:
                state._transform(self.kraus_repr, qubits_idx, 'kraus')
        
        return state

