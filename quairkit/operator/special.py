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
The source file of the class for the special quantum operator.
"""

from typing import Iterable, Optional, Union

import numpy as np
import torch

from ..core import Operator, State
from ..core.intrinsic import _alias, _digit_to_int, _int_to_digit


class ResetState(Operator):
    r"""The class to reset the quantum state. It will be implemented soon.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **kwargs):
        r"""The forward function.

        Returns:
            NotImplemented.
        """
        return NotImplemented


class Collapse(Operator):
    r"""The class to compute the collapse of the quantum state.

    Args:
        system_idx: list of systems to be collapsed.
        desired_result: The desired result you want to collapse. Defaults to ``None`` meaning randomly choose one.
        if_print: whether print the information about the collapsed state. Defaults to ``False``.
        measure_basis: The basis of the measurement. The quantum state will collapse to the corresponding eigenstate.

    Raises:
        NotImplementedError: If the basis of measurement is not z. Other bases will be implemented in future.
        
    Note:
        When desired_result is `None`, Collapse does not support gradient calculation
    """
    @_alias({"system_idx": "qubits_idx"})
    def __init__(self, system_idx: Union[int, Iterable[int]],
                 desired_result: Union[int, str] = None, if_print: bool = False,
                 measure_basis: Optional[torch.Tensor] = None):
        super().__init__()
        self.measure_basis = []

        if isinstance(system_idx, int):
            self.system_idx = [system_idx]
        else:
            self.system_idx = list(system_idx)

        self.desired_result = desired_result
        self.if_print = if_print

        self.measure_basis = measure_basis
        
    def forward(self, state: State) -> State:
        r"""Compute the collapse of the input state.

        Args:
            state: The input state, which will be collapsed

        Returns:
            The collapsed quantum state.
        """
        system_dim = [state.system_dim[idx] for idx in self.system_idx]
        dim = int(np.prod(system_dim))

        prob_array, measured_state = state.measure(self.measure_basis, self.system_idx, keep_state=True)
        
        desired_result = self.desired_result
        if desired_result is None:
            assert measured_state.batch_dim is None, \
                    f"batch computation is not supported for random collapse: received {desired_result}"
            desired_result = np.random.choice(range(dim), p=prob_array)
            digits_str = _int_to_digit(desired_result, base=system_dim)
        elif isinstance(desired_result, str):
            digits_str = desired_result
            desired_result = _digit_to_int(desired_result, system_dim)
        else:
            digits_str = _int_to_digit(desired_result, base=system_dim)
        desired_result = torch.tensor(desired_result)
        state_str = '>'.join(f'|{d}' for d in digits_str) + '>'
        
        prob_collapse = prob_array.index_select(-1, desired_result)
        assert torch.all(prob_collapse > 1e-10).item(), (
            f"It is computationally infeasible for some states in systems {self.system_idx} "
            f"to collapse to state {state_str}")

        # whether print the collapsed result
        if self.if_print:
            prob = prob_collapse.mean().item()
            print(f"systems {self.system_idx} collapse to the state {state_str} with (average) probability {prob}")

        return measured_state.index_select(dim=-1, index=desired_result)
