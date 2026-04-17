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
The source file of the class for the distance.
"""

from typing import Callable, Literal

import torch

from ..core import Operator, StateSimulator, utils
from ..core.intrinsic import _ensure_mat_3d, _ensure_vec_3d

__all__ = ["TraceDistance", "StateFidelity"]


def _align_batch_pair(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_size: int,
    rhs_size: int,
    lhs_name: str,
    rhs_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = max(lhs_size, rhs_size)
    if lhs_size not in (1, B) or rhs_size not in (1, B):
        raise ValueError(
            f"Incompatible batch sizes: {lhs_name} {lhs_size} vs {rhs_name} {rhs_size}"
        )
    if lhs_size == 1 and B > 1:
        lhs = lhs.expand(B, *([-1] * (lhs.ndim - 1)))
    if rhs_size == 1 and B > 1:
        rhs = rhs.expand(B, *([-1] * (rhs.ndim - 1)))
    return lhs, rhs


def _finalize_batch_output(
    out_b: torch.Tensor, batch_shape_a: list[int], batch_shape_b: list[int]
) -> torch.Tensor:
    batch_shape = batch_shape_a if batch_shape_a else batch_shape_b
    return out_b.reshape(batch_shape) if batch_shape else out_b.squeeze(0)


def _extract_state_tensor(
    state: StateSimulator, state_kind: Literal["ket", "density"]
) -> tuple[torch.Tensor, list[int], int]:
    if state_kind == "ket":
        data3, _ = _ensure_vec_3d(state.ket)
    else:
        data3, _ = _ensure_mat_3d(state.density_matrix)
    return data3, list(state.batch_dim), state.numel()


def _check_pair_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_kind: Literal["ket", "density"],
    rhs_kind: Literal["ket", "density"],
) -> None:
    if lhs_kind == rhs_kind:
        if lhs.shape[-2:] != rhs.shape[-2:]:
            raise ValueError(
                f"Two quantum states mismatch: received {lhs.shape[-2:]} and {rhs.shape[-2:]}"
            )
        return
    ket = lhs if lhs_kind == "ket" else rhs
    rho = lhs if lhs_kind == "density" else rhs
    if ket.shape[-2] != rho.shape[-1]:
        raise ValueError(
            f"Two quantum states mismatch: received {ket.shape[-2]} and {rho.shape[-1]}"
        )


def _run_pair_kernel(
    lhs_state: StateSimulator,
    rhs_state: StateSimulator,
    lhs_kind: Literal["ket", "density"],
    rhs_kind: Literal["ket", "density"],
    lhs_name: str,
    rhs_name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    lhs3, batch_shape_lhs, lhs_size = _extract_state_tensor(lhs_state, lhs_kind)
    rhs3, batch_shape_rhs, rhs_size = _extract_state_tensor(rhs_state, rhs_kind)
    _check_pair_shape(lhs3, rhs3, lhs_kind, rhs_kind)
    lhs3, rhs3 = _align_batch_pair(lhs3, rhs3, lhs_size, rhs_size, lhs_name, rhs_name)
    out_b = kernel(lhs3, rhs3.to(dtype=lhs3.dtype))
    return _finalize_batch_output(out_b, batch_shape_lhs, batch_shape_rhs)


def _trace_distance_mm(state: StateSimulator, target_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        state, target_state, "density", "density", "rho", "sigma", utils.qinfo._trace_distance
    )


def _state_fidelity_pp(state: StateSimulator, target_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        state, target_state, "ket", "ket", "psi", "phi", utils.qinfo._state_fidelity_pp
    )


def _trace_distance_pp(state: StateSimulator, target_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        state, target_state, "ket", "ket", "psi", "phi", utils.qinfo._trace_distance_pp
    )


def _trace_distance_pm(pure_state: StateSimulator, mixed_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        pure_state, mixed_state, "ket", "density", "psi", "rho", utils.qinfo._trace_distance_pm
    )


def _state_fidelity_mm(state: StateSimulator, target_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        state, target_state, "density", "density", "rho", "sigma", utils.qinfo._state_fidelity
    )


def _state_fidelity_pm(pure_state: StateSimulator, mixed_state: StateSimulator) -> torch.Tensor:
    return _run_pair_kernel(
        pure_state, mixed_state, "ket", "density", "psi", "rho", utils.qinfo._state_fidelity_pm
    )


class TraceDistance(Operator):
    r"""The class of the loss function to compute the trace distance.

    This interface allows you to use the trace distance as a loss function for training quantum circuits.
    The trace distance is defined as :math:`D(\rho, \sigma) = \frac{1}{2}\text{tr}|\rho-\sigma|`.

    Args:
        target_state: The target quantum state to be used to compute the trace distance.
            Can be a single state or a batch of states.

    Note:
        This class supports batch operations. When both target and input states have batch dimensions,
        the trace distance is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import one_state, bell_state
            from quairkit.loss import TraceDistance

            # Single state example
            target_state = one_state(1)  # Define target
            input_state = bell_state(1)  # Define input
            trace_distance = TraceDistance(target_state)
            result = trace_distance(input_state)
            print('Single state distance:', result)

        ::

            Single state distance: tensor(0.7071)

        .. code-block:: python

            # Batched states example
            from quairkit.database import random_state

            target_state_batch = random_state(num_systems=1, size=2)  # Batch of 2 targets
            input_state_batch = random_state(num_systems=1, size=2)   # Batch of 2 inputs
            trace_distance = TraceDistance(target_state_batch)
            batch_result = trace_distance(input_state_batch)
            print('Batched distances:', batch_result)

        ::

            Batched distances: tensor([0.7912, 0.7283])
    """
    def __init__(self, target_state: StateSimulator):
        r"""Output target state.
        """
        super().__init__()
        self.target_state = target_state
        
    def __call__(self, state: StateSimulator) -> torch.Tensor:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)

    def forward(self, state: StateSimulator) -> torch.Tensor:
        r"""Compute the trace distance between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize quantum circuits.

        Args:
            state: The input quantum state which will be used to compute the trace distance with the target state.
                Can be a single state or a batch of states. Must have the same number of systems as the target state.

        Returns:
            The trace distance between the input state and the target state. Returns a scalar for single states,
            or a tensor for batched states.

        Raises:
            AssertionError: If the number of systems in the input state does not match the target state.
            NotImplementedError: If the backend is wrong or not supported.
        """
        assert (
            state.num_systems == self.target_state.num_systems
        ), f"The number of systems does not match: received {state.num_systems}, expect {self.target_state.num_systems}."

        state_pure = state.backend == "default-pure"
        target_pure = self.target_state.backend == "default-pure"
        if state_pure and target_pure:
            return _trace_distance_pp(state, self.target_state)
        if state_pure and not target_pure:
            return _trace_distance_pm(state, self.target_state)
        if not state_pure and target_pure:
            return _trace_distance_pm(self.target_state, state)
        return _trace_distance_mm(state, self.target_state)

class StateFidelity(Operator):
    r"""The class of the loss function to compute the state fidelity.

    This interface allows you to use the state fidelity as a loss function for training quantum circuits.
    The state fidelity is defined as :math:`F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})`.

    Args:
        target_state: The target quantum state to be used to compute the state fidelity.
            Can be a single state or a batch of states.

    Note:
        This class supports batch operations. When both target and input states have batch dimensions,
        the fidelity is computed element-wise along the batch dimension.

    Examples:
        .. code-block:: python

            from quairkit.database import one_state, bell_state
            from quairkit.loss import StateFidelity

            # Single state example
            target_state = one_state(1)  # Define target
            input_state = bell_state(1)  # Define input
            fidelity_calculator = StateFidelity(target_state)
            result = fidelity_calculator(input_state)
            print('Single state fidelity:', result)

        ::

            Single state fidelity: tensor(0.7071)

        .. code-block:: python

            # Batched states example
            from quairkit.database import random_state

            target_batch = random_state(num_systems=1, size=2)  # Batch of 2 targets
            input_batch = random_state(num_systems=1, size=2)   # Batch of 2 inputs
            fidelity_calculator = StateFidelity(target_batch)
            batch_result = fidelity_calculator(input_batch)
            print('Batched fidelities:', batch_result)

        ::

            Batched fidelities: tensor([0.5658, 0.7090])
    """
    def __init__(self, target_state: StateSimulator):
        r"""Output target state.
        """
        super().__init__()
        self.target_state = target_state
        
    def __call__(self, state: StateSimulator) -> torch.Tensor:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)

    def forward(self, state: StateSimulator) -> torch.Tensor:
        r"""Compute the state fidelity between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize quantum circuits.

        Args:
            state: The input quantum state which will be used to compute the state fidelity with the target state.
                Can be a single state or a batch of states. Must have the same number of systems as the target state.

        Returns:
            The state fidelity between the input state and the target state. Returns a scalar for single states,
            or a tensor for batched states. The fidelity ranges from 0 to 1.

        Raises:
            AssertionError: If the number of systems in the input state does not match the target state.
            NotImplementedError: If the backend is wrong or not supported.
        """
        assert (
            state.num_systems == self.target_state.num_systems
        ), f"The number of systems does not match: received {state.num_systems}, expect {self.target_state.num_systems}."

        state_pure = state.backend == "default-pure"
        target_pure = self.target_state.backend == "default-pure"
        if state_pure and target_pure:
            return _state_fidelity_pp(state, self.target_state)
        if state_pure and not target_pure:
            return _state_fidelity_pm(state, self.target_state)
        if not state_pure and target_pure:
            return _state_fidelity_pm(self.target_state, state)
        return _state_fidelity_mm(state, self.target_state)
