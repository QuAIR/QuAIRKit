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
The basic function of the QuAIRKit.
"""

import enum
import random
import warnings
from typing import Optional, Union

import numpy as np
import torch


class Backend(enum.Enum):
    r"""Backend classifier in the QuAIRKit.
    """
    Simulator = 'default'
    QPU = 'qpu' # TODO: to be added

    StateVector = 'state_vector' # TODO: depreciated in the future
    DensityMatrix = 'density_matrix' # TODO: depreciated in the future


DEFAULT_DEVICE = "cpu"
DEFAULT_SIMULATOR = Backend.Simulator
DEFAULT_DTYPE = torch.complex64
SEED = None


def set_device(device: "str") -> None:
    r"""Set the device to save the tensor.

    Args:
        device: The name of the device.
    """
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device
    torch.set_default_device(device)


def get_device() -> "str":
    r"""Get the current device to save the tensor.

    Returns:
        The name of the current device.
    """
    return DEFAULT_DEVICE


def set_backend(backend: Union[str, Backend]) -> None:
    r"""Set the backend implementation of QuAIRKit.

    Args:
        backend: The name of the backend.
    """
    global DEFAULT_SIMULATOR
    if backend in ["state_vector", "density_matrix"]:
        warnings.warn(
            "The usage for 'state_vector' and 'density_matrix' will be deprecated in the future."
            + "Please use 'default' instead.",
            DeprecationWarning,
        )
        backend = 'default'
    DEFAULT_SIMULATOR = Backend(backend) if isinstance(backend, str) else backend


def get_backend() -> str:
    r"""Get the current backend of QuAIRKit.

    Returns:
        The name of currently used backend.
    """
    return DEFAULT_SIMULATOR.value


def set_seed(seed: int) -> None:
    r"""Set the global seed of QuAIRKit.

    Args:
        seed: the random seed used in QuAIRKit.

    Note:
        The seed is set for the following modules:
        ``torch``, ``torch.cuda``, ``numpy``, ``random``

    """
    global SEED
    SEED = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_seed() -> None:
    r"""Get the currently used seed of QuAIRKit.

    Returns:
        The Currently used seed.

    """
    return SEED


def set_dtype(dtype: str) -> None:
    r"""Set the data type .

    Args:
        dtype: The dtype can be ``complex64`` and ``complex128``.

    Raises:
        ValueError: The dtype should be complex64 or complex128.
    """
    global DEFAULT_DTYPE

    if dtype == "complex64":
        dtype = torch.complex64
    elif dtype == "complex128":
        dtype = torch.complex128
    else:
        raise ValueError("The dtype should be 'complex64' or 'complex128'.")
    DEFAULT_DTYPE = dtype

    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    torch.set_default_dtype(float_dtype)


def get_dtype() -> torch.dtype:
    r"""Return currently used data type.

    Returns:
        Currently used data type.
    """
    return DEFAULT_DTYPE


def get_float_dtype() -> torch.dtype:
    r"""Return currently used float data type.

    Returns:
        Currently used data type.
    """
    return torch.float32 if DEFAULT_DTYPE == torch.complex64 else torch.float64
