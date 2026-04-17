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
Representations of channels
"""

import torch
import importlib


def _require_cpp_submodule(submodule: str):
    """Return a required C++ extension submodule.

    This module no longer supports a Python fallback implementation for kernels
    that have C++ equivalents. If the C++ extension is not available, raise an
    ImportError with actionable guidance.
    """
    try:
        mod = importlib.import_module("quairkit._C")
    except Exception as e:
        raise ImportError(
            "QuAIRKit requires the compiled C++ extension (quairkit._C). "
            "Please build it first (e.g. `python setup.py build_ext --inplace`)."
        ) from e
    cpp_mod = getattr(mod, submodule, None)
    if cpp_mod is None:
        raise ImportError(
            f"quairkit._C is available but missing submodule '{submodule}'. "
            "Please rebuild the C++ extension (e.g. `python setup.py build_ext --inplace`)."
        )
    return cpp_mod


_CPP_REPR = _require_cpp_submodule("representation")


def _bit_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([-1, 1])
    if prob.ndim != 2 or prob.shape[1] != 1:
        raise ValueError(f"prob must have shape [B, 1], got {list(prob.shape)}")
    return _CPP_REPR.bit_flip_kraus(prob)


def _phase_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([-1, 1])
    if prob.ndim != 2 or prob.shape[1] != 1:
        raise ValueError(f"prob must have shape [B, 1], got {list(prob.shape)}")
    return _CPP_REPR.phase_flip_kraus(prob)


def _bit_phase_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([-1, 1])
    if prob.ndim != 2 or prob.shape[1] != 1:
        raise ValueError(f"prob must have shape [B, 1], got {list(prob.shape)}")
    return _CPP_REPR.bit_phase_flip_kraus(prob)


def _amplitude_damping_kraus(gamma:  torch.Tensor) -> torch.Tensor:
    if not isinstance(gamma, torch.Tensor):
        raise TypeError(f"gamma must be torch.Tensor, got {type(gamma)}")
    if gamma.ndim == 1:
        gamma = gamma.reshape([-1, 1])
    if gamma.ndim != 2 or gamma.shape[1] != 1:
        raise ValueError(f"gamma must have shape [B, 1], got {list(gamma.shape)}")
    return _CPP_REPR.amplitude_damping_kraus(gamma)


def _generalized_amplitude_damping_kraus(
    gamma:  torch.Tensor, prob:  torch.Tensor) -> torch.Tensor:
    if not isinstance(gamma, torch.Tensor):
        raise TypeError(f"gamma must be torch.Tensor, got {type(gamma)}")
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if gamma.ndim == 1:
        gamma = gamma.reshape([-1, 1])
    if prob.ndim == 1:
        prob = prob.reshape([-1, 1])
    if gamma.ndim != 2 or gamma.shape[1] != 1:
        raise ValueError(f"gamma must have shape [B, 1], got {list(gamma.shape)}")
    if prob.ndim != 2 or prob.shape[1] != 1:
        raise ValueError(f"prob must have shape [B, 1], got {list(prob.shape)}")
    if gamma.shape[0] != prob.shape[0]:
        raise ValueError(f"Batch size mismatch: gamma B={gamma.shape[0]} vs prob B={prob.shape[0]}")
    out = _CPP_REPR.generalized_amplitude_damping_kraus(gamma, prob)
    return out.unsqueeze(0) if out.ndim == 3 else out


def _phase_damping_kraus(gamma:  torch.Tensor) -> torch.Tensor:
    if not isinstance(gamma, torch.Tensor):
        raise TypeError(f"gamma must be torch.Tensor, got {type(gamma)}")
    if gamma.ndim == 1:
        gamma = gamma.reshape([1, -1])
    if gamma.ndim != 2 or gamma.shape[1] != 1:
        raise ValueError(f"gamma must have shape [B, 1], got {list(gamma.shape)}")
    return _CPP_REPR.phase_damping_kraus(gamma)


def _depolarizing_kraus(prob:  torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([1, -1])
    if prob.ndim != 2 or prob.shape[1] != 1:
        raise ValueError(f"prob must have shape [B, 1], got {list(prob.shape)}")
    return _CPP_REPR.depolarizing_kraus(prob)

    
def _pauli_kraus(prob: torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([1, -1])
    if prob.ndim != 2 or prob.shape[1] != 3:
        raise ValueError(f"prob must have shape [B, 3], got {list(prob.shape)}")
    return _CPP_REPR.pauli_kraus(prob)


def _reset_kraus(prob: torch.Tensor) -> torch.Tensor:
    if not isinstance(prob, torch.Tensor):
        raise TypeError(f"prob must be torch.Tensor, got {type(prob)}")
    if prob.ndim == 1:
        prob = prob.reshape([1, -1])
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError(f"prob must have shape [B, 2], got {list(prob.shape)}")
    return _CPP_REPR.reset_kraus(prob)


def _thermal_relaxation_kraus(
    const_t: torch.Tensor, exec_time: torch.Tensor) -> torch.Tensor:
    if not isinstance(const_t, torch.Tensor):
        raise TypeError(f"const_t must be torch.Tensor, got {type(const_t)}")
    if not isinstance(exec_time, torch.Tensor):
        raise TypeError(f"exec_time must be torch.Tensor, got {type(exec_time)}")
    if exec_time.ndim == 0:
        exec_time = exec_time.reshape([1, 1])
    if const_t.ndim == 1:
        const_t = const_t.reshape([-1, const_t.shape[-1]])
    if exec_time.ndim == 1:
        exec_time = exec_time.reshape([-1, 1])
    if const_t.ndim != 2 or const_t.shape[1] != 2:
        raise ValueError(f"const_t must have shape [B, 2], got {list(const_t.shape)}")
    if exec_time.ndim != 2 or exec_time.shape[1] != 1:
        raise ValueError(f"exec_time must have shape [B, 1], got {list(exec_time.shape)}")
    if const_t.shape[0] != exec_time.shape[0]:
        raise ValueError(f"Batch size mismatch: const_t B={const_t.shape[0]} vs exec_time B={exec_time.shape[0]}")

    t1, t2 = const_t[:, 0:1], const_t[:, 1:2]
    assert torch.all(t2 <= t1 + 1e-6), (
        f"The relaxation time T2 and T1 must satisfy T2 <= T1: received max(T2-T1) {torch.max(t2 - t1).item()}"
    )

    exec_time_s = exec_time / 1000
    prob_reset = 1 - torch.exp(-exec_time_s / t1)
    prob_z = (1 - prob_reset) * (1 - torch.exp(-exec_time_s / t2) * torch.exp(exec_time_s / t1)) / 2
    prob_z = prob_z.clamp(min=0)
    prob_i = 1 - prob_reset - prob_z
    _0 = torch.zeros_like(exec_time_s)

    kraus_oper = [
        torch.sqrt(prob_i), _0,
        _0, torch.sqrt(prob_i),
        torch.sqrt(prob_z), _0,
        _0, -torch.sqrt(prob_z),
        torch.sqrt(prob_reset), _0,
        _0, _0,
        _0, torch.sqrt(prob_reset),
        _0, _0,
    ]
    B = const_t.shape[0]
    kraus_oper = torch.cat(kraus_oper, dim=-1).view([B, 4, 2, 2])
    return kraus_oper + 0j


def _replacement_choi(sigma: torch.Tensor) -> torch.Tensor:
    if not isinstance(sigma, torch.Tensor):
        raise TypeError(f"sigma must be torch.Tensor, got {type(sigma)}")
    if sigma.ndim == 2:
        sigma = sigma.unsqueeze(0)
    if sigma.ndim != 3 or sigma.shape[-1] != sigma.shape[-2]:
        raise ValueError(f"sigma must have shape [B, d, d] (or [d, d]), got {list(sigma.shape)}")
    B, d, _ = sigma.shape
    eye = torch.eye(d, dtype=sigma.dtype, device=sigma.device)
    choi = torch.einsum("ij,bac->biajc", eye, sigma).reshape(B, d * d, d * d)
    return choi

