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


def _bit_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    prob = prob.view([1])
    
    _0 = torch.zeros_like(prob)
    kraus_oper = [
        # E0
        torch.sqrt(1 - prob), _0,
        _0, torch.sqrt(1 - prob),
        # E1
        _0, torch.sqrt(prob),
        torch.sqrt(prob), _0
    ]
    kraus_oper = torch.cat(kraus_oper).view([2, 2, 2])
    return kraus_oper + 0j


def _phase_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    prob = prob.view([1])
    _0 = torch.zeros_like(prob)
    kraus_oper = [
        # E0
        torch.sqrt(1 - prob), _0,
        _0, torch.sqrt(1 - prob),
        # E1
        torch.sqrt(prob), _0,
        _0, -torch.sqrt(prob)
    ]
    kraus_oper = torch.cat(kraus_oper).view([2, 2, 2])
    return kraus_oper + 0j


def _bit_phase_flip_kraus(prob:  torch.Tensor) -> torch.Tensor:
    prob = prob.view([1])
    _0 = torch.zeros_like(prob)
    kraus_oper = [
        # E0
        torch.sqrt(1 - prob), _0,
        _0, torch.sqrt(1 - prob),
        # E1
        _0, -1j * torch.sqrt(prob),
        1j * torch.sqrt(prob), _0,
    ]
    kraus_oper = torch.cat(kraus_oper).view([2, 2, 2])
    return kraus_oper + 0j


def _amplitude_damping_kraus(gamma:  torch.Tensor) -> torch.Tensor:
    gamma = gamma.view([1])
    _0, _1 = torch.zeros_like(gamma), torch.ones_like(gamma)
    kraus_oper = [
        # E0
        _1, _0,
        _0, torch.sqrt(1 - gamma),
        # E1
        _0, torch.sqrt(gamma),
        _0, _0,
    ]
    kraus_oper = torch.cat(kraus_oper).view([2, 2, 2])
    return kraus_oper + 0j


def _generalized_amplitude_damping_kraus(
    gamma:  torch.Tensor, prob:  torch.Tensor) -> torch.Tensor:
    gamma, prob = gamma.view([1]), prob.view([1])
    _0 = torch.zeros_like(prob)
    kraus_oper = [
        # E0
        torch.sqrt(prob), _0,
        _0, torch.sqrt(prob) * torch.sqrt(1 - gamma),
        # E1
        _0, torch.sqrt(prob) * torch.sqrt(gamma),
        _0, _0,
        # E2
        torch.sqrt(1 - prob) * torch.sqrt(1 - gamma), _0,
        _0, torch.sqrt(1 - prob),
        # E3
        _0, _0,
        torch.sqrt(1 - prob) * torch.sqrt(gamma), _0,
    ]
    kraus_oper = torch.cat(kraus_oper).view([4, 2, 2])
    return kraus_oper + 0j


def _phase_damping_kraus(gamma:  torch.Tensor) -> torch.Tensor:
    gamma = gamma.view([1])
    _0, _1=torch.zeros_like(gamma), torch.ones_like(gamma)
    kraus_oper = [
        # E0
        _1, _0,
        _0, torch.sqrt(1 - gamma),
        # E1
        _0, _0,
        _0, torch.sqrt(gamma),
    ]
    kraus_oper = torch.cat(kraus_oper).view([2, 2, 2])
    return kraus_oper + 0j


def _depolarizing_kraus(prob:  torch.Tensor) -> torch.Tensor:
    prob = prob.view([1])
    _0 = torch.zeros_like(prob)
    kraus_oper = [
        # E0
        torch.sqrt(1 - 3 * prob / 4), _0,
        _0, torch.sqrt(1 - 3 * prob / 4),
        # E1
        _0, torch.sqrt(prob / 4),
        torch.sqrt(prob / 4), _0,
        # E2
        _0, -1j * torch.sqrt(prob / 4),
        1j * torch.sqrt(prob / 4), _0,
        # E3
        torch.sqrt(prob / 4), _0,
        _0, (-1 * torch.sqrt(prob / 4)),
    ]
    kraus_oper = torch.cat(kraus_oper).view([4, 2, 2])
    return kraus_oper

    
def _pauli_kraus(prob: torch.Tensor) -> torch.Tensor:
    prob_x, prob_y, prob_z = prob.view([3, 1])
    assert (prob_sum := torch.sum(prob)) <= 1, \
        f"The sum of input probabilities should not be greater than 1: received {prob_sum.item()}"
    prob_i = (1 - prob_sum).view([1])
    _0 = torch.zeros_like(prob_i)
    
    kraus_oper = [
        # E0
        torch.sqrt(prob_i), _0,
        _0, torch.sqrt(prob_i),
        # E1
        _0, torch.sqrt(prob_x),
        torch.sqrt(prob_x), _0,
        # E2
        _0, -1j * torch.sqrt(prob_y),
        1j * torch.sqrt(prob_y), _0,
        # E3
        torch.sqrt(prob_z), _0,
        _0, (-torch.sqrt(prob_z)),
    ]
    kraus_oper = torch.cat(kraus_oper).view([4, 2, 2])
    return kraus_oper


def _reset_kraus(prob: torch.Tensor) -> torch.Tensor:
    prob_0, prob_1 = prob.view([2, 1])
    assert (prob_sum := torch.sum(prob)) <= 1, \
        f"The sum of input probabilities should not be greater than 1: received {prob_sum.item()}"
    prob_i = (1 - prob_sum).view([1])
    _0 = torch.zeros_like(prob_i)
    kraus_oper = [
        # E0
        torch.sqrt(prob_0), _0,
        _0, _0,
        # E1
        _0, torch.sqrt(prob_0),
        _0, _0,
        # E2
        _0, _0,
        torch.sqrt(prob_1), _0,
        # E3
        _0, _0,
        _0, torch.sqrt(prob_1),
        # E4
        torch.sqrt(prob_i), _0,
        _0, torch.sqrt(prob_i),
    ]
    kraus_oper = torch.cat(kraus_oper).view([5, 2, 2])
    return kraus_oper + 0j


def _thermal_relaxation_kraus(
    const_t: torch.Tensor, exec_time: torch.Tensor) -> torch.Tensor:
    
    t1, t2 = const_t.view([2, 1])
    assert t2 <= t1, \
        f"The relaxation time T2 and T1 must satisfy T2 <= T1: received T2 {t2} and T1{t1}"
    
    exec_time = exec_time.view([1]) / 1000
    prob_reset = 1 - torch.exp(-exec_time / t1)
    prob_z = (1 - prob_reset) * (1 - torch.exp(-exec_time / t2) * torch.exp(exec_time / t1)) / 2
    prob_z = prob_z.clamp(min=0)
    prob_i = 1 - prob_reset - prob_z
    _0 = torch.zeros_like(exec_time)
    kraus_oper = [
        # E0
        torch.sqrt(prob_i), _0,
        _0, torch.sqrt(prob_i),
        # E1
        torch.sqrt(prob_z), _0,
        _0, -torch.sqrt(prob_z),
        # E2
        torch.sqrt(prob_reset), _0,
        _0, _0,
        # E3
        _0, torch.sqrt(prob_reset),
        _0, _0,
    ]
    kraus_oper = torch.cat(kraus_oper).view([4, 2, 2])
    return kraus_oper + 0j


def _replacement_choi(sigma: torch.Tensor) -> torch.Tensor:
    return torch.kron(torch.eye(sigma.shape[-1]), sigma)

