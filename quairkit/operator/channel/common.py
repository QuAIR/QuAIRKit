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
The source file of the classes for common quantum channels.
"""

from typing import Iterable, Union

import numpy as np
import torch

from ...core import State
from ...database.representation import (amplitude_damping_kraus,
                                        bit_flip_kraus, bit_phase_flip_kraus,
                                        depolarizing_kraus,
                                        generalized_amplitude_damping_kraus,
                                        generalized_depolarizing_kraus,
                                        pauli_kraus, phase_damping_kraus,
                                        phase_flip_kraus, replacement_choi,
                                        reset_kraus, thermal_relaxation_kraus)
from .base import Channel


class BitFlip(Channel):
    r"""A collection of bit flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-p} I,
        E_1 = \sqrt{p} X.

    Args:
        prob: Probability of a bit flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'bit_flip',
            'tex': r'\mathcal{E}_{p = ' + str(p) + r'}^{\textrm{\tiny{(BF)}}}',
            'api': 'bit_flip',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', bit_flip_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class PhaseFlip(Channel):
    r"""A collection of phase flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Z.

    Args:
        prob: Probability of a phase flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'phase_flip',
            'tex': r'\mathcal{E}_{p = ' + str(p) + r'}^{\textrm{\tiny{(PF)}}}',
            'api': 'phase_flip',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', phase_flip_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class BitPhaseFlip(Channel):
    r"""A collection of bit phase flip channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1 - p} I,
        E_1 = \sqrt{p} Y.

    Args:
        prob: Probability of a bit phase flip. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'bit_phase_flip',
            'tex': r'\mathcal{E}_{p = ' + str(p) + r'}^{\textrm{\tiny{(BPF)}}}',
            'api': 'bit_phase_flip',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', bit_phase_flip_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class AmplitudeDamping(Channel):
    r"""A collection of amplitude damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{bmatrix}.

    Args:
        gamma: Damping probability. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))
    
    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, gamma: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        g = np.round(gamma.item() if isinstance(gamma, torch.Tensor) else gamma, decimals=2)
        channel_info = {
            'name': 'amplitude_damping',
            'tex': r'\mathcal{E}_{\gamma = ' + str(g) + r'}^{\textrm{\tiny{(AD)}}}',
            'api': 'amplitude_damping',
            'kwargs': {'gamma': gamma}
        }
        super().__init__('kraus', amplitude_damping_kraus(gamma), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class GeneralizedAmplitudeDamping(Channel):
    r"""A collection of generalized amplitude damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{p}
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
        E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
        E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

    Args:
        gamma: Damping probability. Its value should be in the range :math:`[0, 1]`.
        prob: Excitation probability. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))
    
    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, gamma: Union[torch.Tensor, float], prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        g = np.round(gamma.item() if isinstance(gamma, torch.Tensor) else gamma, decimals=2)
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'generalized_amplitude_damping',
            'tex': r'\mathcal{E}_{\gamma = ' + str(g) + r', p = ' + str(p) + r'}^{\textrm{\tiny{(GAD)}}}',
            'api': 'generalized_amplitude_damping',
            'kwargs': {'gamma': gamma, 'prob': prob}
        }
        super().__init__(
            'kraus', generalized_amplitude_damping_kraus(gamma, prob), qubits_idx, 
            acted_system_dim=2, check_legality=False, channel_info=channel_info)


class PhaseDamping(Channel):
    r"""A collection of phase damping channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{\gamma}
        \end{bmatrix}.

    Args:
        gamma: Parameter of the phase damping channels. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, gamma: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        g = np.round(gamma.item() if isinstance(gamma, torch.Tensor) else gamma, decimals=2)
        channel_info = {
            'name': 'phase_damping',
            'tex': r'\mathcal{E}_{\gamma = ' + str(g) + r'}^{\textrm{\tiny{(PD)}}}',
            'api': 'phase_damping',
            'kwargs': {'gamma': gamma}
        }
        super().__init__('kraus', phase_damping_kraus(gamma), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class Depolarizing(Channel):
    r"""A collection of depolarizing channels.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-3p/4} I,
        E_1 = \sqrt{p/4} X,
        E_2 = \sqrt{p/4} Y,
        E_3 = \sqrt{p/4} Z.

    Args:
        prob: Parameter of the depolarizing channels. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        The implementation logic for this feature has been updated.
        The current version refers to formula (8.102) in Quantum Computation and Quantum Information 10th
        edition by M.A.Nielsen and I.L.Chuang.
        Reference: Nielsen, M., & Chuang, I. (2010). Quantum Computation and Quantum Information: 10th
        Anniversary Edition. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511976667
        Unless input state is pure, this class will not create a new instance for output.

    """
    def __init__(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'depolarizing',
            'tex': r'\mathcal{D}_{p = ' + str(p) + r'}',
            'api': 'depolarizing',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', depolarizing_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class GeneralizedDepolarizing(Channel):
    r"""A generalized depolarizing channel.

    Such a channel's Kraus operators are

    .. math::

        E_0 = \sqrt{1-(D - 1)p/D} I, \text{ where } D = 4^n, \\
        E_k = \sqrt{p/D} \sigma_k, \text{ for } 0 < k < D.

    Args:
        prob: probability :math:`p`. Its value should be in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act, the length of which is :math:`n`.
            Defaults to be ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
    
    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str], num_qubits: int = None
    ):
        p = np.round(prob.item() if isinstance(prob, torch.Tensor) else prob, decimals=2)
        channel_info = {
            'name': 'generalized_depolarizing',
            'tex': r'\mathcal{D}_{p = ' + str(p) + r'}',
            'api': 'generalized_depolarizing',
            'kwargs': {'prob': prob}
        }
        if qubits_idx in ['full', 'cycle']:
            raise NotImplementedError(
                'The generalized depolarizing channel should act on all qubits of the system')
        num_acted_qubits = np.size(np.array(qubits_idx))
        super().__init__(
            'kraus', generalized_depolarizing_kraus(prob, num_acted_qubits), qubits_idx, 
            acted_system_dim=2, check_legality=False, channel_info=channel_info)


class PauliChannel(Channel):
    r"""A collection of Pauli channels.

    Args:
        prob: Probabilities corresponding to the Pauli X, Y, and Z operators. Each value should be in the
            range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        The sum of three input probabilities should be less than or equal to 1.
    
    Note:
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, Iterable[float]],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        channel_info = {
            'name': 'pauli_channel',
            'tex': r'\mathcal{N}',
            'api': 'pauli_channel',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', pauli_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class ResetChannel(Channel):
    r"""A collection of reset channels.

    Such a channel reset the state to :math:`|0\rangle` with a probability of p and to :math:`|1\rangle` with
    a probability of q. Its Kraus operators are

    .. math::

        E_0 =
        \begin{bmatrix}
            \sqrt{p} & 0 \\
            0 & 0
        \end{bmatrix},
        E_1 =
        \begin{bmatrix}
            0 & \sqrt{p} \\
            0 & 0
        \end{bmatrix},\\
        E_2 =
        \begin{bmatrix}
            0 & 0 \\
            \sqrt{q} & 0
        \end{bmatrix},
        E_3 =
        \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{q}
        \end{bmatrix},\\
        E_4 = \sqrt{1-p-q} I.

    Args:
        prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`. Each value should be
            in the range :math:`[0, 1]`.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        The sum of two input probabilities should be less than or equal to 1.
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, prob: Union[torch.Tensor, Iterable[float]],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        channel_info = {
            'name': 'reset_channel',
            'tex': r'\mathcal{N}',
            'api': 'reset_channel',
            'kwargs': {'prob': prob}
        }
        super().__init__('kraus', reset_kraus(prob), qubits_idx, 
                         acted_system_dim=2, check_legality=False, channel_info=channel_info)


class ThermalRelaxation(Channel):
    r"""A collection of thermal relaxation channels.

    Such a channel simulates the mixture of the :math:`T_1` and the :math:`T_2` processes on superconducting devices.

    Args:
        const_t: :math:`T_1` and :math:`T_2` relaxation time in microseconds.
        exec_time: Quantum gate execution time in the process of relaxation in nanoseconds.
        qubits_idx: Indices of the qubits on which the channels act. Defaults to list(range(# of acted qubits))

    Note:
        Relaxation time must satisfy :math:`T_2 \le T_1`. For reference please see https://arxiv.org/abs/2101.02109.
        Unless input state is pure, this class will not create a new instance for output.
    """
    def __init__(
            self, const_t: Union[torch.Tensor, Iterable[float]], exec_time: Union[torch.Tensor, float],
            qubits_idx: Union[Iterable[int], int, str] = None
    ):
        channel_info = {
            'name': 'thermal_relaxation',
            'tex': r'\mathcal{N}',
            'api': 'thermal_relaxation',
            'kwargs': {'const_t': const_t, 'exec_time': exec_time}
        }
        super().__init__(
            'kraus', thermal_relaxation_kraus(const_t, exec_time),
            qubits_idx, acted_system_dim=2, check_legality=False, channel_info=channel_info)


class ReplacementChannel(Channel):
    r"""A collection of quantum replacement channels.

    For a quantum state :math:`\sigma`, the corresponding replacement channel :math:`R` is defined as

    .. math::

        R(\rho) = \text{tr}(\rho)\sigma

    Args:
        sigma: The state to be replaced.
        system_idx: Indices of the qubits on which the channels act, the length of which is :math:`n`.
            Defaults to list(range(# of acted qubits)).

    """
    def __init__(
            self, sigma: State,
            system_idx: Union[Iterable[int], int, str] = None
    ):
        channel_info = {
            'name': 'replace_channel',
            'tex': r'\mathcal{N}',
        }
        self.replace_state = sigma.clone()
        super().__init__('choi', replacement_choi(sigma.density_matrix), system_idx, 
                         acted_system_dim=sigma.dim, channel_info=channel_info)
        
    def forward(self, state):
        for system_idx in self.system_idx:
            state = state._reset(system_idx, self.replace_state)
        return state
