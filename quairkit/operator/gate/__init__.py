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
The module of the quantum gates.
"""

from ...database.matrix import *
from .base import Gate, ParamGate
from .custom import ControlOracle, Oracle, ParamOracle
from .encoding import (AmplitudeEncoding, AngleEncoding, BasisEncoding,
                       IQPEncoding)
from .multi_qubit_gate import (CCX, CNOT, CP, CRX, CRY, CRZ, CSWAP, CU, CX, CY,
                               CZ, MS, RXX, RYY, RZZ, SWAP, Toffoli,
                               UniversalQudits, UniversalThreeQubits,
                               UniversalTwoQubits)
from .single_qubit_gate import RX, RY, RZ, U3, H, P, S, Sdg, T, Tdg, X, Y, Z
from .visual import _circuit_plot
