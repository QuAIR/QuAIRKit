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
QuAIRKit
========

QuAIRKit is a Python SDK for algorithm development in quantum computing,
quantum information, and quantum machine learning. It focuses on flexible
design, real-time simulation, and rapid verification of quantum and classical
algorithms.

Links
-----
- Documentation: https://quairkit.com/QuAIRKit/latest/index.html
- PyPI: https://pypi.org/project/quairkit/
- Source: https://github.com/QuAIR/QuAIRKit
- License: Apache-2.0

Key features
------------
- Quantum algorithm simulation and optimization
- Quantum circuit simulation and visualization
- Quantum channel simulation
- Quantum algorithm/information tools
- Batch and probabilistic execution; qubit and qudit support
- Seamless integration with PyTorch tensors and autograd
- Optional third-party quantum cloud backends

Installation
------------
Minimum Python: 3.8 (recommended 3.10).

.. code-block:: bash

    pip install quairkit

Or using conda and an editable install:

.. code-block:: bash

    conda create -n quair python=3.10
    conda activate quair
    conda install jupyter notebook
    git clone https://github.com/QuAIR/QuAIRKit
    cd QuAIRKit
    pip install -e . --config-settings editable_mode=strict

Quick start
-----------
Import and basic setup:

.. code-block:: python

    import quairkit as qkit
    from quairkit import Circuit
    from quairkit.database import zero_state, pauli_group
    from quairkit.qinfo import state_fidelity

    # Global defaults
    qkit.set_dtype('complex128')  # default is 'complex64'
    qkit.set_device('cpu')        # or 'cuda' if available
    qkit.set_seed(73)

    # Build and run a simple circuit (batched over I/X/Y/Z)
    cir = Circuit(1)
    cir.oracle(pauli_group(1), 0)
    cir.ry(param=[0.0, 1.0, 2.0, 3.0])
    out = cir(zero_state(1))
    fid = state_fidelity(out, zero_state(1))

Qudit computation
-----------------
Work with heterogeneous system dimensions:

.. code-block:: python

    from quairkit.database import heisenberg_weyl, h

    cir = Circuit(2, system_dim=[2, 3])  # qubit + qutrit
    cir.oracle(heisenberg_weyl(6), [0, 1])
    cir.oracle(h(), [1, 0], control_idx=0)  # H on qubit, controlled by qutrit
    rho_qubit = cir().trace(1)  # trace out qutrit

Probabilistic / LOCC workflows
------------------------------
Model measurement, post-selection, and LOCC:

.. code-block:: python

    import torch
    from quairkit.database import eye, x, z, bell_state, random_state, nkron
    from quairkit.qinfo import state_fidelity

    M1 = torch.stack([eye(), x()])  # apply X for outcome 1
    M2 = torch.stack([eye(), z()])  # apply Z for outcome 1

    cir = Circuit(3)
    cir.cnot([0, 1])
    cir.h(0)
    cir.locc(M1, [1, 2])
    cir.locc(M2, [0, 2])

    psi = random_state(1, size=100)
    inp = nkron(psi, bell_state(2))
    out = cir(inp).trace([0, 1]).expec_state()
    avg_fid = state_fidelity(out, psi).mean().item()

Visualization
-------------
Render circuits via LaTeX/Quantikz:

.. code-block:: python

    cir: Circuit = ...
    cir.plot(print_code=True)

Cloud backends
--------------
Integrate third-party quantum cloud providers for execution:

.. code-block:: python

    class YourState(qkit.StateOperator):
        def _execute(self, qasm: str, shots: int):
            '''Execute the qasm on your backend and return a counts dict.'''

    qkit.set_backend(YourState)

Noise and mixed-state tools
---------------------------
Backends are selected implicitly based on operations:

.. code-block:: python

    cir = Circuit(3)
    cir.complex_entangled_layer(depth=3)
    _ = cir()                     # state_vector backend
    _ = cir().transpose([0, 1])   # switches to density_matrix
    cir.depolarizing(prob=0.1)
    _ = cir()                     # stays density_matrix

Module overview
---------------
- quairkit.circuit: Quantum circuit interface (torch.nn.Module)
- quairkit.database: Common matrices, sets, Hamiltonians, states, etc.
- quairkit.qinfo: Quantum information and algorithm utilities
- quairkit.loss: Loss operators for training
- quairkit.ansatz: Layer templates and ansätze

Tutorials
---------
For notebooks and examples, see:

- https://quairkit.com/QuAIRKit/latest/index.html
- https://github.com/QuAIR/QuAIRKit/tree/main/tutorials
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from .core import set_backend, set_device, set_dtype, set_seed
from .core import get_backend, get_device, get_dtype, get_seed, get_float_dtype
from .core import Hamiltonian, StateSimulator as State, Operator, to_state, StateOperator
from . import ansatz
from . import database
from . import operator
from . import loss
from . import qinfo
from .circuit import Circuit
from . import application

name = "quairkit"
__version__ = "0.4.4"


def print_info() -> None:
    r"""Print the information of QuAIRKit, its dependencies and current environment.

    """
    import torch
    import numpy
    import scipy
    import matplotlib
    print("\n---------VERSION---------")
    print("quairkit:", __version__)
    print("torch:", torch.__version__)
    if torch.cuda.is_available():
        print("torch cuda:", torch.version.cuda)
    print("numpy:", numpy.__version__)
    print("scipy:", scipy.__version__)
    print("matplotlib:", matplotlib.__version__)

    import platform
    print("---------SYSTEM---------")
    print("Python version:", platform.python_version())
    print("OS:", platform.system())
    print("OS version:", platform.version())

    
    import subprocess
    import re
    # stack overflow #4842448
    print("---------DEVICE---------")
    if platform.system() == "Windows":
        cpu_name = platform.processor()
        
    elif platform.system() == "Darwin":
        model = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        cpu_name = model.decode('utf-8')

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                cpu_name = re.sub(".*model name.*:", "", line, 1)
    print("CPU:", cpu_name)

    if torch.cuda.is_available():
        print("GPU: (0)", torch.cuda.get_device_name(0))
        for i in range(torch.cuda.device_count() - 1):
            print(f"     ({i})", torch.cuda.get_device_name(i))
