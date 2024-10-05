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

QuAIRKit is a Python research framework for quantum computing, quantum information,
and quantum machine learning algorithm development. It focuses on flexible design,
real-time simulation and rapid verification of quantum and classical algorithms.

Installation
------------

The minimum Python environment for QuAIRKit is ``3.8``.
We recommend installing QuAIRKit with Python ``3.10``.

.. code-block:: bash

    conda create -n quair python=3.10
    conda activate quair
    conda install jupyter notebook

We recommend the following way of installing QuAIRKit with pip,

.. code-block:: bash

    pip install quairkit

or download all the files and finish the installation locally,

.. code-block:: bash

    git clone https://github.com/QuAIR/QuAIRKit
    cd QuAIRKit
    pip install -e .

Functionality
-------------

- Quantum neural network algorithm simulation
- Quantum circuit simulation & visualization
- Quantum channel simulation
- Quantum algorithm/information tools

Modules
-------

``quairkit``: QuAIRKit source code

- ``ansatz``: module of circuit templates
- ``database``: module of useful matrices & sets
- ``operator``: module of quantum operators
- ``qinfo``: library of quantum algorithms & information tools
- ``circuit``: quantum circuit interface

Tutorials
---------

Check out the tutorial folder on `GitHub <https://github.com/QuAIR/QuAIRKit>`_ for more information.

Relations with Paddle Quantum
-----------------------------

`Paddle Quantum <https://github.com/PaddlePaddle/Quantum>`_ is the world's first cloud-integrated
quantum machine learning platform based on Baidu PaddlePaddle. As most contributors to this project
are also contributors to Paddle Quantum, QuAIRKit incorporates key architectural elements and
interface designs from its predecessor. QuAIRKit focuses more on providing specialized tools and
resources for researchers and developers engaged in cutting-edge quantum algorithm design and
theoretical explorations in quantum information science.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from .core import set_backend, set_device, set_dtype, set_seed
from .core import get_backend, get_device, get_dtype, get_seed, get_float_dtype
from .core import Hamiltonian, State, Operator, visual, to_state
from . import ansatz
from . import database
from . import operator
from . import loss
from . import qinfo
from .circuit import Circuit

name = "quairkit"
__version__ = "0.2.0"


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
