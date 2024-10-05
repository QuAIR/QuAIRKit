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

Batch computation
-----------------

QuAIRKit supports batch computations for quantum circuit simulations, state measurement and quantum information processing. It is easy to use and can be customized for different quantum (machine learning) algorithms.

Below is an example of batch computation for quantum circuit simulation. Here a zero state is passed through four different quantum circuits, and compared with the target state.

```python
import quairkit as qkit
from quairkit.database import *
from quairkit.qinfo import *

target_state = zero_state(1)
unitary_data = pauli_group(1)

cir = qkit.Circuit(1)
cir.oracle(unitary_data, 0)
cir.ry(param=[0, 1, 2, 3])

print(state_fidelity(cir(), target_state)) # zero-state input by default
```

```text
tensor([1.0000, 0.4794, 0.8415, 0.0707])
```

Qudit computation
-----------------

QuAIRKit also supports batch computations for quantum circuit simulations and most of the quantum information processing tools in qudit quantum computing. Note that qudit computation can be used with batch computation, as shown below

```python
# claim three systems, with 1 qubit and 1 qutrit
cir = qkit.Circuit(2, system_dim=[2, 3])

# apply the Heisenberg-Weyl operators on all systems
cir.oracle(heisenberg_weyl(6), [0, 1])

# apply the H gate on the first system, controlled by the second system
cir.control_oracle(h(), [1, 0])

# trace out the qutrit system and get the qubit state
traced_state = cir().trace(1)

print('The 6th and 7th state for the batched qubit state is', traced_state[5:7])
```

```text
The 6th and 7th state for the batched qubit state is 
---------------------------------------------------
 Backend: density_matrix
 System dimension: [2]
 System sequence: [0]
 Batch size: [2]

 # 0:
[[1.+0.j 0.+0.j]
 [0.+0.j 0.+0.j]]
 # 1:
[[0.5+0.j 0.5+0.j]
 [0.5+0.j 0.5+0.j]]
---------------------------------------------------
```

Fast construction
-----------------

QuAIRKit provides a fast and flexible way to construct quantum circuits, by self-managing the parameters. All parameters would be created randomly if not specified. QuAIRKit also supports built-in layer ansatzes, such as `complex_entangled_layer`.

```python
cir = qkit.Circuit(3)

cir.h() # apply Hadamard gate on all qubits
cir.complex_entangled_layer(depth=3) # apply complex entangled layers of depth 3
cir.universal_three_qubits() # apply universal three-qubit gate with random parameters
```

`qkit.Circuit` is a child class of `torch.nn.Module`, so you can access its parameters and other attributes directly, or use it as a layer in a hybrid neural network.

Implicit transition
-------------------

If you want to perform noise simulation or mixed-state-related tools, there is no need to specify the backend, or import other libraries. Just call the function, and QuAIRKit will transit the backend for you.

```python
cir = qkit.Circuit(3)

cir.complex_entangled_layer(depth=3)
print(cir().backend)

# partial transpose on the first two qubits
print(cir().transpose([0, 1]).backend)

cir.depolarizing(prob=0.1)
print(cir().backend)
```

```text
state_vector
density_matrix
density_matrix
```

Global setup
------------

QuAIRKit provides global setup functions to set the default data type, device and random seed.

```python
qkit.set_dtype('complex128') # default data type is complex64
qkit.set_device('cuda') # make sure CUDA is setup with torch
qkit.set_seed(73) # set seeds for all random number generators
```

Overall Structure
-----------------

QuAIRKit provides the following functionalities,

- Quantum neural network algorithm simulation
- Quantum circuit simulation & visualization
- Quantum channel simulation
- Quantum algorithm/information tools

`quairkit`: QuAIRKit source code

- `database`: module of useful matrices & sets
- `loss`: module of quantum loss functions
- `qinfo`: library of quantum algorithms & information tools
- `circuit`: quantum circuit interface

Tutorials
---------

- `Hamiltonian in QuAIRKit <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/Hamiltonian.ipynb>`_
- `Constructing Quantum Circuits in QuAIRKit <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/circuit.ipynb>`_
- `Measuring quantum states in QuAIRKit <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/measure.ipynb>`_
- `Quantum gates and quantum channels <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/operator.ipynb>`_
- `Quantum information tools <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/qinfo.ipynb>`_
- `Manipulation of Quantum States in QuAIRKit <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/state.ipynb>`_
- `Training parameterized quantum circuits <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/introduction/training.ipynb>`_
- `Batch Computation <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/feature/batch.ipynb>`_
- `Neural network setup customization <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/feature/custom.ipynb>`_
- `Introduction to qudit quantum computing <https://github.com/QuAIR/QuAIRKit/tree/v0.2.0/tutorials/feature/qudit.ipynb>`_

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
    import matplotlib
    import numpy
    import scipy
    import torch
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

    
    import re
    import subprocess

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
