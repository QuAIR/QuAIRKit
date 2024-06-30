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
==============

QuAIRKit is a Python research framework for quantum computing, quantum information, 
and quantum machine learning algorithm development. It focuses on flexible design, 
real-time simulation and rapid verification of quantum and classical algorithms.

## Installation

The minimum Python environment for QuAIRKit is `3.8`.
We recommend installing QuAIRKit with Python `3.10`.

```bash
conda create -n quair python=3.10
conda activate quair
conda install jupyter notebook
```

We recommend the following way of installing QuAIRKit with pip,

```bash
pip install quairkit
```

or download all the files and finish the installation locally,

```bash
git clone https://github.com/QuAIR/QuAIRKit
cd QuAIRKit
pip install -e .
```

## Functionality

- Quantum neural network algorithm simulation
- Quantum circuit simulation & visualization
- Quantum channel simulation
- Quantum algorithm/information tools

### Modules

`quairkit`: QuAIRKit source code

- `ansatz`: module of circuit templates
- `database`: module of useful matrices & sets
- `operator`: module of quantum operators
- `qinfo`: library of quantum algorithms & information tools
- `circuit`: quantum circuit interface

### Tutorials

Check out the tutorial folder on [GitHub](https://github.com/QuAIR/QuAIRKit) for more information.

## Relations with Paddle Quantum

[Paddle Quantum](https://github.com/PaddlePaddle/Quantum) is the world's first cloud-integrated 
quantum machine learning platform based on Baidu PaddlePaddle. As most contributors to this project 
are also contributors to Paddle Quantum, QuAIRKit incorporates key architectural elements and 
interface designs from its predecessor. QuAIRKit focuses more on providing specialized tools and 
resources for researchers and developers engaged in cutting-edge quantum algorithm design and 
theoretical explorations in quantum information science.
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from .core import set_backend, set_device, set_dtype, set_seed
from .core import get_backend, get_device, get_dtype, get_seed, get_float_dtype
from .core import Hamiltonian, State, Operator, visual, to_state
from . import ansatz
from . import database
from . import operator
from . import loss
from . import qinfo
from .circuit import Circuit

name = 'quairkit'
__version__ = '0.1.0'
