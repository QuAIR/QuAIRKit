# QuAIRKit

QuAIRKit is a Python research framework for quantum computing, quantum information, and quantum machine learning algorithm development. It focuses on flexible design, real-time simulation and rapid verification of quantum and classical algorithms.

## Installation

The minimum Python environment for QuAIRKit is `3.8`. We recommend installing QuAIRKit with Python `3.10`.

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

## Features

### Batch computation

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

Above output is equivalent to

```math
\left|\bra{0} R_z(0)\,I \ket{0} \right|,\,\,
\left|\bra{0} R_z(1)\,X \ket{0} \right|,\,\,
\left|\bra{0} R_z(2)\,Y \ket{0} \right|,\,\,
\left|\bra{0} R_z(3)\,Z \ket{0} \right|
```

### Qudit computation

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

### Fast construction

QuAIRKit provides a fast and flexible way to construct quantum circuits, by self-managing the parameters. All parameters would be created randomly if not specified. QuAIRKit also supports built-in layer ansatzes, such as `complex_entangled_layer`.

```python
cir = qkit.Circuit(3)

cir.h() # apply Hadamard gate on all qubits
cir.complex_entangled_layer(depth=3) # apply complex entangled layers of depth 3
cir.universal_three_qubits() # apply universal three-qubit gate with random parameters
```

`qkit.Circuit` is a child class of `torch.nn.Module`, so you can access its parameters and other attributes directly, or use it as a layer in a hybrid neural network.

### Implicit transition

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

### Global setup

QuAIRKit provides global setup functions to set the default data type, device and random seed.

```python
qkit.set_dtype('complex128') # default data type is complex64
qkit.set_device('cuda') # make sure CUDA is setup with torch
qkit.set_seed(73) # set seeds for all random number generators
```

## Overall Structure

QuAIRKit provides the following functionalities,

- Quantum neural network algorithm simulation
- Quantum circuit simulation & visualization
- Quantum channel simulation
- Quantum algorithm/information tools

### Modules

`quairkit`: QuAIRKit source code

- `database`: module of useful matrices & sets
- `loss`: module of quantum loss functions
- `qinfo`: library of quantum algorithms & information tools
- `circuit`: quantum circuit interface

### Tutorials

- [Introduction](tutorials/introduction)
  - [Hamiltonian in QuAIRKit](tutorials/introduction/Hamiltonian.ipynb)
  - [Constructing Quantum Circuits in QuAIRKit](tutorials/introduction/circuit.ipynb)
  - [Measuring quantum states in QuAIRKit](tutorials/introduction/measure.ipynb)
  - [Quantum gates and quantum channels](tutorials/introduction/operator.ipynb)
  - [Quantum information tools](tutorials/introduction/qinfo.ipynb)
  - [Manipulation of Quantum States in QuAIRKit](tutorials/introduction/state.ipynb)
  - [Training parameterized quantum circuits](tutorials/introduction/training.ipynb)

- [Feature](tutorials/feature)
  - [Batch Computation](tutorials/feature/batch.ipynb)
  - [Neural network setup customization](tutorials/feature/custom.ipynb)
  - [Introduction to qudit quantum computing](tutorials/feature/qudit.ipynb)

## Acknowledgement

We appreciate the kind support from the [Sourcery AI](https://sourcery.ai/) that greatly enhances the coding & review quality of the QuAIRKit project.
