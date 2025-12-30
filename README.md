# QuAIRKit

QuAIRKit is a Python SDK for algorithm development in quantum computing, quantum information, and quantum machine learning. It focuses on flexible design, real-time simulation and rapid verification of quantum and classical algorithms.

<p align="center">
  <!-- docs -->
  <a href="https://quairkit.com/QuAIRKit/latest/index.html">
    <img src="https://img.shields.io/badge/docs-link-green.svg?style=flat-square&logo=read-the-docs"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/quairkit/">
    <img src="https://img.shields.io/badge/pypi-v0.4.4-orange.svg?style=flat-square&logo=pypi"/>
  </a>
  <!-- Python -->
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python"/>
  </a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square&logo=apache"/>
  </a>
  <!-- Platform -->
  <a href="https://quairkit.com/QuAIRKit">
    <img src="https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-lightgrey.svg?style=flat-square"/>
  </a>
  <!-- Statistics -->
  <a href="https://pepy.tech/projects/quairkit">
    <img src="https://static.pepy.tech/badge/quairkit"/>
  </a>
</p>

QuAIRKit provides the following functionalities,

- Quantum algorithm simulation & optimization
- Quantum circuit simulation & visualization
- Quantum channel simulation
- Quantum algorithm/information tools

## Installation

The minimum Python environment for QuAIRKit is `3.9`. We recommend installing QuAIRKit with Python `3.10`.

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
pip install -e . --config-settings editable_mode=strict
```

## Setup

After installation, you can import QuAIRKit in your Python code as follows:

```python
import quairkit as qkit
import torch # library for tensor manipulation

from quairkit import Circuit # standard quantum circuit interface

from quairkit.database import ... # common matrices, sets, Hamiltonian, states, etc.
from quairkit.qinfo import ... # common functions in quantum information processing
from quairkit.loss import ... # common loss operators in neural network training
```

QuAIRKit provides global setup functions to set the default data type, device and random seed.

```python
qkit.set_dtype('complex128') # default data type is 'complex64'
qkit.set_device('cuda') # make sure CUDA is setup with torch
qkit.set_seed(73) # set seeds for all random number generators
```

## Features

QuAIRKit provides a wide range of features for quantum computing, quantum information processing and quantum machine learning. Below are some of the key features:

### Batch computation

QuAIRKit supports batch computations for quantum circuit simulations, state measurement and quantum information processing. It is easy to use and can be customized for different quantum (machine learning) algorithms.

Below is an example of batch computation for quantum circuit simulation. Here a zero state is passed through four different quantum circuits, and compared with the target state.

```python
target_state = zero_state(1)
unitary_data = pauli_group(1) # I, Pauli-X/Y/Z

cir = Circuit(1)
cir.oracle(unitary_data, 0)
cir.ry(param=[0, 1, 2, 3])

output_state = cir() # zero-state input by default
print(state_fidelity(cir(), target_state)) 
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

QuAIRKit also supports batch computations for quantum circuit simulations and most of the quantum information processing tools in qudit quantum computing, as shown below

```python
# claim three systems, with 1 qubit and 1 qutrit
cir = Circuit(2, system_dim=[2, 3])

# apply 6^2 Heisenberg-Weyl operators on all systems
cir.oracle(heisenberg_weyl(6), [0, 1])

# apply the H gate on the qubit, controlled by the qutrit
cir.oracle(h(), [1, 0], control_idx=0)

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

### Probabilistic computation

QuAIRKit supports probabilistic quantum circuit simulation, which allows you to simulate quantum circuits with probabilistic operations, such as measurement, partial post-selection, LOCC. This is useful in quantum communication protocols or quantum algorithm design. This functional is also compatible with batch computation and qudit computation.

Below is the implementation of a qubit teleportation protocol in QuAIRKit.

```python
M1_locc = torch.stack([eye(), x()]) # apply X gate for measure outcome 1
M2_locc = torch.stack([eye(), z()]) # apply Z gate for measure outcome 1

# setup protocol
cir = Circuit(3)
cir.cnot([0, 1])
cir.h(0)
cir.locc(M1_locc, [1, 2]) # measure on qubit 1, apply local operations on qubit 2
cir.locc(M2_locc, [0, 2]) # measure on qubit 0, apply local operations on qubit 2

# test with 100 random single-qubit (mixed) states
psi = random_state(1, size=100)
input_state = nkron(psi, bell_state(2))
output_state = cir(input_state).trace([0, 1]) # discard first two qubits

fid = state_fidelity(output_state.expec_state(), psi).mean().item()
print('The average fidelity of the teleportation protocol is', fid)
```

```text
The average fidelity of the teleportation protocol is 0.9999999999998951
```

### Other functionalities

#### Plot circuit with LaTeX

Circuit in QuAIRKIt can be plotted with Quantikz, a LaTeX package for quantum circuit visualization, which is useful for academic presentation. You can use the `plot` function to visualize the circuit. Make sure you have up-to-date LaTeX installed on your system, so that the `quantikz` package is available.

```python
cir: Circuit = ...
cir.plot(print_code=True)  # plot the circuit with LaTeX code
```

See the [tutorial](tutorials/feature/plot.ipynb) for more details.

#### Third-party Cloud Integration

QuAIRKit supports third-party cloud integration, which allows you to run quantum circuits on real quantum devices with QuAIRKit interfaces. You can use the `set_backend` function to set the backend for measurement and computation of expectation value.

```python
class YourState(qkit.StateOperator):
    def _execute(self, qasm: str, shots: int) -> Dict[str, int]:
        r"""IMPLEMENT HERE to execute the circuit on the quantum cloud."""

qkit.set_backend(YourState)
```

See the [tutorial](tutorials/feature/cloud.ipynb) for more details.

#### Fast construction

QuAIRKit provides a fast and flexible way to construct quantum circuits, by self-managing the parameters. All parameters would be created randomly if not specified. QuAIRKit also supports built-in layer ansatz, such as `complex_entangled_layer`.

```python
cir = Circuit(2)

cir.rx() # apply Rx gates on all qubits with random parameters
cir.complex_entangled_layer(depth=2) # apply complex entangled layers of depth 2
cir.universal_two_qubits() # apply universal two-qubit gate with random parameters
```

`Circuit` is a child class of `torch.nn.Module`, so you can access its parameters and other attributes directly, or use it as a layer in a hybrid neural network.

#### Implicit transition

If you want to perform noise simulation or mixed-state-related tools, there is no need to specify the backend, or import other libraries. Just call the function, and QuAIRKit will transit the backend for you.

```python
cir = Circuit(3)

cir.complex_entangled_layer(depth=3)
print(cir().backend)

# partial transpose on the first two qubits
print(cir().transpose([0, 1]).backend)

cir.depolarizing(prob=0.1)
print(cir().backend)
```

```text
default-pure
default-mixed
default-mixed
```

## Tutorials

- [Introduction](tutorials/introduction)
  - [Constructing quantum circuits in QuAIRKit](tutorials/introduction/circuit.ipynb)
  - [Manipulation of quantum states in QuAIRKit](tutorials/introduction/state.ipynb)
  - [Measuring quantum states in QuAIRKit](tutorials/introduction/measure.ipynb)
  - [Hamiltonian in QuAIRKit](tutorials/introduction/Hamiltonian.ipynb)
  - [Quantum information tools](tutorials/introduction/qinfo.ipynb)
  - [Quantum gates and quantum channels](tutorials/introduction/operator.ipynb)
  - [Training parameterized quantum circuits](tutorials/introduction/training.ipynb)

- [Feature](tutorials/feature)
  - [Batch computation](tutorials/feature/batch.ipynb)
  - [Drawing Quantum Circuits with QuAIRKit](tutorials/feature/plot.ipynb)
  - [Neural network setup customization](tutorials/feature/custom.ipynb)
  - [Introduction to qudit quantum computing](tutorials/feature/qudit.ipynb)
  - [Running QuAIRKit with third-party quantum cloud platforms](tutorials/feature/cloud.ipynb)

- [Research](tutorials/research)
  - [Analyze Barren Plateau in quantum neural networks](tutorials/research/bp.ipynb)
  - [Hamiltonian simulation via Trotter decomposition](tutorials/research/trotter.ipynb)
  - [Quantum State Teleportation and Distribution](tutorials/research/locc.ipynb)
  - [Search quantum information protocols with LOCCNet](tutorials/research/loccnet.ipynb)
  - [Rediscovering Simon's algorithm with PQC](tutorials/research/simon.ipynb)
  - [Training quantum process transformation with PQC](tutorials/research/comb.ipynb)

- [Tutorials for AIAA 5072](tutorials/AIAA-5072), a quantum computing course instructed in HKUST(GZ)
  - [Brief Introduction to Quantum Computing](tutorials/AIAA-5072/week%201%20qubit.ipynb)
  - [Quantum error mitigation](tutorials/AIAA-5072/week%202%20quasi.ipynb)
  - [Variational quantum eigensolver](tutorials/AIAA-5072/week%203%20vqe.ipynb)
  - [Entanglement](tutorials/AIAA-5072/week%204%20entanglement.ipynb)
  - [Superdense coding](tutorials/AIAA-5072/week%205%20superdense-coding.ipynb)
  - [Quantum state teleportation](tutorials/AIAA-5072/week%206%20teleportation.ipynb)
  - [Quantum state amplitude amplification](tutorials/AIAA-5072/week%207%20qaa.ipynb)
  - [Quantum phase estimation](tutorials/AIAA-5072/week%208%20qpe.ipynb)
  - [Hamiltonian simulation](tutorials/AIAA-5072/week%209%20hamiltonian.ipynb)

## Acknowledgements

We appreciate the kind support from the [Sourcery AI](https://sourcery.ai) that greatly enhances the coding & review quality of the QuAIRKit project.
