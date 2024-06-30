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

[Paddle Quantum](https://github.com/PaddlePaddle/Quantum) is the world's first cloud-integrated quantum machine learning platform based on Baidu PaddlePaddle. As most contributors to this project are also contributors to Paddle Quantum, QuAIRKit incorporates key architectural elements and interface designs from its predecessor. QuAIRKit focuses more on providing specialized tools and resources for researchers and developers engaged in cutting-edge quantum algorithm design and theoretical explorations in quantum information science.
