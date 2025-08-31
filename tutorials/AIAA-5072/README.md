# Tutorials for AIAA 5072

Welcome! This folder contains coding tutorials for the course [Quantum Computing (AIAA 5072)](https://prog-crs.hkust.edu.hk/pgcourse) at HKUST (Guangzhou). The course is offered by the Thrust of Artificial Intelligence, to provide a brief introduction to quantum computing.

These tutorials use [QuAIRKit](https://github.com/QuAIR/QuAIRKit) to help you learn quantum algorithms, ranging from basic computational problems, to communication networks and high-level machine learning applications.

## Getting Started

To run these tutorials, you need Python and QuAIRKit installed. Here are two ways to set up your environment:

### Option 1: Online Jupyter Server

HKUST(GZ) offers an [online Jupyter server](https://jupyter.hpc.hkust-gz.edu.cn) for all students. Log in with your university account. The server already has Python set up. Simply upload the tutorial files to your home directory and start working.

Note: The first time you run a notebook, it may take a while to install QuAIRKit and its dependencies. Also, the server resets after periods of inactivity. Circuit visualization may not work online due to missing LaTeX-related support, but you can skip those steps or use a local setup.

### Option 2: Local Setup

#### Python Environment

We recommend [Anaconda](https://www.anaconda.com/download/success) for managing Python. After installing Anaconda, create a new environment and install QuAIRKit:

```bash
conda create -n quair python=3.10
conda activate quair
conda install jupyter notebook
pip install quairkit
```

*Mac users: Check if your computer uses an Apple silicon or an Intel chip and download the correct Anaconda version.*

#### Code Editor

Use [Visual Studio Code](https://code.visualstudio.com/) (VSCode) for editing and running Jupyter notebooks. Install these helpful extensions:

- Jupyter
- Python Extension Pack
- Code Spell Checker
- Error Lens

You can also get free access to GitHub Copilot via GitHub Education to help you write code faster.

#### LaTeX (Optional)

QuAIRKit uses LaTeX to create academic-level circuit diagrams. To enable this, install a LaTeX distribution on your computer. See [Texlive+VSCode Configuration in Mac and Win10](https://github.com/kuxuanwang/Texlive-VSCode_Configuration_in_Mac_and_Win10) for instructions.

You can still run all tutorials if you don’t install LaTeX. Circuit visualization won’t work, but you can use `Circuit.plot(print_code=True)` to get the diagram code and paste it into an online LaTeX editor like [Overleaf](https://www.overleaf.com/) to view the circuit.

---

For more information, visit our [group website](https://www.quair.group) for research opportunities and collaborations.

Please give a star to the [QuAIRKit](https://github.com/QuAIR/QuAIRKit) GitHub repository if you find it useful!
