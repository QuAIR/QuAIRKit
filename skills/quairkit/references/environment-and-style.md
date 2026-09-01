# QuAIRKit 0.5.1 Environment And Style

Use this reference when setup, imports, plotting, or final-answer style matters. Run `python scripts/check_version.py` from the skill directory before applying its guidance.

## Runtime Install

- Target exactly QuAIRKit 0.5.1. Do not silently substitute another release.
- Use an official wheel matching the operating system and Python interpreter; do not install from a repository tree.
- The verified baseline is Python 3.10 with PyTorch 2.11.
- When installing from the official package index, use a pinned command:

  ```bash
  python -m pip install "quairkit==0.5.1"
  python -m pip check
  ```

- After installation, run `python scripts/check_version.py` and stop if it reports anything other than 0.5.1.

## Default Imports

Use explicit public imports in final code:

```python
import quairkit as qkit
from quairkit import Circuit, Hamiltonian, State, to_state
from quairkit.database import bell_state, zero_state
from quairkit.qinfo import state_fidelity
```

Use `scripts/recommend_import.py` when a symbol's public route is unclear. Avoid wildcard imports in final user-facing code.

## Default Global Setup

- Use `qkit.set_dtype("complex128")` when numerical stability matters.
- Use `qkit.set_device("cpu")` or `qkit.set_device("cuda")` with PyTorch-style device strings.
- Use `qkit.set_seed(seed)` when reproducibility matters.
- Backend, dtype, device, and seed are global settings; reset them in shared tests when needed.

## Plotting And Paper Integration

- `Circuit.to_latex()` returns Quantikz code.
- `Circuit.plot()` depends on `pdflatex`.
- Recommended TeX distributions include TeX Live and MacTeX.
- On macOS, install `poppler` if PDF page-count errors appear.
- For arXiv, include Quantikz support files when the archive does not provide them.
- Exporting Quantikz code and compiling it in Overleaf is a valid fallback.

## Deliverable Style

- Match the user's language in discussion.
- Keep code, comments, identifiers, and repository-facing artifacts in English unless explicitly requested otherwise.
- Keep examples short and runnable, use QuAIRKit 0.5.1 public APIs, and make external dependencies explicit.
- If behavior is uncertain, identify the 0.5.1 check needed instead of guessing.
