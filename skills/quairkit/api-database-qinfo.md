# Database And Qinfo APIs

## Scope
Use this file for:

- `quairkit.database`
- `quairkit.qinfo`
- matrix/state/channel generation
- random data generation
- analysis and post-processing utilities
- NumPy/Torch interoperability

## `quairkit.database`
`database` is the data-generation toolbox.

Objects returned from `database` may be:

- `Hamiltonian`
- `State`
- `torch.Tensor`
- NumPy arrays for some parameterized array-returning helpers when the input is NumPy

## Database Public Families

### Hamiltonian Generators
- `ising_hamiltonian(edges: torch.Tensor, vertices: torch.Tensor) -> Hamiltonian`
- `xy_hamiltonian(edges: torch.Tensor) -> Hamiltonian`
- `heisenberg_hamiltonian(edges: torch.Tensor) -> Hamiltonian`

Important distinction:

- `ising_hamiltonian` uses both an interaction matrix `edges` and a local-field vector `vertices`
- `xy_hamiltonian` uses only `edges`
- `heisenberg_hamiltonian` uses only `edges`

Runtime-verified caveat:

- in the current environment, `heisenberg_hamiltonian` expects a 3D `edges` tensor whose leading dimension stores the XX / YY / ZZ coupling weights

### Matrix Generators
- `phase`
- `shift`
- `grover_matrix`
- `qft_matrix`
- `h`, `s`, `sdg`, `t`, `tdg`, `eye`, `x`, `y`, `z`
- `p`, `rx`, `ry`, `rz`, `u3`
- `cnot`, `cy`, `cz`, `swap`
- `cp`, `crx`, `cry`, `crz`, `cu`
- `rxx`, `ryy`, `rzz`, `ms`
- `cswap`, `toffoli`, `ccx`
- `universal2`, `universal3`, `universal_qudit`
- `Uf`, `Of`
- `permutation_matrix`

## Database Signature Patterns

### Fixed Matrix Constructors
These return `torch.Tensor` and do not mirror NumPy input types:

- `h() -> torch.Tensor`
- `s() -> torch.Tensor`
- `sdg() -> torch.Tensor`
- `t() -> torch.Tensor`
- `tdg() -> torch.Tensor`
- `x() -> torch.Tensor`
- `y() -> torch.Tensor`
- `z() -> torch.Tensor`
- `cnot() -> torch.Tensor`
- `cy() -> torch.Tensor`
- `cz() -> torch.Tensor`
- `swap() -> torch.Tensor`
- `cswap() -> torch.Tensor`
- `toffoli() -> torch.Tensor`
- `ccx() -> torch.Tensor`
- `eye(dim: int = 2) -> torch.Tensor`
- `phase(dim: int) -> torch.Tensor`
- `shift(dim: int) -> torch.Tensor`
- `grover_matrix(oracle: np.ndarray | torch.Tensor, dtype: torch.dtype | None = None) -> np.ndarray | torch.Tensor`
- `qft_matrix(num_systems: int | None = None, system_dim: int | list[int] = 2) -> torch.Tensor`

Important oracle distinctions:

- `Of(f, n)` is the `n`-qubit phase oracle for unstructured search and satisfies `U|x> = (-1)^{f(x)}|x>`
- `Uf(f, n)` is Simon's algorithm oracle and returns a `2n`-qubit XOR-style unitary satisfying `U|x, y> = |x, y xor f(x)>`
- `grover_matrix(oracle)` does not expect the phase oracle from `Of(...)`; its `oracle` argument is the unitary `A` in QuAIRKit's documented Grover-operator formula
- for textbook phase-oracle Grover search, use `Of(...)` plus a manually constructed diffusion operator unless you have verified that a different `grover_matrix(...)` wiring matches your intended convention

### Parameterized Matrix Constructors
These usually follow the `_ArrayLike` mirror rule:

- `rx(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `ry(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `rz(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `p(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `u3(theta: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor`
- `cp(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `crx(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `cry(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `crz(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `cu(theta: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor`
- `rxx(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `ryy(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `rzz(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `ms(theta: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `universal2(theta: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor`
- `universal3(theta: np.ndarray | torch.Tensor | Iterable[float]) -> np.ndarray | torch.Tensor`
- `universal_qudit(theta: np.ndarray | torch.Tensor | Iterable[float], dimension: int) -> np.ndarray | torch.Tensor`

Practical rule:

- NumPy parameter input usually gives NumPy output
- tensor or scalar input usually gives tensor output
- fixed matrix constructors such as `cnot()` or `h()` still return `torch.Tensor`

### State Factories
These return QuAIRKit state objects, not bare tensors:

- `zero_state(...) -> State`
- `one_state(...) -> State`
- `computational_state(...) -> State`
- `bell_state(num_systems: int | None = None, system_dim: int | list[int] = 2) -> State`
- `ghz_state(...) -> State`
- `w_state(...) -> State`
- `isotropic_state(...) -> State`

Other named factories in this family include `bell_diagonal_state`, `completely_mixed_computational`, `r_state`, and `s_state`.

Important caveat:

- `bell_state(num_systems)` requires an even number of systems
- in the current runtime, `bell_state()` with no explicit argument does not mean "the smallest Bell pair"; use `bell_state(2)` when you want the standard two-qubit Bell state

### Random Generators
Representative signatures:

- `random_state(num_systems: int, rank: int | None = None, is_real: bool = False, size: int | list[int] = 1, system_dim: int | list[int] = 2) -> State`
- `random_unitary(num_systems: int, size: int | list[int] = 1, system_dim: int | list[int] = 2) -> torch.Tensor`
- `random_channel(num_systems: int, rank: int | None = None, target: str = "kraus", size: int | None = 1, system_dim: int | list[int] = 2) -> torch.Tensor`

Other names in this family include `random_pauli_str_generator`, `random_hamiltonian_generator`, `random_hermitian`, `random_projector`, `random_orthogonal_projection`, `random_density_matrix`, `random_unitary_hermitian`, `random_unitary_with_hermitian_block`, `random_lcu`, `haar_orthogonal`, `haar_unitary`, `haar_state_vector`, `haar_density_operator`, and `random_clifford`.

Important note:

- `random_state(...)` currently returns a simulator-backed state object such as `MixedState`
- do not assume `random_state(...)` will expose a pure-state `ket`; in the current runtime it commonly returns a mixed-state object
- many random generators accept `size` for batch generation
- `num_qubits` may exist as an alias of `num_systems`

### Kraus And Representation Generators
Typical pattern:

- `bit_flip_kraus(prob: float | np.ndarray | torch.Tensor) -> list[np.ndarray | torch.Tensor]`
- `depolarizing_kraus(prob: float | np.ndarray | torch.Tensor) -> list[np.ndarray | torch.Tensor]`
- `replacement_choi(sigma: State | torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray`

Do not assume every representation helper returns exactly the same container type. Check the specific family when writing code.

Other helpers in this family include `phase_flip_kraus`, `bit_phase_flip_kraus`, `amplitude_damping_kraus`, `generalized_amplitude_damping_kraus`, `phase_damping_kraus`, `generalized_depolarizing_kraus`, `pauli_kraus`, `reset_kraus`, and `thermal_relaxation_kraus`.

### Bases / Sets
Useful basis/set constructors include `pauli_basis`, `pauli_group`, `pauli_str_basis`, `pauli_str_povm`, `qft_basis`, `std_basis`, `bell_basis`, `heisenberg_weyl`, `phase_space_point`, and `gell_mann`.

## Database Usage Patterns

### Standard-State Construction
Use `database` when the target state has a known name:

- `zero_state`
- `bell_state`
- `ghz_state`
- `w_state`

### Random Training / Benchmark Data
Use:

- `random_state`
- `random_unitary`
- `random_channel`
- `random_hamiltonian_generator`

Batch generation is commonly controlled by `size`, for example:

```python
states = random_state(2, size=100)
unitaries = random_unitary(1, size=32)
```

### Matrix-Level Customization
Use `database.matrix` helpers when:

- building custom oracles
- checking expected matrix forms
- producing fixed gate/channel examples

For input/output type behavior, rely on the signature patterns above instead of assuming universal NumPy round-tripping across all `database` helpers.

## `quairkit.qinfo`
`qinfo` is the main analysis and post-processing toolbox.

The functions are best understood in three families:

- validation / checks
- linear algebra and state manipulation
- quantum-information quantities

## Qinfo Public Families

### Check Functions
- `is_choi`
- `is_density_matrix`
- `is_hermitian`
- `is_linear`
- `is_positive`
- `is_povm`
- `is_projector`
- `is_ppt`
- `is_pvm`
- `is_state_vector`
- `is_unitary`

### Linear-Algebra Functions
- `abs_norm`
- `block_enc_herm`
- `create_matrix`
- `dagger`
- `direct_sum`
- `gradient`
- `hessian`
- `herm_transform`
- `kron_power`
- `logm`
- `nkron`
- `NKron`
- `p_norm`
- `partial_trace`
- `partial_trace_discontiguous`
- `partial_transpose`
- `pauli_decomposition`
- `permute_systems`
- `prob_sample`
- `schmidt_decompose`
- `sqrtm`
- `trace`
- `trace_norm`

### Quantum-Information Functions
- `channel_repr_convert`
- `create_choi_repr`
- `decomp_1qubit`
- `decomp_ctrl_1qubit`
- `diamond_norm`
- `gate_fidelity`
- `general_state_fidelity`
- `link`
- `logarithmic_negativity`
- `mana`
- `mutual_information`
- `negativity`
- `pauli_str_convertor`
- `purity`
- `relative_entropy`
- `stab_nullity`
- `stab_renyi`
- `state_fidelity`
- `trace_distance`
- `von_neumann_entropy`

## Qinfo Signature Patterns

### Check Function Signatures
Typical pattern:

- `is_unitary(mat: np.ndarray | torch.Tensor, eps: float | None = 1e-4) -> bool | list[bool]`
- `is_density_matrix(mat: np.ndarray | torch.Tensor, eps: float | None = 1e-4) -> bool | list[bool]`

Use these as validation helpers, not as differentiable objectives.

### Single-Input Linear-Algebra Functions
Representative exact signatures:

- `dagger(mat: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`
- `partial_trace(state: np.ndarray | torch.Tensor | State, trace_idx: int | list[int], system_dim: int | list[int] = 2) -> np.ndarray | torch.Tensor | State`
- `von_neumann_entropy(rho: np.ndarray | torch.Tensor | State, base: float | int | np.ndarray | torch.Tensor | None = 2) -> np.ndarray | torch.Tensor`

`partial_trace(...)` interprets `trace_idx` as the subsystems to trace out (remove). The remaining subsystems form the output object.

Typical rule:

- NumPy input often leads to NumPy output
- tensor input often leads to tensor output
- `State` input often leads to tensor or state-aware output depending on the function

### Multi-Input Information Functions
Representative exact signatures:

- `state_fidelity(rho: np.ndarray | torch.Tensor | State, sigma: np.ndarray | torch.Tensor | State) -> np.ndarray | torch.Tensor`
- `trace_distance(rho: np.ndarray | torch.Tensor | State, sigma: np.ndarray | torch.Tensor | State) -> np.ndarray | torch.Tensor`
- `gate_fidelity(U: np.ndarray | torch.Tensor, V: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor`

Typical rule:

- all data inputs NumPy -> NumPy output
- any tensor or `State` among the inputs -> tensor output

### Utility Functions With Special Return Types
- `prob_sample(distribution: np.ndarray | torch.Tensor, shots: int = 1024, binary: bool = True, proportional: bool = False) -> dict[str, int | float | torch.Tensor]`

In practice, `prob_sample(...)` may return dictionary values backed by `torch.Tensor`, so do not over-trust the static annotation.

## Recommended High-Frequency Qinfo Functions
When writing user-facing examples, these are especially common:

- `state_fidelity`
- `trace_distance`
- `permute_systems`
- `partial_trace`
- `trace`
- `nkron`
- `is_unitary`
- `channel_repr_convert`
- `gate_fidelity`
- `von_neumann_entropy`
- `purity`

## Qinfo Versus Loss
- `qinfo` is analysis-oriented
- `loss` is training-oriented

Prefer:

- `qinfo.*` for scripts, diagnostics, post-processing, and NumPy-mixed analysis
- `loss.*` for reusable trainable modules inside model code

## NumPy / Tensor / State Interop Rules
Use the signature sections above as the source of truth:

- `Database Signature Patterns` for matrix/channel generators and state factories
- `Qinfo Signature Patterns` for analysis helpers and metrics
- `State` inputs usually become tensor-based outputs unless a function explicitly preserves a state-aware path such as `partial_trace`

The only safe global summary is that `qinfo` mirrors NumPy/Tensor inputs more consistently than `database`, while fixed `database` constructors such as `h()` and `cnot()` remain tensor-first.

## Practical Interop Examples

### Array Analysis
Use NumPy arrays when the result must stay in NumPy for external tooling:

```python
rho_np = rho.density_matrix.detach().cpu().numpy()
fid = state_fidelity(rho_np, rho_np)
```

### Tensor Analysis
Use tensors or `State` inputs when results should remain inside PyTorch/QuAIRKit workflows:

```python
fid = state_fidelity(state_a, state_b)
```

### High-Frequency Runtime Facts
Verified in the current environment:

- `type(rx(0.5)).__name__ == "Tensor"`
- `type(rx(np.array([0.5]))).__name__ == "ndarray"`
- `type(u3(np.array([0.1, 0.2, 0.3]))).__name__ == "ndarray"`
- `type(cnot()).__name__ == "Tensor"`
- `type(ising_hamiltonian(torch.ones(2, 2), torch.ones(2))).__name__ == "Hamiltonian"`
- `type(random_state(2)).__name__ == "MixedState"`
- `type(state_fidelity(np.eye(2), np.eye(2))).__name__ == "ndarray"`
- `type(state_fidelity(np.eye(2), torch.eye(2, dtype=torch.complex128))).__name__ == "Tensor"`
- `type(trace_distance(np.eye(2), np.eye(2))).__name__ == "ndarray"`
- `type(von_neumann_entropy(np.eye(2))).__name__ == "ndarray"`
- `type(dagger(np.eye(2))).__name__ == "ndarray"`
- `inspect.signature(ising_hamiltonian) == (edges, vertices)`
- `inspect.signature(xy_hamiltonian) == (edges)`
- `inspect.signature(heisenberg_hamiltonian) == (edges)`

## Notes On Validation Functions
Many `is_*` functions return:

- a Python boolean-like result for a single input
- a list/batch-structured boolean result for batched input

These are analysis helpers, not differentiable training objectives.

## Notes On Batch Behavior
Many `qinfo` functions support batch inputs directly.

Examples:

- batched state fidelity
- batched trace distance
- batched partial traces
- batched validation checks

If a function accepts two batched objects, expect broadcast-like or pairwise matching rules rather than arbitrary shape mixing.

## Guidance For Writing Examples
- Use `database` to build named states and matrices.
- Use `qinfo` to analyze what those objects mean.
- Keep generation and analysis conceptually separate in user-facing code.
- Mention NumPy compatibility only when it matters to the task; otherwise keep examples tensor-first.

## Common Pitfalls
- `database.rx(...)` returns a matrix, not a callable gate object. `database.rx(0.5)(state)` is wrong.
- `Circuit.rx(...)`, `database.rx(...)`, and `quairkit.operator.RX(...)` are different abstractions. Use `Circuit` for circuit construction and `database` for matrices.
- Do not assume every `database` helper mirrors NumPy input types. Fixed constructors such as `h()` and `cnot()` return `torch.Tensor`.
- Do not assume all Hamiltonian generators share one signature. In the current runtime: `ising_hamiltonian(edges, vertices)`, `xy_hamiltonian(edges)`, `heisenberg_hamiltonian(edges)`.
- `bell_state()` without an explicit even `num_systems` is easy to misuse. Use `bell_state(2)` when you want the standard two-qubit Bell pair.
- `Uf` and `Of` are not interchangeable. `Uf(f, n)` is Simon's `2n`-qubit XOR oracle, while `Of(f, n)` is the `n`-qubit phase oracle for unstructured search.
- `grover_matrix(Of(...))` is not automatically the textbook phase-oracle Grover iterate. Check the documented `oracle` convention first, or build diffusion plus `Of(...)` manually.
- `state_fidelity` and `trace_distance` return NumPy only when all data inputs are NumPy. Mixed NumPy/Tensor input returns a tensor.
- `von_neumann_entropy` follows a different single-input rule from two-input metrics, so do not generalize from `state_fidelity`.
- For pure or nearly pure inputs, `von_neumann_entropy` can emit matrix-log warnings from the underlying numerical routine in some environments even when the returned entropy is correctly near zero.
- `random_state(...)` returns a state object, not a raw tensor.
