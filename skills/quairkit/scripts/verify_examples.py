import numpy as np
import torch

import quairkit as qkit
from quairkit import Circuit, Hamiltonian, to_state
from quairkit.application import OneWayLOCCNet
from quairkit.core import SimpleStateOperator
from quairkit.database import bell_state, rx
from quairkit.loss import ExpecVal, Measure
from quairkit.qinfo import state_fidelity, trace_distance


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_index_semantics() -> None:
    c = Circuit(3)
    c.x([0, 1])
    check(len(list(c.children())) == 1, "x([0,1]) should create one child module")
    check(c.operator_history[0]["system_idx"] == [[0], [1]], "x([0,1]) should record two single-system targets")

    c2 = Circuit(3)
    c2.cnot([0, 1])
    check(c2.operator_history[0]["system_idx"] == [[0, 1]], "cnot([0,1]) should be one two-system operator")

    c3 = Circuit(3)
    c3.cnot([[0, 1], [1, 2]])
    check(c3.operator_history[0]["system_idx"] == [[0, 1], [1, 2]], "batched cnot indices mismatch")


def verify_param_registration() -> None:
    c = Circuit(2)
    c.ry([0, 1], param=None)
    check(len(list(c.parameters())) == 1, "param=None should register trainable parameters")

    p = torch.nn.Parameter(torch.tensor([0.1, 0.2], dtype=qkit.get_float_dtype()))
    c2 = Circuit(2)
    c2.ry([0, 1], param=p)
    check(len(list(c2.parameters())) == 1, "explicit Parameter should stay registered")

    fixed = torch.tensor([0.1, 0.2], dtype=qkit.get_float_dtype())
    c3 = Circuit(2)
    c3.ry([0, 1], param=fixed)
    check(len(list(c3.parameters())) == 0, "plain tensor should not be registered as a module parameter")


def verify_append_extend() -> None:
    layer = Circuit(2)
    layer.h(0)
    layer.cnot([0, 1])

    a = Circuit(2)
    a.append(layer)
    check(len(list(a.children())) == 1, "append should keep the layer as one child")
    check(type(a[0]).__name__ == "Circuit", "append should preserve the child type")

    b = Circuit(2)
    b.extend(layer)
    check(len(list(b.children())) == 2, "extend should flatten into operator children")
    check([type(x).__name__ for x in b.children()] == ["H", "CNOT"], "extend flatten order mismatch")


def verify_numpy_interop() -> None:
    arr = np.array([0.1, 0.2])
    mat = rx(arr)
    check(isinstance(mat, np.ndarray), "rx(np.ndarray) should return np.ndarray")

    rho_np = bell_state(2).density_matrix.detach().cpu().numpy()
    rho_t = bell_state(2).density_matrix
    rho_s = to_state(rho_t)

    check(isinstance(state_fidelity(rho_np, rho_np), np.ndarray), "numpy+numpy state_fidelity should return numpy")
    check(isinstance(state_fidelity(rho_t, rho_t), torch.Tensor), "tensor+tensor state_fidelity should return tensor")
    check(isinstance(state_fidelity(rho_s, rho_s), torch.Tensor), "state+state state_fidelity should return tensor")
    check(isinstance(trace_distance(rho_np, rho_t), torch.Tensor), "mixed numpy/tensor trace_distance should return tensor")


def verify_loccnet() -> None:
    net = OneWayLOCCNet({"Alice": 1, "Bob": 1})
    net["Alice"].u3([0])
    net.param_locc(qkit.database.u3, 3, [("Alice", 0), ("Bob", 0)], support_batch=False)
    net.set_init_state([("Alice", 0), ("Bob", 0)])

    check(sorted(net.keys()) == ["Alice", "Bob"], "LOCCNet party names mismatch")
    check(type(net.physical_circuit).__name__ == "Circuit", "physical_circuit should be a Circuit")

    out = net()
    check(out.system_dim == [2], "Teleportation-style LOCCNet output should leave one qubit after trace")


def verify_stateoperator_backend() -> None:
    qkit.set_backend(SimpleStateOperator)
    try:
        ham = Hamiltonian([[-1.0, "Z0,Z1"]])
        measure = Measure()
        energy = ExpecVal(ham)

        cir = Circuit(2)
        cir.h(0)
        cir.cnot([0, 1])

        state = cir()
        prob = measure(state=state, system_idx=[0, 1], shots=64)
        value = energy(state, shots=64)

        check(type(state).__name__ == "SimpleStateOperator", "SimpleStateOperator backend not active")
        check(tuple(prob.shape) == (4,), "shot-based measurement should return 4 probabilities for two qubits")
        check(isinstance(value, torch.Tensor), "shot-based expectation should return a tensor")

        try:
            to_state(torch.tensor([1.0, 0.0], dtype=torch.complex128))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("to_state should fail for numeric input on StateOperator backends")
    finally:
        qkit.set_backend("default")


def main() -> None:
    qkit.set_dtype("complex128")
    qkit.set_device("cpu")
    qkit.set_backend("default")

    verify_index_semantics()
    verify_param_registration()
    verify_append_extend()
    verify_numpy_interop()
    verify_loccnet()
    verify_stateoperator_backend()

    print("OK")


if __name__ == "__main__":
    main()
