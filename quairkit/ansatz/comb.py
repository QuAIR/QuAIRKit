# !/usr/bin/env python3
# Copyright (c) 2024 QuAIR team. All Rights Reserved.
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
The source file of the PQCombNet class.
"""

import csv
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import ModuleList

from ..circuit import Circuit
from ..core import set_seed, to_state, utils
from ..database import bell_state, random_unitary, zero_state
from ..operator import ParamOracle
from ..qinfo import channel_repr_convert


class PQCombNet:
    r"""
    Parameterized Quantum Comb Net.

    Args:
        target_function: The function to apply to each tensor in the dataset.
        num_slots: The parameter 'num_slots' used in the specific quantum computation context.
        num_aux_qubits: The parameter 'num_aux_qubits' also used in the quantum computation context.
        num_qubits_U: The number of qubits of the unitaries to be queried.
        train_unitary_info: Information for generating or providing training unitaries, either as an integer or a tensor.
        test_unitary_info: Information for generating or providing testing unitaries, similar to train_unitary_info.
        train_mode: The mode to use for training, default is 'choi'.
        LR: Learning rate for the optimization algorithm.
        NUM_ITR: Number of iterations to run the training.
        name_task: Optional name for the task, useful for data logging and storage.
        seed: Optional seed for random number generation, enhancing reproducibility.
        is_save_data: Flag to determine if data should be saved during training.
        is_auto_stop: Flag to enable stopping the training process automatically based on certain criteria.
        is_ctrl_U: Flag to indicate if a controlled-U operation is used in the training process.
    """
    def __init__(
        self,
        target_function: Callable[[torch.Tensor], torch.Tensor],
        num_slots: int,
        num_aux_qubits: int,
        num_qubits_U: int = 1,
        train_unitary_info: Union[int, torch.Tensor] = 200,
        test_unitary_info: Union[int, torch.Tensor] = 1000,
        train_mode: str = "choi",
        LR: float = 0.1,
        NUM_ITR: int = 1000,
        name_task: str = None,
        seed: Optional[int] = None,
        is_save_data: bool = False,
        is_auto_stop: bool = True,
        is_ctrl_U: bool = False,
    ) -> None:
        _setup_parameters(
            self,
            target_function,
            num_qubits_U,
            num_slots,
            num_aux_qubits,
            name_task,
            is_ctrl_U,
            LR,
            NUM_ITR,
            train_mode,
            is_save_data,
            is_auto_stop,
            seed,
        )
        _setup_unitary_sets(self, train_unitary_info, test_unitary_info)
        _initialize_training_environment(self)

    def update_V_circuit(
        self,
        index: int,
        new_V: Union[
            Circuit, ParamOracle, torch.Tensor, Tuple[torch.Tensor, List[int]]
        ],
    ) -> None:
        r"""
        Update the V circuit at the specified index with a new circuit.

        Args:
            index: The index of the V circuit to update.
            new_V: The new V circuit, which can be a ParamOracle, Circuit, torch.Tensor or Tuple[torch.Tensor, List[int]].

        Raises:
            ValueError: If the index is out of range or if the number of qubits in the provided Circuit does not match the number of qubits in the existing Circuit.
            TypeError: If new_V is not a ParamOracle, Circuit, torch.Tensor or Tuple[torch.Tensor, List[int]].
        """
        if not (0 <= index < len(self.V_circuit_list)):
            raise ValueError(f"Index out of range: {index}")

        if isinstance(V := new_V, Circuit):
            if new_V.num_qubits != self.num_qubits:
                raise ValueError(
                    f"The number of qubits in the provided Circuit does not match the number of qubits in the existing Circuit: {self.num_qubits}"
                )
        else:
            V = Circuit(self.num_qubits)
            if isinstance(new_V, ParamOracle):
                V.append(new_V)
            elif isinstance(new_V, torch.Tensor):
                V.oracle(new_V, list(range(self.num_qubits)))
            elif (
                isinstance(new_V, tuple)
                and isinstance(new_V[0], torch.Tensor)
                and isinstance(new_V[1], list)
            ):
                V.oracle(new_V[0], new_V[1])
            else:
                raise TypeError(
                    "new_V must be a Circuit, ParamOracle, torch.Tensor, Tuple[torch.Tensor, or List[int]]"
                )

        self.V_circuit_list[index] = V

    def train(self) -> None:
        r"""
        Train the PQCombNet model.
        """

        opt = torch.optim.Adam(self.V_circuit_list.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")

        time_list = []
        for itr in range(self.NUM_ITR):
            _print_progress_bar(itr + 1, self.NUM_ITR, prefix="Progress")

            start_time = time.time()
            loss = 1 - _average_fidelity(self, self.train_unitary_set, self.omega_train)
            current_lr = scheduler.get_last_lr()[0]
            time_list.append(time.time() - start_time)

            if _log_progress(
                self,
                itr,
                loss,
                time_list,
                current_lr,
            ):
                break

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(loss)
        fidelity = _average_fidelity(
            self, self.test_unitary_set, self.omega_test
        ).item()
        print(
            f"[{self.name_task} | {self.train_mode} | {self.seed}] Finished training with Fidelity: {fidelity:.8f}"
        )

    def extract_highest_fidelity(self) -> None:
        r"""
        Call the _extract_highest_fidelity function to generate the fidelity tables.
        If the file does not exist, prompt the user to set is_save_data to True.
        """
        filepath = os.path.join(
            self.data_directory_name, f"{self.name_task}_train_log.csv"
        )
        if not os.path.exists(filepath):
            print(
                f"File {filepath} does not exist. Consider setting is_save_data to True."
            )
            return

        _extract_highest_fidelity(
            self.data_directory_name, f"{self.name_task}_train_log.csv"
        )


def _prepare_initial_state(self: PQCombNet, is_choi_mode: bool) -> torch.Tensor:
    r"""
    Prepare the initial state for the quantum circuit.

    Args:
        is_choi_mode: A boolean flag indicating whether to prepare a Choi state.

    Returns:
        The initial state tensor.
    """
    ancilla_state = (
        [zero_state(self.num_aux_qubits).ket]
        if self.num_aux_qubits > 0
        else [torch.eye(1)]
    )
    bell_states = [bell_state(2 * self.num_qubits_U).ket] * (
        self.num_V if is_choi_mode else 1
    )

    return to_state(utils.linalg._nkron(*(ancilla_state + bell_states)))


def _construct_circuit(
    self: PQCombNet,
    num_qubits_loss: int,
    unitary_set: torch.Tensor,
    V_list_applied_index: list,
) -> Circuit:
    r"""
    Construct the quantum circuit for the loss calculation.

    Args:
        num_qubits_loss: Number of qubits in the circuit.
        unitary_set: The set of unitary matrices to apply.
        V_list_applied_index: List of indices for applying V circuits.

    Returns:
        The constructed quantum circuit.
    """
    cir_loss = Circuit(num_qubits_loss)
    _apply_V_circuits(self, cir_loss, V_list_applied_index, unitary_set)
    return cir_loss


def _compute_output_density_matrix(
    self: PQCombNet,
    circuit: Circuit,
    Psi_in: torch.Tensor,
    is_choi_mode: bool,
    num_qubits_loss: int,
) -> torch.Tensor:
    r"""
    Compute the output density matrix after applying the circuit to the input state.

    Args:
        circuit: The quantum circuit to apply.
        Psi_in: The input quantum state.
        is_choi_mode: Boolean flag indicating whether it's in Choi mode.
        num_qubits_loss: The total number of qubits used in the loss calculation.

    Returns:
        The output density matrix.
    """
    Psi_out_dm = circuit(Psi_in).density_matrix

    return utils.linalg._partial_trace(
        Psi_out_dm,
        list(range(self.num_aux_qubits)),
        [2] * num_qubits_loss,
    ) * (2 ** (self.num_qubits_U * self.num_V) if is_choi_mode else 1)


def _compute_fidelity(
    self: PQCombNet,
    psi_out_density_matrix: torch.Tensor,
    omega: torch.Tensor,
    is_choi_mode: bool,
) -> torch.Tensor:
    r"""
    Compute the fidelity between the output density matrix and the omega tensor.

    Args:
        psi_out_density_matrix: The output density matrix.
        omega: The omega tensor used for computing the fidelity.
        is_choi_mode: Boolean flag indicating whether it's in Choi mode.

    Returns:
        The fidelity value as a real-valued tensor.
    """
    if is_choi_mode:
        return (
            utils.linalg._trace(psi_out_density_matrix @ omega)
            / (2 ** (2 * self.num_qubits_U))
        ).real
    else:
        return torch.mean(utils.linalg._trace(psi_out_density_matrix @ omega)).real


def _average_fidelity(
    self: PQCombNet, unitary_set: torch.Tensor, omega: torch.Tensor
) -> torch.Tensor:
    r"""
    Compute the average fidelity for a given set of unitaries and omega tensor.

    Args:
        unitary_set: The set of unitary matrices.
        omega: The omega tensor used for computing the fidelity.

    Returns:
        The average fidelity as a real-valued tensor.
    """
    # Set up the parameters
    is_choi_mode = self.train_mode == "choi"
    num_qubits_loss = self.num_aux_qubits + 2 * self.num_qubits_U * (
        self.num_V if is_choi_mode else 1
    )
    V_list_applied_index = _get_V_list_applied_index(self, is_choi_mode)

    # Prepare the input state
    Psi_in = _prepare_initial_state(self, is_choi_mode)

    # Construct the circuit
    circuit = _construct_circuit(
        self, num_qubits_loss, unitary_set, V_list_applied_index
    )

    # Compute the output density matrix
    psi_out_density_matrix = _compute_output_density_matrix(
        self, circuit, Psi_in, is_choi_mode, num_qubits_loss
    )

    # Compute and return the fidelity
    return _compute_fidelity(self, psi_out_density_matrix, omega, is_choi_mode)


def _save_results(self: PQCombNet, itr, fidelity, loss, current_lr):
    data = {
        "num_qubits_U": self.num_qubits_U,
        "num_slots": self.num_slots,
        "num_aux_qubits": self.num_aux_qubits,
        "train_mode": self.train_mode,
        "LR": self.LR,
        "current_lr": current_lr,
        "num_test_unitary": len(self.train_unitary_set),
        "num_train_unitary": len(self.train_unitary_set),
        "seed": self.seed,
        "NUM_ITR": itr,
        "loss": loss,
        "fidelity": fidelity,
    }
    _save_results_to_csv(
        data=data,
        filename=f"{self.name_task}_train_log.csv",
        directory=self.data_directory_name,
    )
    if self.is_save_data:
        V_circuit_lists_dir = os.path.join(self.data_directory_name, "V_circuit_lists")
        os.makedirs(V_circuit_lists_dir, exist_ok=True)
        save_path = os.path.join(
            V_circuit_lists_dir,
            f"V_circuit_list_{self.train_mode}_nqu{self.num_qubits_U}_m{self.num_slots}_na{self.num_aux_qubits}_itr{itr}_fid{fidelity:.5f}.pt",
        )
        torch.save(self.V_circuit_list, save_path)


def _setup_parameters(
    self: PQCombNet,
    target_function: Callable[[torch.Tensor], torch.Tensor],
    num_qubits_U: int,
    num_slots: int,
    num_aux_qubits: int,
    name_task: Optional[str] = None,
    is_ctrl_U: bool = False,
    LR: float = 0.1,
    NUM_ITR: int = 10000,
    train_mode: str = "choi",
    is_save_data: bool = False,
    is_auto_stop: bool = True,
    seed: Optional[int] = None,
) -> None:
    r"""
    Combines the setup of basic parameters and training-specific parameters for quantum computation and training.

    Args:
        target_function: Function applied to tensors.
        num_qubits_U, num_slots, num_aux_qubits: Parameters defining the quantum circuit's structure and number of qubits.
        name_task: Task name for logging and data management.
        is_ctrl_U: Determines if a controlled-U operation is used.
        LR: Learning rate for the optimizer.
        NUM_ITR: Total iterations for the training process.
        train_mode: Specifies the training mode to use.
        is_save_data: Enables or disables data saving.
        is_auto_stop: Enables automatic stopping of training.
        seed: Random seed for reproducibility.
    """
    # Basic parameters setup
    self.target_function = target_function
    self.is_ctrl_U = is_ctrl_U
    self.num_qubits_U = num_qubits_U
    self.num_slots = num_slots
    self.num_aux_qubits = num_aux_qubits
    self.num_V = num_slots + 1
    self.num_qubits = num_aux_qubits + num_qubits_U
    self.name_task = (
        name_task
        or f"pqcomb_search_{target_function.__name__}{'_ctrl' if is_ctrl_U else ''}"
    )

    # Training parameters setup
    self.LR = LR
    self.NUM_ITR = NUM_ITR
    self.is_save_data = is_save_data
    self.is_auto_stop = is_auto_stop
    self.seed = seed or np.random.randint(1e6)
    set_seed(self.seed)

    _validate_training_mode(self, train_mode)


def _setup_unitary_sets(
    self: PQCombNet,
    train_unitary_info: Union[int, torch.Tensor],
    test_unitary_info: Union[int, torch.Tensor],
) -> None:
    r"""
    Prepares the unitary sets for training and testing.

    Args:
        train_unitary_info: Information or data for training unitaries.
        test_unitary_info: Information or data for testing unitaries.
    """
    self.train_unitary_set = _generate_unitary_set(
        train_unitary_info, self.num_qubits_U
    )
    self.test_unitary_set = _generate_unitary_set(test_unitary_info, self.num_qubits_U)

    _calculate_omegas(self)


def _calculate_omegas(self: PQCombNet) -> None:
    r"""
    Calculate omega for train and test unitary sets.
    """
    try:
        self.omega_train = _get_omega(self, self.train_unitary_set)
        self.omega_test = _get_omega(self, self.test_unitary_set)
    except RuntimeError as e:
        if "not enough memory" not in str(e) or self.train_mode != "choi":
            raise e

        print(
            f"[{self.name_task} | {self.train_mode} | {self.seed}] "
            f"Out of memory error caught, switching train_mode from '{self.train_mode}' to ",
            end="",
        )
        self.train_mode = "pqc"
        print(f"'{self.train_mode}'...")
        self.omega_train = _get_omega(self, self.train_unitary_set)
        self.omega_test = _get_omega(self, self.test_unitary_set)


def _validate_training_mode(self: PQCombNet, train_mode: str) -> None:
    r"""
    Validates the training mode against supported modes.

    Args:
        train_mode: Training mode to validate.
    """
    train_mode_list = ["choi", "pqc"]
    if train_mode.lower() in train_mode_list:
        self.train_mode = "pqc" if self.is_ctrl_U else train_mode.lower()
    else:
        raise ValueError(
            f"Invalid train_mode: {train_mode}, must be one of {train_mode_list}"
        )


def _initialize_training_environment(self: PQCombNet) -> None:
    r"""
    Sets up the data directories and initializes the training environment.

    Initializes the omega tensors and the list of variable quantum circuits for the simulation.
    """
    self.data_directory_name = f"{self.name_task}_data"
    self.V_circuit_list = _create_V_circuit_list(self)


def _get_omega(self: PQCombNet, unitary_set: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the omega tensor for a given task.

    Args:
        unitary_set: The set of unitary matrices.

    Returns:
        torch.Tensor: The omega tensor.
    """
    if self.train_mode == "choi":
        return _compute_omega_choi(
            self.target_function, unitary_set, self.num_qubits_U, self.num_slots
        )
    else:
        return _compute_omega_pqc(self.target_function, unitary_set, self.num_qubits_U)


def _create_V_circuit_list(self: PQCombNet) -> ModuleList:
    r"""
    Create a list of V circuits.

    Returns:
        ModuleList: The list of V circuits.
    """
    V_circuit_list = ModuleList()
    qubit_index_list = list(range(self.num_qubits))

    for _ in range(self.num_V):
        V = Circuit(self.num_qubits)

        if self.num_qubits == 1:
            V.u3(qubit_index_list)
        elif self.num_qubits == 2:
            V.universal_two_qubits(qubit_index_list)
        elif self.num_qubits == 3:
            V.universal_three_qubits(qubit_index_list)
        else:
            V.real_entangled_layer(qubit_index_list, depth=self.num_qubits * 10)
        V_circuit_list.append(V)

    return V_circuit_list


def _get_V_list_applied_index(
    self: PQCombNet, is_choi_mode: bool
) -> Union[List[List[int]], List[int]]:
    r"""
    Returns the list of indices where V circuits are applied, depending on the training mode.

    Args:
        is_choi_mode: Indicates whether the 'choi' mode is used for determining the index list.

    Returns:
        Union[List[List[int]], List[int]]: The list of indices where V circuits are applied.
    """
    if is_choi_mode:
        return [
            list(range(self.num_aux_qubits))
            + list(
                range(
                    self.num_aux_qubits + self.num_qubits_U * (2 * j + 1),
                    self.num_aux_qubits + self.num_qubits_U * (2 * j + 2),
                )
            )
            for j in range(self.num_V)
        ]
    else:
        return list(range(self.num_qubits))


def _apply_V_circuits(
    self: PQCombNet,
    cir_loss: Circuit,
    V_list_applied_index: List[Union[List[int], int]],
    unitary_set: torch.Tensor,
) -> None:
    r"""
    Applies the V circuits to the circuit loss object.

    Args:
        cir_loss: The circuit to which the V circuits are applied.
        V_list_applied_index: A list or a list of lists of indices where V circuits should be applied.
        unitary_set: The set of unitary matrices used in the control operations.
    """
    for index, V_circuit in enumerate(self.V_circuit_list):
        cir_loss.oracle(
            V_circuit.unitary_matrix(),
            (
                V_list_applied_index[index]
                if self.train_mode == "choi"
                else V_list_applied_index
            ),
            latex_name=f"$V_{index}$",
        )
        if self.train_mode == "pqc" and index < self.num_V - 1:
            _apply_controlled_U(self, cir_loss, unitary_set, index)


def _apply_controlled_U(
    self: PQCombNet, cir_loss: Circuit, unitary_set: torch.Tensor, index: int
) -> None:
    r"""
    Applies the controlled U or U† depending on the configuration.

    Args:
        cir_loss: The circuit to which the controlled operations are applied.
        unitary_set: The set of unitary matrices.
        index: The index of the current operation in the sequence.
    """
    if self.is_ctrl_U:
        cir_loss.control_oracle(
            unitary_set if index % 2 == 0 else utils.linalg._dagger(unitary_set),
            list(
                range(self.num_aux_qubits - 1, self.num_aux_qubits + self.num_qubits_U)
            ),
            latex_name=(r"$U$" if index % 2 == 0 else r"$U^{\dagger}$"),
        )
    else:
        cir_loss.oracle(
            unitary_set,
            list(range(self.num_aux_qubits, self.num_aux_qubits + self.num_qubits_U)),
            latex_name=(r"$U$"),
        )


def _log_progress(
    self: PQCombNet, itr: int, loss: torch.Tensor, time_list: list, current_lr: float
) -> bool:
    r"""
    Logs the training progress at specified intervals and saves the results.
    It provides insights into the current state of training, including loss, fidelity, and learning rate.

    This function checks if the current iteration is a multiple of 40 or the last iteration.
    If so, it calculates the average fidelity, constructs a log message with relevant metrics,
    and prints it. The function also saves the results and may determine if training should stop early.

    Args:
        itr: The current iteration number.
        loss: The current loss value.
        time_list: A list of time taken for each iteration.
        current_lr: The current learning rate.

    Returns:
        Returns True if training should be stopped early, otherwise None.
    """
    if itr % 40 == 0 or itr == self.NUM_ITR - 1:
        fidelity = _average_fidelity(
            self, self.test_unitary_set, self.omega_test
        ).item()

        log_message = (
            f"[{self.name_task} | {self.train_mode} | {self.seed} | \033[90m{itr}\t{np.mean(time_list):.4f}s\033[0m] "
            f"num_qubits_U: {self.num_qubits_U}, num_slots: {self.num_slots}, num_aux_qubits: {self.num_aux_qubits}, "
            f"\033[93mLR: {current_lr:.2e}\033[0m, "
            f"\033[91mLoss: {loss.item():.8f}\033[0m, "
            f"\033[92mFid: {fidelity:.8f}\033[0m"
        )

        print(log_message)
        time_list.clear()
        # Save results and possibly stop training
        if self.is_save_data:
            _save_results(self, itr, fidelity, loss, f"{current_lr:.2e}")
        if current_lr < 1e-3 * self.LR and self.is_auto_stop:
            return True


def _compute_omega_choi(
    target_function: Callable[[torch.Tensor], torch.Tensor],
    unitary_set: torch.Tensor,
    num_qubits_U: int,
    num_slots: int,
) -> torch.Tensor:
    r"""
    Compute the omega tensor using the 'choi' mode.

    Args:
        target_function: The function to apply to each tensor in the dataset.
        unitary_set: The set of unitary matrices.
        num_qubits_U: The number of qubits of the unitaries to be queried.
        num_slots: The parameter 'num_slots' used in computation.

    Returns:
        torch.Tensor: The computed omega tensor.
    """
    omega = 0
    for u in unitary_set:
        u_transformed_choi = channel_repr_convert(
            target_function(u), source="kraus", target="choi"
        )
        u_conj_choi = channel_repr_convert(u.conj(), source="kraus", target="choi")
        omega += utils.linalg._nkron(
            *([u_transformed_choi] + [u_conj_choi] * num_slots)
        )
    omega /= len(unitary_set)

    perm_list = list(range(2 * num_qubits_U * (num_slots + 1)))
    end = num_qubits_U * 2

    start = num_qubits_U
    perm_list = perm_list[:start] + perm_list[end:] + perm_list[start:end]
    return utils.linalg._permute_systems(
        omega, perm_list, [2] * (2 * start) * (num_slots + 1)
    )


def _generate_unitary_set(
    unitary_info: Union[int, torch.Tensor], num_qubits_U: int
) -> torch.Tensor:
    r"""
    Generates a set of unitary matrices based on provided info.

    Args:
        unitary_info: Details to generate or directly provide the unitaries.
        num_qubits_U: num_qubits_U for the unitary matrices.
    """
    return (
        random_unitary(num_qubits_U, unitary_info)
        if isinstance(unitary_info, int)
        else unitary_info
    )


def _compute_omega_pqc(
    target_function: Callable[[torch.Tensor], torch.Tensor],
    unitary_set: torch.Tensor,
    num_qubits_U: int,
) -> torch.Tensor:
    r"""
    Compute the omega tensor using the 'pqc' mode.

    Args:
        target_function: The function to apply to each tensor in the dataset.
        unitary_set: The set of unitary matrices.
        num_qubits_U: The number of qubits of the unitaries to be queried.

    Returns:
        torch.Tensor: The computed omega tensor.
    """
    target_unitary = target_function(unitary_set)
    average_output_ket = (
        target_unitary.kron(torch.eye(2**num_qubits_U))
        @ bell_state(2 * num_qubits_U).ket
    )
    return average_output_ket @ utils.linalg._dagger(average_output_ket)


def _print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "█",
) -> None:
    r"""
    Call in a loop to create a terminal progress bar.

    Args:
        iteration: Current iteration.
        total: Total iterations.
        prefix: Prefix string.
        suffix: Suffix string.
        decimals: Positive number of decimals in percent complete.
        length: Character length of the bar.
        fill: Bar fill character.
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()


def _save_results_to_csv(data: Dict[str, Any], filename: str, directory: str) -> None:
    r"""
    Save the results to a CSV file.

    Args:
        data: A dictionary containing the data to be saved.
        filename: The name of the CSV file.
        directory: The directory where the CSV file will be saved.
    """
    filepath = os.path.join(directory, filename)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Define the column names
    fieldnames = list(data.keys())

    # Write to the CSV file
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def _extract_highest_fidelity(data_directory, filename):
    r"""
    Extract the highest fidelity for each combination of num_slots and num_aux_qubits and generate a table.

    Args:
        data_directory: The directory where the CSV file is saved.
        filename: The name of the CSV file.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "The pandas library is required to run this function. Please install it using 'pip install pandas'."
        ) from e

    # Construct the full file path
    filepath = os.path.join(data_directory, filename)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Find unique num_qubits_U values
    num_qubits_U_values = df["num_qubits_U"].unique()

    # Create a directory to save the result tables
    result_dir = os.path.join(data_directory, "fidelity_tables")
    os.makedirs(result_dir, exist_ok=True)

    # Iterate over each unique num_qubits_U value
    for num_qubits_U in num_qubits_U_values:
        # Filter the DataFrame for the current num_qubits_U
        df_filtered = df[df["num_qubits_U"] == num_qubits_U]

        # Pivot the table to have num_slots as rows and num_aux_qubits as columns, with max fidelity as values
        pivot_table = df_filtered.pivot_table(
            index="num_slots",
            columns="num_aux_qubits",
            values="fidelity",
            aggfunc="max",
        )

        # Save the pivot table to a CSV file
        output_filename = f"fidelity_table_num_qubits_U_{num_qubits_U}.csv"
        output_filepath = os.path.join(result_dir, output_filename)
        pivot_table.to_csv(output_filepath)
        print(f"Saved table for num_qubits_U = {num_qubits_U} to {output_filepath}")
