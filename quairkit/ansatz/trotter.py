# !/usr/bin/env python3
# Copyright (c) 2025 QuAIR team. All Rights Reserved.
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


import re
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch

from ..core import Hamiltonian, utils
from ..operator import CNOT, RX, RY, RZ, H
from .container import Layer

__all__ = ['TrotterLayer']


def map_hamilton_to_qubits_idx(hamiltonian: Hamiltonian, qubits_idx: List[int]) -> List[Hamiltonian]:
    r"""Implement the Hamiltonian on the computational qubits specified by qubit-idx, return the new Hamiltonian after mapping.

    Note:       
        This is an intrinsic function, user do not need to call this directly

    Args:
        hamiltonian (Hamiltonian): The original Hamiltonian to be mapped. It should have a method `decompose_with_sites()`
            that returns three components:
                - coeffs: A list of coefficients corresponding to each term in the Hamiltonian.
                - pauli_words: A list of Pauli strings (e.g., ['X', 'Y', 'Z']) representing the Pauli operators.
                - sites: A list of tuples specifying the qubit indices associated with each Pauli word.
        qubits_idx (List[int]): A list of integers representing the indices of the computational qubits 
            onto which the Hamiltonian will be mapped. The length of this list must be greater than the maximum 
            index present in the `sites` component of the Hamiltonian.

    Returns:
        List[Hamiltonian]: A list containing a single Hamiltonian object. This new Hamiltonian represents the 
            original Hamiltonian remapped to the specified qubits.

    Raises:
        AssertionError: If the number of qubits specified in `qubits_idx` is insufficient to represent all 
            the qubits in the original Hamiltonian.
    """
    coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    assert len(qubits_idx) > np.max([item for sublist in sites for item in sublist if isinstance(item, int)]), "The number of qubits_idx must be bigger than the number of qubits in the Hamiltonian"
    new_terms = []

    for coeff, pauli_word, site in zip(coeffs, pauli_words, sites):
        if all(p == 'I' for p in pauli_word):
            continue
        filtered_pauli_word = [pauli for pauli in pauli_word if pauli != 'I']
        filtered_site = [s for pauli, s in zip(pauli_word, site) if pauli != 'I']
        new_site = tuple(qubits_idx[s] for s in filtered_site)

        pauli_str_parts = [
            f"{pauli}{site}"
            for pauli, site in zip(filtered_pauli_word, new_site)
        ]
        pauli_str = ', '.join(pauli_str_parts)
        new_terms.append([coeff, pauli_str])

    return [Hamiltonian(new_terms)]

def sort_pauli_word(pauli_word, site) -> tuple:
    r"""reordering the ``pauli_word`` by the value of ``site``, return the new pauli_word and site after sort.

    Note:
        This is an intrinsic function, user do not need to call this directly
    """
    sort_index = np.argsort(np.array(site))
    return ''.join(np.array(list(pauli_word))[sort_index].tolist()), np.array(site)[sort_index]


class TrotterLayer(Layer):
    r"""Add time-evolving circuits to a user-specified circuit.
    
    This circuit could approximate the time-evolving operator of a system given its Hamiltonian H,
    i.e., :math:`U_{\rm cir}~ e^{-iHt}`.

    Args:
        hamiltonian: Hamiltonian of the system whose time evolution is to be simulated.
        qubits_idx: Indices of the qubits on which the layer is applied.
        tau: Evolution time of each trotter block.
        num_steps: Number of trotter blocks that will be added in total, where ``num_steps * tau`` should be the total evolution time.
        order: Order of the Trotter-Suzuki decomposition.
        name: Name of the Hamiltonian. Defaults to 'H'.
    
    Raises:
        ValueError: The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)
        ValueError: The number of qubits_idx must be bigger than the number of qubits in the Hamiltonian
        ValueError: Grouping method ``grouping`` is not supported, valid key words: 'xyz', 'even_odd'
        ValueError: The method ``method`` is not supported, valid method keywords: 'suzuki', 'custom'

    """
    def __init__(self, hamiltonian: Hamiltonian, qubits_idx: List[int], 
                 tau: float, num_steps: int, order: int, name: str) -> None:
        super().__init__(qubits_idx, depth=num_steps, name=name)
        self._assert_qubits()
        self.tau = tau
        self.hamiltonian = hamiltonian
        
        assert hamiltonian.n_qubits == len(qubits_idx), \
            f"Number of qubits does not match: receive {qubits_idx}, expect of length {hamiltonian.n_qubits}"

        if order > 2 and order % 2 != 0 and type(order) != int:
            raise ValueError('The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)'
                             ', got order = %i' % order)
        if qubits_idx is None:
            grouped_hamiltonian = [hamiltonian]
        else:
            grouped_hamiltonian = map_hamilton_to_qubits_idx(hamiltonian, qubits_idx)

        for _ in range(num_steps):
            self._add_trotter_block(grouped_hamiltonian=grouped_hamiltonian, order=order)
    
    @property
    def fidelity(self) -> float:
        r"""The fidelity of the Trotterization.
        """
        t = self.tau * self._depth
        ideal_gate = torch.linalg.matrix_exp(-1j * t * self.hamiltonian.matrix)
        return utils.qinfo._gate_fidelity(ideal_gate, self.matrix).item()
        
    def get_latex_name(self, style: str = 'standard') -> str:
        depth = self._depth
        tau_rounded = np.round(self.tau, 3)
        time_rounded = np.round(depth * self.tau, 3)

        if style == "compact":
            if depth > 1:
                name = rf"$\approx \exp(-iH\tau) \times {depth}$, with $\tau = {tau_rounded}$"
            else:
                name = rf"$\approx \exp(-iH\tau)$, with $\tau = {tau_rounded}$"
        else:
            name = rf"$\approx \exp(-iHt)$, with $t = {time_rounded}$"
            if style == "detailed" and self.num_systems <= 6:
                fidelity_rounded = np.round(self.fidelity, 3)
                name += rf" and fidelity ${fidelity_rounded}$"
        return name

    def _add_trotter_block(self, grouped_hamiltonian, order: int) -> None:
        r"""add a Trotter block, i.e. :math:`e^{-iH\tau}`, use Trotter-Suzuki decomposition to expand it.

        Args:
            grouped_hamiltonian: list of Hamiltonian objects, this function uses these as the basic terms of Trotter-Suzuki expansion by default
            order: The order of Trotter-Suzuki expansion

        Note:
            About how to use grouped_hamiltonian: 
            For example, consider Trotter-Suzuki decomposition of the second order S2(t), if grouped_hamiltonian = [H_1, H_2], it will add Trotter circuit
            with (H_1, t/2)(H_2, t/2)(H_2, t/2)(H_1, t/2). Specifically, if user does not pre-grouping the Hamiltonians and put a single Hamiltonian object,
            this function will make canonical decomposition according to the order of this Hamiltonian: for second order, if put a single Hamiltonian H, 
            the circuit will be added with (H[0:-1:1], t/2)(H[-1:0:-1], t/2)

        Warning:
            This function is usually an intrinsic function, it does not check or correct the input. 
            To build time evolution circuit, function ``construct_trotter_circuit()`` is recommended
        """
        tau = self.tau
        if order == 1:
            self.__add_first_order_trotter_block(tau, grouped_hamiltonian)
        elif order == 2:
            self.__add_second_order_trotter_block(tau, grouped_hamiltonian)
        else:
            self.__add_higher_order_trotter_block(tau, grouped_hamiltonian, order)

    def __add_first_order_trotter_block(self, tau: float, grouped_hamiltonian: List[Hamiltonian], reverse=False) -> None:
        r"""Add a time evolution block of the first order Trotter-Suzuki decomposition

        Note:
            This is an intrinsic function, user do not need to call this directly
        """
        if not reverse:
            for hamiltonian in grouped_hamiltonian:
                assert isinstance(hamiltonian, Hamiltonian)
                # decompose the Hamiltonian into 3 lists
                coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
                term_index = 0
                while term_index < len(coeffs):
                    # get the sorted pauli_word and site (an array of qubit indices) according to their qubit indices
                    pauli_word, site = sort_pauli_word(pauli_words[term_index], sites[term_index])
                    self.add_n_pauli_gate(2 * tau * coeffs[term_index], pauli_word, site)
                    term_index += 1
        elif len(grouped_hamiltonian) == 1:
            coeffs, pauli_words, sites = grouped_hamiltonian[0].decompose_with_sites()
            for term_index in reversed(range(len(coeffs))):
                pauli_word, site = sort_pauli_word(pauli_words[term_index], sites[term_index])
                self.add_n_pauli_gate(2 * tau * coeffs[term_index], pauli_word, site)
        else:
            for hamiltonian in reversed(grouped_hamiltonian):
                assert isinstance(hamiltonian, Hamiltonian)
                coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
                for term_index in range(len(coeffs)):
                    pauli_word, site = sort_pauli_word(pauli_words[term_index], sites[term_index])
                    self.add_n_pauli_gate(2 * tau * coeffs[term_index], pauli_word, site)


    def __add_second_order_trotter_block(self, tau, grouped_hamiltonian) -> None:
        r"""Add a time evolution block of the second order Trotter-Suzuki decomposition
        
        Note:
            This is an intrinsic function, user do not need to call this directly
        """
        self.__add_first_order_trotter_block(tau / 2, grouped_hamiltonian)
        self.__add_first_order_trotter_block(tau / 2, grouped_hamiltonian, reverse=True)


    def __add_higher_order_trotter_block(self, tau, grouped_hamiltonian, order) -> None:
        r"""Add a time evolution block of the higher order (2k) Trotter-Suzuki decomposition 

        Note:
            This is an intrinsic function, user do not need to call this directly
        """
        assert order % 2 == 0
        p_values = self.get_suzuki_p_values(order)
        if order != 4:
            for p in p_values:
                self.__add_higher_order_trotter_block(p * tau, grouped_hamiltonian, order - 2)
        else:
            for p in p_values:
                self.__add_second_order_trotter_block(p * tau, grouped_hamiltonian)

    
    def get_suzuki_p_values(self, k: int) -> List:
        r"""Calculate the parameter p(k) in the Suzuki recurrence relationship.

        Args:
            k: Order of the Suzuki decomposition.

        Returns:
            A list of length five of form [p, p, (1 - 4 * p), p, p].
        """
        p = 1 / (4 - 4 ** (1 / (k - 1)))
        return [p, p, (1 - 4 * p), p, p]


    def add_n_pauli_gate(self, theta: Union[torch.Tensor, float], pauli_word: str, which_qubits: Iterable) -> None:
        r"""Add a rotation gate for a tensor product of Pauli operators, for example :math:`e^{-\theta/2 * X \otimes I \otimes X \otimes Y}`.

        Args:
            theta: Rotation angle.
            pauli_word: Pauli operators in a string format, e.g., ``"XXZ"``.
            which_qubits: List of the index of the qubit that each Pauli operator in the ``pauli_word`` acts on.

        Raises:
            ValueError: The ``which_qubits`` should be either ``list``, ``tuple``, or ``np.ndarray``.
        """
        if isinstance(which_qubits, (Tuple, List)):
            which_qubits = np.array(which_qubits)
        elif not isinstance(which_qubits, np.ndarray):
            raise ValueError('which_qubits should be either a list, tuple or np.ndarray')

        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        # if it is a single-Pauli case, apply the single qubit rotation gate accordingly
        if len(which_qubits) == 1:
            if re.match(r'X', pauli_word[0], flags=re.I):
                self.append(RX(which_qubits[0], param=theta))
            elif re.match(r'Y', pauli_word[0], flags=re.I):
                self.append(RY(which_qubits[0], param=theta))
            elif re.match(r'Z', pauli_word[0], flags=re.I):
                self.append(RZ(which_qubits[0], param=theta))

        else:
            self.__add_n_pauli(which_qubits, pauli_word, theta)

    def __add_n_pauli(self, which_qubits, pauli_word, theta):
        which_qubits.sort()

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                self.append(H([which_qubits[qubit_index]]))
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                self.append(RX(which_qubits[qubit_index], param=torch.pi / 2))

        # Add a Z tensor n rotational gate
        for i in range(len(which_qubits) - 1):
            self.append(CNOT([which_qubits[i], which_qubits[i + 1]]))
        self.append(RZ(which_qubits[-1], param=theta))
        for i in reversed(range(len(which_qubits) - 1)):
            self.append(CNOT([which_qubits[i], which_qubits[i + 1]]))

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                self.append(H([which_qubits[qubit_index]]))
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                self.append(RX(which_qubits[qubit_index], param=-torch.pi / 2))
