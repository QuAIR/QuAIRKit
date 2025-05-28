# !/usr/bin/env python3
# Copyright (c) 2023 Paddle Quantum Authors. All Rights Reserved.
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
The module of the hamiltonian class.
"""

import copy
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse

from . import get_device, get_dtype


class Hamiltonian:
    r"""Hamiltonian ``class`` in QuAIRKit.

    User can instantiate the ``class`` with a Pauli string.

    Args:
        pauli_str: A list of Hamiltonian information, e.g. ``[(1, 'Z0, Z1'), (2, 'I')]``
        compress: Determines whether the input list will be automatically merged (e.g. ``(1, 'Z0, Z1')`` and ``(2, 'Z1, Z0')``, these two items will be automatically merged).
         Defaults to ``True``.

    Returns:
        Create a Hamiltonian class

    Note:
        If ``compress=False``, the legitimacy of the input will not be checked.
    """

    def __init__(self, pauli_str: list, compress: Optional[bool] = True):
        self.__coefficients = None
        self.__terms = None
        self.__pauli_words_r = []
        self.__pauli_words = []
        self.__sites = []
        self.__nqubits = None
        # when internally updating the __pauli_str, be sure to set __update_flag to True
        self.__pauli_str = pauli_str
        self.__update_flag = True
        self.__decompose()
        if compress:
            self.__compress()

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(self.n_terms)[indices])
        elif isinstance(indices, tuple):
            indices = list(indices)

        new_pauli_str = [[self.coefficients[index], ','.join(self.terms[index])] for index in indices]

        return Hamiltonian(new_pauli_str, compress=False)

    def __add__(self, h_2):
        new_pauli_str = self.pauli_str.copy()
        if isinstance(h_2, (float, int)):
            new_pauli_str.extend([[float(h_2), 'I']])
        else:
            new_pauli_str.extend(h_2.pauli_str)
        return Hamiltonian(new_pauli_str)

    def __mul__(self, other):
        new_pauli_str = copy.deepcopy(self.pauli_str)
        for i in range(len(new_pauli_str)):
            new_pauli_str[i][0] *= other
        return Hamiltonian(new_pauli_str, compress=False)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __str__(self):
        str_out = ''
        for idx in range(self.n_terms):
            str_out += f'{self.coefficients[idx]} '
            for _ in range(len(self.terms[idx])):
                str_out += self.terms[idx][_]
                if _ != len(self.terms[idx]) - 1:
                    str_out += ', '
            if idx != self.n_terms - 1:
                str_out += '\n'
        return str_out

    def __repr__(self):
        return 'quairkit.Hamiltonian object: \n' + self.__str__()

    @property
    def n_terms(self) -> int:
        r"""Number of terms of the hamiltonian.
        """
        return len(self.__pauli_str)

    @property
    def pauli_str(self) -> List:
        r"""The Pauli string corresponding to the hamiltonian.
        """
        return self.__pauli_str

    @property
    def terms(self) -> Iterable[float]:
        r"""All items in hamiltonian, i.e. ``[['Z0, Z1'], ['I']]``.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__terms

    @property
    def coefficients(self) -> List[float]:
        r"""The coefficient of each term in the Hamiltonian, i.e. ``[1.0, 2.0]``.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__coefficients

    @property
    def pauli_words(self) -> List[str]:
        r"""The Pauli word of each term, i.e. ``['ZIZ', 'IIX']``.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__pauli_words

    @property
    def pauli_words_r(self) -> List[str]:
        r"""A list of Pauli word (exclude I), i.e. ``['ZXZZ', 'Z', 'X']``.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__pauli_words_r

    @property
    def pauli_words_matrix(self) -> List[torch.Tensor]:
        r"""The list of matrices with respect to simplied Pauli words.
        """

        def to_matrix(string):
            matrix = torch.tensor([1 + 0j], dtype=get_dtype())
            for i in range(len(string)):
                if string[i] == 'X':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[0, 1], [1, 0]], dtype=get_dtype())
                    )
                elif string[i] == 'Y':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[0j, -1j], [1j, 0j]], dtype=get_dtype())
                    )
                elif string[i] == 'Z':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[1, 0], [0, -1]], dtype=get_dtype())
                    )
                elif string[i] == 'I':
                    matrix = torch.kron(
                        matrix,
                        torch.tensor([[1, 0], [0, 1]], dtype=get_dtype())
                    )
                else:
                    raise ValueError(f'wrong format of string {string}')
            return matrix

        return list(map(to_matrix, copy.deepcopy(self.pauli_words_r)))
    
    @property
    def matrix(self) -> torch.Tensor:
        r"""The matrix form of the Hamiltonian.
        """
        return torch.from_numpy(self.construct_h_matrix()).to(device=get_device(), dtype=get_dtype())

    @property
    def sites(self) -> List:
        r"""A list of qubits index corresponding to the hamiltonian.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__sites

    @property
    def n_qubits(self) -> int:
        r"""Number of qubits.
        """
        if self.__update_flag:
            self.__decompose()
        return self.__nqubits

    def __decompose(self):
        r"""decompose the Hamiltonian into vairious forms

        Raises:
            Exception: Operators should be defined with a string composed of Pauli operators followed by qubit index on which it act, separated with ",". i.e. "Z0, X1"

        Note:
            This is an intrinsic function, user do not need to call this directly
            This is a fundamental function, it decomposes the input Pauli string into different forms and stores them into private variables.
        """
        self.__pauli_words = []
        self.__pauli_words_r = []
        self.__sites = []
        self.__terms = []
        self.__coefficients = []
        self.__nqubits = 1
        new_pauli_str = []
        for coefficient, pauli_term in self.__pauli_str:
            pauli_word_r = ''
            site = []
            single_pauli_terms = re.split(r',\s*', pauli_term.upper())
            self.__coefficients.append(float(coefficient))
            self.__terms.append(single_pauli_terms)
            for single_pauli_term in single_pauli_terms:
                match_i = re.match(r'I', single_pauli_term, flags=re.I)
                if match_i:
                    assert single_pauli_term[0].upper() == 'I', \
                            'The offset is defined with a sole letter "I", i.e. (3.0, "I")'
                    pauli_word_r += 'I'
                    site.append('')
                else:
                    match = re.match(r'([XYZ])([0-9]+)', single_pauli_term, flags=re.I)
                    if not match:
                        raise Exception(
                            'Operators should be defined with a string composed of Pauli operators followed' +
                            'by qubit index on which it act, separated with ",". i.e. "Z0, X1"')
                    pauli_word_r += match[1].upper()
                    assert int(match[2]) not in site, 'each Pauli operator should act on different qubit'

                    site.append(int(match[2]))
                    self.__nqubits = max(self.__nqubits, int(match[2]) + 1)
            self.__pauli_words_r.append(pauli_word_r)
            self.__sites.append(site)
            new_pauli_str.append([float(coefficient), pauli_term.upper()])

        for term_index in range(len(self.__pauli_str)):
            pauli_word = ['I' for _ in range(self.__nqubits)]
            site = self.__sites[term_index]
            for site_index in range(len(site)):
                if type(site[site_index]) == int:
                    pauli_word[site[site_index]] = self.__pauli_words_r[term_index][site_index]
            self.__pauli_words.append(''.join(pauli_word))
            self.__pauli_str = new_pauli_str
            self.__update_flag = False

    def __compress(self):
        r"""combine like terms

        Note:
            This is an intrinsic function, user do not need to call this directly
        """
        if self.__update_flag:
            self.__decompose()
        new_pauli_str = []
        flag_merged = [False for _ in range(self.n_terms)]
        for term_idx_1 in range(self.n_terms):
            if not flag_merged[term_idx_1]:
                for term_idx_2 in range(term_idx_1 + 1, self.n_terms):
                    if not flag_merged[term_idx_2] and self.pauli_words[term_idx_1] == self.pauli_words[term_idx_2]:
                        self.__coefficients[term_idx_1] += self.__coefficients[term_idx_2]
                        flag_merged[term_idx_2] = True
                if self.__coefficients[term_idx_1] != 0:
                    new_pauli_str.append([self.__coefficients[term_idx_1], ','.join(self.__terms[term_idx_1])])
        self.__pauli_str = new_pauli_str
        self.__update_flag = True

    def decompose_with_sites(self) -> Tuple[list]:
        r"""Decompose pauli_str into coefficients, a simplified form of Pauli strings, and the indices of qubits on which the Pauli operators act on.

        Returns:
            A tuple containing the following elements:
                - coefficients: the coefficient for each term.
                - pauli_words_r: the simplified form of the Pauli string for each item, e.g. the Pauli word of 'Z0, Z1, X3' is 'ZZX'. 
                - sites: a list of qubits index, e.g. the site of 'Z0, Z1, X3' is [0, 1, 3].
        """
        if self.__update_flag:
            self.__decompose()
        return self.coefficients, self.__pauli_words_r, self.__sites

    def decompose_pauli_words(self) -> Tuple[list]:
        r"""Decompose pauli_str into coefficients and Pauli strings.

        Returns:
            A tuple containing the following elements:
                - coefficients: the coefficient for each term.
                - the Pauli string for each item, e.g. the Pauli word of 'Z0, Z1, X3' is 'ZZIX'.
        """
        if self.__update_flag:
            self.__decompose()
        return self.coefficients, self.__pauli_words

    def construct_h_matrix(self, qubit_num: Optional[int] = None) -> np.ndarray:
        r"""Construct a matrix form of the Hamiltonian in Z-basis.

        Args:
            qubit_num: The number of qubits. Defaults to ``None``.

        Returns:
            The matrix form of the Hamiltonian in Z-basis.
        """
        coefs, pauli_words, sites = self.decompose_with_sites()
        if qubit_num is None:
            qubit_num = 1
            for site in sites:
                if type(site[0]) is int:
                    qubit_num = max(qubit_num, max(site) + 1)
        else:
            assert qubit_num >= self.n_qubits, "the input number of qubits must be no less than the number of qubits of this Hamiltonian"
        n_qubit = qubit_num
        h_matrix = np.zeros([2 ** n_qubit, 2 ** n_qubit])
        spin_ops = SpinOps(n_qubit, use_sparse=True)
        for idx in range(len(coefs)):
            op = coefs[idx] * sparse.eye(2 ** n_qubit)
            for site_idx in range(len(sites[idx])):
                if re.match(r'X', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigx_p[sites[idx][site_idx]])
                elif re.match(r'Y', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigy_p[sites[idx][site_idx]])
                elif re.match(r'Z', pauli_words[idx][site_idx], re.I):
                    op = op.dot(spin_ops.sigz_p[sites[idx][site_idx]])
            h_matrix += op
        return h_matrix


class SpinOps:
    r"""The spin operators in matrix forms, could be used to construct Hamiltonian matrix or spin observables.
    
    Args:
        size: Size of the system (number of qubits).
        use_sparse: Decide whether to use the sparse matrix to calculate. Default is ``False``.
    """

    def __init__(self, size: int, use_sparse: Optional[bool] = False):
        self.size = size
        self.id = sparse.eye(2)
        self.__sigz = sparse.bsr_matrix([[1, 0], [0, -1]])
        self.__sigy = sparse.bsr_matrix([[0, -1j], [1j, 0]])
        self.__sigx = sparse.bsr_matrix([[0, 1], [1, 0]])
        self.__sigz_p = []
        self.__sigy_p = []
        self.__sigx_p = []
        self.__sparse = use_sparse
        for i in range(self.size):
            self.__sigz_p.append(self.__direct_prod_op(spin_op=self.__sigz, spin_index=i))
            self.__sigy_p.append(self.__direct_prod_op(spin_op=self.__sigy, spin_index=i))
            self.__sigx_p.append(self.__direct_prod_op(spin_op=self.__sigx, spin_index=i))

    @property
    def sigz_p(self) -> List:
        r""" A list of :math:`S^z_i` operators, different elements correspond to different indices :math:`i`.
        """
        return self.__sigz_p

    @property
    def sigy_p(self) -> List:
        r""" A list of :math:`S^y_i` operators, different elements correspond to different indices :math:`i`.
        """
        return self.__sigy_p

    @property
    def sigx_p(self) -> List:
        r""" A list of :math:`S^x_i` operators, different elements correspond to different indices :math:`i`.
        """
        return self.__sigx_p

    def __direct_prod_op(self, spin_op, spin_index):
        r"""get spin operators on n-th spin (qubit) with direct product

        Args:
            spin_op: single body spin operator
            spin_index: on which spin (qubit)

        Returns:
            scipy.sparse or np.ndarray: spin operator with direct product form. The data type is specified by self.__use_sparse
        """
        s_p = copy.copy(spin_op)
        for i in range(self.size):
            if i < spin_index:
                s_p = sparse.kron(self.id, s_p)
            elif i > spin_index:
                s_p = sparse.kron(s_p, self.id)
        if self.__sparse:
            return s_p
        else:
            return s_p.toarray()


def _pauli_str_to_matrix(pauli_str: list, n: int) -> torch.Tensor:
    r"""Convert the input list form of an observable to its matrix form.

    For example, if the input ``pauli_str`` is ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` and ``n=3``,
    then this function returns the observable :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`
    in matrix form.

    Args:
        pauli_str: A list form of an observable.
        n: Number of qubits.

    Raises:
        ValueError: Only Pauli operator "I" can be accepted without specifying its position.

    Returns:
        The matrix form of the input observable.
    """
    pauli_dict = {'i': np.eye(2) + 0j, 'x': np.array([[0, 1], [1, 0]]) + 0j,
                  'y': np.array([[0, -1j], [1j, 0]]), 'z': np.array([[1, 0], [0, -1]]) + 0j}

    # Parse pauli_str; 'x0,z1,y4' to 'xziiy'
    new_pauli_str = []
    for coeff, op_str in pauli_str:
        init = list('i' * n)
        op_list = re.split(r',\s*', op_str.lower())
        for op in op_list:
            if len(op) > 1:
                pos = int(op[1:])
                assert pos < n, 'n is too small'
                init[pos] = op[0]
            elif op.lower() != 'i':
                raise ValueError('Only Pauli operator "I" can be accepted without specifying its position')
        new_pauli_str.append([coeff, ''.join(init)])

    # Convert new_pauli_str to matrix; 'xziiy' to NKron(x, z, i, i, y)
    matrices = []
    for coeff, op_str in new_pauli_str:
        sub_matrices = [pauli_dict[op.lower()] for op in op_str]
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            mat = sub_matrices[0]
            for idx in range(1, len(sub_matrices)):
                mat = np.kron(mat, sub_matrices[idx])
            matrices.append(coeff * mat)

    return torch.tensor(sum(matrices), dtype=get_dtype())
