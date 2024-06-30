# !/usr/bin/env python3
# Copyright (c) 2023 QuAIR team. All Rights Reserved.
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
The library of common Hamiltonians.
"""

import torch

from ..core import Hamiltonian


def ising_hamiltonian(edges: torch.Tensor, vertices: torch.Tensor) -> Hamiltonian:
    r"""Compute the Ising Hamiltonian

    .. math::

        \begin{align}
            H_{Ising}= \sum_{(u,v) \in E(u>v)}\gamma_{uv}Z_u Z_v + \sum_{k \in V}\beta_k X_k
        \end{align}

    Args:
            edges: A tensor E shape=[V, V], where E[u][v] is \gamma_{uv}.
            vertices: A tensor E shape=[V], where V[k] is \beta_{k}.

    Returns:
        H_{Ising}
    """
    h_list = []
    shape_of_edges = edges.shape

    assert len(shape_of_edges) == 2, \
    f'The input variable edges should be a 2-dimension torch.tensor, but receive {len(shape_of_edges)}-dimension'

    edges_param_list = edges.tolist()
    vertices_param_list = vertices.tolist()

    for i in range(shape_of_edges[0]):
        for j in range(i + 1, shape_of_edges[1]):
            para = edges_param_list[i][j]
            if para == 0:
                continue
            h_list.append((para, f'Z{i},Z{j}'))

    for i in range(vertices.shape[0]):
        para = vertices_param_list[i]
        if para == 0:
            continue
        h_list.append((para, f'X{i}'))

    return Hamiltonian(h_list)


def xy_hamiltonian(edges: torch.Tensor) -> Hamiltonian:
    r"""Compute the Ising Hamiltonian

    .. math::

        \begin{align}
            H_{XY}= \sum_{(u,v) \in E(u>v)}(\alpha_{uv}X_u X_v + \beta_{uv}Y_u Y_v)
        \end{align}

    Args:
            edges: A tensor E shape=[2, V, V], where E[0][u][v] is \alpha_{uv} and E[1][u][v] is \beta_{uv}.

    Returns:
        H_{XY}
    """
    h_list = []
    shape_of_edges = edges.shape

    assert len(shape_of_edges) == 3, \
    f'The input variable edges should be a 3-dimension torch.tensor, but receive {len(shape_of_edges)}-dimension'

    edges_param_list = edges.tolist()

    for i in range(shape_of_edges[1]):
        for j in range(i + 1, shape_of_edges[1]):
            alpha = edges_param_list[0][i][j]
            beta = edges_param_list[1][i][j]
            if alpha == 0 and beta == 0:
                continue
            h_list.extend(((alpha, f'X{i},X{j}'), (beta, f'Y{i},Y{j}')))
    return Hamiltonian(h_list)


def heisenberg_hamiltonian(edges: torch.Tensor) -> Hamiltonian:
    r"""Compute the Heisenberg Hamiltonian

    .. math::

        \begin{align}
            H_{Heisenberg}= \sum_{(u,v) \in E(u>v)}(\alpha_{uv}X_u X_v + \beta_{uv}Y_u Y_v, + \gamma_{uv}Z_u Z_v)
        \end{align}

    Args:
            edges: A tensor E shape=[3, V, V], where E[0][u][v] is \alpha_{uv}, E[1][u][v] is \beta_{uv} and E[2][u][v] is \gamma_{uv}.

    Returns:
        H_{Heisenberg}
    """
    h_list = []
    shape_of_edges = edges.shape

    assert len(shape_of_edges) == 3, \
    f'The input variable edges should be a 3-dimension torch.tensor, but receive {len(shape_of_edges)}-dimension'

    edges_param_list = edges.tolist()

    for i in range(shape_of_edges[1]):
        for j in range(i + 1, shape_of_edges[1]):
            alpha = edges_param_list[0][i][j]
            beta = edges_param_list[1][i][j]
            gamma = edges_param_list[2][i][j]
            if alpha == 0 and beta == 0 and gamma == 0:
                continue
            h_list.extend(
                (
                    (alpha, f'X{i},X{j}'),
                    (beta, f'Y{i},Y{j}'),
                    (gamma, f'Z{i},Z{j}'),
                )
            )
    return Hamiltonian(h_list)
