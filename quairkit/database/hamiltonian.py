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

__all__ = ["ising_hamiltonian", "xy_hamiltonian", "heisenberg_hamiltonian"]


def ising_hamiltonian(edges: torch.Tensor, vertices: torch.Tensor) -> Hamiltonian:
    r"""Compute the Ising Hamiltonian.

    .. math::

        \begin{align}
            H_{Ising}= \sum_{(u,v) \in E(u>v)}\gamma_{uv}Z_u Z_v + \sum_{k \in V}\beta_k X_k
        \end{align}

    Args:
        edges: A tensor E shape=[V, V], where E[u][v] is \gamma_{uv}.
        vertices: A tensor V shape=[V], where V[k] is \beta_{k}.

    Returns:
        Hamiltonian representation of the Ising Hamiltonian.

    Examples:
        .. code-block:: python

            import torch
            from quairkit.database.hamiltonian import ising_hamiltonian

            edges = torch.tensor([[0, 1, 0.5],
                                  [1, 0, 0.2],
                                  [0.5, 0.2, 0]])
            vertices = torch.tensor([0.3, 0.4, 0.1])
            hamiltonian = ising_hamiltonian(edges, vertices)
            print(f'The Ising_Hamiltonian is \n {hamiltonian}')

        ::

            The Ising_Hamiltonian is 
            1.0 Z0, Z1
            0.5 Z0, Z2
            0.20000000298023224 Z1, Z2
            0.30000001192092896 X0
            0.4000000059604645 X1
            0.10000000149011612 X2
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
    r"""Compute the XY Hamiltonian.

    .. math::

        \begin{align}
            H_{XY}= \sum_{(u,v) \in E(u>v)}(\alpha_{uv}X_u X_v + \beta_{uv}Y_u Y_v)
        \end{align}

    Args:
        edges: A tensor E shape=[2, V, V], where E[0][u][v] is \alpha_{uv} and E[1][u][v] is \beta_{uv}.

    Returns:
        Hamiltonian representation of the XY Hamiltonian.

    Examples:
        .. code-block:: python

            import torch
            from quairkit.database.hamiltonian import xy_hamiltonian

            edges = torch.tensor([
                [
                    [0, 0.7, 0],
                    [0.7, 0, 0.2],
                    [0, 0.2, 0]
                ],
                [
                    [0, 0.5, 0],
                    [0.5, 0, 0.3],
                    [0, 0.3, 0]
                ]
            ])
            H_XY = xy_hamiltonian(edges)
            print(f'The XY Hamiltonian is:\n{H_XY}')

        ::

            The XY Hamiltonian is:
            0.699999988079071 X0, X1
            0.5 Y0, Y1
            0.20000000298023224 X1, X2
            0.30000001192092896 Y1, Y2
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
    r"""Compute the Heisenberg Hamiltonian.

    .. math::

        \begin{align}
            H_{Heisenberg}= \sum_{(u,v) \in E(u>v)}(\alpha_{uv}X_u X_v + \beta_{uv}Y_u Y_v + \gamma_{uv}Z_u Z_v)
        \end{align}

    Args:
        edges: A tensor E shape=[3, V, V], where
               E[0][u][v] is \alpha_{uv},
               E[1][u][v] is \beta_{uv}, and
               E[2][u][v] is \gamma_{uv}.

    Returns:
        Hamiltonian representation of the Heisenberg Hamiltonian.

    Examples:
        .. code-block:: python

            import torch
            from quairkit.database.hamiltonian import heisenberg_hamiltonian

            edges = torch.tensor([
                [
                    [0, 0.5, 0],
                    [0.5, 0, 0.2],
                    [0, 0.2, 0]
                ],
                [
                    [0, 0.3, 0],
                    [0.3, 0, 0.4],
                    [0, 0.4, 0]
                ],
                [
                    [0, 0.7, 0],
                    [0.7, 0, 0.1],
                    [0, 0.1, 0]
                ]
            ])
            H_Heisenberg = heisenberg_hamiltonian(edges)
            print(f'The Heisenberg Hamiltonian is:\n{H_Heisenberg}')

        ::

            The Heisenberg Hamiltonian is:
            0.5 X0, X1
            0.30000001192092896 Y0, Y1
            0.699999988079071 Z0, Z1
            0.20000000298023224 X1, X2
            0.4000000059604645 Y1, Y2
            0.10000000149011612 Z1, Z2
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
                    (gamma, f'Z{i},Z{j}')
                )
            )
    return Hamiltonian(h_list)
