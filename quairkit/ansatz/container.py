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

r"""
The source file of the Sequential class.
"""

import itertools
import warnings
from typing import Dict, Generator, Iterable, List, Optional, Union

import numpy as np
import torch
from torch.nn.parameter import Parameter

from ..core import get_device, get_dtype, get_float_dtype
from ..core.intrinsic import (_display_png, _format_sequential_idx,
                              _merge_qasm, _replace_indices, _State)
from ..core.latex import OperatorListDrawer
from ..core.operator import Operator, OperatorInfoType
from ..core.state import StateSimulator
from ..database import std_basis, zero_state


class OperatorList(torch.nn.Sequential):
    r"""Sequential container for QuAIRKit operators.

    Args:
        num_systems: number of systems in the operator list.
        system_dim: dimension of systems of this list. Can be a list of system dimensions 
            or an int representing the dimension of all systems.
        physical_idx: physical indices of systems.
    
    """
    def __init__(self, num_systems: Optional[int], 
                 system_dim: Union[List[int], int],
                 physical_idx: Optional[List[int]]) -> None:
        self.__register_list(num_systems, system_dim, physical_idx)

        super().__init__()
        self.dtype = get_dtype()
        self.device = get_device()
    
    def __getitem__(self, key) -> Union[Operator, 'OperatorList']:
        return super().__getitem__(key)
    
    def __register_list(self, num_systems: Optional[int], 
                        system_dim: Union[List[int], int],
                        physical_idx: Optional[List[int]] = None) -> None:
        r"""Register the operator list with input information.
        """
        if isinstance(system_dim, int):
            assert num_systems, \
                ("Since QuAIRKit v0.4.0, system information need to be specified during circuit initialization. " + 
                 "One can change the number of systems by attribute `num_systems`.")
            system_dim = [system_dim] * num_systems
                
        elif num_systems is None:
            num_systems = len(system_dim)
        
        else:
            assert num_systems == len(system_dim), \
                f"num_systems and system_dim do not agree: received {num_systems} and {system_dim}"
        
        self.__system_dim: List[int] = system_dim
        
        if physical_idx is None:
            physical_idx = list(range(num_systems))
        else:
            assert len(set(physical_idx)) == len(physical_idx) == num_systems, \
                f"Duplicate or too less system indices: received {physical_idx}, expected of length {num_systems}"
        self.__system_idx: List[int] = physical_idx # physical index of systems

    @property
    def num_qubits(self) -> int:
        r"""Number of qubits in this circuit.
        """
        return 0 if isinstance(self.__system_dim, int) else self.__system_dim.count(2)
    
    @property
    def num_qutrits(self) -> int:
        r"""Number of qutrits in this circuit.
        """
        return 0 if isinstance(self.__system_dim, int) else self.__system_dim.count(3)

    @property
    def system_dim(self) -> List[int]:
        r"""Dimension of systems in this circuit.
        """
        return self.__system_dim.copy()
    
    @property
    def equal_dim(self) -> bool:
        r"""Whether the systems in this circuit have the same dimension.
        """
        return len(set(self.__system_dim)) == 1
    
    @property
    def system_idx(self) -> List[int]:
        r"""List of physical indices of systems.
        """
        return self.__system_idx.copy()
    
    @system_idx.setter
    def system_idx(self, system_idx: List[int]) -> None:
        r"""Set the physical indices of systems.
        
        Args:
            system_idx: physical indices of systems.
        
        """
        assert (
            len(set(system_idx)) == len(system_idx) == self.num_systems
        ), f"Duplicate or too less system indices: received {system_idx}"

        new_map = {physical: system_idx[logical] for logical, physical in enumerate(self.__system_idx)}
        for op in self.children():
            op.system_idx = _replace_indices(op.system_idx, new_map)
        self.__system_idx = system_idx
    
    @property
    def num_systems(self) -> int:
        r"""Number of logical systems.
        """
        return len(self.__system_idx)
    
    def add_systems(self, num_new_systems: int, 
                    new_system_dim: Optional[Union[int, List[int]]] = None, 
                    new_physical_idx: Optional[List[int]] = None) -> None:
        r"""Add new systems to the list.
        
        Args:
            num_new_systems: number of new systems to be added.
            new_system_dim: dimension of new systems. Defaults to be the same as other systems.
            new_physical_idx: physical indices of new systems. Defaults to start from the largest index.
        """
        if new_system_dim is None:
            assert self.equal_dim, \
                "Need to specify dimensions of new systems when circuit systems have different dimensions"
        elif isinstance(new_system_dim, int):
            new_system_dim = [new_system_dim] * num_new_systems
        else:
            assert len(new_system_dim) == num_new_systems, \
                f"Wrong format of new_system_dim: received {new_system_dim}, expected of length {num_new_systems}"
        
        num_old_systems = self.num_systems
        if new_physical_idx is None:
            new_physical_idx = list(range(num_old_systems, num_old_systems + num_new_systems))
        else:
            assert len(set(new_physical_idx)) == len(new_physical_idx) == num_new_systems, \
                f"Duplicate or too less system indices: received {new_physical_idx}"
            assert set(new_physical_idx).isdisjoint(self.__system_idx), \
                f"Duplicate physical system indices: received {new_physical_idx}, existing {self.__system_idx}"
        
        self.__system_idx = self.__system_idx + new_physical_idx
        self.__system_dim = self.__system_dim + new_system_dim
    
    def sort(self) -> None:
        r"""Sort the systems in the circuit by their physical indices.
        """
        for op in self.children():
            if isinstance(op, OperatorList):
                op.sort()
        
        _system_idx, _system_dim = zip(*sorted(zip(self.system_idx, self.system_dim)))
        self.__system_idx, self.__system_dim = list(_system_idx), list(_system_dim)
    
    @num_systems.setter
    def num_systems(self, num_systems: int) -> None:
        r"""Set the number of logical systems.
        
        Args:
            num_new_systems: number of total systems.
        
        """
        if (old_num_systems := self.num_systems) == num_systems:
            return
        assert num_systems >= old_num_systems, \
            f"Incorrect number of systems: received {num_systems}, expected >= {old_num_systems}"
        
        self.add_systems(num_systems - old_num_systems)

    def register_idx(self, operator_idx: Optional[Union[Iterable[int], int, str]], num_acted_system: Optional[int]) -> List[List[int]]:
        r"""Update sequential according to input operator index information, or report error.

        Args:
            operator_idx: input system indices of the operator. None means acting on all systems. 
                Supports negative indices (counting from the end).
            num_acted_system: number of systems that one operator acts on. None means just check the input.
        
        Returns:
            the formatted system indices.
        
        """
        if operator_idx is None or isinstance(operator_idx, str):
            assert self.equal_dim, \
                f"The sequential's systems have different dimensions. Invalid input qubit idx: {operator_idx}"
        if operator_idx is None:
            return self.system_idx

        num_systems = self.num_systems
        operator_idx = _format_sequential_idx(operator_idx, num_systems, num_acted_system)
        return _replace_indices(operator_idx, self.__system_idx)

    def append(self, op: Union[Operator, 'OperatorList']) -> 'OperatorList':
        r"""Appends an operator or an operator sub-list to the end.

        Args:
            op: module to append
        
        """
        if isinstance(op, (Operator, OperatorList)):
            return super().append(op)
        elif isinstance(op, torch.nn.Module):
            warnings.warn(
                "OperatorList shall only append quantum operator: received normal module", UserWarning)
        else:
            warnings.warn(
                f"Unrecognized input: received {type(op)}", UserWarning)
        return super().append(op)
            
    def __iadd__(self, other: 'OperatorList') -> 'OperatorList':
        return self.extend(other)

    def extend(self, sequential: 'OperatorList') -> 'OperatorList':
        r"""Extend the list with another sequential

        Args:
            sequential: a sequential of operators to be extended

        Returns:
            Concatenation of two quantum operator sequential
        
        """
        new_idx, new_dim = [], []
        for idx, physical_idx in enumerate(sequential.system_idx):
            if physical_idx in self.system_idx:
                physical_dim = sequential.system_dim[idx]
                expected_dim = self.__system_dim[self.system_idx.index(physical_idx)]
                assert physical_dim == expected_dim, \
                    f"Physical system {physical_idx}: received dim {physical_dim}, expected {expected_dim}"
            else:
                new_idx.append(physical_idx)
                new_dim.append(sequential.system_dim[idx])
        
        self.__system_idx = self.__system_idx + new_idx
        self.__system_dim = self.__system_dim + new_dim
        
        return super().extend(sequential)
    
    def operators(self) -> Generator[Operator, None, None]:
        r"""Yield all operators in this list.

        Returns:
            A generator of operators
        
        """
        for op in self:
            if isinstance(op, OperatorList):
                yield from op.operators()
            else:
                yield op

    def to(self, dtype: Optional[str] = None):
        if dtype is not None:
            self.dtype = dtype

        for operator in self:
            operator.to(dtype=self.dtype)

    @property
    def operator_history(self) -> List[Union[OperatorInfoType, List[OperatorInfoType]]]:
        r"""Return the operator history of this Sequential
        """
        operator_history = []
        operator_history.extend(
            op.operator_history if isinstance(op, OperatorList) else op.info
            for op in self.children()
        )
        return operator_history

    @property
    def param(self) -> torch.Tensor:
        r"""Flattened parameters in this list.
        """
        assert self._modules, \
                    "The operator list is empty, please add some operators first."

        list_params = []
        for op in self.children():
            if isinstance(op, OperatorList):
                list_params.append(op.param)
                continue
            list_params.extend([torch.flatten(param.detach()) for param in op.parameters()])
        
        return torch.cat(list_params) if list_params else torch.tensor([])

    @property
    def grad(self) -> torch.Tensor:
        r"""Gradients with respect to the flattened parameters.
        """
        assert self._modules, \
            "The operator list is empty, please add some operators first."
        grad_list = []
        for param in self.parameters():
            assert param.grad is not None, (
                'The gradient is None, run the backward first before calling this property, '
                'otherwise check where the gradient chain is broken.')
            grad_list.append(param.grad.detach().flatten())
        return torch.cat(grad_list) if grad_list else torch.tensor([])

    def update_param(self, theta: Union[torch.Tensor, np.ndarray, float], 
                     idx: Optional[Union[int, None]] = None) -> None:
        r"""Replace parameters of all/one layer(s) by ``theta``.

        Args:
            theta: New parameters
            idx: Index of replacement. Defaults to None, referring to all layers.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        theta = theta.reshape([-1]).to(dtype=get_float_dtype())

        if idx is None:
            assert self.param.shape == theta.shape, \
                f"the shape of input parameters is not correct: expect {self.param.shape}, received {theta.shape}"
            for layer in self.children():
                if isinstance(layer, OperatorList):
                    num_param = int(torch.numel(layer.param))
                    layer.update_param(theta[:num_param], None)
                    continue
                
                for name, param in layer.named_parameters():
                    num_param = int(torch.numel(param))
                    layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))

                    if num_param == theta.shape[0]:
                        return
                    theta = theta[num_param:]
        
        elif isinstance(idx, int):
            layer = next(itertools.islice(self.children(), idx, None))
            if isinstance(layer, OperatorList):
                layer.update_param(theta, None)
                return
            
            assert theta.shape == torch.cat([torch.flatten(param) for param in layer.parameters()]).shape, (
                "The shape of input parameters is not correct.")

            for name, param in layer.named_parameters():
                num_param = int(torch.numel(param))
                layer.register_parameter(name, Parameter(theta[:num_param].reshape(param.shape)))

                if num_param == theta.shape[0]:
                    return
                theta = theta[num_param:]
        else:
            raise ValueError("idx must be an integer or None")

    def transfer_static(self) -> None:
        r"""
        set ``stop_gradient`` of all parameters of the list as ``True``
        """
        for layer in self.children():
            if isinstance(layer, OperatorList):
                layer.transfer_static()
                continue
            
            for name, param in layer.named_parameters():
                param.requires_grad = False
                self.register_parameter(name, param)

    def randomize_param(self, arg0: float = 0, arg1: float = 2 * np.pi, 
                        method: str = 'uniform') -> None:
        r"""Randomize parameters of the list based on the initializer.  
        Current we only support Uniform and Normal initializer. 

        Args:
            arg0: first argument of the initializer. Defaults to 0.
            arg1: first argument of the initializer. Defaults to 2 pi.
            method: The sampling method. Defaults to 'uniform'.
        """
        assert method in {
            "uniform",
            "normal",
        }, "The initializer should be uniform or normal."

        for layer in self.children():
            if isinstance(layer, OperatorList):
                layer.randomize_param(arg0, arg1, method)
                continue
            
            for name, param in layer.named_parameters():

                if method == "normal":
                    new_param = Parameter(
                        torch.normal(
                            mean=arg0,
                            std=arg1,
                            size=param.shape,
                            dtype=param.dtype,
                            device=param.device,
                        )
                    )
                elif method == "uniform":
                    new_param = Parameter(
                        torch.rand(param.shape, dtype=param.dtype, device=param.device)
                        * (arg1 - arg0) + arg0
                    )
                else:
                    raise NotImplementedError
                
                layer.register_parameter(name, new_param)
    
    def __call__(self, state: Optional[_State] = None) -> _State:
        r"""Same as forward of Neural Network
        """
        return self.forward(state)

    def forward(self, state: Optional[_State] = None) -> _State:
        r"""Passing a physical input state.

        Args:
            state: initial state. Defaults to zero state.

        Returns:
            output quantum state
        """
        if state is None:
            assert self.num_systems == max(self.system_idx) + 1, \
                "Cannot provide default input state: dimensions of some physical systems are not specified."
            state = zero_state(self.num_systems, self.system_dim)
        
        if len(self) == 0:
            return state.clone()

        assert self.num_systems <= state.num_systems, \
            f"Insufficient system: received {state.num_systems}, expected >= {self.num_systems}"
        assert self.system_dim == (applied_dim := [state.system_dim[i] for i in self.system_idx]), \
            f"Dimension for systems {self.system_idx} does not agree: received {applied_dim}, expected {self.system_dim}"
        
        state = state.clone()
        if isinstance(state, StateSimulator):
            state = super().forward(state)
        else:
            state.apply(self.operator_history)
        return state
    
    @property
    def matrix(self) -> torch.Tensor:
        r"""Get the unitary matrix form of the operator list.
        """
        dim = int(np.prod(self.system_dim))
        input_basis = std_basis(self.num_systems, self.system_dim)
        input_basis._keep_dim = True
        
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        output = self.forward(input_basis)
        self.system_idx = physical_idx
        
        assert output.backend == 'default-pure', \
            f"The circuit seems to be a noisy circuit: expect 'default-pure', output {output_basis.backend}"
        input_basis = input_basis.bra.view([dim, 1, 1, dim])
        output_basis = output.ket.view([dim, -1, dim, 1])
        return torch.sum(output_basis @ input_basis, dim=0).view(output.batch_dim[1:] + [dim, dim])
    
    @property
    def _system_depth(self) -> Dict[int, int]:
        r"""Depth of each system in the circuit.
        
        Returns:
            depth of each system
        """
        system_depth = {x: 0 for x in self.system_idx}
        for op in self.children():
            if isinstance(op, OperatorList):
                for idx, depth in op._system_depth.items():
                    system_depth[idx] += depth
                continue
            else:
                for system_idx in op.system_idx:
                    for idx in system_idx:
                        system_depth[idx] += 1
        
        return system_depth

    @property
    def depth(self) -> int:
        r"""Depth of gate sequences.
        
        Returns:
            depth of this circuit
        
        Note:
            The measurement is omitted, and all gates are assumed to have depth 1. 
            See Niel's answer in the [StackExchange](https://quantumcomputing.stackexchange.com/a/5772).
        
        """
        return max(self._system_depth.values()) if self._system_depth else 0
    
    def get_qasm(self, transpile: bool) -> str:
        r"""Get the OpenQASM-like string representation of the circuit.

        Args:
            transpile: whether to transpile the circuit to OpenQASM 2.0 format. Defaults to ``True``.

        Returns:
            OpenQASM-like string representation of the circuit.
        """
        qasm_str = ''
        for op in self.children():
            if isinstance(op, OperatorList):
                if qasm_str and qasm_str[-1] != '\n':
                    qasm_str = _merge_qasm(qasm_str, '\n')
                if isinstance(op, Layer):
                    qasm_str = _merge_qasm(qasm_str, '\n// ' + op.get_latex_name('standard'))
                op_qasm = op.get_qasm(transpile) + '\n'
            else:
                op_qasm = _merge_qasm('\n', (op.info.qasm2 if transpile else op.info.qasm))
            qasm_str = _merge_qasm(qasm_str, op_qasm)

        return qasm_str
        
    def __str__(self):
        return self.__repr__()

    def dagger(self) -> None:
        r"""Reverse the entire operator list.

        The dagger is obtained by reversing the order of operators and taking the dagger of each operator.
        """
        i, length = 0, len(self)
        while i < length:
            op = self.pop(i)
            op.dagger()
            self.insert(0, op)
            i += 1

class Layer(OperatorList):
    r"""Base class for built-in trainable quantum circuit ansatz.
    
    Args:
        physical_idx: Physical indices of the systems on which this layer is applied. Supports negative indices (counting from the end).
        depth: Depth of the layer.
        name: Name of the layer. Defaults to 'Layer'.
        system_dim: Dimension of the systems. Defaults to be qubit-systems.
        
    Note:
        A Circuit instance needs to extend this Layer instance to be used in a circuit. 
    
    """
    def __init__(self, physical_idx: List[int],
                 depth: int, name: str = 'Layer',
                 system_dim: Union[int, List[int]] = 2) -> None:
        assert len(physical_idx) > 1, \
            f"Acted systems in a built-in layer needs more than 1: received {len(physical_idx)}."
        
        self.name = name
        self._depth = depth
        super().__init__(len(physical_idx), system_dim, physical_idx)
        
    @property
    def depth(self) -> int:
        r"""Depth of the layer.
        
        Note:
            The depth of the layer is defined as the layer depth.
            It is not the same as the depth of the circuit, which is defined as the maximum depth of all systems.
        """
        return self._depth
    
    def get_latex_name(self, style: str = 'standard') -> str:
        r"""Return the LaTeX name of the layer.
        
        Args:
            style: the style of the plot, can be 'standard', 'compact' or 'detailed'. Defaults to ``standard``.
        """
        depth, name = self._depth, self.name
        if depth > 1:
            if style == 'compact':
                name += r'$\times ' + str(depth) + r'$'
            else:
                name = f"{depth} {name}s"
        return name
        
    def _assert_qubits(self) -> None:
        r"""Assert the input systems are qubits.
        """
        assert self.equal_dim and self.system_dim[0] == 2, \
            f"Incorrect dimension for system {self.system_idx}: received {self.system_dim}, expected all qubits."
    
    def _format_layer_param(self, param: Optional[torch.Tensor], param_shape: List[int]) -> torch.Tensor:
        r"""Format the input parameters for the layer.
        
        Args:
            param: Initial parameters for the layer. Defaults to be self-generated.
            param_shape: Shape of the parameters.
        
        Returns:
            formatted parameters
        """
        if param is None:
            param = torch.rand(param_shape) * 2 * np.pi
        else:
            assert param.numel() == np.prod(param_shape), \
                f"Number of parameters does not match: received {param.shape}, expected size {np.prod(param_shape)}."
        return param
    
    def _get_drawer(self, style: str, decimal: int) -> OperatorListDrawer:
        history = self.operator_history
        if len(history) > 15 and style == 'standard':
            warnings.warn(
                "The circuit may be too large to be readable. " +
                "Will transfer to compact style. You may enforce this by setting style as 'detailed'", UserWarning)
            style = 'compact'
        
        drawer = OperatorListDrawer(style, decimal)
        drawer = drawer.draw_layer(history, self.get_latex_name(style), self._depth)
        return drawer
    
    def plot(self, style: str = 'standard', decimal: int = 2,
             dpi: int = 300, print_code: bool = False, show_plot: bool = True) -> None:
        r"""Plot the circuit layer using LaTeX
        
        Args:
            style: the style of the plot, can be 'standard', 'compact' or 'detailed'. Defaults to ``standard``.
            decimal: number of decimal places to display. Defaults to 2.
            dpi: dots per inches of plot image. Defaults to 300.
            print_code: whether print the LaTeX code of the circuit, default to ``True``.
            show_plot: whether show the plotted circuit, default to ``True``.
        """
        physical_idx, self.system_idx = self.system_idx, list(range(self.num_systems))
        
        drawer = self._get_drawer(style, decimal)
        drawer.fill_all()
        drawer.add_end()
        
        self.system_idx = physical_idx
        
        _fig = drawer.plot(dpi, print_code)
        if show_plot:
            _display_png(_fig)
