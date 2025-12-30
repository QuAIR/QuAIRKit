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
Source file for transforming OperatorInfo to string representing quantikz.

Quantikz is a LaTeX package for typesetting quantum circuit diagrams.
See https://ctan.org/pkg/quantikz for more details.
"""

import datetime
import os
import subprocess
import uuid
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import IPython
import numpy as np
import torch
from pdf2image import convert_from_path

from .operator.base import OperatorInfoType

__all__ = ['code_to_str', 'OperatorListDrawer']


def _format_line(list_s: List[str], begin_code: str, next_line: bool = True) -> str:
    r"""Format a list of strings into a LaTeX line, connected by & (and \\ if contain next line)
    """
    begin_code = f"{begin_code} & " if begin_code else "& "
    return begin_code + " & ".join(list_s) + (r" \\" +  "\n" if next_line else "")

def code_to_str(code: Dict[int, List[str]], 
               begin_code: Optional[Dict[int, str]] = None) -> str:
    r"""Translate the given quantikz code into complete LaTeX commands
    
    Args:
        code: The quantikz code to translate, where the key i is the i-th line of the code, 
                and the value is the list of gate commands on that line.
        begin_code: a dictionary or a string representing the beginning code for each line. 
                Defaults to empty.
        
    Returns:
        The complete LaTeX code for the given quantikz code, connected by & and \\
    
    """
    system_idx = sorted(code.keys())
    assert len(system_idx) == system_idx[-1] + 1, \
        f"The system indices must be consecutive and start from 0: received {system_idx}"
    
    begin_code = {} if begin_code is None else begin_code
    last_line = _format_line(code[system_idx[-1]], begin_code.get(system_idx[-1], ''), next_line=False)
    other_lines = ''.join(_format_line(code[idx], begin_code.get(idx, '')) for idx in system_idx[:-1])
    return ''.join(other_lines) + last_line


def _format_number(num: Union[int, float, complex], decimals: int, simplify_zero: bool = False) -> str:
    if simplify_zero and np.abs(num) < (10 ** -decimals):
        return '0'
    return f"{num:.{decimals}f}".replace('(', '').replace(')', '').replace('j', 'i')

def _format_vector(data: np.ndarray, decimals: int) -> str:
    present_data = data[:3]
    row_str = [_format_number(x, decimals) for x in present_data]
    if len(data) > 3:
        row_str.append(r'\ldots')
    return r', '.join(row_str)

def _format_matrix(data: np.ndarray, decimals: int) -> str:
    assert len(data.shape) == 2, \
        f"Only 2D arrays are supported: received {data.shape}"
    
    text = r'\begin{bmatrix}' + '\n'
    for x in range(len(data)):
        row_str = [_format_number(data[x, y], decimals, simplify_zero=True) for y in range(len(data[x]))]
        text += r' & '.join(row_str)
        if x != len(data) - 1:
            text += r' \\' + '\n'
    text += '\n' + r'\end{bmatrix}'
    return text

def _format_data(data: torch.Tensor, decimals: int) -> str:
    r"""Format the given data tensor into flatten LaTex numbers or LaTeX matrix
    
    Args:
        data: The data tensor to format
        decimals: The number of decimals to show
        to_matrix: Whether to format the data into a 2D matrix
    
    """
    data = np.round(data.squeeze().detach().cpu().numpy(), decimals=decimals)
    
    if len(data.shape) == 0:
        return _format_number(data, decimals)
    
    elif len(data.shape) == 1:
        return _format_vector(data, decimals)
    
    elif len(data.shape) == 2:
        return _format_matrix(data, decimals)
    
    raise ValueError(
        f"Only 1D and 2D tensors are supported: received {data.shape}")


def _permute_str(perm: List[int], system_idx: List[int]) -> Tuple[str, Tuple[List[int], List[int]]]:
    r"""Give the quantikz permutation of the systems
    
    Args:
        perm: permutation
        system_idx: systems to permute
    
    """
    assert len(perm) == len(system_idx), \
        f"Lengths of permutation and system_idx mismatch: received {perm} and {system_idx}"
    min_idx = min(system_idx)
    perm_map = {system_idx[i]: system_idx[perm[i]] for i in range(len(perm))}
    
    for idx in range(min_idx, max(system_idx) + 1):
        perm_map.setdefault(idx, idx)
    target_param = [str(perm_map[idx] - min_idx + 1) for idx in sorted(perm_map)]
    return fr"\permute{{{','.join(target_param)}}}"


def __clear_temp_files(file_name: str) -> None:
    for ext in [".aux", ".log", ".pdf", ".tex", ".png"]:
        temp_file = f"{file_name}{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)

def _plot_code(latex_code: str, dpi: int) -> IPython.display.Image:
    r"""Plot the given LaTeX code using local LaTeX distribution
    """
    # Write the LaTeX code to a file
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    uid = uuid.uuid4()
    file_name = f"quairkit{date}temp_{uid}"
    with open(f"{file_name}.tex", "w") as f:
        f.write(latex_code)

    # Run pdflatex to compile the TeX file
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{file_name}.tex"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0:
        log_path = f"{file_name}.log"
        log_output = ""
        if os.path.exists(log_path):
            with open(log_path, "r") as log_file:
                log_output = log_file.read()
            print("LaTeX compilation failed. Log output:")
            print(log_output)
        else:
            print("Compilation failed, and no log file was generated.")
            
        __clear_temp_files(file_name)
        raise RuntimeError("LaTeX compilation failed. See log output for details.")

    # Convert the resulting PDF to an image using pdf2image
    pdf_path = f"{file_name}.pdf"
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        images[0].save(f"{file_name}.png", "PNG")
    except Exception as e:
        __clear_temp_files(file_name)
        raise RuntimeError(f"Failed to convert PDF to PNG: {e}") from e

    obj = IPython.display.Image(filename=f"{file_name}.png")
    __clear_temp_files(file_name)
    return obj


class OperatorListDrawer:
    r"""Latex plot for a list of operators in QuAIRKit
    
    Args:
        style: the plot style of the circuit, can be 'standard', 'compact' or 'detailed'.
    
    Note:
        The LaTeX package used is [quantikz](https://ctan.org/pkg/quantikz).
    
    """
    def __init__(self, style: str, decimals: int) -> None:
        style = style.lower()
        if style not in ['standard', 'compact', 'detailed']:
            raise ValueError(
                f"The style must be 'standard', 'compact' or 'detailed': received {style}."
            )
        
        self.style = style
        self.decimals = decimals
        self._code: Dict[int, List[str]] = {}
        
    def _fill_empty(self, system_idx: List[int], classical_idx: Optional[List[int]] = None) -> None:
        r"""Fill empty &s for incoming multi-system operators
        """
        min_idx, max_idx = min(system_idx), max(system_idx)
        if min_idx == max_idx:
            self._code.setdefault(min_idx, [])
            return
        
        lengths = {}
        for idx in range(min_idx, max_idx + 1):
            current_str = self._code.setdefault(idx, [])
            lengths[idx] = len(current_str)
        max_len = max(lengths.values()) + 1
        
        # Pad each line up to max_len
        classical_idx = classical_idx if classical_idx is not None else []
        for idx, old_len in lengths.items():
            pad_amount = max_len - (old_len + 1)
            
            if idx in classical_idx:
                self._code[idx] += [r"\wireoverride{c}"] * pad_amount
            else:
                self._code[idx] += [r"{}"] * pad_amount
            
    def fill_all(self) -> None:
        r"""Fill empty &s for all lines covered by the current codes
        """
        min_idx, max_idx = min(self._code.keys()), max(self._code.keys())
        self._fill_empty(list(range(min_idx, max_idx + 1)))
            
    def clone(self) -> 'OperatorListDrawer':
        r"""Clone the current drawer
        """
        new_drawer = OperatorListDrawer(self.style, self.decimals)
        new_drawer._code = deepcopy(self._code)
        return new_drawer
        
    @property
    def code(self) -> Dict[int, List[str]]:
        r"""The dictionary of the current circuit code
        
        Note:
            Empty lines are filled with '{}'
        """
        if not list(self._code.keys()):
            return {}
        self.fill_all()
        return deepcopy(self._code)
    
    def add_end(self, rstick_str: Optional[Dict[int, str]] = None) -> None:
        r"""Add end sticks to the circuit
        
        Args:
            rstick_str: The string to add to the right stick at idx, such as \ket{0}
        
        """
        rstick_str = {} if rstick_str is None else rstick_str
        for idx in self._code:
            self._code[idx].append(r"\rstick{" + rstick_str.get(idx, '') +  r'}')
    
    __plot_begin_code = r'''\documentclass[border=2pt]{standalone}

% WARNING: please disable other quantum circuit packages, such as qcircuit
\usepackage{tikz}
\usetikzlibrary{quantikz2}

\begin{document}

\begin{quantikz}[transparent]
'''
    
    __end_code = r'''
\end{quantikz}

\end{document}
'''
    
    def plot(self, dpi: int = 300, print_code: bool = True, 
             begin_code: Optional[Dict[int, str]] = None) -> IPython.display.Image:
        r"""Plot the given quantikz code using LaTeX
        
        Args:
            dpi: The DPI to use for the resulting image. Defaults to 300.
            print_code: Whether to print the code to the console. Defaults to True.
            begin_code: a dictionary or a string representing the beginning code for each line. 
                Defaults to empty.
            
        Returns:
            The image object of the plot
        
        """
        if begin_code is None:
            begin_code = {idx: r'\lstick{}' for idx in self._code.keys()}
        latex_code = self.__plot_begin_code + code_to_str(self.code, begin_code) + self.__end_code
        
        if print_code:
            print(latex_code)
        return _plot_code(latex_code, dpi)
    
    def _catenate(self, drawer: 'OperatorListDrawer') -> 'OperatorListDrawer':
        r"""Catenate the given drawers
        
        Args:
            *drawers: The drawers to catenate
        
        """
        if not list(drawer._code.keys()):
            return self.clone()
        if not list(self._code.keys()):
            return drawer.clone()
        
        new_drawer = self.clone()
        
        if list(set(new_drawer._code.keys()) & set(drawer._code.keys())) :
            new_drawer._fill_empty(list(drawer._code.keys()))
            
            for idx, code in drawer._code.items():
                new_drawer._code[idx] = new_drawer._code.get(idx, []) + code
        else:
            for idx, code in drawer._code.items():
                new_drawer._code[idx] = code
        return new_drawer
    
    def __add__(self, drawer: 'OperatorListDrawer') -> 'OperatorListDrawer':
        return self._catenate(drawer)
    
    def _add_ctrl(self, system_idx: List[int], num_ctrl_system: int, control_label: str, classical: bool = False) -> Tuple[int, int]:
        r"""Add control lines to the given system index
        
        Returns:
            The minimum and maximum system index the input operator applies
        
        """
        if num_ctrl_system == 0:
            return min(system_idx), max(system_idx)
        assert control_label and len(control_label) == num_ctrl_system, \
                f"Control label must be provided for each control system: received {control_label} with {num_ctrl_system} control systems"
        ctrl_system_idx, apply_system_idx = system_idx[:num_ctrl_system], system_idx[num_ctrl_system:]

        min_apply_idx, max_apply_idx = min(apply_system_idx), max(apply_system_idx)
        min_ctrl_idx, max_ctrl_idx = min(ctrl_system_idx), max(ctrl_system_idx)

        if min_ctrl_idx > max_apply_idx:
            start_idx, end_idx = max_ctrl_idx, max_apply_idx
            min_idx, max_idx = max_apply_idx + 1, max_ctrl_idx
        elif max_ctrl_idx < min_apply_idx:
            start_idx, end_idx = min_ctrl_idx, min_apply_idx
            min_idx, max_idx = min_ctrl_idx, min_apply_idx - 1
        else:
            raise ValueError(
                f"Control systems and apply systems must be separated: received {ctrl_system_idx} controlling {apply_system_idx}")
        across_system = end_idx - start_idx

        classical_prefix = r"vertical wire=c" if classical else ''

        for idx in range(min_idx, max_idx + 1):
            if idx == start_idx: # the first control system
                label = control_label[ctrl_system_idx.index(idx)]
                if label == '0':
                    self._code[idx].append(r"\octrl[" + classical_prefix + r"]{" + str(across_system) + r'}')
                elif label == '1':
                    self._code[idx].append(r"\ctrl[" + classical_prefix + r"]{" + str(across_system) + r'}')
                else:
                    self._code[idx].append(r"\ctrl[style={fill=white,draw=black,inner sep=0.3pt}," + classical_prefix + 
                                           r"]{" + str(across_system) + r"} \push{\scriptscriptstyle " + label + r"}")
            elif idx in ctrl_system_idx:
                label = control_label[ctrl_system_idx.index(idx)]
                if label == '0':
                    self._code[idx].append(r"\control[style={fill=white,draw=black,inner sep=1pt}]{}")
                elif label == '1':
                    self._code[idx].append(r"\control{}")
                else:
                    self._code[idx].append(r"\control[style={fill=white,draw=black,inner sep=0.3pt}]{} \push{\scriptscriptstyle " + 
                                           label + r"}")
            else:
                self._code[idx].append(r"\push{\,\,}")

        return min_apply_idx, max_apply_idx
    
    __empty_through = (r'\gateinput[label style={xshift=-8.2pt}]{$\textcolor{white}{\rule{4pt}{4pt}}$}' + 
                       r'\gateoutput[label style={xshift=8.2pt}]{$\textcolor{white}{\rule{4pt}{4pt}}$}')
        
    def _append_general_op(self, info: OperatorInfoType) -> None:
        r"""Append a general fixed operator
        
        Note:
            Presentation rule is as follows:
            - If the style is 'detailed', the matrix is provided and not controlled, show the matrix;
            - Otherwise, just show the name.
        
        """
        info.setdefault("tex", fr"\operatorname{{{info['name']}}}")
        num_ctrl_system, control_label = info.get("num_ctrl_system", 0), info.get("label", None)
        show_matrix = (
            self.style == "detailed"
            and ("matrix" in info and len(info["matrix"].shape) <= 2 and info["matrix"].shape[-1] <= 10)
            and num_ctrl_system == 0
        )
        _empty_through = r"{}" if "permute" in info else self.__empty_through
        
        for idx, system_idx in enumerate(info["system_idx"]):
            self._fill_empty(system_idx)
            min_idx, max_idx = self._add_ctrl(system_idx, num_ctrl_system, control_label)
            num_across_system = max_idx - min_idx + 1
            
            if show_matrix:
                matrix = info["matrix"]
                if len(matrix.shape) > 2:
                    matrix = matrix[idx]
                gate_name = _format_data(matrix, self.decimals)
            elif "permute" in info:
                gate_name = _permute_str(info["permute"], system_idx[num_ctrl_system:])
            else:
                gate_name = info["tex"]
            
            if gate_name == r"\targ{}":
                gate_command = gate_name
            else:
                gate_prefix = str(num_across_system)
                if "permute" in info:
                    style_str = "draw=gray, dashed" if num_ctrl_system != 0 else "draw=none"
                    gate_prefix += r',style={' + style_str + r'}'
                gate_command = fr"\gate[{gate_prefix}]{{{gate_name}}}"
            
            self._code[min_idx].append(gate_command)
            for idx in range(min_idx + 1, max_idx + 1):
                self._code[idx].append(r"{}" if idx in system_idx else _empty_through)
    
    def _append_general_param_op(self, info: OperatorInfoType) -> None:
        r"""Append a general parameterized operator
        
        Note:
            Presentation rule is as follows:
            - If the style is 'detailed', the matrix is provided, 
            the number of acted parameters is greater than 3 and not controlled, show the matrix;
            - If the style is 'compact', just show the name;
            - Otherwise, show the name with parameters.
        """
        assert "param" in info, \
            "The parameterized operator must have a parameter"
        if "param_sharing" not in info:
            warnings.warn(
                f"info for param_sharing of operator {info['name']} is not provided, set to False", UserWarning)
            info["param_sharing"] = False
        
        if "tex" not in info:
            info["tex"] = rf"\operatorname{{{info['name']}}}"
        num_ctrl_system, control_label = info.get("num_ctrl_system", 0), info.get("label", None)
        
        if self.style == 'compact' or info['name'] == "universal":
            show_matrix = False
            show_param = False
        elif (self.style == 'detailed' and "matrix" in info and 
            info['param'].shape[-1] > 3 and num_ctrl_system == 0):
            show_matrix = True
            show_param = False
        else:
            show_matrix = False
            show_param = True
        
        for op_idx, system_idx in enumerate(info["system_idx"]):
            self._fill_empty(system_idx)
            min_idx, max_idx = self._add_ctrl(system_idx, num_ctrl_system, control_label)
            num_across_system = max_idx - min_idx + 1
            
            if show_matrix:
                gate_name = _format_data(info['matrix'][op_idx], self.decimals)
            elif show_param:
                param = info['param'][0, 0] if info['param_sharing'] else info['param'][op_idx, 0]
                param_str = _format_data(param, self.decimals)
                gate_name = info['tex'] + '(' + param_str + ')'
            else:
                gate_name = info['tex']
            
            gate_prefix = str(num_across_system)
            if show_param and self.style == 'detailed':
                param_str = _format_data((param / np.pi) % 2, self.decimals)
                param_str = ', '.join([rf"{num}\pi" for num in param_str.split(', ')]).replace(r'\ldots\pi', r'\ldots')
                gate_prefix += r',label style={label={[gray]below:$\scriptscriptstyle ' + param_str + r' $}}'
            gate_command = r"\gate[" + gate_prefix + r']{' +  gate_name + r'}'
            
            self._code[min_idx].append(gate_command)
            for idx in range(min_idx + 1, max_idx + 1):
                self._code[idx].append(r"{}" if idx in system_idx else self.__empty_through)
    
    def _append_measure_op(self, info: OperatorInfoType) -> None:
        r"""Append a measurement operator
        """
        system_idx = info["system_idx"][0]
        self._fill_empty(system_idx)
        
        if (collapse_label := info.get("label", None)):
            for i, idx in enumerate(system_idx):
                self._code[idx].append(r"\measuretab{" + collapse_label[i] + r"}")
                self._code[idx].append(r"\wireoverride{n}")
        else:
            min_idx, max_idx = min(system_idx), max(system_idx)
            num_across_system = max_idx - min_idx + 1
            measure_command = r"\meter[" + str(num_across_system) + r"]{" + str(info.get('label', '')) + "}"
            self._code[min_idx].append(measure_command)
            for idx in range(min_idx + 1, max_idx + 1):
                self._code[idx].append(r"{}" if idx in system_idx else self.__empty_through)
                
    def _append_locc_op(self, info: OperatorInfoType) -> None:
        r"""Append a local operation
        """
        system_idx, num_measure_system = info["system_idx"][0], info["num_ctrl_system"]
        measure_idx = system_idx[:num_measure_system]
        self._fill_empty(measure_idx)
        
        label = info["label"]
        if len(measure_idx) == 1:
            self._code[measure_idx[0]].append(r"\meter{}")
            self._code[measure_idx[0]].append(r"\push{" + label + r"} \wireoverride{c}")
        else:
            for i, idx in enumerate(measure_idx):
                self._code[idx].append(r"\meter{}")
                self._code[idx].append(r"\push{{" + label + r"}_" + str(i) + r"} \wireoverride{c}")
        self._fill_empty(system_idx, classical_idx=measure_idx)
        
        min_idx, max_idx = self._add_ctrl(system_idx, num_measure_system, ['1'] * num_measure_system, classical=True)
        for idx in measure_idx:
            self._code[idx][-1] += r"\wireoverride{c}"
        num_across_system = max_idx - min_idx + 1
        gate_name = r'{' + info["tex"] + r"}^{(" + label + r")}"
        gate_command = r"\gate[" + str(num_across_system) + r']{' + gate_name + r'}'
        self._code[min_idx].append(gate_command)
        for idx in range(min_idx + 1, max_idx + 1):
            self._code[idx].append(r"{}" if idx in system_idx else self.__empty_through)
            
    def _append_reset_op(self, info: OperatorInfoType) -> None:
        reset_idx, state_label = info["system_idx"][0], info["tex"]
        self._fill_empty(reset_idx)
        
        for idx in reset_idx:
            self._code[idx].append(r"\ground{}")
            self._code[idx].append(r"\wireoverride{n} \push{" + state_label + r"\,\,}")
                
    def append(self, info: OperatorInfoType) -> None:
        r"""Append an operator to the current circuit code
        
        Args:
            info: The information of the operator to append
        
        """
        if info['name'] == "measure":
            self._append_measure_op(info)
        elif info['name'] == "locc":
            self._append_locc_op(info)
        elif info['name'] == "reset":
            self._append_reset_op(info)
        elif "param" in info:
            self._append_general_param_op(info)
        else:
            self._append_general_op(info)

    def draw_layer(self, list_info: List[OperatorInfoType], 
                   name: str, depth: int = 1) -> 'OperatorListDrawer':
        r"""Give the drawer for a layer, i.e., a list of operators
        
        Args:
            list_info: The list of operators to draw
            name: The name of the layer.
            depth: The depth of the layer. Defaults to 1.
        
        Note:
            if name is given, the layer name will be added to the code by
            - command '\gate' with layer name if the style is 'compact';
            - command '\gate' with matrix if the style is 'detailed' and matrix is given.
            - command '\gategroup' with name above if the style is 'standard';
        
        """
        assert depth > 0, \
                f"Layer must have non-zero depth: received {depth}"
        num_op_per_layer, num_remain_layer = divmod(len(list_info), depth)
        
        layer_style = self.style

        drawer = OperatorListDrawer(layer_style, self.decimals)

        if layer_style == 'compact':
            list_info, list_end_info = list_info[:num_op_per_layer], list_info[-num_remain_layer:]
        for info in list_info:
            drawer.append(info)
        min_idx, max_idx = min(drawer._code.keys()), max(drawer._code.keys())
        label_pos = 'above' if min_idx == 0 else 'below'
        max_len = max((len(lst) for lst in drawer._code.values()), default=0)
        
        drawer._code[min_idx][0] += (r'\gategroup[' + str(len(drawer._code)) + 
                                     r',steps=' + str(max_len) + 
                                     r',style={inner sep=4pt,dashed,label={' + label_pos + 
                                     r':{' + name + r'}}}]{}')
        drawer.fill_all()
        
        if layer_style == 'compact' and num_remain_layer > 0:
            for info in list_end_info:
                drawer.append(info)
            drawer.fill_all()
        return drawer
