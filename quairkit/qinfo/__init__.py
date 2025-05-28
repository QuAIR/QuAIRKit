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
The library of functions in quantum information theory and quantum computing.
"""

from .check import *
from .linalg import *
from .qinfo import *

from .check import __all__ as check_fcn
from .linalg import __all__ as linalg_fcn
from .qinfo import __all__ as qinfo_fcn

__all__ = check_fcn + linalg_fcn + qinfo_fcn

