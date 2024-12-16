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
The library of torch functions used in QuAIRKit.
"""

# -------------------------- Conventions --------------------------
#
# In principle, functions inside `utils` submodule shall be   
#     - protected i.e. the name should start with "_" ;
#     - used for developers only ;
#     - used without calling other functions outside ``utils`` ;
#     - added with "utils.xxx" when used outside ``core`` module ;
#     - related to torch in/outputs only, no numpy in/outputs allowed .
#
# -----------------------------------------------------------------

from . import check, linalg, matrix, qinfo, representation
