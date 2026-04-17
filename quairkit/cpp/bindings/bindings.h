// Copyright (c) 2026 QuAIR team. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace quairkit_cpp {

void bind_linalg(py::module_ &m);
void bind_matrix(py::module_ &m);
void bind_representation(py::module_ &m);
void bind_qinfo(py::module_ &m);
void bind_state(py::module_ &m);
void bind_counts(py::module_ &m);

}