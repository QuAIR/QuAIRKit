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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <string>

#include "bindings.h"

namespace py = pybind11;

static std::string version() { return "quairkit._C"; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QuAIRKit PyTorch C++ extension.";

  m.def(
      "version",
      &version,
      R"doc(
Return the build-time identifier of the QuAIRKit C++ extension.

Returns:
    A short string identifying the extension build.
)doc");

  quairkit_cpp::bind_linalg(m);
  quairkit_cpp::bind_matrix(m);
  quairkit_cpp::bind_representation(m);
  quairkit_cpp::bind_qinfo(m);
  quairkit_cpp::bind_state(m);
  quairkit_cpp::bind_counts(m);
}