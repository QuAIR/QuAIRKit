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

#include <torch/extension.h>

#include <cstdint>

namespace quairkit_cpp {

inline torch::Tensor trace_two_dims(const torch::Tensor &x, int64_t dim1,
                                    int64_t dim2) {
  auto d1 = dim1 < 0 ? dim1 + x.dim() : dim1;
  auto d2 = dim2 < 0 ? dim2 + x.dim() : dim2;
  auto diag = x.diagonal( 0, d1, d2);
  return diag.sum(-1);
}

}