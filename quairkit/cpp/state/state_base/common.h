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

#include <cstdint>
#include <vector>

namespace quairkit_cpp {

inline int64_t product_int64(const std::vector<int64_t> &xs) {
  int64_t p = 1;
  for (auto v : xs) {
    p *= v;
  }
  return p;
}

inline int64_t normalize_dim(int64_t dim, int64_t rank) {
  if (dim < 0) {
    dim += rank;
  }
  return dim;
}

}