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

#include <cstdint>
#include <string>

#include "bindings.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

namespace quairkit_cpp {

namespace {

inline int64_t popcount64(uint64_t x) {
#if defined(_MSC_VER)
  return static_cast<int64_t>(__popcnt64(x));
#else
  return static_cast<int64_t>(__builtin_popcountll(x));
#endif
}

inline void ensure_cpu_contig(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU.");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous.");
}

torch::Tensor dot_prob_grad(const torch::Tensor &offsets,
                            const torch::Tensor &indices,
                            const torch::Tensor &counts,
                            const torch::Tensor &grad,
                            const torch::Tensor &shots) {
  ensure_cpu_contig(offsets, "offsets");
  ensure_cpu_contig(indices, "indices");
  ensure_cpu_contig(counts, "counts");
  ensure_cpu_contig(grad, "grad");
  ensure_cpu_contig(shots, "shots");

  TORCH_CHECK(offsets.scalar_type() == torch::kInt64,
              "offsets must be int64.");
  TORCH_CHECK(grad.dim() == 1, "grad must be 1-D.");
  TORCH_CHECK(shots.numel() == 1 || shots.dim() == 1,
              "shots must be a scalar tensor or 1-D tensor.");
  TORCH_CHECK(shots.scalar_type() == torch::kInt64,
              "shots must be int64 tensor.");

  const int64_t n_circuits = offsets.numel() - 1;
  TORCH_CHECK(n_circuits >= 0, "offsets must have at least 1 element.");
  if (n_circuits == 0) {
    return torch::empty({0}, grad.options());
  }
  if (shots.dim() == 1) {
    TORCH_CHECK(shots.numel() == n_circuits,
                "shots length must equal number of circuits.");
  }

  auto out = torch::zeros({n_circuits}, grad.options());

  const auto *off = offsets.data_ptr<int64_t>();
  const auto *shots_ptr = shots.data_ptr<int64_t>();
  const bool shots_is_scalar = (shots.numel() == 1);

  const int64_t grad_size = grad.size(0);

  const auto idx_type = indices.scalar_type();
  TORCH_CHECK(idx_type == torch::kInt32 || idx_type == torch::kInt64,
              "indices must be int32 or int64.");
  const auto cnt_type = counts.scalar_type();
  TORCH_CHECK(cnt_type == torch::kInt32 || cnt_type == torch::kInt64,
              "counts must be int32 or int64.");

  AT_DISPATCH_FLOATING_TYPES(
      grad.scalar_type(), "quairkit_counts_dot_prob_grad", [&] {
        const scalar_t *g = grad.data_ptr<scalar_t>();
        scalar_t *o = out.data_ptr<scalar_t>();

        auto get_shots = [&](int64_t c) -> int64_t {
          return shots_is_scalar ? shots_ptr[0] : shots_ptr[c];
        };

        auto acc_one = [&](int64_t c, auto get_idx, auto get_cnt) {
          const int64_t start = off[c];
          const int64_t end = off[c + 1];
          int64_t s = get_shots(c);
          if (s == 0) {
            o[c] = static_cast<scalar_t>(0);
            return;
          }
          double acc = 0.0;
          for (int64_t p = start; p < end; ++p) {
            const int64_t idx = static_cast<int64_t>(get_idx(p));
            TORCH_CHECK(idx >= 0 && idx < grad_size,
                        "index out of bounds in counts: ", idx,
                        " (grad size ", grad_size, ")");
            const int64_t cnt = static_cast<int64_t>(get_cnt(p));
            acc += static_cast<double>(cnt) * static_cast<double>(g[idx]);
          }
          o[c] = static_cast<scalar_t>(acc / static_cast<double>(s));
        };

        if (idx_type == torch::kInt64 && cnt_type == torch::kInt64) {
          const int64_t *idx = indices.data_ptr<int64_t>();
          const int64_t *cnt = counts.data_ptr<int64_t>();
          for (int64_t c = 0; c < n_circuits; ++c) {
            acc_one(c, [&](int64_t p) { return idx[p]; },
                    [&](int64_t p) { return cnt[p]; });
          }
        } else if (idx_type == torch::kInt64 && cnt_type == torch::kInt32) {
          const int64_t *idx = indices.data_ptr<int64_t>();
          const int32_t *cnt = counts.data_ptr<int32_t>();
          for (int64_t c = 0; c < n_circuits; ++c) {
            acc_one(c, [&](int64_t p) { return idx[p]; },
                    [&](int64_t p) { return cnt[p]; });
          }
        } else if (idx_type == torch::kInt32 && cnt_type == torch::kInt64) {
          const int32_t *idx = indices.data_ptr<int32_t>();
          const int64_t *cnt = counts.data_ptr<int64_t>();
          for (int64_t c = 0; c < n_circuits; ++c) {
            acc_one(c, [&](int64_t p) { return idx[p]; },
                    [&](int64_t p) { return cnt[p]; });
          }
        } else {
          const int32_t *idx = indices.data_ptr<int32_t>();
          const int32_t *cnt = counts.data_ptr<int32_t>();
          for (int64_t c = 0; c < n_circuits; ++c) {
            acc_one(c, [&](int64_t p) { return idx[p]; },
                    [&](int64_t p) { return cnt[p]; });
          }
        }
      });

  return out;
}

torch::Tensor dot_parity(const torch::Tensor &offsets,
                         const torch::Tensor &indices,
                         const torch::Tensor &counts,
                         const torch::Tensor &masks) {
  ensure_cpu_contig(offsets, "offsets");
  ensure_cpu_contig(indices, "indices");
  ensure_cpu_contig(counts, "counts");
  ensure_cpu_contig(masks, "masks");

  TORCH_CHECK(offsets.scalar_type() == torch::kInt64,
              "offsets must be int64.");
  TORCH_CHECK(masks.scalar_type() == torch::kInt64,
              "masks must be int64.");
  TORCH_CHECK(masks.dim() == 1, "masks must be 1-D.");

  const int64_t n_circuits = offsets.numel() - 1;
  TORCH_CHECK(masks.numel() == n_circuits,
              "masks length must equal number of circuits.");
  if (n_circuits == 0) {
    return torch::empty({0}, masks.options());
  }

  const auto *off = offsets.data_ptr<int64_t>();
  const auto *mask_ptr = masks.data_ptr<int64_t>();

  const auto idx_type = indices.scalar_type();
  TORCH_CHECK(idx_type == torch::kInt32 || idx_type == torch::kInt64,
              "indices must be int32 or int64.");
  const auto cnt_type = counts.scalar_type();
  TORCH_CHECK(cnt_type == torch::kInt32 || cnt_type == torch::kInt64,
              "counts must be int32 or int64.");

  auto out = torch::zeros({n_circuits}, masks.options());
  int64_t *o = out.data_ptr<int64_t>();

  auto acc_one = [&](int64_t c, auto get_idx, auto get_cnt) {
    const int64_t start = off[c];
    const int64_t end = off[c + 1];
    const uint64_t mask = static_cast<uint64_t>(mask_ptr[c]);
    int64_t acc = 0;
    for (int64_t p = start; p < end; ++p) {
      const uint64_t idx = static_cast<uint64_t>(get_idx(p));
      const int64_t cnt = static_cast<int64_t>(get_cnt(p));
      const bool odd = (popcount64(idx & mask) & 1) != 0;
      acc += odd ? -cnt : cnt;
    }
    o[c] = acc;
  };

  if (idx_type == torch::kInt64 && cnt_type == torch::kInt64) {
    const int64_t *idx = indices.data_ptr<int64_t>();
    const int64_t *cnt = counts.data_ptr<int64_t>();
    for (int64_t c = 0; c < n_circuits; ++c) {
      acc_one(c, [&](int64_t p) { return idx[p]; },
              [&](int64_t p) { return cnt[p]; });
    }
  } else if (idx_type == torch::kInt64 && cnt_type == torch::kInt32) {
    const int64_t *idx = indices.data_ptr<int64_t>();
    const int32_t *cnt = counts.data_ptr<int32_t>();
    for (int64_t c = 0; c < n_circuits; ++c) {
      acc_one(c, [&](int64_t p) { return idx[p]; },
              [&](int64_t p) { return cnt[p]; });
    }
  } else if (idx_type == torch::kInt32 && cnt_type == torch::kInt64) {
    const int32_t *idx = indices.data_ptr<int32_t>();
    const int64_t *cnt = counts.data_ptr<int64_t>();
    for (int64_t c = 0; c < n_circuits; ++c) {
      acc_one(c, [&](int64_t p) { return idx[p]; },
              [&](int64_t p) { return cnt[p]; });
    }
  } else {
    const int32_t *idx = indices.data_ptr<int32_t>();
    const int32_t *cnt = counts.data_ptr<int32_t>();
    for (int64_t c = 0; c < n_circuits; ++c) {
      acc_one(c, [&](int64_t p) { return idx[p]; },
              [&](int64_t p) { return cnt[p]; });
    }
  }

  return out;
}

}

void bind_counts(py::module_ &m) {
  auto counts_mod =
      m.def_submodule("counts", "Measurement counts aggregation kernels.");

  counts_mod.def(
      "dot_prob_grad", &dot_prob_grad,
      R"doc(
Compute per-circuit dot products between empirical probabilities and a gradient vector.

This is optimized for shot-based gradients: for each circuit c,

    out[c] = sum_k (counts[k] / shots[c]) * grad[k]

Args:
    offsets: int64 1-D tensor of length n_circuits+1 (CSR offsets).
    indices: int32/int64 1-D tensor of outcome indices (CSR indices).
    counts: int32/int64 1-D tensor of counts aligned with indices.
    grad: 1-D float tensor of length num_outcomes.
    shots: int64 scalar tensor or int64 1-D tensor of length n_circuits.

Returns:
    1-D float tensor of length n_circuits.
)doc",
      py::arg("offsets"), py::arg("indices"), py::arg("counts"),
      py::arg("grad"), py::arg("shots"));

  counts_mod.def(
      "dot_parity", &dot_parity,
      R"doc(
Compute per-circuit sums of counts weighted by a parity sign.

For each circuit c:

    out[c] = sum_k counts[k] * (-1)^(popcount(indices[k] & masks[c]))

This matches the observable expectation after basis change for Pauli strings,
where masks[c] encodes which measured bits contribute a -1 sign when set.

Args:
    offsets: int64 1-D tensor of length n_circuits+1 (CSR offsets).
    indices: int32/int64 1-D tensor of outcome indices (CSR indices).
    counts: int32/int64 1-D tensor of counts aligned with indices.
    masks: int64 1-D tensor of length n_circuits.

Returns:
    1-D int64 tensor of length n_circuits.
)doc",
      py::arg("offsets"), py::arg("indices"), py::arg("counts"),
      py::arg("masks"));
}

}