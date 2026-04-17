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
#include <stdexcept>
#include <tuple>
#include <vector>

#include "common.h"
#include "tensor_utils.h"

namespace quairkit_cpp {

inline torch::Tensor dagger(const torch::Tensor &mat) {
  return mat.transpose(-2, -1).conj().contiguous();
}

inline torch::Tensor pure_evolve(const torch::Tensor &data,
                                 const torch::Tensor &unitary, int64_t dim,
                                 int64_t applied_dim, int64_t prob_dim,
                                 bool on_batch) {
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, applied_dim, dim / applied_dim});
  return at::matmul(u, d).view({-1, dim});
}

inline torch::Tensor pure_evolve_keep_dim(const torch::Tensor &data,
                                          const torch::Tensor &unitary,
                                          int64_t dim, int64_t applied_dim,
                                          int64_t prob_dim, bool on_batch) {
  const auto num_unitary = (unitary.dim() >= 3) ? unitary.size(-3) : 1;
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, num_unitary, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, num_unitary, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, 1, applied_dim, dim / applied_dim});
  return at::matmul(u, d).view({-1, dim});
}

inline torch::Tensor mixed_evolve(const torch::Tensor &data,
                                  const torch::Tensor &unitary, int64_t dim,
                                  int64_t applied_dim, int64_t prob_dim,
                                  bool on_batch) {
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d = data.view({-1, prob_dim, applied_dim, (dim * dim) / applied_dim});
  d = at::matmul(u, d);
  d = d.view({-1, prob_dim, dim, applied_dim, dim / applied_dim});
  auto u2 = u.unsqueeze(-3).conj();
  return at::matmul(u2, d).view({-1, dim, dim});
}

inline torch::Tensor mixed_evolve_keep_dim(const torch::Tensor &data,
                                           const torch::Tensor &unitary,
                                           int64_t dim, int64_t applied_dim,
                                           int64_t prob_dim, bool on_batch) {
  const auto num_unitary = (unitary.dim() >= 3) ? unitary.size(-3) : 1;
  std::vector<int64_t> u_shape;
  if (on_batch) {
    u_shape = {-1, 1, num_unitary, applied_dim, applied_dim};
  } else {
    u_shape = {1, -1, num_unitary, applied_dim, applied_dim};
  }
  auto u = unitary.view(u_shape);
  auto d =
      data.view({-1, prob_dim, 1, applied_dim, (dim * dim) / applied_dim});
  d = at::matmul(u, d);
  d = d.view({-1, prob_dim, num_unitary, dim, applied_dim, dim / applied_dim});
  auto u2 = u.unsqueeze(-3).conj();
  d = at::matmul(u2, d);
  return d.view({-1, dim, dim});
}

inline std::tuple<torch::Tensor, torch::Tensor>
pure_measure_collapse(const torch::Tensor &data, int64_t dim) {
  auto d = data.view({-1, dim, 1});
  auto prob = at::real(at::matmul(d.transpose(-2, -1).conj(), d));
  prob = prob.view({-1, 1, 1});
  auto mask = at::abs(prob) >= 1e-10;
  auto denom = at::sqrt(at::where(mask, prob, at::ones_like(prob)));
  auto collapsed = d / denom;
  collapsed = collapsed * mask.to(collapsed.scalar_type());
  collapsed = at::where(at::isnan(collapsed), at::zeros_like(collapsed),
                        collapsed);
  return {prob, collapsed.view({-1, dim})};
}

inline std::tuple<torch::Tensor, torch::Tensor>
mixed_measure_collapse(const torch::Tensor &data, int64_t dim) {
  auto diag = data.diagonal( 0, -2, -1);
  auto prob = at::real(diag.sum(-1)).view({-1, 1, 1});
  auto mask = at::abs(prob) >= 1e-10;
  auto denom = at::where(mask, prob, at::ones_like(prob));
  auto collapsed = data / denom;
  collapsed = collapsed * mask.to(collapsed.scalar_type());
  collapsed = at::where(at::isnan(collapsed), at::zeros_like(collapsed),
                        collapsed);
  return {prob, collapsed.view({-1, dim, dim})};
}

inline torch::Tensor pure_expec_val(const torch::Tensor &data,
                                    const torch::Tensor &obs, int64_t dim,
                                    int64_t applied_dim) {
  const auto num_obs = obs.size(-3);
  auto state = data.view({-1, 1, applied_dim, dim / applied_dim});
  auto measured_state = at::matmul(obs, state).view({-1, num_obs, dim});
  auto out = at::sum(data.unsqueeze(-2).conj() * measured_state, -1);
  return out;
}

inline torch::Tensor mixed_expec_val(const torch::Tensor &data,
                                     const torch::Tensor &obs, int64_t dim,
                                     int64_t applied_dim) {
  auto state = data.view({-1, 1, applied_dim, (dim * dim) / applied_dim});
  auto tmp = at::matmul(obs, state).view({-1, dim, dim});
  auto tr = tmp.diagonal(0, -2, -1).sum(-1);
  return tr;
}

inline torch::Tensor mixed_kraus_transform(const torch::Tensor &data,
                                          const torch::Tensor &kraus,
                                          int64_t dim, int64_t applied_dim,
                                          int64_t prob_dim, bool on_batch) {
  auto evolved =
      mixed_evolve_keep_dim(data, kraus, dim, applied_dim, prob_dim, on_batch);
  const auto rank = kraus.size(-3);
  return evolved.view({-1, rank, dim, dim}).sum(-3);
}

inline torch::Tensor ptrace_1(const torch::Tensor &xy, const torch::Tensor &x) {
  const auto dim_x = x.size(-1);
  const auto dim_xy = xy.size(-1);
  const auto dim_y = dim_xy / dim_x;

  auto xy_sizes = xy.sizes().vec();
  if (xy_sizes.size() < 2) {
    throw std::runtime_error("xy must have at least 2 dimensions.");
  }
  std::vector<int64_t> batch_dims(xy_sizes.begin(), xy_sizes.end() - 2);
  std::vector<int64_t> xy_view_shape = batch_dims;
  xy_view_shape.push_back(-1);
  xy_view_shape.push_back(dim_x);
  xy_view_shape.push_back(dim_y);

  auto xy_view = xy.view(xy_view_shape);
  auto x_view = x.view({-1, dim_x, 1});
  auto y = (xy_view * x_view.conj()).sum(-2);
  return y.squeeze(-2);
}

inline std::string create_state_type(const torch::Tensor &data) {
  auto sizes = data.sizes();
  const auto ndim = static_cast<int64_t>(sizes.size());
  if (ndim == 1) {
    return "pure";
  }
  if (ndim == 2) {
    return (sizes[0] == sizes[1]) ? "mixed" : "pure";
  }
  if (sizes[ndim - 1] == 1) {
    return "pure";
  }
  if (sizes[ndim - 1] == sizes[ndim - 2]) {
    return "mixed";
  }
  throw std::runtime_error("Input data shape does not match default backend.");
}

inline torch::Tensor make_controlled_unitary(const torch::Tensor &unitary,
                                             int64_t ctrl_dim,
                                             int64_t index) {
  auto opts =
      unitary.options().dtype(unitary.scalar_type()).device(unitary.device());
  auto proj = torch::zeros({ctrl_dim, ctrl_dim}, opts);
  proj.index_put_({index, index}, 1);
  auto eye_c = torch::eye(ctrl_dim, opts);
  auto eye_u = torch::eye(unitary.size(-1), opts);

  auto u1 = at::kron(proj, unitary);
  auto u2 = at::kron(eye_c - proj, eye_u);
  return u1 + u2;
}

inline torch::Tensor mixed_choi_transform(const torch::Tensor &data,
                                         const torch::Tensor &choi,
                                         int64_t dim_refer, int64_t dim_in,
                                         int64_t dim_out, int64_t prob_dim,
                                         bool on_batch) {
  const auto dim_in_total = dim_refer * dim_in;
  auto d = data.view({-1, dim_in_total, dim_in_total});
  d = d.view({-1, prob_dim, dim_refer, dim_in, dim_refer, dim_in});

  std::vector<int64_t> choi_shape;
  if (on_batch) {
    choi_shape = {-1, 1, dim_in, dim_in * (dim_out * dim_out)};
  } else {
    choi_shape = {1, -1, dim_in, dim_in * (dim_out * dim_out)};
  }
  auto c = choi.view(choi_shape);

  d = d.transpose(-1, -3)
          .reshape({-1, prob_dim, (dim_refer * dim_refer) * dim_in, dim_in});
  d = at::matmul(d, c);

  d = d.view({-1, prob_dim, dim_in, dim_refer, dim_out, dim_in, dim_out});
  d = d.transpose(-3, -4);
  d = trace_two_dims(d, -2, -5);
  return d.view({-1, dim_refer * dim_out, dim_refer * dim_out});
}

inline torch::Tensor mixed_trace_1(const torch::Tensor &data, int64_t trace_dim) {
  const auto dim = data.size(-1);
  const auto rest = dim / trace_dim;
  auto d = data.view({-1, trace_dim, rest, trace_dim, rest});
  d = trace_two_dims(d, 1, 3);
  return d.view({-1, rest, rest});
}

inline torch::Tensor mixed_reset_1(const torch::Tensor &data,
                                   const torch::Tensor &replace_dm,
                                   int64_t trace_dim) {
  auto traced = mixed_trace_1(data, trace_dim);
  return at::kron(traced, replace_dm);
}

}