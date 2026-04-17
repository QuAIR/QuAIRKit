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

#include <vector>

#include "bindings.h"

namespace py = pybind11;
using torch::indexing::Slice;

namespace quairkit_cpp {

namespace {

at::ScalarType complex_dtype_for_prob(const torch::Tensor &t) {
  const auto st = t.scalar_type();
  if (st == at::kComplexFloat || st == at::kComplexDouble) {
    return st;
  }
  if (st == at::kFloat) {
    return at::kComplexFloat;
  }
  if (st == at::kDouble) {
    return at::kComplexDouble;
  }
  return at::kComplexDouble;
}

torch::Tensor bit_flip_kraus(torch::Tensor prob) {
  prob = prob.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(prob);
  auto p = prob.to(cdtype);
  auto z0 = at::zeros_like(p);
  std::vector<torch::Tensor> pieces = {at::sqrt(1 - p), z0, z0, at::sqrt(1 - p),
                                       z0, at::sqrt(p), at::sqrt(p), z0};
  auto out = at::cat(pieces, -1).view({-1, 2, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor phase_flip_kraus(torch::Tensor prob) {
  prob = prob.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(prob);
  auto p = prob.to(cdtype);
  auto z0 = at::zeros_like(p);
  std::vector<torch::Tensor> pieces = {at::sqrt(1 - p), z0, z0, at::sqrt(1 - p),
                                       at::sqrt(p), z0, z0, -at::sqrt(p)};
  auto out = at::cat(pieces, -1).view({-1, 2, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor bit_phase_flip_kraus(torch::Tensor prob) {
  prob = prob.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(prob);
  auto p = prob.to(cdtype);
  auto z0 = at::zeros_like(p);
  auto i = c10::complex<double>(0.0, 1.0);
  std::vector<torch::Tensor> pieces = {at::sqrt(1 - p), z0, z0, at::sqrt(1 - p),
                                       z0, -i * at::sqrt(p), i * at::sqrt(p), z0};
  auto out = at::cat(pieces, -1).view({-1, 2, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor amplitude_damping_kraus(torch::Tensor gamma) {
  gamma = gamma.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(gamma);
  auto g = gamma.to(cdtype);
  auto z0 = at::zeros_like(g);
  auto o1 = at::ones_like(g);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, at::sqrt(1 - g), z0,
                                       at::sqrt(g), z0, z0};
  auto out = at::cat(pieces, -1).view({-1, 2, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor generalized_amplitude_damping_kraus(torch::Tensor gamma,
                                                  torch::Tensor prob) {
  gamma = gamma.view({-1, 1});
  prob = prob.view({-1, 1});
  auto bt = at::broadcast_tensors({gamma, prob});
  gamma = bt[0];
  prob = bt[1];
  const auto cdtype = complex_dtype_for_prob(prob);
  auto g = gamma.to(cdtype);
  auto p = prob.to(cdtype);
  auto z0 = at::zeros_like(p);
  std::vector<torch::Tensor> pieces = {
      at::sqrt(p), z0, z0, at::sqrt(p) * at::sqrt(1 - g),
      z0, at::sqrt(p) * at::sqrt(g), z0, z0,
      at::sqrt(1 - p) * at::sqrt(1 - g), z0, z0, at::sqrt(1 - p),
      z0, z0, at::sqrt(1 - p) * at::sqrt(g), z0};
  auto out = at::cat(pieces, -1).view({-1, 4, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor phase_damping_kraus(torch::Tensor gamma) {
  gamma = gamma.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(gamma);
  auto g = gamma.to(cdtype);
  auto z0 = at::zeros_like(g);
  auto o1 = at::ones_like(g);
  std::vector<torch::Tensor> pieces = {o1, z0, z0, at::sqrt(1 - g), z0, z0,
                                       z0, at::sqrt(g)};
  auto out = at::cat(pieces, -1).view({-1, 2, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor depolarizing_kraus(torch::Tensor prob) {
  prob = prob.view({-1, 1});
  const auto cdtype = complex_dtype_for_prob(prob);
  auto p = prob.to(cdtype);
  auto z0 = at::zeros_like(p);
  auto i = c10::complex<double>(0.0, 1.0);
  std::vector<torch::Tensor> pieces = {
      at::sqrt(1 - 3 * p / 4), z0, z0, at::sqrt(1 - 3 * p / 4),
      z0, at::sqrt(p / 4), at::sqrt(p / 4), z0,
      z0, -i * at::sqrt(p / 4), i * at::sqrt(p / 4), z0,
      at::sqrt(p / 4), z0, z0, -at::sqrt(p / 4)};
  auto out = at::cat(pieces, -1).view({-1, 4, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor pauli_kraus(torch::Tensor prob) {
  if (prob.dim() == 1) {
    prob = prob.unsqueeze(0);
  }
  auto prob_sum = prob.sum(-1, true);
  auto prob_i = 1 - prob_sum;
  auto prob_x = prob.index({Slice(), Slice(0, 1)}).contiguous();
  auto prob_y = prob.index({Slice(), Slice(1, 2)}).contiguous();
  auto prob_z = prob.index({Slice(), Slice(2, 3)}).contiguous();
  const auto cdtype = complex_dtype_for_prob(prob);
  auto pi = prob_i.to(cdtype);
  auto px = prob_x.to(cdtype);
  auto pyv = prob_y.to(cdtype);
  auto pz = prob_z.to(cdtype);
  auto z0 = at::zeros_like(pi);
  auto i = c10::complex<double>(0.0, 1.0);
  std::vector<torch::Tensor> pieces = {at::sqrt(pi), z0, z0, at::sqrt(pi),
                                       z0, at::sqrt(px), at::sqrt(px), z0,
                                       z0, -i * at::sqrt(pyv), i * at::sqrt(pyv), z0,
                                       at::sqrt(pz), z0, z0, -at::sqrt(pz)};
  auto out = at::cat(pieces, -1).view({-1, 4, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

torch::Tensor reset_kraus(torch::Tensor prob) {
  if (prob.dim() == 1) {
    prob = prob.unsqueeze(0);
  }
  auto prob_sum = prob.sum(-1, true);
  auto prob_i = 1 - prob_sum;
  auto prob_0 = prob.index({Slice(), Slice(0, 1)}).contiguous();
  auto prob_1 = prob.index({Slice(), Slice(1, 2)}).contiguous();
  const auto cdtype = complex_dtype_for_prob(prob);
  auto p0 = prob_0.to(cdtype);
  auto p1 = prob_1.to(cdtype);
  auto pi = prob_i.to(cdtype);
  auto z0 = at::zeros_like(pi);
  std::vector<torch::Tensor> pieces = {at::sqrt(p0), z0, z0, z0,
                                       z0, at::sqrt(p0), z0, z0,
                                       z0, z0, at::sqrt(p1), z0,
                                       z0, z0, z0, at::sqrt(p1),
                                       at::sqrt(pi), z0, z0, at::sqrt(pi)};
  auto out = at::cat(pieces, -1).view({-1, 5, 2, 2});
  if (out.size(0) == 1) {
    out = out.squeeze(0);
  }
  return out;
}

}

void bind_representation(py::module_ &m) {
  auto repr_mod =
      m.def_submodule("representation", "Channel representations (Kraus, etc).");
  repr_mod.def(
      "bit_flip_kraus", &bit_flip_kraus,
      R"doc(
Construct Kraus operators for the bit-flip channel.

Args:
    prob: Flip probability. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 2, 2, 2) or (2, 2, 2) for scalar prob.
)doc",
      py::arg("prob"));

  repr_mod.def(
      "phase_flip_kraus", &phase_flip_kraus,
      R"doc(
Construct Kraus operators for the phase-flip channel.

Args:
    prob: Flip probability. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 2, 2, 2) or (2, 2, 2) for scalar prob.
)doc",
      py::arg("prob"));

  repr_mod.def(
      "bit_phase_flip_kraus", &bit_phase_flip_kraus,
      R"doc(
Construct Kraus operators for the bit-phase-flip channel.

Args:
    prob: Flip probability. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 2, 2, 2) or (2, 2, 2) for scalar prob.
)doc",
      py::arg("prob"));

  repr_mod.def(
      "amplitude_damping_kraus", &amplitude_damping_kraus,
      R"doc(
Construct Kraus operators for the amplitude damping channel.

Args:
    gamma: Damping rate. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 2, 2, 2) or (2, 2, 2) for scalar gamma.
)doc",
      py::arg("gamma"));

  repr_mod.def(
      "generalized_amplitude_damping_kraus", &generalized_amplitude_damping_kraus,
      R"doc(
Construct Kraus operators for the generalized amplitude damping channel.

Args:
    gamma: Damping rate. Supports broadcasting.
    prob: Excited-state population probability. Supports broadcasting with gamma.

Returns:
    Kraus operators with shape (..., 4, 2, 2) or (4, 2, 2) for scalar inputs.
)doc",
      py::arg("gamma"), py::arg("prob"));

  repr_mod.def(
      "phase_damping_kraus", &phase_damping_kraus,
      R"doc(
Construct Kraus operators for the phase damping channel.

Args:
    gamma: Damping rate. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 2, 2, 2) or (2, 2, 2) for scalar gamma.
)doc",
      py::arg("gamma"));

  repr_mod.def(
      "depolarizing_kraus", &depolarizing_kraus,
      R"doc(
Construct Kraus operators for the depolarizing channel.

Args:
    prob: Depolarizing probability. Supports broadcasting.

Returns:
    Kraus operators with shape (..., 4, 2, 2) or (4, 2, 2) for scalar prob.
)doc",
      py::arg("prob"));

  repr_mod.def(
      "pauli_kraus", &pauli_kraus,
      R"doc(
Construct Kraus operators for a Pauli channel.

Args:
    prob: Probabilities for X/Y/Z with shape (..., 3) or (3,). The identity probability is inferred as 1 - sum(prob).

Returns:
    Kraus operators with shape (..., 4, 2, 2) or (4, 2, 2) for scalar inputs.
)doc",
      py::arg("prob"));

  repr_mod.def(
      "reset_kraus", &reset_kraus,
      R"doc(
Construct Kraus operators for a reset channel (reset to |0> or |1>).

Args:
    prob: Probabilities for reset-to-|0> and reset-to-|1> with shape (..., 2) or (2,).
        The identity probability is inferred as 1 - sum(prob).

Returns:
    Kraus operators with shape (..., 5, 2, 2) or (5, 2, 2) for scalar inputs.
)doc",
      py::arg("prob"));
}

}