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

#include <cmath>
#include <vector>

#include "bindings.h"

namespace py = pybind11;

namespace quairkit_cpp {

namespace {

torch::Tensor linalg_dagger(const torch::Tensor &mat) {
  return mat.transpose(-2, -1).conj().contiguous();
}

torch::Tensor linalg_trace(const torch::Tensor &mat, int64_t axis1 = -2,
                           int64_t axis2 = -1) {
  auto diag = mat.diagonal( 0, axis1, axis2);
  return diag.sum(-1);
}

torch::Tensor linalg_sqrtm(const torch::Tensor &a) {
  auto eig = at::linalg_eigh(a);
  auto w = std::get<0>(eig);
  auto v = std::get<1>(eig);
  w = at::clamp(w, 1e-20);
  auto sqrt_w = at::sqrt(w);
  return at::matmul(v * sqrt_w.unsqueeze(-2), linalg_dagger(v));
}

torch::Tensor kraus_to_choi(const torch::Tensor &kraus) {
  auto dims = kraus.sizes().vec();
  if (dims.size() < 3) {
    throw std::runtime_error("kraus_to_choi: expected shape [..., r, d, d]");
  }
  const auto d_in = kraus.size(-1);
  const auto d_out = kraus.size(-2);
  const auto r = kraus.size(-3);
  const auto d2 = d_out * d_in;

  auto flat = kraus;
  const bool has_batch = dims.size() > 3;
  if (has_batch) {
    flat = flat.reshape({-1, r, d_out, d_in});
  } else {
    flat = flat.unsqueeze(0);
  }

  auto vec = flat.transpose(-2, -1).contiguous().reshape({-1, r, d2});
  auto choi = at::matmul(vec.transpose(1, 2), vec.conj());

  if (has_batch) {
    std::vector<int64_t> out_shape(dims.begin(), dims.end() - 3);
    out_shape.push_back(d2);
    out_shape.push_back(d2);
    return choi.view(out_shape);
  }
  return choi.squeeze(0);
}

torch::Tensor choi_to_kraus(const torch::Tensor &choi, double tol) {
  auto dims = choi.sizes().vec();
  if (dims.size() < 2) {
    throw std::runtime_error("choi_to_kraus: expected shape [..., d2, d2]");
  }
  const auto d2 = choi.size(-1);
  if (choi.size(-2) != d2) {
    throw std::runtime_error("choi_to_kraus: expected a square Choi matrix.");
  }
  const auto d = static_cast<int64_t>(std::sqrt(static_cast<double>(d2)));
  if (d * d != d2) {
    throw std::runtime_error("choi_to_kraus: expected last dims to be d^2 x d^2.");
  }
  if (tol < 0 || tol >= 1) {
    throw std::runtime_error("choi_to_kraus: tol must be in [0, 1).");
  }

  auto flat = choi;
  const bool has_batch = dims.size() > 2;
  if (has_batch) {
    flat = flat.reshape({-1, d2, d2});
  } else {
    flat = flat.unsqueeze(0);
  }

  auto eig = at::linalg_eigh(flat);
  auto w = std::get<0>(eig);
  auto v = std::get<1>(eig);
  auto wabs = at::abs(w);
  auto total = wabs.sum(-1, true);
  total = at::where(total == 0, at::ones_like(total), total);

  auto rev = at::flip(wabs, {-1});
  auto csum = at::cumsum(rev, -1);
  auto ratio = csum / total;

  std::vector<int64_t> keep_counts(static_cast<size_t>(ratio.size(0)), d2);
  int64_t r_max = 1;
  const double thr = 1.0 - tol;
  for (int64_t b = 0; b < ratio.size(0); ++b) {
    auto row = ratio[b];
    int64_t keep = d2;
    for (int64_t i = 0; i < d2; ++i) {
      if (row[i].item<double>() > thr) {
        keep = i + 1;
        break;
      }
    }
    if (keep < 1) {
      keep = 1;
    }
    keep_counts[static_cast<size_t>(b)] = keep;
    r_max = std::max(r_max, keep);
  }

  auto out = at::zeros({ratio.size(0), r_max, d, d}, choi.options());
  auto w_pos = at::clamp(w, 0);
  auto sqrt_w = at::sqrt(w_pos);

  for (int64_t b = 0; b < ratio.size(0); ++b) {
    const auto keep = keep_counts[static_cast<size_t>(b)];
    const auto start = d2 - keep;
    auto vb = v[b];
    auto wb = sqrt_w[b];
    for (int64_t k = 0; k < keep; ++k) {
      const auto col = start + k;
      auto vec_k = vb.index({torch::indexing::Slice(), col}) * wb[col];
      auto K = vec_k.view({d, d}).transpose(-2, -1).contiguous();
      out.index_put_({b, k}, K);
    }
  }

  if (has_batch) {
    std::vector<int64_t> out_shape(dims.begin(), dims.end() - 2);
    out_shape.push_back(r_max);
    out_shape.push_back(d);
    out_shape.push_back(d);
    return out.view(out_shape);
  }
  return out.squeeze(0);
}

torch::Tensor kraus_to_stinespring(const torch::Tensor &kraus) {
  const auto din = kraus.size(-1);
  const auto dout = kraus.size(-2);
  const auto r = kraus.size(-3);
  auto flat = kraus;
  const bool has_batch = kraus.dim() > 3;
  if (has_batch) {
    flat = flat.reshape({-1, r, dout, din});
  } else {
    flat = flat.unsqueeze(0);
  }

  auto out = flat.permute({0, 2, 1, 3}).contiguous().view({-1, r * dout, din});
  if (has_batch) {
    std::vector<int64_t> out_shape;
    auto dims = kraus.sizes().vec();
    out_shape.insert(out_shape.end(), dims.begin(), dims.end() - 3);
    out_shape.push_back(r * dout);
    out_shape.push_back(din);
    return out.view(out_shape);
  }
  return out.squeeze(0);
}

torch::Tensor stinespring_to_kraus(const torch::Tensor &st) {
  const auto din = st.size(-1);
  const auto d_out = din;
  const auto rd_out = st.size(-2);
  if (rd_out % d_out != 0) {
    throw std::runtime_error("stinespring_to_kraus: stinespring[-2] must be divisible by d_out (= d_in).");
  }
  const auto r = rd_out / d_out;

  auto dims = st.sizes().vec();
  const bool has_batch = st.dim() > 2;
  if (has_batch) {
    auto flat = st.reshape({-1, d_out, r, din}).permute({0, 2, 1, 3}).contiguous();
    std::vector<int64_t> out_shape(dims.begin(), dims.end() - 2);
    out_shape.push_back(r);
    out_shape.push_back(d_out);
    out_shape.push_back(din);
    return flat.view(out_shape);
  }

  return st.view({d_out, r, din}).permute({1, 0, 2}).contiguous();
}

torch::Tensor stinespring_to_choi(const torch::Tensor &st) {
  auto kraus = stinespring_to_kraus(st);
  return kraus_to_choi(kraus);
}

torch::Tensor choi_to_stinespring(const torch::Tensor &choi, double tol) {
  auto kraus = choi_to_kraus(choi, tol);
  return kraus_to_stinespring(kraus);
}

torch::Tensor trace_distance(const torch::Tensor &rho,
                             const torch::Tensor &sigma) {
  auto eig = at::linalg_eigvalsh(rho - sigma);
  auto dist = 0.5 * at::sum(at::abs(eig), -1);
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rho.dim() - 2; ++i) {
    shape.push_back(rho.size(i));
  }
  return shape.empty() ? dist.squeeze() : dist.view(shape);
}

torch::Tensor trace_distance_pp(const torch::Tensor &psi,
                                const torch::Tensor &phi) {
  auto inner = at::matmul(linalg_dagger(psi), phi).squeeze(-1).squeeze(-1);
  auto fidelity_sq = at::pow(at::abs(inner), 2);
  auto dist = at::sqrt(at::clamp_min(1.0 - fidelity_sq, 0.0));
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < psi.dim() - 2; ++i) {
    shape.push_back(psi.size(i));
  }
  return shape.empty() ? dist.squeeze() : dist.view(shape);
}

torch::Tensor trace_distance_pm(const torch::Tensor &psi,
                                const torch::Tensor &rho) {
  auto rho_pure = at::matmul(psi, linalg_dagger(psi));
  return trace_distance(rho_pure, rho);
}

torch::Tensor state_fidelity(const torch::Tensor &rho,
                             const torch::Tensor &sigma) {
  auto sqrt_rho = linalg_sqrtm(rho);
  auto term = linalg_sqrtm(at::matmul(sqrt_rho, at::matmul(sigma, sqrt_rho)));
  auto fid = at::real(linalg_trace(term));
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rho.dim() - 2; ++i) {
    shape.push_back(rho.size(i));
  }
  return shape.empty() ? fid.squeeze() : fid.view(shape);
}

torch::Tensor state_fidelity_pp(const torch::Tensor &psi,
                                const torch::Tensor &phi) {
  auto inner = at::matmul(linalg_dagger(psi), phi).squeeze(-1).squeeze(-1);
  auto fid = at::abs(inner);
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < psi.dim() - 2; ++i) {
    shape.push_back(psi.size(i));
  }
  return shape.empty() ? fid.squeeze() : fid.view(shape);
}

torch::Tensor state_fidelity_pm(const torch::Tensor &psi,
                                const torch::Tensor &rho) {
  auto overlap = at::matmul(linalg_dagger(psi), at::matmul(rho, psi))
                     .squeeze(-1)
                     .squeeze(-1);
  auto fid = at::sqrt(at::clamp_min(at::real(overlap), 0.0));
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rho.dim() - 2; ++i) {
    shape.push_back(rho.size(i));
  }
  return shape.empty() ? fid.squeeze() : fid.view(shape);
}

torch::Tensor gate_fidelity(const torch::Tensor &u, const torch::Tensor &v) {
  const auto dim = static_cast<double>(u.size(-1));
  auto tr = linalg_trace(at::matmul(u, linalg_dagger(v)));
  auto fid = at::abs(tr) / dim;
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < u.dim() - 2; ++i) {
    shape.push_back(u.size(i));
  }
  return shape.empty() ? fid.squeeze() : fid.view(shape);
}

torch::Tensor purity(const torch::Tensor &rho) {
  auto prod = at::matmul(rho, rho);
  auto pur = at::real(linalg_trace(prod, -2, -1));
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rho.dim() - 2; ++i) {
    shape.push_back(rho.size(i));
  }
  return shape.empty() ? pur.squeeze() : pur.view(shape);
}

torch::Tensor transpose_1(const torch::Tensor &mat, int64_t dim1) {
  const auto total_dim = mat.size(-1);
  const auto dim2 = total_dim / dim1;
  std::vector<int64_t> batch_dims;
  batch_dims.reserve(std::max<int64_t>(0, mat.dim() - 2));
  for (int64_t i = 0; i < mat.dim() - 2; ++i) {
    batch_dims.push_back(mat.size(i));
  }
  auto reshaped = mat.view({-1, dim1, dim2, dim1, dim2});
  auto permuted = reshaped.permute({0, 3, 2, 1, 4});
  std::vector<int64_t> out_shape = batch_dims;
  out_shape.push_back(total_dim);
  out_shape.push_back(total_dim);
  return permuted.reshape(out_shape);
}

torch::Tensor negativity(const torch::Tensor &density_op) {
  const auto d = density_op.size(-1);
  const auto half_dim = static_cast<int64_t>(std::sqrt(static_cast<double>(d)));
  auto transposed = transpose_1(density_op, half_dim);
  auto eig = at::linalg_eigvalsh(transposed);
  auto threshold = -1e-16 * static_cast<double>(d);
  auto mask = eig <= threshold;
  auto neg = at::sum(at::where(mask, eig, at::zeros_like(eig)), -1);
  auto out = at::abs(neg);
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < density_op.dim() - 2; ++i) {
    shape.push_back(density_op.size(i));
  }
  return shape.empty() ? out.squeeze() : out.view(shape);
}

torch::Tensor logarithmic_negativity(const torch::Tensor &density_op) {
  auto neg = negativity(density_op);
  return at::log2(2 * neg + 1);
}

}

void bind_qinfo(py::module_ &m) {
  auto qinfo_mod = m.def_submodule("qinfo", "Quantum information utilities.");
  qinfo_mod.def(
      "trace_distance", &trace_distance,
      R"doc(
Compute the trace distance between two density matrices.

Args:
    rho: Density matrix tensor of shape (..., d, d).
    sigma: Density matrix tensor of shape (..., d, d).

Returns:
    The trace distance of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("rho"), py::arg("sigma"));

  qinfo_mod.def(
      "trace_distance_pp", &trace_distance_pp,
      R"doc(
Compute the trace distance between two pure states.

Args:
    psi: Pure-state tensor of shape (..., d, 1).
    phi: Pure-state tensor of shape (..., d, 1).

Returns:
    The trace distance of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("psi"), py::arg("phi"));

  qinfo_mod.def(
      "trace_distance_pm", &trace_distance_pm,
      R"doc(
Compute the trace distance between a pure state and a mixed state.

Args:
    psi: Pure-state tensor of shape (..., d, 1).
    rho: Mixed-state density matrix tensor of shape (..., d, d).

Returns:
    The trace distance of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("psi"), py::arg("rho"));

  qinfo_mod.def(
      "state_fidelity", &state_fidelity,
      R"doc(
Compute the quantum state fidelity between two density matrices.

Args:
    rho: Density matrix tensor of shape (..., d, d).
    sigma: Density matrix tensor of shape (..., d, d).

Returns:
    The fidelity of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("rho"), py::arg("sigma"));

  qinfo_mod.def(
      "state_fidelity_pp", &state_fidelity_pp,
      R"doc(
Compute the quantum state fidelity between two pure states.

Args:
    psi: Pure-state tensor of shape (..., d, 1).
    phi: Pure-state tensor of shape (..., d, 1).

Returns:
    The fidelity of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("psi"), py::arg("phi"));

  qinfo_mod.def(
      "state_fidelity_pm", &state_fidelity_pm,
      R"doc(
Compute the quantum state fidelity between a pure state and a mixed state.

Args:
    psi: Pure-state tensor of shape (..., d, 1).
    rho: Mixed-state density matrix tensor of shape (..., d, d).

Returns:
    The fidelity of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("psi"), py::arg("rho"));

  qinfo_mod.def(
      "gate_fidelity", &gate_fidelity,
      R"doc(
Compute the gate fidelity between two unitary matrices.

Args:
    u: Unitary tensor of shape (..., d, d).
    v: Unitary tensor of shape (..., d, d).

Returns:
    The fidelity |tr(u v^H)| / d of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("u"), py::arg("v"));

  qinfo_mod.def(
      "purity", &purity,
      R"doc(
Compute the purity tr(rho^2) of a density matrix.

Args:
    rho: Density matrix tensor of shape (..., d, d).

Returns:
    The purity of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("rho"));

  qinfo_mod.def(
      "negativity", &negativity,
      R"doc(
Compute the entanglement negativity for a bipartite density matrix.

Args:
    density_op: Density matrix tensor of shape (..., d, d) where d is assumed to be a square number.

Returns:
    The negativity of shape (...) (or a scalar if inputs are unbatched).

Notes:
    This implementation assumes a 2-party split with equal local dimensions: d = (sqrt(d))^2.
)doc",
      py::arg("density_op"));

  qinfo_mod.def(
      "logarithmic_negativity", &logarithmic_negativity,
      R"doc(
Compute the logarithmic negativity log2(2*N + 1).

Args:
    density_op: Density matrix tensor of shape (..., d, d).

Returns:
    The logarithmic negativity of shape (...) (or a scalar if inputs are unbatched).
)doc",
      py::arg("density_op"));

  qinfo_mod.def(
      "kraus_to_choi", &kraus_to_choi, py::arg("kraus"),
      R"doc(
Convert Kraus representation to Choi.

Args:
    kraus: Tensor with shape (..., r, d, d) or (r, d, d).

Returns:
    Tensor with shape (..., d^2, d^2) or (d^2, d^2).
)doc");

  qinfo_mod.def(
      "choi_to_kraus", &choi_to_kraus, py::arg("choi"), py::arg("tol") = 1e-6,
      R"doc(
Convert Choi representation to Kraus representation.

Args:
    choi: Tensor with shape (..., d_out^2, d_in^2) or (d_out^2, d_in^2).
    tol:  Cumulative eigenvalue tolerance to truncate small components.

Returns:
    Tensor with shape (..., r, d_out, d_in) or (r, d_out, d_in).
)doc");

  qinfo_mod.def(
      "kraus_to_stinespring", &kraus_to_stinespring, py::arg("kraus"),
      R"doc(
Convert Kraus representation to Stinespring representation.

Args:
    kraus: Tensor with shape (..., r, d_out, d_in) or (r, d_out, d_in).

Returns:
    Tensor with shape (..., r * d_out, d_in) or (r * d_out, d_in).
)doc");

  qinfo_mod.def(
      "stinespring_to_kraus", &stinespring_to_kraus, py::arg("stinespring"),
      R"doc(
Convert Stinespring representation to Kraus representation.

Args:
    stinespring: Tensor with shape (..., r * d_out, d_in).

Returns:
    Tensor with shape (..., r, d_out, d_in) (same rank inferred from leading dim).
)doc");

  qinfo_mod.def(
      "stinespring_to_choi", &stinespring_to_choi, py::arg("stinespring"),
      R"doc(
Convert Stinespring representation to Choi representation.

Args:
    stinespring: Tensor with shape (..., r * d_out, d_in).

Returns:
    Tensor with shape (..., d_out^2, d_in^2).
)doc");

  qinfo_mod.def(
      "choi_to_stinespring", &choi_to_stinespring, py::arg("choi"),
      py::arg("tol") = 1e-6,
      R"doc(
Convert Choi representation to Stinespring representation.

Args:
    choi: Tensor with shape (..., d_out^2, d_in^2).
    tol:  Cumulative eigenvalue tolerance to truncate small components.

Returns:
    Tensor with shape (..., r * d_out, d_in).
)doc");
}

}