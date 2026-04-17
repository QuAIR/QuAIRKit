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

#include "product_state.h"

#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>

#include "../state_base/common.h"
#include "../state_base/kernels.h"

namespace quairkit_cpp {

namespace {

std::vector<int64_t> arange_vec(int64_t n) {
  std::vector<int64_t> v(static_cast<size_t>(n));
  std::iota(v.begin(), v.end(), 0);
  return v;
}

std::vector<int64_t> flatten_nested(const std::vector<std::vector<int64_t>> &xs) {
  std::vector<int64_t> out;
  for (const auto &group : xs) {
    out.insert(out.end(), group.begin(), group.end());
  }
  return out;
}

std::vector<int64_t> broadcast_batch_dims(const std::vector<int64_t> &a,
                                          const std::vector<int64_t> &b) {
  if (a.empty()) {
    return b;
  }
  if (b.empty()) {
    return a;
  }
  std::vector<int64_t> left = a;
  std::vector<int64_t> right = b;
  if (right.size() > left.size()) {
    std::swap(left, right);
  }
  std::vector<int64_t> result = left;
  const auto left_len = static_cast<int64_t>(left.size());
  const auto right_len = static_cast<int64_t>(right.size());
  for (int64_t i = 0; i < right_len; ++i) {
    const auto idx_left = left_len - 1 - i;
    const auto x = right[static_cast<size_t>(right_len - 1 - i)];
    const auto y = left[static_cast<size_t>(idx_left)];
    if (x == 1) {
      continue;
    }
    if (y == 1) {
      result[static_cast<size_t>(idx_left)] = x;
      continue;
    }
    if (x != y) {
      throw std::runtime_error("Block batch dims are not broadcastable.");
    }
  }
  return result;
}

std::string lower_copy(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v;
}

bool is_pure_state(const ProductDenseState &state) {
  return std::holds_alternative<PureState>(state);
}

ProductDenseState clone_dense(const ProductDenseState &state) {
  return std::visit(
      [](const auto &st) -> ProductDenseState { return st.clone(); }, state);
}

std::vector<int64_t> dense_system_dim(const ProductDenseState &state) {
  return std::visit([](const auto &st) { return st.system_dim(); }, state);
}

std::vector<int64_t> dense_system_seq(const ProductDenseState &state) {
  return std::visit([](const auto &st) { return st.system_seq(); }, state);
}

std::vector<int64_t> dense_batch_dim(const ProductDenseState &state) {
  return std::visit([](const auto &st) { return st.batch_dim(); }, state);
}

void dense_set_batch_dim(ProductDenseState &state,
                         const std::vector<int64_t> &batch_dim) {
  std::visit([&](auto &st) { st.set_batch_dim(batch_dim); }, state);
}

int64_t dense_dim(const ProductDenseState &state) {
  return std::visit([](const auto &st) { return st.dim(); }, state);
}

torch::Tensor dense_data(const ProductDenseState &state) {
  return std::visit([](const auto &st) { return st.data(); }, state);
}

void dense_set_data(ProductDenseState &state, const torch::Tensor &data) {
  std::visit([&](auto &st) { st.set_data(data); }, state);
}

ProbabilityDataImpl &dense_prob(ProductDenseState &state) {
  return std::visit([](auto &st) -> ProbabilityDataImpl & { return st.prob(); }, state);
}

const ProbabilityDataImpl &dense_prob_const(const ProductDenseState &state) {
  return std::visit(
      [](const auto &st) -> const ProbabilityDataImpl & { return st.prob(); }, state);
}

torch::Tensor dense_ket(const ProductDenseState &state) {
  return std::visit(
      [](const auto &st) -> torch::Tensor {
        auto copy = st.clone();
        copy.reset_sequence();
        using T = std::decay_t<decltype(st)>;
        if constexpr (std::is_same_v<T, PureState>) {
          return copy.ket();
        } else {
          return copy.density_matrix();
        }
      },
      state);
}

torch::Tensor dense_density_matrix(const ProductDenseState &state) {
  return std::visit(
      [](const auto &st) {
        auto copy = st.clone();
        copy.reset_sequence();
        return copy.density_matrix();
      },
      state);
}

void dense_reset_sequence(ProductDenseState &state) {
  std::visit([](auto &st) { st.reset_sequence(); }, state);
}

void dense_set_system_seq(ProductDenseState &state,
                          const std::vector<int64_t> &target_seq) {
  std::visit([&](auto &st) { st.set_system_seq(target_seq); }, state);
}

void dense_set_system_seq_metadata(ProductDenseState &state,
                                   const std::vector<int64_t> &target_seq) {
  std::visit([&](auto &st) { st.set_system_seq_metadata(target_seq); }, state);
}

ProductDenseState dense_to_mixed(const ProductDenseState &state) {
  if (std::holds_alternative<MixedState>(state)) {
    return clone_dense(state);
  }
  const auto &pure = std::get<PureState>(state);
  return MixedState(pure.density_matrix(), pure.system_dim(), pure.system_seq(),
                    pure.prob().clone_list());
}

ProductDenseState dense_permute(const ProductDenseState &state,
                                const std::vector<int64_t> &target_seq) {
  auto cloned = clone_dense(state);
  dense_reset_sequence(cloned);
  dense_set_system_seq(cloned, target_seq);
  dense_set_system_seq_metadata(cloned, arange_vec(static_cast<int64_t>(target_seq.size())));
  return cloned;
}

void expand_dense_to_batch(ProductDenseState &state,
                           const std::vector<int64_t> &target_batch_dim) {
  const auto cur_bd = dense_batch_dim(state);
  if (cur_bd == target_batch_dim || target_batch_dim.empty()) {
    return;
  }

  const auto d = dense_dim(state);
  const auto is_pure = is_pure_state(state);
  auto data = dense_data(state);

  std::vector<int64_t> cur_shape;
  std::vector<int64_t> target_shape = target_batch_dim;
  if (is_pure) {
    target_shape.push_back(d);
  } else {
    target_shape.push_back(d);
    target_shape.push_back(d);
  }

  if (cur_bd.empty()) {
    cur_shape.assign(target_batch_dim.size(), 1);
    if (is_pure) {
      cur_shape.push_back(d);
    } else {
      cur_shape.push_back(d);
      cur_shape.push_back(d);
    }
    data = data.view(cur_shape);
  } else {
    const auto pad_len =
        static_cast<int64_t>(target_batch_dim.size()) - static_cast<int64_t>(cur_bd.size());
    std::vector<int64_t> cur_bd_padded = cur_bd;
    if (pad_len > 0) {
      bool is_prefix = true;
      for (size_t i = 0; i < cur_bd.size(); ++i) {
        if (target_batch_dim[i] != cur_bd[i]) {
          is_prefix = false;
          break;
        }
      }
      if (is_prefix) {
        cur_bd_padded.insert(cur_bd_padded.end(), static_cast<size_t>(pad_len), 1);
      } else {
        cur_bd_padded.insert(cur_bd_padded.begin(), static_cast<size_t>(pad_len), 1);
      }
    }
    cur_shape = cur_bd_padded;
    if (is_pure) {
      cur_shape.push_back(d);
    } else {
      cur_shape.push_back(d);
      cur_shape.push_back(d);
    }
    data = data.view(cur_shape);
  }

  auto expanded = data.expand(target_shape).contiguous();
  if (is_pure) {
    dense_set_data(state, expanded.view({-1, d}));
  } else {
    dense_set_data(state, expanded.view({-1, d, d}));
  }
  dense_set_batch_dim(state, target_batch_dim);
}

torch::Tensor nkron_list(const std::vector<torch::Tensor> &xs) {
  if (xs.empty()) {
    throw std::runtime_error("nkron requires non-empty tensor list.");
  }
  auto expand_to_batch = [](const torch::Tensor &t,
                            const std::vector<int64_t> &target_batch) {
    auto cur_shape = t.sizes().vec();
    if (cur_shape.size() < 2) {
      throw std::runtime_error("kron tensor must have at least 2 trailing dims.");
    }
    std::vector<int64_t> lead(cur_shape.begin(), cur_shape.end() - 2);
    if (lead == target_batch) {
      return t;
    }
    const auto pad = static_cast<int64_t>(target_batch.size()) -
                     static_cast<int64_t>(lead.size());
    std::vector<int64_t> view_shape;
    if (pad > 0) {
      view_shape.assign(static_cast<size_t>(pad), 1);
    }
    view_shape.insert(view_shape.end(), lead.begin(), lead.end());
    view_shape.push_back(cur_shape[cur_shape.size() - 2]);
    view_shape.push_back(cur_shape[cur_shape.size() - 1]);

    auto target_shape = target_batch;
    target_shape.push_back(cur_shape[cur_shape.size() - 2]);
    target_shape.push_back(cur_shape[cur_shape.size() - 1]);
    return t.view(view_shape).expand(target_shape);
  };

  auto kron_two = [&](const torch::Tensor &a, const torch::Tensor &b) {
    auto a_shape = a.sizes().vec();
    auto b_shape = b.sizes().vec();
    std::vector<int64_t> a_batch(a_shape.begin(), a_shape.end() - 2);
    std::vector<int64_t> b_batch(b_shape.begin(), b_shape.end() - 2);
    auto target_batch = broadcast_batch_dims(a_batch, b_batch);

    auto a_exp = expand_to_batch(a, target_batch);
    auto b_exp = expand_to_batch(b, target_batch);
    const auto m = a_shape[a_shape.size() - 2];
    const auto n = a_shape[a_shape.size() - 1];
    const auto p = b_shape[b_shape.size() - 2];
    const auto q = b_shape[b_shape.size() - 1];

    auto kron = a_exp.unsqueeze(-1).unsqueeze(-3) * b_exp.unsqueeze(-2).unsqueeze(-4);
    auto out_shape = target_batch;
    out_shape.push_back(m * p);
    out_shape.push_back(n * q);
    return kron.contiguous().view(out_shape);
  };

  auto out = xs.front();
  for (size_t i = 1; i < xs.size(); ++i) {
    out = kron_two(out, xs[i]);
  }
  return out;
}

void dense_evolve(ProductDenseState &state, const torch::Tensor &unitary,
                  const std::vector<int64_t> &sys_idx, bool on_batch) {
  std::visit([&](auto &st) { st.evolve(unitary, sys_idx, on_batch); }, state);
}

void dense_evolve_with_broadcast(ProductDenseState &state,
                                 const torch::Tensor &unitary,
                                 const std::vector<int64_t> &sys_idx,
                                 bool on_batch) {
  if (!on_batch) {
    const auto block_batch = dense_batch_dim(state);
    std::vector<int64_t> unitary_batch(unitary.sizes().begin(),
                                       unitary.sizes().end() - 2);
    auto u = unitary;
    if (!block_batch.empty() && !unitary_batch.empty()) {
      const auto d = unitary.size(-1);
      const auto num_prior = static_cast<int64_t>(block_batch.size()) -
                             static_cast<int64_t>(unitary_batch.size());
      if (num_prior > 0) {
        std::vector<int64_t> expanded_shape(static_cast<size_t>(num_prior), 1);
        for (auto s : unitary.sizes().vec()) {
          expanded_shape.push_back(s);
        }
        auto target_shape = block_batch;
        target_shape.push_back(d);
        target_shape.push_back(d);
        u = u.view(expanded_shape).expand(target_shape).contiguous();
      }
      dense_evolve(state, u, sys_idx, true);
      return;
    }
    if (!block_batch.empty()) {
      const auto d = unitary.size(-1);
      std::vector<int64_t> expanded_shape(block_batch.size(), 1);
      expanded_shape.push_back(d);
      expanded_shape.push_back(d);
      auto target_shape = block_batch;
      target_shape.push_back(d);
      target_shape.push_back(d);
      u = u.view(expanded_shape).expand(target_shape).contiguous();
      dense_evolve(state, u, sys_idx, true);
      return;
    }
    dense_evolve(state, u, sys_idx, on_batch);
    return;
  }
  dense_evolve(state, unitary, sys_idx, on_batch);
}

void dense_evolve_keep_dim(ProductDenseState &state, const torch::Tensor &unitary,
                           const std::vector<int64_t> &sys_idx, bool on_batch) {
  std::visit([&](auto &st) { st.evolve_keep_dim(unitary, sys_idx, on_batch); }, state);
}

void dense_evolve_ctrl(ProductDenseState &state, const torch::Tensor &unitary,
                       int64_t index,
                       const std::vector<std::vector<int64_t>> &sys_idx,
                       bool on_batch) {
  std::visit(
      [&](auto &st) { st.evolve_ctrl(unitary, index, sys_idx, on_batch); }, state);
}

void dense_transform(ProductDenseState &state, const torch::Tensor &op,
                     const std::vector<int64_t> &sys_idx,
                     const std::string &repr_type, bool on_batch) {
  if (std::holds_alternative<PureState>(state)) {
    state = dense_to_mixed(state);
  }
  auto &mixed = std::get<MixedState>(state);
  if (repr_type == "kraus") {
    mixed.transform_kraus(op, sys_idx, on_batch);
    return;
  }
  mixed.transform_choi(op, sys_idx, on_batch);
}

std::tuple<torch::Tensor, ProductDenseState>
dense_measure(const ProductDenseState &state, const torch::Tensor &measure_op,
              const std::vector<int64_t> &sys_idx) {
  return std::visit(
      [&](const auto &st) -> std::tuple<torch::Tensor, ProductDenseState> {
        auto copy = st.clone();
        auto out = copy.measure(measure_op, sys_idx);
        return {std::get<0>(out), std::get<1>(out)};
      },
      state);
}

std::tuple<torch::Tensor, ProductDenseState>
dense_measure_many(const ProductDenseState &state, const torch::Tensor &measure_op,
                   const std::vector<std::vector<int64_t>> &sys_idx_list) {
  return std::visit(
      [&](const auto &st) -> std::tuple<torch::Tensor, ProductDenseState> {
        auto copy = st.clone();
        auto out = copy.measure_many(measure_op, sys_idx_list);
        return {std::get<0>(out), std::get<1>(out)};
      },
      state);
}

std::tuple<torch::Tensor, std::optional<ProductDenseState>>
dense_measure_by_state(const ProductDenseState &state, const PureState &basis,
                       const std::vector<int64_t> &sys_idx, bool keep_rest) {
  if (std::holds_alternative<PureState>(state)) {
    auto st = std::get<PureState>(state).clone();
    auto out = st.measure_by_state(basis.ket(), sys_idx, keep_rest);
    auto prob = std::get<0>(out);
    auto rest_opt = std::get<1>(out);
    if (!rest_opt.has_value()) {
      return {prob, std::nullopt};
    }
    return {prob, ProductDenseState(std::move(*rest_opt))};
  }
  auto st = std::get<MixedState>(state).clone();
  auto out = st.measure_by_state(basis.ket(), sys_idx, keep_rest);
  auto prob = std::get<0>(out);
  auto rest_opt = std::get<1>(out);
  if (!rest_opt.has_value()) {
    return {prob, std::nullopt};
  }
  return {prob, ProductDenseState(std::move(*rest_opt))};
}

std::tuple<torch::Tensor, std::optional<ProductDenseState>>
dense_measure_by_state_product(const ProductDenseState &state,
                               const std::vector<torch::Tensor> &list_kets,
                               const std::vector<std::vector<int64_t>> &list_sys_idx,
                               bool keep_rest) {
  if (std::holds_alternative<PureState>(state)) {
    auto st = std::get<PureState>(state).clone();
    auto out = st.measure_by_state_product(list_kets, list_sys_idx, keep_rest);
    auto prob = std::get<0>(out);
    auto rest_opt = std::get<1>(out);
    if (!rest_opt.has_value()) {
      return {prob, std::nullopt};
    }
    return {prob, ProductDenseState(std::move(*rest_opt))};
  }
  auto st = std::get<MixedState>(state).clone();
  auto out = st.measure_by_state_product(list_kets, list_sys_idx, keep_rest);
  auto prob = std::get<0>(out);
  auto rest_opt = std::get<1>(out);
  if (!rest_opt.has_value()) {
    return {prob, std::nullopt};
  }
  return {prob, ProductDenseState(std::move(*rest_opt))};
}

ProductDenseState dense_trace(const ProductDenseState &state,
                              const std::vector<int64_t> &trace_idx) {
  if (std::holds_alternative<MixedState>(state)) {
    auto st = std::get<MixedState>(state).clone();
    return st.trace(trace_idx);
  }
  auto mixed = dense_to_mixed(state);
  return std::get<MixedState>(mixed).trace(trace_idx);
}

ProductDenseState dense_transpose(const ProductDenseState &state,
                                  const std::vector<int64_t> &transpose_idx) {
  if (std::holds_alternative<MixedState>(state)) {
    auto st = std::get<MixedState>(state).clone();
    return st.transpose(transpose_idx);
  }
  auto mixed = dense_to_mixed(state);
  return std::get<MixedState>(mixed).transpose(transpose_idx);
}

torch::Tensor dense_expec_val(const ProductDenseState &state, const torch::Tensor &obs,
                              const std::vector<int64_t> &sys_idx) {
  return std::visit(
      [&](const auto &st) {
        auto copy = st.clone();
        return copy.expectation_value(obs, sys_idx);
      },
      state);
}

torch::Tensor dense_expec_val_product(
    const ProductDenseState &state, const std::vector<torch::Tensor> &obs_list,
    const std::vector<std::vector<int64_t>> &sys_idx_list) {
  return std::visit(
      [&](const auto &st) {
        auto copy = st.clone();
        return copy.expectation_value_product(obs_list, sys_idx_list);
      },
      state);
}

torch::Tensor dense_expec_val_pauli_terms(
    const ProductDenseState &state,
    const std::vector<std::string> &pauli_words_r,
    const std::vector<std::vector<int64_t>> &sites) {
  return std::visit(
      [&](const auto &st) {
        auto copy = st.clone();
        return copy.expec_val_pauli_terms(pauli_words_r, sites);
      },
      state);
}

ProductDenseState dense_prob_select(const ProductDenseState &state,
                                    const torch::Tensor &outcome_idx,
                                    int64_t prob_idx) {
  return std::visit(
      [&](const auto &st) -> ProductDenseState {
        auto copy = st.clone();
        return copy.prob_select(outcome_idx, prob_idx);
      },
      state);
}

ProductDenseState dense_expec_state(const ProductDenseState &state,
                                    const std::vector<int64_t> &prob_idx) {
  if (std::holds_alternative<MixedState>(state)) {
    auto copy = std::get<MixedState>(state).clone();
    return copy.expec_state(prob_idx);
  }
  auto mixed = dense_to_mixed(state);
  return std::get<MixedState>(mixed).expec_state(prob_idx);
}

void dense_normalize(ProductDenseState &state) {
  if (std::holds_alternative<PureState>(state)) {
    auto &st = std::get<PureState>(state);
    auto data = st.data();
    st.set_data(torch::div(data, torch::norm(data, 2, -1, true)));
    return;
  }
  auto &st = std::get<MixedState>(state);
  auto dm = st.density_matrix();
  auto tr = dm.diagonal(0, -2, -1).sum(-1);
  auto tr_shape = tr.sizes().vec();
  tr_shape.push_back(1);
  tr_shape.push_back(1);
  dm = dm / tr.view(tr_shape);
  st.set_data(dm.view({-1, st.dim(), st.dim()}));
}

}

ProductState::ProductState(const std::vector<ProductDenseState> &blocks,
                           const std::vector<std::vector<int64_t>> &block_indices,
                           const std::vector<torch::Tensor> &probability,
                           const std::vector<std::string> &roles, bool keep_dim)
    : blocks_(blocks), block_indices_(block_indices), prob_(probability),
      roles_(normalize_roles(roles, prob_)), keep_dim_(keep_dim) {
  if (blocks_.size() != block_indices_.size()) {
    throw std::runtime_error(
        "ProductState requires blocks and block_indices with the same length.");
  }
  std::set<int64_t> seen_idx;
  int64_t max_idx = -1;
  for (size_t b = 0; b < blocks_.size(); ++b) {
    const auto local_dim = dense_system_dim(blocks_[b]);
    if (local_dim.size() != block_indices_[b].size()) {
      throw std::runtime_error(
          "Each ProductState block index list must match block system dimension length.");
    }
    for (auto idx : block_indices_[b]) {
      if (idx < 0) {
        throw std::runtime_error("ProductState indices must be non-negative.");
      }
      if (!seen_idx.insert(idx).second) {
        throw std::runtime_error("ProductState indices must form a disjoint partition.");
      }
      max_idx = std::max(max_idx, idx);
    }
  }
  if (max_idx >= 0) {
    for (int64_t i = 0; i <= max_idx; ++i) {
      if (seen_idx.find(i) == seen_idx.end()) {
        throw std::runtime_error(
            "ProductState indices must cover a contiguous range [0..n-1].");
      }
    }
  }

  for (const auto &st : blocks_) {
    if (dense_prob_const(st).size() != 0) {
      throw std::runtime_error(
          "ProductState blocks must not carry local probability history.");
    }
  }

  std::vector<int64_t> temp_batch_dim;
  for (const auto &st : blocks_) {
    const auto bd = dense_batch_dim(st);
    if (temp_batch_dim.empty()) {
      temp_batch_dim = bd;
      continue;
    }
    temp_batch_dim = broadcast_batch_dims(temp_batch_dim, bd);
  }

  const auto num_prob = static_cast<int64_t>(prob_.size());
  if (num_prob > 0 && static_cast<int64_t>(temp_batch_dim.size()) >= num_prob) {
    data_batch_dim_.assign(temp_batch_dim.begin(),
                           temp_batch_dim.end() - num_prob);
  } else {
    data_batch_dim_ = temp_batch_dim;
  }
  sync_data_batch_dim();
}

ProductState ProductState::clone() const {
  std::vector<ProductDenseState> cloned;
  cloned.reserve(blocks_.size());
  for (const auto &st : blocks_) {
    cloned.push_back(clone_dense(st));
  }
  return ProductState(cloned, block_indices_, prob_.clone_list(), roles_, keep_dim_);
}

std::vector<std::string>
ProductState::normalize_roles(const std::vector<std::string> &roles,
                              const ProbabilityDataImpl &prob) {
  const auto num_prob = static_cast<int64_t>(prob.size());
  if (roles.empty()) {
    return std::vector<std::string>(static_cast<size_t>(num_prob), kRoleClassical);
  }
  if (static_cast<int64_t>(roles.size()) != num_prob) {
    throw std::runtime_error(
        "roles length must match probability length for ProductState.");
  }
  std::vector<std::string> normalized;
  normalized.reserve(roles.size());
  for (const auto &role : roles) {
    auto v = lower_copy(role);
    if (v != kRoleClassical && v != kRoleProdSum) {
      throw std::runtime_error("Unsupported ProductState probability role.");
    }
    normalized.push_back(v);
  }
  return normalized;
}

void ProductState::invalidate_cache() const { merged_cache_ = c10::nullopt; }

int64_t ProductState::num_systems() const {
  int64_t max_idx = -1;
  for (const auto &idxs : block_indices_) {
    for (auto v : idxs) {
      max_idx = std::max(max_idx, v);
    }
  }
  return max_idx + 1;
}

std::vector<int64_t> ProductState::system_dim() const {
  const auto n = num_systems();
  if (n <= 0) {
    return {};
  }
  std::vector<int64_t> out(static_cast<size_t>(n), 0);
  for (size_t b = 0; b < blocks_.size(); ++b) {
    const auto dims = dense_system_dim(blocks_[b]);
    const auto &idxs = block_indices_[b];
    for (size_t i = 0; i < idxs.size(); ++i) {
      out[static_cast<size_t>(idxs[i])] = dims[i];
    }
  }
  return out;
}

std::vector<int64_t> ProductState::batch_dim() const {
  auto out = data_batch_dim_;
  const auto prob_shape = prob_.shape();
  out.insert(out.end(), prob_shape.begin(), prob_shape.end());
  return out;
}

std::string ProductState::backend() const {
  return std::all_of(blocks_.begin(), blocks_.end(), is_pure_state)
             ? "default-pure"
             : "default-mixed";
}

std::vector<int64_t> ProductState::classical_indices() const {
  std::vector<int64_t> out;
  for (size_t i = 0; i < roles_.size(); ++i) {
    if (roles_[i] == kRoleClassical) {
      out.push_back(static_cast<int64_t>(i));
    }
  }
  return out;
}

std::vector<int64_t> ProductState::prod_sum_indices() const {
  std::vector<int64_t> out;
  for (size_t i = 0; i < roles_.size(); ++i) {
    if (roles_[i] == kRoleProdSum) {
      out.push_back(static_cast<int64_t>(i));
    }
  }
  return out;
}

std::vector<ProductExportBlock> ProductState::export_blocks(bool clone_state) const {
  std::vector<ProductExportBlock> out;
  out.reserve(blocks_.size());
  for (size_t i = 0; i < blocks_.size(); ++i) {
    out.push_back(
        ProductExportBlock{clone_state ? clone_dense(blocks_[i]) : blocks_[i],
                           block_indices_[i]});
  }
  return out;
}

torch::Tensor ProductState::probability() const {
  const auto classical = classical_indices();
  if (classical.empty()) {
    if (!prob_.empty()) {
      const auto &ref = prob_.list().front();
      return torch::tensor(1.0, ref.options());
    }
    return torch::tensor(1.0);
  }
  return prob_.joint(classical);
}

void ProductState::sync_data_batch_dim() {
  if (blocks_.empty()) {
    return;
  }
  std::vector<int64_t> blocks_batch;
  for (const auto &st : blocks_) {
    const auto bd = dense_batch_dim(st);
    if (blocks_batch.empty()) {
      blocks_batch = bd;
      continue;
    }
    blocks_batch = broadcast_batch_dims(blocks_batch, bd);
  }

  const auto current_total = batch_dim();
  if (blocks_batch.size() > current_total.size()) {
    const auto num_prob = static_cast<int64_t>(prob_.size());
    if (num_prob > 0 &&
        static_cast<int64_t>(blocks_batch.size()) >= num_prob) {
      data_batch_dim_.assign(blocks_batch.begin(), blocks_batch.end() - num_prob);
    } else {
      data_batch_dim_ = blocks_batch;
    }
  }

  const auto target = batch_dim();
  if (target.empty()) {
    return;
  }
  for (auto &st : blocks_) {
    if (dense_batch_dim(st) != target) {
      expand_dense_to_batch(st, target);
    }
  }
}

std::pair<int64_t, std::vector<int64_t>>
ProductState::global_to_block_local(const std::vector<int64_t> &sys_idx) const {
  int64_t target_block = -1;
  std::vector<int64_t> local;
  local.reserve(sys_idx.size());
  for (auto s : sys_idx) {
    int64_t found = -1;
    for (size_t b = 0; b < block_indices_.size(); ++b) {
      const auto &idxs = block_indices_[b];
      if (std::find(idxs.begin(), idxs.end(), s) != idxs.end()) {
        found = static_cast<int64_t>(b);
        break;
      }
    }
    if (found < 0) {
      throw std::runtime_error("Unknown system index for ProductState.");
    }
    if (target_block < 0) {
      target_block = found;
    } else if (target_block != found) {
      return {-1, {}};
    }
    const auto &idxs = block_indices_[static_cast<size_t>(found)];
    auto it = std::find(idxs.begin(), idxs.end(), s);
    local.push_back(static_cast<int64_t>(std::distance(idxs.begin(), it)));
  }
  return {target_block, local};
}

ProductDenseState ProductState::make_dense_from_data(
    const torch::Tensor &data, const std::vector<int64_t> &sys_dim,
    bool force_mixed) const {
  if (force_mixed) {
    return MixedState(data, sys_dim, arange_vec(static_cast<int64_t>(sys_dim.size())),
                      {});
  }
  if (create_state_type(data) == "mixed") {
    return MixedState(data, sys_dim, arange_vec(static_cast<int64_t>(sys_dim.size())),
                      {});
  }
  return PureState(data, sys_dim, arange_vec(static_cast<int64_t>(sys_dim.size())),
                   {});
}

std::pair<ProductDenseState, std::vector<int64_t>>
ProductState::merge_blocks(const std::vector<int64_t> &block_ids,
                           bool attach_real_prob) const {
  if (block_ids.empty()) {
    throw std::runtime_error("merge_blocks requires non-empty block_ids.");
  }

  std::vector<int64_t> merged_global;
  for (auto b : block_ids) {
    const auto &idxs = block_indices_[static_cast<size_t>(b)];
    merged_global.insert(merged_global.end(), idxs.begin(), idxs.end());
  }
  std::sort(merged_global.begin(), merged_global.end());

  std::vector<int64_t> sorted_ids = block_ids;
  std::sort(sorted_ids.begin(), sorted_ids.end(), [&](int64_t l, int64_t r) {
    return *std::min_element(block_indices_[static_cast<size_t>(l)].begin(),
                             block_indices_[static_cast<size_t>(l)].end()) <
           *std::min_element(block_indices_[static_cast<size_t>(r)].begin(),
                             block_indices_[static_cast<size_t>(r)].end());
  });

  std::vector<int64_t> concat_global;
  std::vector<std::pair<std::vector<int64_t>, ProductDenseState>> ordered_items;
  for (auto b : sorted_ids) {
    auto idxs = block_indices_[static_cast<size_t>(b)];
    auto state = clone_dense(blocks_[static_cast<size_t>(b)]);
    std::vector<int64_t> perm(static_cast<size_t>(idxs.size()));
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](int64_t i, int64_t j) { return idxs[i] < idxs[j]; });
    bool identity = true;
    for (size_t i = 0; i < perm.size(); ++i) {
      if (perm[i] != static_cast<int64_t>(i)) {
        identity = false;
        break;
      }
    }
    if (!identity) {
      state = dense_permute(state, perm);
      std::vector<int64_t> sorted_idxs;
      sorted_idxs.reserve(idxs.size());
      for (auto p : perm) {
        sorted_idxs.push_back(idxs[static_cast<size_t>(p)]);
      }
      idxs = sorted_idxs;
    }
    ordered_items.push_back({idxs, state});
    concat_global.insert(concat_global.end(), idxs.begin(), idxs.end());
  }

  const bool all_pure = std::all_of(
      ordered_items.begin(), ordered_items.end(),
      [](const auto &x) { return is_pure_state(x.second); });

  std::vector<torch::Tensor> tensors;
  tensors.reserve(ordered_items.size());
  for (const auto &item : ordered_items) {
    tensors.push_back(all_pure ? dense_ket(item.second)
                               : dense_density_matrix(item.second));
  }
  const auto data = nkron_list(tensors);

  const auto all_sys_dim = system_dim();
  std::vector<int64_t> sys_dim_concat;
  sys_dim_concat.reserve(concat_global.size());
  for (auto g : concat_global) {
    sys_dim_concat.push_back(all_sys_dim[static_cast<size_t>(g)]);
  }

  auto merged = make_dense_from_data(data, sys_dim_concat, !all_pure);
  if (concat_global != merged_global) {
    std::vector<int64_t> target_seq;
    target_seq.reserve(merged_global.size());
    for (auto g : merged_global) {
      auto it = std::find(concat_global.begin(), concat_global.end(), g);
      target_seq.push_back(
          static_cast<int64_t>(std::distance(concat_global.begin(), it)));
    }
    merged = dense_permute(merged, target_seq);
    merged = make_dense_from_data(
        all_pure ? dense_ket(merged) : dense_density_matrix(merged),
        [&]() {
          std::vector<int64_t> dims;
          for (auto g : merged_global) {
            dims.push_back(all_sys_dim[static_cast<size_t>(g)]);
          }
          return dims;
        }(),
        !all_pure);
  }

  if (attach_real_prob && !prob_.empty()) {
    auto cur_batch = dense_batch_dim(merged);
    const auto target_prob_len = static_cast<int64_t>(prob_.size());
    if (static_cast<int64_t>(cur_batch.size()) >= target_prob_len) {
      cur_batch.resize(static_cast<size_t>(cur_batch.size() - target_prob_len));
      dense_set_batch_dim(merged, cur_batch);
    }
    auto &merged_prob = dense_prob(merged);
    for (const auto &p : prob_.clone_list()) {
      merged_prob.append(p, false, c10::nullopt, c10::nullopt,
                         false);
    }
  }
  return {merged, merged_global};
}

void ProductState::replace_blocks_with_merged(
    const std::vector<int64_t> &block_ids, const ProductDenseState &merged_state,
    const std::vector<int64_t> &merged_global) {
  std::vector<ProductDenseState> keep_state;
  std::vector<std::vector<int64_t>> keep_idx;
  std::set<int64_t> remove(block_ids.begin(), block_ids.end());
  const auto insert_pos = *std::min_element(block_ids.begin(), block_ids.end());
  bool inserted = false;
  for (size_t i = 0; i < blocks_.size(); ++i) {
    if (remove.find(static_cast<int64_t>(i)) != remove.end()) {
      if (static_cast<int64_t>(i) == insert_pos && !inserted) {
        keep_state.push_back(clone_dense(merged_state));
        keep_idx.push_back(merged_global);
        inserted = true;
      }
      continue;
    }
    keep_state.push_back(clone_dense(blocks_[i]));
    keep_idx.push_back(block_indices_[i]);
  }
  if (!inserted) {
    keep_state.push_back(clone_dense(merged_state));
    keep_idx.push_back(merged_global);
  }
  blocks_ = keep_state;
  block_indices_ = keep_idx;
}

void ProductState::ensure_merged_for_indices(const std::vector<int64_t> &sys_idx,
                                             bool on_batch) {
  (void)on_batch;
  std::set<int64_t> involved;
  for (size_t b = 0; b < block_indices_.size(); ++b) {
    const auto &idxs = block_indices_[b];
    for (auto g : sys_idx) {
      if (std::find(idxs.begin(), idxs.end(), g) != idxs.end()) {
        involved.insert(static_cast<int64_t>(b));
        break;
      }
    }
  }
  if (involved.size() <= 1) {
    return;
  }
  std::vector<int64_t> block_ids(involved.begin(), involved.end());
  auto merged = merge_blocks(block_ids, false);
  replace_blocks_with_merged(block_ids, std::get<0>(merged), std::get<1>(merged));
  invalidate_cache();
}

ProductDenseState ProductState::merged_dense() const {
  if (merged_cache_.has_value()) {
    return clone_dense(*merged_cache_);
  }

  std::vector<int64_t> all_ids;
  all_ids.reserve(blocks_.size());
  for (size_t i = 0; i < blocks_.size(); ++i) {
    all_ids.push_back(static_cast<int64_t>(i));
  }

  const auto prod_indices = prod_sum_indices();
  if (prod_indices.empty()) {
    auto merged = merge_blocks(all_ids, true);
    merged_cache_ = clone_dense(std::get<0>(merged));
    return clone_dense(*merged_cache_);
  }

  auto merged_no_prob = merge_blocks(all_ids, false);
  const bool all_pure = std::all_of(blocks_.begin(), blocks_.end(), is_pure_state);
  auto dense = all_pure ? dense_ket(std::get<0>(merged_no_prob))
                        : dense_density_matrix(std::get<0>(merged_no_prob));

  auto coeff = prob_.joint(prod_indices);
  auto view_shape = coeff.sizes().vec();
  view_shape.insert(view_shape.end(),
                    static_cast<size_t>(dense.dim() - coeff.dim()), 1);
  dense = dense * coeff.view(view_shape);

  std::vector<int64_t> sum_axes;
  sum_axes.reserve(prod_indices.size());
  for (auto idx : prod_indices) {
    sum_axes.push_back(static_cast<int64_t>(data_batch_dim_.size()) + idx);
  }
  std::sort(sum_axes.begin(), sum_axes.end(), std::greater<int64_t>());
  for (auto axis : sum_axes) {
    dense = dense.sum(axis);
  }

  auto merged = make_dense_from_data(dense, system_dim(), !all_pure);
  const auto classical = classical_indices();
  if (!classical.empty()) {
    auto cur_batch = dense_batch_dim(merged);
    if (static_cast<int64_t>(cur_batch.size()) >=
        static_cast<int64_t>(classical.size())) {
      cur_batch.resize(cur_batch.size() - classical.size());
      dense_set_batch_dim(merged, cur_batch);
    }
    auto &merged_prob = dense_prob(merged);
    for (auto idx : classical) {
      merged_prob.append(prob_.list()[static_cast<size_t>(idx)],
                         false, c10::nullopt, c10::nullopt,
                         false);
    }
  }
  merged_cache_ = clone_dense(merged);
  return merged;
}

int64_t ProductState::normalize_classical_prob_idx(int64_t prob_idx) const {
  const auto classical = classical_indices();
  const auto n = static_cast<int64_t>(classical.size());
  if (n == 0) {
    throw std::runtime_error(
        "This ProductState has no classical probability dimensions.");
  }
  auto normalized = prob_idx < 0 ? (n + prob_idx) : prob_idx;
  if (normalized < 0 || normalized >= n) {
    throw std::runtime_error("Classical probability index out of range.");
  }
  return normalized;
}

std::vector<int64_t> ProductState::normalize_classical_prob_indices(
    const std::vector<int64_t> &prob_idx) const {
  const auto classical = classical_indices();
  if (classical.empty()) {
    if (prob_idx.empty()) {
      return {};
    }
    throw std::runtime_error(
        "This ProductState has no classical probability dimensions.");
  }
  if (prob_idx.empty()) {
    return classical;
  }
  std::vector<int64_t> normalized;
  normalized.reserve(prob_idx.size());
  for (auto idx : prob_idx) {
    normalized.push_back(normalize_classical_prob_idx(idx));
  }
  std::sort(normalized.begin(), normalized.end());
  std::vector<int64_t> out;
  out.reserve(normalized.size());
  for (auto idx : normalized) {
    out.push_back(classical[static_cast<size_t>(idx)]);
  }
  return out;
}

ProductDenseState ProductState::prob_select_dense(
    const torch::Tensor &outcome_idx, int64_t prob_idx) const {
  const auto classical = classical_indices();
  const auto mapped = normalize_classical_prob_idx(prob_idx);
  auto merged = merged_dense();
  return dense_prob_select(merged, outcome_idx,
                           mapped - static_cast<int64_t>(classical.size()));
}

ProductDenseState ProductState::expec_state_dense(
    const std::vector<int64_t> &classical_prob_idx) const {
  const auto classical = classical_indices();
  if (classical.empty()) {
    return clone_dense(merged_dense());
  }
  std::map<int64_t, int64_t> idx_map;
  for (size_t i = 0; i < classical.size(); ++i) {
    idx_map[classical[i]] = static_cast<int64_t>(i);
  }
  std::vector<int64_t> mapped;
  mapped.reserve(classical_prob_idx.size());
  for (auto idx : classical_prob_idx) {
    auto it = idx_map.find(idx);
    if (it == idx_map.end()) {
      throw std::runtime_error(
          "expec_state_dense received a non-classical probability index.");
    }
    mapped.push_back(it->second);
  }
  return dense_expec_state(merged_dense(), mapped);
}

ProductDenseState ProductState::export_block_dense() const {
  if (blocks_.size() != 1) {
    throw std::runtime_error(
        "export_block_dense only supports ProductState with exactly one block.");
  }
  return clone_dense(merged_dense());
}

torch::Tensor ProductState::ket() const { return dense_ket(merged_dense()); }

torch::Tensor ProductState::density_matrix() const {
  return dense_density_matrix(merged_dense());
}

torch::Tensor ProductState::expec_val(const torch::Tensor &obs,
                                      const std::vector<int64_t> &sys_idx) const {
  auto st = clone();
  st.ensure_merged_for_indices(sys_idx, true);
  auto mapping = st.global_to_block_local(sys_idx);
  return dense_expec_val(st.blocks_[static_cast<size_t>(std::get<0>(mapping))],
                         obs, std::get<1>(mapping));
}

torch::Tensor ProductState::expec_val_product(
    const std::vector<torch::Tensor> &obs_list,
    const std::vector<std::vector<int64_t>> &sys_idx_list) const {
  auto st = clone();
  const auto flat = flatten_nested(sys_idx_list);
  st.ensure_merged_for_indices(flat, true);
  auto mapping = st.global_to_block_local(flat);
  const auto &local_flat = std::get<1>(mapping);
  std::map<int64_t, int64_t> g2l;
  for (size_t i = 0; i < flat.size(); ++i) {
    g2l[flat[i]] = local_flat[i];
  }
  std::vector<std::vector<int64_t>> local_sys_idx;
  local_sys_idx.reserve(sys_idx_list.size());
  for (const auto &group : sys_idx_list) {
    std::vector<int64_t> g;
    g.reserve(group.size());
    for (auto idx : group) {
      g.push_back(g2l[idx]);
    }
    local_sys_idx.push_back(g);
  }
  return dense_expec_val_product(
      st.blocks_[static_cast<size_t>(std::get<0>(mapping))], obs_list,
      local_sys_idx);
}

torch::Tensor ProductState::expec_val_pauli_terms(
    const std::vector<std::string> &pauli_words_r,
    const std::vector<std::vector<int64_t>> &sites) const {
  if (pauli_words_r.size() != sites.size()) {
    throw std::runtime_error("pauli_words_r and sites must have same length.");
  }
  if (pauli_words_r.empty()) {
    auto options = blocks_.empty() ? torch::TensorOptions().dtype(torch::kComplexFloat)
                                   : dense_data(blocks_.front()).options();
    const auto n = blocks_.empty() ? int64_t{1} : dense_data(blocks_.front()).size(0);
    return torch::empty({0, n}, options);
  }

  auto st = clone();
  std::vector<int64_t> flat;
  for (const auto &site : sites) {
    flat.insert(flat.end(), site.begin(), site.end());
  }
  st.ensure_merged_for_indices(flat, true);
  auto mapping = st.global_to_block_local(flat);
  const auto &local_flat = std::get<1>(mapping);
  std::map<int64_t, int64_t> g2l;
  for (size_t i = 0; i < flat.size(); ++i) {
    g2l[flat[i]] = local_flat[i];
  }

  std::vector<std::vector<int64_t>> local_sites;
  local_sites.reserve(sites.size());
  for (const auto &site : sites) {
    std::vector<int64_t> local_site;
    local_site.reserve(site.size());
    for (auto idx : site) {
      local_site.push_back(g2l[idx]);
    }
    local_sites.push_back(local_site);
  }

  return dense_expec_val_pauli_terms(
      st.blocks_[static_cast<size_t>(std::get<0>(mapping))], pauli_words_r,
      local_sites);
}

ProductState ProductState::to(c10::optional<torch::Dtype> dtype,
                              c10::optional<torch::Device> device) const {
  auto st = clone();
  for (auto &block : st.blocks_) {
    std::visit([&](auto &s) { s.to(dtype, device); }, block);
  }
  std::vector<torch::Tensor> new_prob;
  new_prob.reserve(prob_.size());
  for (const auto &p : prob_.clone_list()) {
    auto q = p;
    if (dtype.has_value() || device.has_value()) {
      auto opts = q.options();
      if (dtype.has_value()) {
        opts = opts.dtype(*dtype);
      }
      if (device.has_value()) {
        opts = opts.device(*device);
      }
      q = q.to(opts);
    }
    new_prob.push_back(q);
  }
  return ProductState(st.blocks_, st.block_indices_, new_prob, st.roles_, st.keep_dim_);
}

void ProductState::normalize() {
  invalidate_cache();
  for (auto &block : blocks_) {
    dense_normalize(block);
  }
}

void ProductState::add_probability(const torch::Tensor &prob) {
  invalidate_cache();
  if (!prod_sum_indices().empty()) {
    throw std::runtime_error(
        "add_probability is not supported for prod_sum ProductState.");
  }
  const auto num_outcomes = prob.numel();
  for (auto &st : blocks_) {
    auto cur_bd = dense_batch_dim(st);
    auto new_bd = cur_bd;
    new_bd.push_back(num_outcomes);
    const auto d = dense_dim(st);
    auto data = dense_data(st);
    if (is_pure_state(st)) {
      data = data.unsqueeze(-2).repeat({1, num_outcomes, 1}).view({-1, d});
    } else {
      data = data.unsqueeze(-3).repeat({1, num_outcomes, 1, 1}).view({-1, d, d});
    }
    dense_set_data(st, data);
    dense_set_batch_dim(st, new_bd);
  }
  prob_.append(prob, false, c10::nullopt, c10::nullopt,
               true);
  roles_.push_back(kRoleClassical);
}

void ProductState::evolve(const torch::Tensor &unitary,
                          const std::vector<int64_t> &sys_idx, bool on_batch) {
  invalidate_cache();
  ensure_merged_for_indices(sys_idx, on_batch);
  auto mapping = global_to_block_local(sys_idx);
  auto b = std::get<0>(mapping);
  auto local = std::get<1>(mapping);
  auto &block = blocks_[static_cast<size_t>(b)];
  dense_evolve_with_broadcast(block, unitary, local, on_batch);
  sync_data_batch_dim();
}

void ProductState::evolve_many(
    const torch::Tensor &unitary,
    const std::vector<std::vector<int64_t>> &sys_idx_list,
    bool on_batch) {
  if (sys_idx_list.empty()) {
    return;
  }

  invalidate_cache();
  std::vector<int64_t> cached_sys_idx;
  std::vector<int64_t> cached_local_idx;
  int64_t cached_block = -1;
  bool has_cached = false;
  for (const auto &sys_idx : sys_idx_list) {
    int64_t block_id = -1;
    const std::vector<int64_t> *local = nullptr;
    if (has_cached && sys_idx == cached_sys_idx) {
      block_id = cached_block;
      local = &cached_local_idx;
    } else {
      ensure_merged_for_indices(sys_idx, on_batch);
      auto mapping = global_to_block_local(sys_idx);
      block_id = std::get<0>(mapping);
      cached_local_idx = std::get<1>(mapping);
      cached_sys_idx = sys_idx;
      cached_block = block_id;
      has_cached = true;
      local = &cached_local_idx;
    }
    auto &block = blocks_[static_cast<size_t>(block_id)];
    dense_evolve_with_broadcast(block, unitary, *local, on_batch);
  }
  sync_data_batch_dim();
}

void ProductState::evolve_many_batched(
    const std::vector<torch::Tensor> &unitary_groups,
    const std::vector<std::vector<std::vector<int64_t>>> &sys_idx_groups,
    bool on_batch) {
  if (unitary_groups.size() != sys_idx_groups.size()) {
    throw std::runtime_error(
        "evolve_many_batched: unitary_groups and sys_idx_groups size mismatch.");
  }
  if (unitary_groups.empty()) {
    return;
  }

  invalidate_cache();
  std::vector<int64_t> cached_sys_idx;
  std::vector<int64_t> cached_local_idx;
  int64_t cached_block = -1;
  bool has_cached = false;
  auto apply_one = [&](const torch::Tensor &u,
                       const std::vector<int64_t> &sys_idx) {
    int64_t block_id = -1;
    const std::vector<int64_t> *local = nullptr;
    if (has_cached && sys_idx == cached_sys_idx) {
      block_id = cached_block;
      local = &cached_local_idx;
    } else {
      ensure_merged_for_indices(sys_idx, on_batch);
      auto mapping = global_to_block_local(sys_idx);
      block_id = std::get<0>(mapping);
      cached_local_idx = std::get<1>(mapping);
      cached_sys_idx = sys_idx;
      cached_block = block_id;
      has_cached = true;
      local = &cached_local_idx;
    }
    auto &block = blocks_[static_cast<size_t>(block_id)];
    dense_evolve_with_broadcast(block, u, *local, on_batch);
  };

  for (size_t g = 0; g < unitary_groups.size(); ++g) {
    const auto &unitary = unitary_groups[g];
    const auto &idx_list = sys_idx_groups[g];
    if (idx_list.empty()) {
      continue;
    }
    if (unitary.dim() == 2) {
      for (const auto &idx : idx_list) {
        apply_one(unitary, idx);
      }
      continue;
    }
    if (unitary.dim() < 3) {
      throw std::runtime_error(
          "evolve_many_batched: unitary must be 2D or batched with leading dim.");
    }
    if (static_cast<int64_t>(idx_list.size()) != unitary.size(0)) {
      throw std::runtime_error(
          "evolve_many_batched: leading dim of unitary must match sys_idx_list length.");
    }
    for (size_t i = 0; i < idx_list.size(); ++i) {
      auto u_i = unitary.select(0, static_cast<int64_t>(i));
      apply_one(u_i, idx_list[i]);
    }
  }
  sync_data_batch_dim();
}

void ProductState::evolve_keep_dim(const torch::Tensor &unitary,
                                   const std::vector<int64_t> &sys_idx,
                                   bool on_batch) {
  invalidate_cache();
  ensure_merged_for_indices(sys_idx, on_batch);
  auto mapping = global_to_block_local(sys_idx);
  auto &block = blocks_[static_cast<size_t>(std::get<0>(mapping))];
  dense_evolve_keep_dim(block, unitary, std::get<1>(mapping), on_batch);
  if (on_batch) {
    sync_data_batch_dim();
  }
}

void ProductState::evolve_ctrl(
    const torch::Tensor &unitary, int64_t index,
    const std::vector<std::vector<int64_t>> &sys_idx, bool on_batch) {
  invalidate_cache();
  const auto flat = flatten_nested(sys_idx);
  ensure_merged_for_indices(flat, true);
  auto mapping = global_to_block_local(flat);
  const auto &local_flat = std::get<1>(mapping);
  std::map<int64_t, int64_t> g2l;
  for (size_t i = 0; i < flat.size(); ++i) {
    g2l[flat[i]] = local_flat[i];
  }
  std::vector<std::vector<int64_t>> local_groups;
  local_groups.reserve(sys_idx.size());
  for (const auto &group : sys_idx) {
    std::vector<int64_t> g;
    g.reserve(group.size());
    for (auto idx : group) {
      g.push_back(g2l[idx]);
    }
    local_groups.push_back(g);
  }
  dense_evolve_ctrl(blocks_[static_cast<size_t>(std::get<0>(mapping))], unitary,
                    index, local_groups, on_batch);
  invalidate_cache();
  sync_data_batch_dim();
}

void ProductState::transform(const torch::Tensor &op,
                             const std::vector<int64_t> &sys_idx,
                             const std::string &repr_type, bool on_batch) {
  invalidate_cache();
  ensure_merged_for_indices(sys_idx, on_batch);
  auto mapping = global_to_block_local(sys_idx);
  auto &block = blocks_[static_cast<size_t>(std::get<0>(mapping))];
  dense_transform(block, op, std::get<1>(mapping), repr_type, on_batch);
  if (on_batch) {
    sync_data_batch_dim();
  }
}

void ProductState::transform_many(
    const torch::Tensor &op,
    const std::vector<std::vector<int64_t>> &sys_idx_list,
    const std::string &repr_type, bool on_batch) {
  if (sys_idx_list.empty()) {
    return;
  }

  invalidate_cache();
  std::vector<int64_t> cached_sys_idx;
  std::vector<int64_t> cached_local_idx;
  int64_t cached_block = -1;
  bool has_cached = false;
  for (const auto &sys_idx : sys_idx_list) {
    int64_t block_id = -1;
    const std::vector<int64_t> *local = nullptr;
    if (has_cached && sys_idx == cached_sys_idx) {
      block_id = cached_block;
      local = &cached_local_idx;
    } else {
      ensure_merged_for_indices(sys_idx, on_batch);
      auto mapping = global_to_block_local(sys_idx);
      block_id = std::get<0>(mapping);
      cached_local_idx = std::get<1>(mapping);
      cached_sys_idx = sys_idx;
      cached_block = block_id;
      has_cached = true;
      local = &cached_local_idx;
    }
    auto &block = blocks_[static_cast<size_t>(block_id)];
    dense_transform(block, op, *local, repr_type, on_batch);
  }
  if (on_batch) {
    sync_data_batch_dim();
  }
}

std::tuple<torch::Tensor, ProductState>
ProductState::measure(const torch::Tensor &measure_op,
                      const std::vector<int64_t> &sys_idx) const {
  auto new_state = clone();
  new_state.ensure_merged_for_indices(sys_idx, true);
  auto mapping = new_state.global_to_block_local(sys_idx);
  const auto b = std::get<0>(mapping);
  const auto local = std::get<1>(mapping);
  auto &block = new_state.blocks_[static_cast<size_t>(b)];
  const auto prior_batch = dense_batch_dim(block);

  auto measure_out = dense_measure(block, measure_op, local);
  auto prob = std::get<0>(measure_out);
  auto collapsed = std::get<1>(measure_out);

  auto new_prob = new_state.prob_.clone();
  new_prob.append(prob, false, c10::nullopt, c10::nullopt,
                  true);

  dense_prob(collapsed).clear();
  auto target_batch = prior_batch;
  target_batch.push_back(prob.size(-1));
  dense_set_batch_dim(collapsed, target_batch);
  for (size_t i = 0; i < new_state.blocks_.size(); ++i) {
    if (static_cast<int64_t>(i) == b) {
      continue;
    }
    expand_dense_to_batch(new_state.blocks_[i], target_batch);
  }
  new_state.blocks_[static_cast<size_t>(b)] = collapsed;

  auto roles = new_state.roles_;
  roles.push_back(kRoleClassical);
  auto out = ProductState(new_state.blocks_, new_state.block_indices_,
                          new_prob.clone_list(), roles, keep_dim_);
  out.sync_data_batch_dim();
  return {prob, out};
}

std::tuple<torch::Tensor, ProductState>
ProductState::measure_many(
    const torch::Tensor &measure_op,
    const std::vector<std::vector<int64_t>> &sys_idx_list) const {
  const auto num_measure = measure_op.size(0);
  if (static_cast<int64_t>(sys_idx_list.size()) != num_measure) {
    throw std::runtime_error(
        "measure_many requires measure_op.size(0) == len(sys_idx_list).");
  }
  auto out = clone();
  const auto prob_start = out.prob_.size();
  for (int64_t i = 0; i < num_measure; ++i) {
    auto op_i = measure_op.select(0, i);
    auto measured = out.measure(op_i, sys_idx_list[static_cast<size_t>(i)]);
    out = std::get<1>(measured);
  }
  std::vector<int64_t> idx;
  for (int64_t i = prob_start; i < out.prob_.size(); ++i) {
    idx.push_back(i);
  }
  return {out.prob_.joint(idx), out};
}

std::tuple<torch::Tensor, std::optional<ProductState>>
ProductState::measure_by_state(const PureState &measure_basis,
                               const std::vector<int64_t> &sys_idx,
                               bool keep_rest) const {
  auto new_state = clone();
  new_state.ensure_merged_for_indices(sys_idx, true);
  auto mapping = new_state.global_to_block_local(sys_idx);
  const auto b = std::get<0>(mapping);
  const auto local = std::get<1>(mapping);
  const auto block_global = new_state.block_indices_[static_cast<size_t>(b)];

  auto out_local = dense_measure_by_state(
      new_state.blocks_[static_cast<size_t>(b)], measure_basis, local, keep_rest);
  auto prob = std::get<0>(out_local);
  auto rest_state_opt = std::get<1>(out_local);
  if (!rest_state_opt.has_value()) {
    return {prob, std::nullopt};
  }
  auto rest_state = std::move(*rest_state_opt);

  auto new_prob = new_state.prob_.clone();
  new_prob.append(prob, false, c10::nullopt, c10::nullopt,
                  true);

  ProductDenseState measured_state = measure_basis.clone();
  const auto num_outcomes = prob.size(-1);
  auto target_batch = new_state.batch_dim();
  target_batch.push_back(num_outcomes);
  expand_dense_to_batch(measured_state, target_batch);
  dense_prob(measured_state).clear();

  dense_set_batch_dim(rest_state, target_batch);
  dense_prob(rest_state).clear();

  for (size_t i = 0; i < new_state.blocks_.size(); ++i) {
    if (static_cast<int64_t>(i) == b) {
      continue;
    }
    expand_dense_to_batch(new_state.blocks_[i], target_batch);
  }

  std::vector<int64_t> rest_global;
  for (auto g : block_global) {
    if (std::find(sys_idx.begin(), sys_idx.end(), g) == sys_idx.end()) {
      rest_global.push_back(g);
    }
  }

  std::vector<ProductDenseState> states;
  std::vector<std::vector<int64_t>> idxs;
  for (int64_t i = 0; i < b; ++i) {
    states.push_back(new_state.blocks_[static_cast<size_t>(i)]);
    idxs.push_back(new_state.block_indices_[static_cast<size_t>(i)]);
  }
  states.push_back(measured_state);
  idxs.push_back(sys_idx);
  if (!rest_global.empty()) {
    states.push_back(rest_state);
    idxs.push_back(rest_global);
  }
  for (size_t i = static_cast<size_t>(b + 1); i < new_state.blocks_.size(); ++i) {
    states.push_back(new_state.blocks_[i]);
    idxs.push_back(new_state.block_indices_[i]);
  }

  auto roles = new_state.roles_;
  roles.push_back(kRoleClassical);
  auto out = ProductState(states, idxs, new_prob.clone_list(), roles, keep_dim_);
  out.sync_data_batch_dim();
  return {prob, std::optional<ProductState>(std::move(out))};
}

std::tuple<torch::Tensor, std::optional<ProductState>>
ProductState::measure_by_product_state(const ProductState &measure_basis,
                                       const std::vector<int64_t> &sys_idx,
                                       bool keep_rest) const {
  auto new_state = clone();
  new_state.ensure_merged_for_indices(sys_idx, true);
  auto mapping = new_state.global_to_block_local(sys_idx);
  const auto b = std::get<0>(mapping);
  const auto local = std::get<1>(mapping);
  const auto block_global = new_state.block_indices_[static_cast<size_t>(b)];

  std::vector<std::vector<int64_t>> mapped_local_idx;
  int64_t flat = 0;
  for (const auto &group : measure_basis.block_indices_) {
    if (flat + static_cast<int64_t>(group.size()) >
        static_cast<int64_t>(local.size())) {
      throw std::runtime_error(
          "measure_by_product_state: measure basis does not match measured systems.");
    }
    std::vector<int64_t> mapped_group;
    mapped_group.reserve(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
      mapped_group.push_back(local[static_cast<size_t>(flat + static_cast<int64_t>(i))]);
    }
    flat += static_cast<int64_t>(group.size());
    mapped_local_idx.push_back(mapped_group);
  }

  std::vector<torch::Tensor> basis_kets;
  basis_kets.reserve(measure_basis.blocks_.size());
  for (const auto &st : measure_basis.blocks_) {
    if (!is_pure_state(st)) {
      throw std::runtime_error(
          "measure_by_product_state requires pure blocks in measurement basis.");
    }
    basis_kets.push_back(std::get<PureState>(st).ket());
  }

  auto out_local = dense_measure_by_state_product(
      new_state.blocks_[static_cast<size_t>(b)], basis_kets, mapped_local_idx,
      keep_rest);
  auto prob = std::get<0>(out_local);
  auto rest_state_opt = std::get<1>(out_local);
  if (!rest_state_opt.has_value()) {
    return {prob, std::nullopt};
  }
  auto rest_state = std::move(*rest_state_opt);

  auto new_prob = new_state.prob_.clone();
  new_prob.append(prob, false, c10::nullopt, c10::nullopt,
                  true);

  const auto num_outcomes = prob.size(-1);
  auto target_batch = new_state.batch_dim();
  target_batch.push_back(num_outcomes);

  auto measured_blocks = measure_basis.blocks_;
  for (auto &st : measured_blocks) {
    expand_dense_to_batch(st, target_batch);
    dense_prob(st).clear();
  }

  dense_set_batch_dim(rest_state, target_batch);
  dense_prob(rest_state).clear();

  for (size_t i = 0; i < new_state.blocks_.size(); ++i) {
    if (static_cast<int64_t>(i) == b) {
      continue;
    }
    expand_dense_to_batch(new_state.blocks_[i], target_batch);
  }

  std::vector<std::vector<int64_t>> mapped_global_idx;
  flat = 0;
  for (const auto &group : measure_basis.block_indices_) {
    if (flat + static_cast<int64_t>(group.size()) >
        static_cast<int64_t>(sys_idx.size())) {
      throw std::runtime_error(
          "measure_by_product_state: sys_idx length does not match measure basis.");
    }
    std::vector<int64_t> mapped_group;
    mapped_group.reserve(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
      mapped_group.push_back(sys_idx[static_cast<size_t>(flat + static_cast<int64_t>(i))]);
    }
    flat += static_cast<int64_t>(group.size());
    mapped_global_idx.push_back(mapped_group);
  }
  std::vector<int64_t> measured_global = flatten_nested(mapped_global_idx);
  std::vector<int64_t> rest_global;
  for (auto g : block_global) {
    if (std::find(measured_global.begin(), measured_global.end(), g) ==
        measured_global.end()) {
      rest_global.push_back(g);
    }
  }

  std::vector<ProductDenseState> states;
  std::vector<std::vector<int64_t>> idxs;
  for (int64_t i = 0; i < b; ++i) {
    states.push_back(new_state.blocks_[static_cast<size_t>(i)]);
    idxs.push_back(new_state.block_indices_[static_cast<size_t>(i)]);
  }
  for (size_t i = 0; i < measured_blocks.size(); ++i) {
    states.push_back(measured_blocks[i]);
    idxs.push_back(mapped_global_idx[i]);
  }
  if (!rest_global.empty()) {
    states.push_back(rest_state);
    idxs.push_back(rest_global);
  }
  for (size_t i = static_cast<size_t>(b + 1); i < new_state.blocks_.size(); ++i) {
    states.push_back(new_state.blocks_[i]);
    idxs.push_back(new_state.block_indices_[i]);
  }

  auto roles = new_state.roles_;
  roles.push_back(kRoleClassical);
  auto out = ProductState(states, idxs, new_prob.clone_list(), roles, keep_dim_);
  out.sync_data_batch_dim();
  return {prob, std::optional<ProductState>(std::move(out))};
}

ProductState ProductState::trace(const std::vector<int64_t> &trace_idx) const {
  std::set<int64_t> trace_set(trace_idx.begin(), trace_idx.end());
  std::vector<ProductDenseState> new_states;
  std::vector<std::vector<int64_t>> new_idxs;
  for (size_t b = 0; b < blocks_.size(); ++b) {
    const auto &idxs = block_indices_[b];
    std::vector<int64_t> local_to_trace;
    for (size_t i = 0; i < idxs.size(); ++i) {
      if (trace_set.find(idxs[i]) != trace_set.end()) {
        local_to_trace.push_back(static_cast<int64_t>(i));
      }
    }
    if (local_to_trace.empty()) {
      new_states.push_back(clone_dense(blocks_[b]));
      new_idxs.push_back(idxs);
      continue;
    }
    if (local_to_trace.size() == idxs.size()) {
      continue;
    }
    auto reduced = dense_trace(blocks_[b], local_to_trace);
    std::vector<int64_t> remain;
    for (auto g : idxs) {
      if (trace_set.find(g) == trace_set.end()) {
        remain.push_back(g);
      }
    }
    new_states.push_back(reduced);
    new_idxs.push_back(remain);
  }

  std::vector<int64_t> kept;
  for (const auto &idxs : new_idxs) {
    kept.insert(kept.end(), idxs.begin(), idxs.end());
  }
  std::sort(kept.begin(), kept.end());
  std::map<int64_t, int64_t> mapping;
  for (size_t i = 0; i < kept.size(); ++i) {
    mapping[kept[i]] = static_cast<int64_t>(i);
  }
  for (auto &idxs : new_idxs) {
    for (auto &g : idxs) {
      g = mapping[g];
    }
  }
  return ProductState(new_states, new_idxs, prob_.clone_list(), roles_, keep_dim_);
}

ProductState ProductState::reset(const std::vector<int64_t> &reset_idx,
                                 const ProductDenseState &replace_state) const {
  auto new_state = clone();
  new_state.ensure_merged_for_indices(reset_idx, true);
  auto mapping = new_state.global_to_block_local(reset_idx);
  const auto b = std::get<0>(mapping);
  const auto local = std::get<1>(mapping);
  auto block_idx = new_state.block_indices_[static_cast<size_t>(b)];
  auto replace_block = clone_dense(replace_state);

  if (local.size() == block_idx.size()) {
    new_state.blocks_[static_cast<size_t>(b)] = replace_block;
  } else {
    new_state.blocks_[static_cast<size_t>(b)] =
        dense_trace(new_state.blocks_[static_cast<size_t>(b)], local);
    std::vector<int64_t> remain;
    for (auto idx : block_idx) {
      if (std::find(reset_idx.begin(), reset_idx.end(), idx) == reset_idx.end()) {
        remain.push_back(idx);
      }
    }
    new_state.block_indices_[static_cast<size_t>(b)] = remain;
    new_state.blocks_.push_back(replace_block);
    new_state.block_indices_.push_back(reset_idx);
  }
  return new_state;
}

ProductState ProductState::reset_with_product(
    const std::vector<int64_t> &reset_idx, const ProductState &replace_state) const {
  return reset(reset_idx, replace_state.merged_dense());
}

ProductState ProductState::transpose(
    const std::vector<int64_t> &transpose_idx) const {
  auto new_state = clone();
  new_state.ensure_merged_for_indices(transpose_idx, true);
  auto mapping = new_state.global_to_block_local(transpose_idx);
  new_state.blocks_[static_cast<size_t>(std::get<0>(mapping))] =
      dense_transpose(new_state.blocks_[static_cast<size_t>(std::get<0>(mapping))],
                      std::get<1>(mapping));
  return new_state;
}

ProductState ProductState::permute(const std::vector<int64_t> &target_seq) const {
  const auto n = num_systems();
  auto sorted = target_seq;
  std::sort(sorted.begin(), sorted.end());
  if (sorted != arange_vec(n)) {
    throw std::runtime_error("target_seq must be a permutation of [0..n-1].");
  }
  if (target_seq == arange_vec(n)) {
    return clone();
  }
  if (keep_dim_) {
    auto dense = dense_permute(merged_dense(), target_seq);
    return ProductState({dense}, {arange_vec(n)}, prob_.clone_list(), roles_,
                        keep_dim_);
  }
  std::map<int64_t, int64_t> old_of_new;
  for (int64_t i = 0; i < n; ++i) {
    old_of_new[i] = target_seq[static_cast<size_t>(i)];
  }
  std::map<int64_t, int64_t> new_of_old;
  for (const auto &kv : old_of_new) {
    new_of_old[kv.second] = kv.first;
  }
  auto new_idxs = block_indices_;
  for (auto &idxs : new_idxs) {
    for (auto &g : idxs) {
      g = new_of_old[g];
    }
  }
  std::vector<ProductDenseState> states;
  states.reserve(blocks_.size());
  for (const auto &s : blocks_) {
    states.push_back(clone_dense(s));
  }
  return ProductState(states, new_idxs, prob_.clone_list(), roles_, keep_dim_);
}

ProductState ProductState::kron_dense(const ProductDenseState &other) const {
  if (!prob_.empty() && dense_prob_const(other).size() != 0) {
    throw std::runtime_error("Cannot kron two probabilistic states.");
  }
  auto out = clone();
  const auto shift = out.num_systems();
  out.blocks_.push_back(clone_dense(other));
  std::vector<int64_t> idxs;
  const auto other_n = static_cast<int64_t>(dense_system_dim(other).size());
  for (int64_t i = 0; i < other_n; ++i) {
    idxs.push_back(i + shift);
  }
  out.block_indices_.push_back(idxs);
  if (out.prob_.empty() && dense_prob_const(other).size() != 0) {
    out.prob_ = dense_prob_const(other).clone();
    out.roles_ = std::vector<std::string>(static_cast<size_t>(out.prob_.size()),
                                          kRoleClassical);
  }
  out.sync_data_batch_dim();
  return out;
}

ProductState ProductState::kron_product(const ProductState &other) const {
  if (!prob_.empty() && !other.prob_.empty()) {
    throw std::runtime_error("Cannot kron two probabilistic states.");
  }
  auto out = clone();
  const auto shift = out.num_systems();
  for (size_t i = 0; i < other.blocks_.size(); ++i) {
    out.blocks_.push_back(clone_dense(other.blocks_[i]));
    std::vector<int64_t> idxs;
    idxs.reserve(other.block_indices_[i].size());
    for (auto g : other.block_indices_[i]) {
      idxs.push_back(g + shift);
    }
    out.block_indices_.push_back(idxs);
  }
  if (out.prob_.empty() && !other.prob_.empty()) {
    out.prob_ = other.prob_.clone();
    out.roles_ = other.roles_;
  }
  out.keep_dim_ = out.keep_dim_ || other.keep_dim_;
  out.sync_data_batch_dim();
  return out;
}

}