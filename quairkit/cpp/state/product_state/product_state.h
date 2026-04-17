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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "../mixed_state/mixed_state.h"
#include "../pure_state/pure_state.h"
#include "../state_base/probability_data.h"

namespace quairkit_cpp {

using ProductDenseState = std::variant<PureState, MixedState>;

struct ProductExportBlock {
  ProductDenseState state;
  std::vector<int64_t> global_indices;
};

class ProductState {
public:
  static constexpr const char *kRoleClassical = "classical";
  static constexpr const char *kRoleProdSum = "prod_sum";

  ProductState(const std::vector<ProductDenseState> &blocks,
               const std::vector<std::vector<int64_t>> &block_indices,
               const std::vector<torch::Tensor> &probability = {},
               const std::vector<std::string> &roles = {},
               bool keep_dim = false);

  ProductState clone() const;

  std::string backend() const;
  int64_t num_systems() const;
  std::vector<int64_t> system_dim() const;
  std::vector<int64_t> batch_dim() const;
  std::vector<std::string> roles() const { return roles_; }
  std::vector<torch::Tensor> prob_list() const { return prob_.list(); }
  std::vector<int64_t> classical_indices() const;
  std::vector<int64_t> prod_sum_indices() const;
  bool keep_dim() const { return keep_dim_; }
  void set_keep_dim(bool keep_dim) { keep_dim_ = keep_dim; }

  int64_t num_blocks() const { return static_cast<int64_t>(blocks_.size()); }
  std::vector<std::vector<int64_t>> block_layout() const { return block_indices_; }
  std::vector<ProductExportBlock> export_blocks(bool clone_state = true) const;

  torch::Tensor probability() const;

  ProductDenseState merged_dense() const;
  ProductDenseState prob_select_dense(const torch::Tensor &outcome_idx,
                                      int64_t prob_idx = -1) const;
  ProductDenseState expec_state_dense(
      const std::vector<int64_t> &classical_prob_idx) const;
  ProductDenseState export_block_dense() const;

  torch::Tensor ket() const;
  torch::Tensor density_matrix() const;
  torch::Tensor expec_val(const torch::Tensor &obs,
                          const std::vector<int64_t> &sys_idx) const;
  torch::Tensor expec_val_product(
      const std::vector<torch::Tensor> &obs_list,
      const std::vector<std::vector<int64_t>> &sys_idx_list) const;
  torch::Tensor expec_val_pauli_terms(
      const std::vector<std::string> &pauli_words_r,
      const std::vector<std::vector<int64_t>> &sites) const;

  ProductState to(c10::optional<torch::Dtype> dtype,
                  c10::optional<torch::Device> device) const;
  void normalize();
  void add_probability(const torch::Tensor &prob);

  void evolve(const torch::Tensor &unitary, const std::vector<int64_t> &sys_idx,
              bool on_batch = true);
  void evolve_many(const torch::Tensor &unitary,
                   const std::vector<std::vector<int64_t>> &sys_idx_list,
                   bool on_batch = true);
  void evolve_many_batched(
      const std::vector<torch::Tensor> &unitary_groups,
      const std::vector<std::vector<std::vector<int64_t>>> &sys_idx_groups,
      bool on_batch = true);
  void evolve_keep_dim(const torch::Tensor &unitary,
                       const std::vector<int64_t> &sys_idx,
                       bool on_batch = true);
  void evolve_ctrl(const torch::Tensor &unitary, int64_t index,
                   const std::vector<std::vector<int64_t>> &sys_idx,
                   bool on_batch = true);
  void transform(const torch::Tensor &op, const std::vector<int64_t> &sys_idx,
                 const std::string &repr_type, bool on_batch = true);
  void transform_many(const torch::Tensor &op,
                      const std::vector<std::vector<int64_t>> &sys_idx_list,
                      const std::string &repr_type, bool on_batch = true);

  std::tuple<torch::Tensor, ProductState>
  measure(const torch::Tensor &measure_op,
          const std::vector<int64_t> &sys_idx) const;
  std::tuple<torch::Tensor, ProductState>
  measure_many(const torch::Tensor &measure_op,
               const std::vector<std::vector<int64_t>> &sys_idx_list) const;
  std::tuple<torch::Tensor, std::optional<ProductState>>
  measure_by_state(const PureState &measure_basis,
                   const std::vector<int64_t> &sys_idx, bool keep_rest) const;
  std::tuple<torch::Tensor, std::optional<ProductState>>
  measure_by_product_state(const ProductState &measure_basis,
                           const std::vector<int64_t> &sys_idx,
                           bool keep_rest) const;

  ProductState trace(const std::vector<int64_t> &trace_idx) const;
  ProductState reset(const std::vector<int64_t> &reset_idx,
                     const ProductDenseState &replace_state) const;
  ProductState reset_with_product(const std::vector<int64_t> &reset_idx,
                                  const ProductState &replace_state) const;
  ProductState transpose(const std::vector<int64_t> &transpose_idx) const;
  ProductState permute(const std::vector<int64_t> &target_seq) const;

  ProductState kron_dense(const ProductDenseState &other) const;
  ProductState kron_product(const ProductState &other) const;

private:
  ProductDenseState make_dense_from_data(const torch::Tensor &data,
                                         const std::vector<int64_t> &sys_dim,
                                         bool force_mixed) const;

  void invalidate_cache() const;
  void sync_data_batch_dim();

  std::pair<int64_t, std::vector<int64_t>>
  global_to_block_local(const std::vector<int64_t> &sys_idx) const;

  std::pair<ProductDenseState, std::vector<int64_t>>
  merge_blocks(const std::vector<int64_t> &block_ids,
               bool attach_real_prob = false) const;

  void replace_blocks_with_merged(const std::vector<int64_t> &block_ids,
                                  const ProductDenseState &merged_state,
                                  const std::vector<int64_t> &merged_global);
  void ensure_merged_for_indices(const std::vector<int64_t> &sys_idx,
                                 bool on_batch);

  int64_t normalize_classical_prob_idx(int64_t prob_idx) const;
  std::vector<int64_t>
  normalize_classical_prob_indices(const std::vector<int64_t> &prob_idx) const;

  static std::vector<std::string>
  normalize_roles(const std::vector<std::string> &roles,
                  const ProbabilityDataImpl &prob);

  std::vector<ProductDenseState> blocks_;
  std::vector<std::vector<int64_t>> block_indices_;
  ProbabilityDataImpl prob_;
  std::vector<std::string> roles_;
  bool keep_dim_ = false;

  std::vector<int64_t> data_batch_dim_;

  mutable c10::optional<ProductDenseState> merged_cache_;
};

}