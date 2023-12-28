# Derived from https://github.com/google-research/long-range-arena
# by Lucas Dax Lingle.
#
# Copyright 2022 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration and hyperparameter sweeps."""
import ml_collections

from lra_benchmarks.listops.configs import base_listops_config


def get_config():
    """Get the default hyperparameter configuration."""
    config = base_listops_config.get_config()
    config.model_type = "transformer_tlb"
    config.model = ml_collections.ConfigDict()
    config.learning_rate = 0.05 / 4.0
    config.model.self_to_cross_ratio_input_updater = 2
    config.model.num_cross_layers_input_updater = 1
    config.model.num_cross_layers_state_updater = 1
    config.model.num_state_tokens = 100
    config.model.block_size = 100
    config.model.use_global_pos_encoding = False
    return config


def get_hyper(hyper):
    return hyper.product([])
