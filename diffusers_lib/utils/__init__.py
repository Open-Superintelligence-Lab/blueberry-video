# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .deprecation_utils import deprecate
from .import_utils import (
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xla_version,
    is_xformers_available,
    is_torch_version,
    DummyObject,
)
from . import logging
from .logging import get_logger
from .torch_utils import maybe_allow_in_graph
from .peft_utils import scale_lora_layers, unscale_lora_layers
from .outputs import BaseOutput
from .constants import CONFIG_NAME, HUGGINGFACE_CO_RESOLVE_ENDPOINT
from .hub_utils import extract_commit_hash, http_user_agent
from .pil_utils import PIL_INTERPOLATION

# Stubs
USE_PEFT_BACKEND = False
MIN_PEFT_VERSION = "0.6.0"

__all__ = [
    "deprecate",
    "is_torch_npu_available",
    "is_torch_xla_available",
    "is_torch_xla_version",
    "is_xformers_available",
    "is_torch_version",
    "DummyObject",
    "logging",
    "get_logger",
    "maybe_allow_in_graph",
    "scale_lora_layers",
    "unscale_lora_layers",
    "BaseOutput",
    "CONFIG_NAME",
    "HUGGINGFACE_CO_RESOLVE_ENDPOINT",
    "extract_commit_hash",
    "http_user_agent",
    "PIL_INTERPOLATION",
    "USE_PEFT_BACKEND",
    "MIN_PEFT_VERSION",
]

