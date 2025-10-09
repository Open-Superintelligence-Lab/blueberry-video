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

"""
Minimal implementation of ModelMixin for standalone usage.
This is a simplified stub that provides basic functionality.
"""

import torch
import torch.nn as nn


class ModelMixin(nn.Module):
    """
    Minimal base class for models. Provides basic PyTorch Module functionality.
    """
    
    def __init__(self):
        super().__init__()
    
    @property
    def device(self):
        """Returns the device of the model."""
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        """Returns the dtype of the model."""
        return next(self.parameters()).dtype


def load_state_dict(path, **kwargs):
    """Simple state dict loader."""
    return torch.load(path, **kwargs)
