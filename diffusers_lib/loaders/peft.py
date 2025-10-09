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
Minimal implementation of PeftAdapterMixin for standalone usage.
This is a simplified stub that provides basic PEFT adapter functionality.
"""

import torch.nn as nn


class PeftAdapterMixin:
    """
    Minimal mixin for PEFT adapter support.
    This is a stub implementation for basic compatibility.
    """
    
    def set_adapters(self, *args, **kwargs):
        """Stub for setting adapters."""
        pass
    
    def disable_adapters(self):
        """Stub for disabling adapters."""
        pass
    
    def enable_adapters(self):
        """Stub for enabling adapters."""
        pass
    
    def add_adapter(self, *args, **kwargs):
        """Stub for adding an adapter."""
        pass
