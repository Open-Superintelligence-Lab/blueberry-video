# Diffusers Library Integration

This repository has been made standalone by copying essential files from the `diffusers` library into a local `diffusers_lib/` directory.

## What Was Done

1. **Cloned diffusers repository** - Temporarily cloned the official diffusers repo to extract needed files
2. **Copied essential files** - Only copied the minimal set of files required for the models to function
3. **Updated imports** - Changed all `from diffusers...` imports to `from diffusers_lib...` throughout the codebase
4. **Created stubs** - Simplified complex classes (ModelMixin, PeftAdapterMixin, FromOriginalModelMixin) to minimal implementations
5. **Removed diffusers dependency** - Updated `requirements.txt` to remove the `diffusers` package dependency
6. **Cleaned up** - Removed the cloned diffusers repository

## Files Copied

### Core Structure
- `diffusers_lib/__init__.py` - Main package init with version and core exports
- `diffusers_lib/configuration_utils.py` - Configuration management (ConfigMixin, register_to_config)
- `diffusers_lib/image_processor.py` - Image processing utilities (IPAdapterMaskProcessor)

### Utils Directory (`diffusers_lib/utils/`)
- `__init__.py` - Utils package exports
- `constants.py` - Configuration constants
- `deprecation_utils.py` - Deprecation utilities
- `hub_utils.py` - HuggingFace Hub utilities
- `import_utils.py` - Import checking utilities
- `logging.py` - Logging utilities
- `outputs.py` - Output classes (BaseOutput)
- `peft_utils.py` - PEFT/LoRA utilities
- `pil_utils.py` - PIL/image utilities
- `torch_utils.py` - PyTorch utilities

### Models Directory (`diffusers_lib/models/`)
- `__init__.py` - Models package exports
- `modeling_outputs.py` - Model output classes (Transformer2DModelOutput)
- `modeling_utils.py` - **Simplified stub** - Minimal ModelMixin implementation

### Loaders Directory (`diffusers_lib/loaders/`)
- `__init__.py` - Loaders package exports
- `peft.py` - **Simplified stub** - Minimal PeftAdapterMixin implementation
- `single_file_model.py` - **Simplified stub** - Minimal FromOriginalModelMixin implementation

## What Was Simplified

The following classes were replaced with minimal stub implementations to avoid copying hundreds of dependency files:

1. **ModelMixin** - Now only provides basic device and dtype properties
2. **PeftAdapterMixin** - Now only provides stub methods for adapter management
3. **FromOriginalModelMixin** - Now only provides a stub for from_single_file (raises NotImplementedError)

These simplifications are sufficient for the video generation models in this repository, which primarily use these classes as base classes without requiring their full functionality.

## Dependencies Added

To replace the `diffusers` dependency, the following were added to `requirements.txt`:
- `huggingface_hub>=0.19.0` - Required for model loading utilities
- `typing_extensions>=4.5.0` - Required for type hints in diffusers_lib

## Files Modified

All model files in `models/` directory were updated to import from `diffusers_lib` instead of `diffusers`:
- `models/attention.py`
- `models/attention_processor.py`
- `models/activations.py`
- `models/cogvideox_transformer.py`
- `models/embeddings.py`
- `models/hunyuan_video_transformer.py`
- `models/normalization.py`

## Testing

All imports were verified to work correctly:
```bash
# Test utils
python3 -c "from diffusers_lib.utils import logging; print('Utils import OK')"

# Test models
python3 -c "from models.activations import get_activation; print('Activations import OK')"
python3 -c "from models.cogvideox_transformer import CogVideoXTransformer3DModel; print('CogVideoX transformer import OK')"
python3 -c "from models.hunyuan_video_transformer import HunyuanVideoTransformer3DModel; print('HunyuanVideo transformer import OK')"
```

## Benefits

1. **Standalone** - No longer requires installing the full diffusers library
2. **Minimal** - Only includes the files actually needed
3. **Customizable** - Can modify diffusers_lib code without affecting the global package
4. **Version locked** - Not affected by breaking changes in future diffusers releases

## License

All files in `diffusers_lib/` retain their original Apache 2.0 license from the HuggingFace diffusers project.
Copyright 2025 The HuggingFace Inc. team.

