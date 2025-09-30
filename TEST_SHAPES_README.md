# Tensor Shape Tests for B3tt3r

This document describes the tensor shape validation tests implemented in `test_shapes.py`.

## Overview

The test suite validates input and output tensor dimensions across various components of the B3tt3r 3D reconstruction system using PyTorch. These tests ensure that tensor shapes are correct throughout the pipeline, from image input to 3D point cloud generation.

## Test Coverage

### 1. `test_check_if_same_size`
**Purpose**: Validates the `check_if_same_size` function from `spann3r/dust3r/inference.py`

**What it tests**:
- Correctly identifies when all image pairs have the same size
- Correctly identifies when image pairs have different sizes
- Returns `True` when all images match dimensions
- Returns `False` when any image pair has mismatched dimensions

**Example shapes tested**:
- Same size: All pairs with `(1, 3, 224, 224)` → Returns `True`
- Different sizes: Mix of `(1, 3, 224, 224)` and `(1, 3, 256, 256)` → Returns `False`

---

### 2. `test_patch_embed_shapes`
**Purpose**: Tests shape validations for `PatchEmbedDust3R` from `spann3r/dust3r/patch_embed.py`

**What it tests**:
- Input tensor format: `(B, C, H, W)` where B=batch, C=channels, H=height, W=width
- Height must be divisible by patch_size (16)
- Width must be divisible by patch_size (16)
- Invalid dimensions are properly detected

**Example shapes tested**:
- Valid: `(2, 3, 224, 224)` with patch_size=16 ✓
- Invalid: `(2, 3, 225, 224)` with patch_size=16 ✗ (225 not divisible by 16)

---

### 3. `test_many_ar_patch_embed_shapes`
**Purpose**: Tests `ManyAR_PatchEmbed` which handles non-square aspect ratios

**What it tests**:
- Images must be in landscape mode (W >= H)
- Both dimensions divisible by patch_size
- `true_shape` tensor has correct dimensions `(B, 2)`
- Token dimension calculations are correct
- Handles mix of landscape and portrait orientations

**Example shapes tested**:
- Image: `(4, 3, 224, 336)` - Landscape mode
- true_shape: `(4, 2)` - Contains actual dimensions for each image
- Token dimensions: H=14, W=21, total=294 tokens

---

### 4. `test_find_opt_scaling_shapes`
**Purpose**: Validates tensor shapes for `find_opt_scaling` function in inference.py

**What it tests**:
- All point cloud tensors are 4D: `(B, H, W, 3)`
- Ground truth and prediction shapes match
- Works with pairs of point clouds (pts1 and pts2)

**Example shapes tested**:
- gt_pts1: `(2, 56, 56, 3)`
- pr_pts1: `(2, 56, 56, 3)`
- gt_pts2: `(2, 56, 56, 3)`
- pr_pts2: `(2, 56, 56, 3)`

---

### 5. `test_spann3r_encode_shapes`
**Purpose**: Tests Spann3R model's encode methods from `spann3r/spann3r/model.py`

**What it tests**:
- View dictionaries contain properly shaped 'img' and 'true_shape' tensors
- Image concatenation for paired encoding works correctly
- Batch dimension doubles when concatenating view pairs
- true_shape concatenation maintains proper dimensions

**Example shapes tested**:
- view1['img']: `(2, 3, 224, 224)`
- view1['true_shape']: `(2, 2)`
- Concatenated: `(4, 3, 224, 224)` - Batch doubled

---

### 6. `test_interleave_imgs`
**Purpose**: Tests the `_interleave_imgs` function that interleaves two image batches

**What it tests**:
- Stacking and flattening produces correct interleaved output
- Batch size doubles in the output
- Other dimensions remain unchanged

**Example shapes tested**:
- img1: `(4, 3, 224, 224)`
- img2: `(4, 3, 224, 224)`
- Interleaved: `(8, 3, 224, 224)` - Batch doubled to 8

---

### 7. `test_dpt_output_adapter_shapes`
**Purpose**: Tests DPT output adapter shape calculations from `spann3r/croco/models/dpt_block.py`

**What it tests**:
- Patch calculation: Number of patches from image size and patch size
- Encoder token shape: `(B, N_tokens, embed_dim)`
- Spatial representation: Reshape from `(B, N, C)` to `(B, C, N_H, N_W)`

**Example shapes tested**:
- Image: 224×224, patch_size=16 → 14×14 patches (196 tokens)
- Encoder tokens: `(2, 196, 768)`
- Spatial: `(2, 768, 14, 14)`

---

### 8. `test_transpose_operations`
**Purpose**: Tests transpose operations for landscape/portrait handling

**What it tests**:
- `swapaxes` correctly converts landscape to portrait
- Reverse transpose restores original orientation
- Dimensions are swapped in correct positions (last two dimensions)

**Example shapes tested**:
- Landscape: `(2, 3, 224, 336)`
- Portrait (swapaxes -1, -2): `(2, 3, 336, 224)`
- Back to landscape: `(2, 3, 224, 336)`

---

## Running the Tests

### Prerequisites
Install required dependencies:
```bash
pip install torch torchvision numpy==1.26.4 tqdm scipy roma einops
```

### Run all tests
```bash
python3 test_shapes.py
```

### Expected Output
All 8 tests should pass:
```
================================================================================
TEST SUMMARY: 8 passed, 0 failed out of 8 tests
================================================================================
```

## Key Insights from Tests

1. **Batch Processing**: The system processes images in batches, and batch dimensions are carefully maintained and sometimes doubled (e.g., for symmetric operations).

2. **Patch-based Processing**: Images are divided into patches (typically 16×16), so all dimensions must be divisible by the patch size.

3. **Aspect Ratio Handling**: The system supports both landscape and portrait orientations, using transposes to normalize processing.

4. **4D Point Clouds**: 3D point cloud data is stored as 4D tensors `(B, H, W, 3)` where the last dimension contains x, y, z coordinates.

5. **Shape Tracking**: The `true_shape` tensor `(B, 2)` tracks actual image dimensions when working with variable aspect ratios.

## Debugging Shape Issues

If you encounter shape-related errors in the codebase:

1. Check that input images have dimensions divisible by 16
2. Verify landscape mode for ManyAR_PatchEmbed (width >= height)
3. Ensure batch dimensions match across paired views
4. Confirm 4D tensors for point cloud operations
5. Validate true_shape tensor has shape `(B, 2)`

## References

- `spann3r/dust3r/inference.py` - Main inference functions
- `spann3r/dust3r/patch_embed.py` - Patch embedding with aspect ratio handling
- `spann3r/spann3r/model.py` - Spann3R model architecture
- `spann3r/croco/models/dpt_block.py` - DPT output adapter
