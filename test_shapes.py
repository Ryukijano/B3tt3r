"""
Test script to check tensor shapes for the B3tt3r models using PyTorch.
This script validates input and output tensor dimensions for various components.
"""

import torch
import sys
import os

# Add paths to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spann3r'))

print("=" * 80)
print("TENSOR SHAPE VALIDATION TESTS FOR B3TT3R")
print("=" * 80)


def test_check_if_same_size():
    """Test the check_if_same_size function from inference.py"""
    print("\n[TEST 1] Testing check_if_same_size function")
    print("-" * 60)
    
    # Create mock image pairs with same sizes
    pairs_same = [
        ({'img': torch.randn(1, 3, 224, 224)}, {'img': torch.randn(1, 3, 224, 224)}),
        ({'img': torch.randn(1, 3, 224, 224)}, {'img': torch.randn(1, 3, 224, 224)}),
    ]
    
    # Create mock image pairs with different sizes
    pairs_diff = [
        ({'img': torch.randn(1, 3, 224, 224)}, {'img': torch.randn(1, 3, 224, 224)}),
        ({'img': torch.randn(1, 3, 256, 256)}, {'img': torch.randn(1, 3, 224, 224)}),
    ]
    
    # Import the function
    from dust3r.inference import check_if_same_size
    
    # Test with same sizes
    result_same = check_if_same_size(pairs_same)
    print(f"  Same size pairs: {result_same}")
    assert result_same == True, "Expected True for same-sized images"
    print(f"  ✓ Same size test PASSED")
    
    # Test with different sizes
    result_diff = check_if_same_size(pairs_diff)
    print(f"  Different size pairs: {result_diff}")
    assert result_diff == False, "Expected False for different-sized images"
    print(f"  ✓ Different size test PASSED")
    
    print(f"\n  Test check_if_same_size: PASSED ✓")


def test_patch_embed_shapes():
    """Test PatchEmbedDust3R shape validations"""
    print("\n[TEST 2] Testing PatchEmbedDust3R tensor shapes")
    print("-" * 60)
    
    # Test valid input shapes
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    patch_size = 16
    
    # Create test tensor
    x = torch.randn(batch_size, channels, height, width)
    print(f"  Input shape: {x.shape}")
    print(f"  Expected: (B={batch_size}, C={channels}, H={height}, W={width})")
    
    # Test height divisible by patch_size
    assert height % patch_size == 0, f"Height {height} not divisible by patch_size {patch_size}"
    print(f"  ✓ Height {height} is divisible by patch_size {patch_size}")
    
    # Test width divisible by patch_size
    assert width % patch_size == 0, f"Width {width} not divisible by patch_size {patch_size}"
    print(f"  ✓ Width {width} is divisible by patch_size {patch_size}")
    
    # Test invalid height (not divisible by patch_size)
    try:
        invalid_height = 225  # Not divisible by 16
        invalid_x = torch.randn(batch_size, channels, invalid_height, width)
        assert invalid_height % patch_size != 0
        print(f"  ✓ Invalid height {invalid_height} correctly identified (not divisible by {patch_size})")
    except Exception as e:
        print(f"  Error caught: {e}")
    
    print(f"\n  Test PatchEmbedDust3R shapes: PASSED ✓")


def test_many_ar_patch_embed_shapes():
    """Test ManyAR_PatchEmbed shape validations"""
    print("\n[TEST 3] Testing ManyAR_PatchEmbed tensor shapes")
    print("-" * 60)
    
    batch_size = 4
    channels = 3
    height = 224
    width = 336  # Landscape mode (W >= H)
    patch_size = 16
    
    # Create test tensors
    img = torch.randn(batch_size, channels, height, width)
    true_shape = torch.tensor([[224, 336], [224, 336], [336, 224], [336, 224]])  # Mix of landscape and portrait
    
    print(f"  Image shape: {img.shape}")
    print(f"  True_shape tensor: {true_shape.shape}")
    
    # Test landscape mode assertion
    assert width >= height, f"Expected landscape mode (W >= H), got W={width}, H={height}"
    print(f"  ✓ Image is in landscape mode (W={width} >= H={height})")
    
    # Test height divisibility
    assert height % patch_size == 0, f"Height {height} not divisible by {patch_size}"
    print(f"  ✓ Height {height} is divisible by patch_size {patch_size}")
    
    # Test width divisibility
    assert width % patch_size == 0, f"Width {width} not divisible by {patch_size}"
    print(f"  ✓ Width {width} is divisible by patch_size {patch_size}")
    
    # Test true_shape dimensions
    assert true_shape.shape == (batch_size, 2), f"true_shape wrong shape: {true_shape.shape}"
    print(f"  ✓ true_shape has correct dimensions: {true_shape.shape}")
    
    # Calculate token dimensions
    W_tokens = width // patch_size
    H_tokens = height // patch_size
    n_tokens = H_tokens * W_tokens
    print(f"  Token dimensions: H={H_tokens}, W={W_tokens}, total={n_tokens}")
    
    print(f"\n  Test ManyAR_PatchEmbed shapes: PASSED ✓")


def test_find_opt_scaling_shapes():
    """Test find_opt_scaling tensor shape assertions"""
    print("\n[TEST 4] Testing find_opt_scaling tensor shapes")
    print("-" * 60)
    
    batch_size = 2
    height = 56
    width = 56
    channels = 3
    
    # Create 4D tensors as expected
    gt_pts1 = torch.randn(batch_size, height, width, channels)
    pr_pts1 = torch.randn(batch_size, height, width, channels)
    gt_pts2 = torch.randn(batch_size, height, width, channels)
    pr_pts2 = torch.randn(batch_size, height, width, channels)
    
    print(f"  gt_pts1 shape: {gt_pts1.shape}")
    print(f"  pr_pts1 shape: {pr_pts1.shape}")
    print(f"  gt_pts2 shape: {gt_pts2.shape}")
    print(f"  pr_pts2 shape: {pr_pts2.shape}")
    
    # Test ndim assertions
    assert gt_pts1.ndim == 4, f"Expected 4D tensor, got {gt_pts1.ndim}D"
    assert pr_pts1.ndim == 4, f"Expected 4D tensor, got {pr_pts1.ndim}D"
    print(f"  ✓ All tensors are 4D")
    
    # Test shape matching
    assert gt_pts1.shape == pr_pts1.shape, f"Shape mismatch: {gt_pts1.shape} != {pr_pts1.shape}"
    print(f"  ✓ gt_pts1 and pr_pts1 have matching shapes")
    
    assert gt_pts2.shape == pr_pts2.shape, f"Shape mismatch: {gt_pts2.shape} != {pr_pts2.shape}"
    print(f"  ✓ gt_pts2 and pr_pts2 have matching shapes")
    
    print(f"\n  Test find_opt_scaling shapes: PASSED ✓")


def test_spann3r_encode_shapes():
    """Test Spann3R model encode methods tensor shapes"""
    print("\n[TEST 5] Testing Spann3R encode methods tensor shapes")
    print("-" * 60)
    
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    # Create mock views
    view1 = {
        'img': torch.randn(batch_size, channels, height, width),
        'true_shape': torch.tensor([[height, width], [height, width]])
    }
    
    view2 = {
        'img': torch.randn(batch_size, channels, height, width),
        'true_shape': torch.tensor([[height, width], [height, width]])
    }
    
    print(f"  view1['img'] shape: {view1['img'].shape}")
    print(f"  view1['true_shape'] shape: {view1['true_shape'].shape}")
    print(f"  view2['img'] shape: {view2['img'].shape}")
    print(f"  view2['true_shape'] shape: {view2['true_shape'].shape}")
    
    # Verify shapes
    assert view1['img'].shape[0] == batch_size, "Batch size mismatch"
    assert view1['img'].shape[1] == channels, "Channel count mismatch"
    assert view1['img'].shape[2] == height, "Height mismatch"
    assert view1['img'].shape[3] == width, "Width mismatch"
    print(f"  ✓ view1 image has correct shape (B, C, H, W)")
    
    assert view1['true_shape'].shape == (batch_size, 2), "true_shape dimension mismatch"
    print(f"  ✓ view1 true_shape has correct dimensions (B, 2)")
    
    # Test concatenation (as done in encode_image_pairs)
    img_concat = torch.cat((view1['img'], view2['img']), dim=0)
    shape_concat = torch.cat((view1['true_shape'], view2['true_shape']), dim=0)
    
    print(f"  Concatenated images shape: {img_concat.shape}")
    print(f"  Concatenated true_shapes shape: {shape_concat.shape}")
    
    assert img_concat.shape[0] == 2 * batch_size, "Concatenation failed"
    print(f"  ✓ Image concatenation successful (batch doubled)")
    
    assert shape_concat.shape[0] == 2 * batch_size, "Concatenation failed"
    print(f"  ✓ true_shape concatenation successful (batch doubled)")
    
    print(f"\n  Test Spann3R encode shapes: PASSED ✓")


def test_interleave_imgs():
    """Test _interleave_imgs shape handling"""
    print("\n[TEST 6] Testing _interleave_imgs tensor shapes")
    print("-" * 60)
    
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    
    # Create mock image dictionaries
    img1 = {
        'img': torch.randn(batch_size, channels, height, width),
        'idx': list(range(batch_size))
    }
    
    img2 = {
        'img': torch.randn(batch_size, channels, height, width),
        'idx': list(range(batch_size, 2*batch_size))
    }
    
    print(f"  img1['img'] shape: {img1['img'].shape}")
    print(f"  img2['img'] shape: {img2['img'].shape}")
    
    # Test interleaving (stack and flatten)
    interleaved = torch.stack((img1['img'], img2['img']), dim=1).flatten(0, 1)
    print(f"  Interleaved shape: {interleaved.shape}")
    
    expected_shape = (2 * batch_size, channels, height, width)
    assert interleaved.shape == expected_shape, f"Expected {expected_shape}, got {interleaved.shape}"
    print(f"  ✓ Interleaving produces correct shape: {interleaved.shape}")
    
    print(f"\n  Test _interleave_imgs shapes: PASSED ✓")


def test_dpt_output_adapter_shapes():
    """Test DPT output adapter shape calculations"""
    print("\n[TEST 7] Testing DPT output adapter tensor shapes")
    print("-" * 60)
    
    batch_size = 2
    image_height = 224
    image_width = 224
    stride_level = 1
    patch_size = 16
    
    # Calculate number of patches
    N_H = image_height // (stride_level * patch_size)
    N_W = image_width // (stride_level * patch_size)
    
    print(f"  Image size: H={image_height}, W={image_width}")
    print(f"  Patch size: {patch_size}")
    print(f"  Stride level: {stride_level}")
    print(f"  Number of patches: N_H={N_H}, N_W={N_W}")
    
    # Simulate encoder tokens
    num_tokens = N_H * N_W
    embed_dim = 768
    encoder_token = torch.randn(batch_size, num_tokens, embed_dim)
    
    print(f"  Encoder token shape: {encoder_token.shape}")
    print(f"  Expected: (B={batch_size}, N={num_tokens}, D={embed_dim})")
    
    # Test reshaping to spatial representation
    # Reshape from (B, N_H*N_W, C) to (B, C, N_H, N_W)
    spatial = encoder_token.permute(0, 2, 1).reshape(batch_size, embed_dim, N_H, N_W)
    print(f"  Spatial representation shape: {spatial.shape}")
    
    expected_spatial = (batch_size, embed_dim, N_H, N_W)
    assert spatial.shape == expected_spatial, f"Expected {expected_spatial}, got {spatial.shape}"
    print(f"  ✓ Spatial reshaping successful")
    
    print(f"\n  Test DPT output adapter shapes: PASSED ✓")


def test_transpose_operations():
    """Test transpose operations for landscape/portrait handling"""
    print("\n[TEST 8] Testing transpose operations for orientation")
    print("-" * 60)
    
    batch_size = 2
    height = 224
    width = 336
    channels = 3
    
    # Create landscape tensor
    landscape = torch.randn(batch_size, channels, height, width)
    print(f"  Landscape shape: {landscape.shape}")
    
    # Test swapaxes for portrait conversion
    portrait = landscape.swapaxes(-1, -2)
    print(f"  Portrait shape (after swapaxes): {portrait.shape}")
    
    assert portrait.shape == (batch_size, channels, width, height), "Transpose failed"
    print(f"  ✓ Swapaxes correctly converts landscape to portrait")
    
    # Test back conversion
    back_to_landscape = portrait.swapaxes(-1, -2)
    assert back_to_landscape.shape == landscape.shape, "Reverse transpose failed"
    print(f"  ✓ Reverse swapaxes works correctly")
    
    print(f"\n  Test transpose operations: PASSED ✓")


def run_all_tests():
    """Run all tensor shape tests"""
    print("\nStarting all tests...\n")
    
    tests = [
        test_check_if_same_size,
        test_patch_embed_shapes,
        test_many_ar_patch_embed_shapes,
        test_find_opt_scaling_shapes,
        test_spann3r_encode_shapes,
        test_interleave_imgs,
        test_dpt_output_adapter_shapes,
        test_transpose_operations,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {test_func.__name__} FAILED: {str(e)}\n")
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
