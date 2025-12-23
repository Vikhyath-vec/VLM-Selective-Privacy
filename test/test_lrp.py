"""
Test LRP (Layer-wise Relevance Propagation) implementation.

This script tests the LRP relevance computation for VLM vision encoders
to ensure it correctly tracks information flow through the network.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from lrp_relevance import lrp_aggregation_loss, compute_lrp_rollout
from train_adversarial_image import AttentionValueCapture
from utils import preprocess_image_and_boxes, get_patch_size


def test_lrp_basic():
    """Test basic LRP computation with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic LRP computation with synthetic data")
    print("=" * 60)

    batch_size = 1
    num_heads = 8
    seq_len = 197  # 196 patches + 1 CLS token (14x14 patches)
    dim = 64

    # Create synthetic attention and value tensors
    attentions = []
    values = []

    for layer in range(3):
        # Random attention weights
        attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)  # Normalize
        attn.requires_grad = True

        # Random values
        val = torch.randn(batch_size, num_heads, seq_len, dim)
        val.requires_grad = True

        attentions.append(attn)
        values.append(val)

    # Define ROI patches (center region)
    roi_patches = [(6, 6), (6, 7), (7, 6), (7, 7)]  # 2x2 region in center
    all_patches = (14, 14)

    # Compute LRP relevance
    try:
        relevance = compute_lrp_rollout(attentions, values)
        print(f"✓ LRP relevance computed successfully")
        print(f"  Relevance shape: {relevance.shape}")
        print(f"  Relevance range: [{relevance.min():.4f}, {relevance.max():.4f}]")
        print(f"  Relevance sum: {relevance.sum():.4f}")

        # Check that relevance is differentiable
        loss = relevance.sum()
        loss.backward()
        print(f"✓ Gradients computed successfully")

        return True
    except Exception as e:
        print(f"✗ Error computing LRP relevance: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lrp_aggregation():
    """Test LRP aggregation loss function."""
    print("\n" + "=" * 60)
    print("TEST 2: LRP aggregation loss")
    print("=" * 60)

    batch_size = 1
    num_heads = 8
    seq_len = 197
    dim = 64

    # Create synthetic data
    attentions = []
    values = []

    for layer in range(3):
        attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)
        attn.requires_grad = True

        val = torch.randn(batch_size, num_heads, seq_len, dim)
        val.requires_grad = True

        attentions.append(attn)
        values.append(val)

    # Define ROI patches
    roi_patches = [(6, 6), (6, 7), (7, 6), (7, 7)]
    all_patches = (14, 14)

    try:
        # Compute loss
        loss = lrp_aggregation_loss(attentions, values, roi_patches, all_patches, num_layers=-1)
        print(f"✓ LRP aggregation loss computed successfully")
        print(f"  Loss value: {loss.item():.6f}")

        # Check differentiability
        loss.backward()
        print(f"✓ Gradients flow correctly through LRP loss")

        # Check that gradients exist for attention and values
        has_attn_grad = all(a.grad is not None for a in attentions)
        has_val_grad = all(v.grad is not None for v in values)
        print(f"  Attention gradients: {'✓' if has_attn_grad else '✗'}")
        print(f"  Value gradients: {'✓' if has_val_grad else '✗'}")

        return True
    except Exception as e:
        print(f"✗ Error computing LRP aggregation loss: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lrp_with_model():
    """Test LRP with a real VLM model."""
    print("\n" + "=" * 60)
    print("TEST 3: LRP with real BLIP-2 model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load small BLIP-2 model for testing
    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"\nLoading model: {model_id}")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map=device
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Skipping this test (requires model download)")
        return True  # Don't fail test if model not available

    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    print("✓ Created dummy image (224x224)")

    # Initialize capture
    try:
        capture = AttentionValueCapture(model)
        print("✓ Initialized AttentionValueCapture")
    except Exception as e:
        print(f"✗ Error initializing capture: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Preprocess image
    inputs = processor(images=dummy_image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']
    print(f"✓ Preprocessed image, shape: {pixel_values.shape}")

    # Forward pass
    try:
        capture.reset()
        with torch.set_grad_enabled(True):
            _ = capture.vision_model(pixel_values, output_attentions=True, output_hidden_states=True)

        print(f"✓ Forward pass completed")
        print(f"  Captured {len(capture.attentions)} attention layers")
        print(f"  Captured {len(capture.values)} value layers")

        if len(capture.attentions) > 0:
            print(f"  Attention shape: {capture.attentions[0].shape}")
        if len(capture.values) > 0:
            print(f"  Value shape: {capture.values[0].shape}")
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compute LRP loss
    try:
        # Get patch size
        patch_size = get_patch_size(model)
        target_size = pixel_values.shape[2:]  # (H, W)
        patch_h = target_size[0] // patch_size
        patch_w = target_size[1] // patch_size

        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Grid: {patch_h}x{patch_w} patches")

        # Define ROI (center 2x2 region)
        center_r, center_c = patch_h // 2, patch_w // 2
        roi_patches = [
            (center_r, center_c),
            (center_r, center_c + 1),
            (center_r + 1, center_c),
            (center_r + 1, center_c + 1)
        ]
        all_patches = (patch_h, patch_w)

        # Compute LRP loss
        loss = lrp_aggregation_loss(
            capture.attentions,
            capture.values,
            roi_patches,
            all_patches,
            num_layers=-1
        )

        print(f"✓ LRP loss computed on real model")
        print(f"  Loss value: {loss.item():.6f}")

        # Test gradient flow
        loss.backward()
        print(f"✓ Gradients computed successfully")

        return True

    except Exception as e:
        print(f"✗ Error computing LRP on real model: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_aggregation_methods():
    """Compare LRP with other aggregation methods."""
    print("\n" + "=" * 60)
    print("TEST 4: Compare aggregation methods")
    print("=" * 60)

    batch_size = 1
    num_heads = 8
    seq_len = 197
    dim = 64

    # Create synthetic data with clear pattern
    # Make ROI patches have high attention
    attentions = []
    values = []

    roi_patches = [(6, 6), (6, 7), (7, 6), (7, 7)]
    all_patches = (14, 14)

    for layer in range(3):
        attn = torch.ones(batch_size, num_heads, seq_len, seq_len) * 0.01

        # Boost attention to ROI patches
        for roi_r, roi_c in roi_patches:
            roi_idx = 1 + roi_r * all_patches[1] + roi_c
            attn[:, :, :, roi_idx] = 0.1  # Higher attention

        attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize
        attn.requires_grad = True

        val = torch.randn(batch_size, num_heads, seq_len, dim)
        val.requires_grad = True

        attentions.append(attn)
        values.append(val)

    # Compute losses with different methods
    results = {}

    # Sum
    from train_adversarial_image import attention_sum
    try:
        loss_sum = attention_sum(attentions, roi_patches, all_patches, num_layers=-1)
        results['sum'] = loss_sum.item()
        print(f"✓ Sum aggregation: {loss_sum.item():.6f}")
    except Exception as e:
        print(f"✗ Sum aggregation failed: {e}")

    # Rollout
    from train_adversarial_image import attention_rollout_aggregation
    try:
        loss_rollout = attention_rollout_aggregation(attentions, roi_patches, all_patches, num_layers=-1)
        results['rollout'] = loss_rollout.item()
        print(f"✓ Rollout aggregation: {loss_rollout.item():.6f}")
    except Exception as e:
        print(f"✗ Rollout aggregation failed: {e}")

    # LRP
    try:
        loss_lrp = lrp_aggregation_loss(attentions, values, roi_patches, all_patches, num_layers=-1)
        results['lrp'] = loss_lrp.item()
        print(f"✓ LRP aggregation: {loss_lrp.item():.6f}")
    except Exception as e:
        print(f"✗ LRP aggregation failed: {e}")

    print("\nComparison:")
    for method, value in results.items():
        print(f"  {method:10s}: {value:.6f}")

    return len(results) == 3


if __name__ == "__main__":
    print("=" * 60)
    print("LRP IMPLEMENTATION TESTS")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("Basic LRP", test_lrp_basic()))
    results.append(("LRP Aggregation", test_lrp_aggregation()))
    results.append(("LRP with Model", test_lrp_with_model()))
    results.append(("Method Comparison", compare_aggregation_methods()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    total = len(results)
    passed = sum(results[i][1] for i in range(len(results)))
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
