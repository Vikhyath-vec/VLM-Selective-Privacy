#!/usr/bin/env python3
"""
Visualize and compare different attention aggregation methods.

This script creates side-by-side visualizations of raw attention, rollout, and LRP
to help understand the differences between methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from transformers import AutoProcessor, AutoModelForVision2Seq
from train_adversarial_image import AttentionValueCapture
from lrp_relevance import compute_lrp_rollout
from attention_rollout import AttentionRollout
from utils import get_patch_size


def visualize_comparison(image_path, model_id="Salesforce/blip2-opt-2.7b"):
    """
    Create side-by-side visualization of different attention methods.

    Args:
        image_path: Path to input image
        model_id: HuggingFace model ID
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and processor
    print(f"\nLoading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map=device
    )
    print("✓ Model loaded")

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']
    print(f"✓ Image loaded: {image.size}")

    # Initialize capture
    capture = AttentionValueCapture(model)
    print("✓ Capture initialized")

    # Forward pass
    print("\nRunning forward pass...")
    capture.reset()
    with torch.no_grad():
        _ = capture.vision_model(pixel_values, output_attentions=True, output_hidden_states=True)

    print(f"✓ Captured {len(capture.attentions)} attention layers")
    print(f"✓ Captured {len(capture.values)} value layers")

    if len(capture.attentions) == 0:
        print("Error: No attentions captured!")
        return

    # Get dimensions
    batch_size, num_heads, seq_len, _ = capture.attentions[0].shape
    num_patches = seq_len - 1  # Exclude CLS token
    patch_size_val = int(np.sqrt(num_patches))
    print(f"✓ Patch grid: {patch_size_val}x{patch_size_val}")

    # 1. Raw Attention (average over layers and heads)
    print("\nComputing raw attention...")
    raw_attention = []
    for attn in capture.attentions:
        # Average over heads, take CLS to patches
        attn_mean = attn.mean(dim=1)  # (batch, seq, seq)
        cls_attn = attn_mean[0, 0, 1:]  # CLS to patches
        raw_attention.append(cls_attn)

    raw_attention = torch.stack(raw_attention).mean(dim=0)  # Average over layers
    raw_attention_map = raw_attention.cpu().numpy().reshape(patch_size_val, patch_size_val)
    print("✓ Raw attention computed")

    # 2. Attention Rollout
    print("\nComputing attention rollout...")
    result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)

    for attention in capture.attentions:
        # Average over heads
        attention_mean = torch.mean(attention, dim=1)

        # Add identity and normalize
        I = torch.eye(seq_len, device=device).unsqueeze(0)
        a = (attention_mean + I) / 2.0
        a = a / a.sum(dim=-1, keepdim=True)

        # Matrix multiplication
        result = torch.bmm(a, result)

    rollout_map = result[0, 0, 1:].cpu().numpy().reshape(patch_size_val, patch_size_val)
    print("✓ Attention rollout computed")

    # 3. LRP
    print("\nComputing LRP relevance...")
    with torch.no_grad():
        lrp_relevance = compute_lrp_rollout(capture.attentions, capture.values)

    lrp_map = lrp_relevance[0, 1:].cpu().numpy().reshape(patch_size_val, patch_size_val)
    print("✓ LRP relevance computed")

    # Normalize all maps to [0, 1]
    raw_attention_map = (raw_attention_map - raw_attention_map.min()) / (raw_attention_map.max() - raw_attention_map.min() + 1e-8)
    rollout_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)
    lrp_map = (lrp_map - lrp_map.min()) / (lrp_map.max() - lrp_map.min() + 1e-8)

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Resize original image to match patch grid
    image_resized = image.resize((patch_size_val * 16, patch_size_val * 16))

    # Row 1: Original image and heatmaps
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(raw_attention_map, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Raw Attention (Sum)\n❌ Ignores Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(rollout_map, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('Attention Rollout\n⚠️ Only Attention Patterns', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Overlays on image
    axes[1, 0].imshow(image_resized)
    axes[1, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image_resized)
    im3 = axes[1, 1].imshow(raw_attention_map, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 1].set_title('Raw Attention Overlay', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(image_resized)
    im4 = axes[1, 2].imshow(lrp_map, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[1, 2].set_title('LRP Relevance Overlay\n✅ Actual Information Flow', fontsize=12, fontweight='bold', color='green')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = os.path.splitext(image_path)[0] + '_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Raw Attention:")
    print(f"  Mean: {raw_attention_map.mean():.4f}")
    print(f"  Std:  {raw_attention_map.std():.4f}")
    print(f"  Max:  {raw_attention_map.max():.4f}")

    print(f"\nAttention Rollout:")
    print(f"  Mean: {rollout_map.mean():.4f}")
    print(f"  Std:  {rollout_map.std():.4f}")
    print(f"  Max:  {rollout_map.max():.4f}")

    print(f"\nLRP Relevance:")
    print(f"  Mean: {lrp_map.mean():.4f}")
    print(f"  Std:  {lrp_map.std():.4f}")
    print(f"  Max:  {lrp_map.max():.4f}")

    # Compute differences
    diff_rollout_raw = np.abs(rollout_map - raw_attention_map).mean()
    diff_lrp_raw = np.abs(lrp_map - raw_attention_map).mean()
    diff_lrp_rollout = np.abs(lrp_map - rollout_map).mean()

    print(f"\n" + "=" * 60)
    print("DIFFERENCES (Mean Absolute)")
    print("=" * 60)
    print(f"Rollout vs Raw:      {diff_rollout_raw:.4f}")
    print(f"LRP vs Raw:          {diff_lrp_raw:.4f}")
    print(f"LRP vs Rollout:      {diff_lrp_rollout:.4f}")

    return {
        'raw_attention': raw_attention_map,
        'rollout': rollout_map,
        'lrp': lrp_map
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare attention aggregation methods"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="HuggingFace model ID (default: Salesforce/blip2-opt-2.7b)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return

    print("=" * 60)
    print("ATTENTION METHOD COMPARISON VISUALIZATION")
    print("=" * 60)

    visualize_comparison(args.image_path, args.model_id)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
