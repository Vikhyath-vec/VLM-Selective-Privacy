"""
Test script for AttentionValueCapture hooks with debug output.
This helps verify that the hooks are correctly capturing attention weights and value matrices.

Usage:
    python test/test_capture_hooks.py --model_id Salesforce/blip2-flan-t5-xl
    python test/test_capture_hooks.py --model_id llava-hf/llava-1.5-7b-hf --debug
    python test/test_capture_hooks.py --model_id Salesforce/instructblip-vicuna-7b
"""

import torch
import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_adversarial_image import AttentionValueCapture
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaForConditionalGeneration,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor
)
from PIL import Image
import numpy as np

# Model configuration
MODEL_CONFIGS = {
    'blip2': {
        'model_id': 'Salesforce/blip2-flan-t5-xl',
        'model_class': AutoModelForVision2Seq,
        'processor_class': AutoProcessor,
        'model_type': 'blip2'
    },
    'llava': {
        'model_id': 'llava-hf/llava-1.5-7b-hf',
        'model_class': LlavaForConditionalGeneration,
        'processor_class': AutoProcessor,
        'model_type': 'llava'
    },
    'instructblip': {
        'model_id': 'Salesforce/instructblip-vicuna-7b',
        'model_class': InstructBlipForConditionalGeneration,
        'processor_class': InstructBlipProcessor,
        'model_type': 'instructblip'
    }
}

def get_model_config(model_id):
    """Get model configuration based on model_id."""
    # Check if it's a shorthand
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    
    # Otherwise, infer from model_id string
    if 'blip2' in model_id.lower():
        return {
            'model_id': model_id,
            'model_class': AutoModelForVision2Seq,
            'processor_class': AutoProcessor,
            'model_type': 'blip2'
        }
    elif 'llava' in model_id.lower():
        return {
            'model_id': model_id,
            'model_class': LlavaForConditionalGeneration,
            'processor_class': AutoProcessor,
            'model_type': 'llava'
        }
    elif 'instructblip' in model_id.lower():
        return {
            'model_id': model_id,
            'model_class': InstructBlipForConditionalGeneration,
            'processor_class': InstructBlipProcessor,
            'model_type': 'instructblip'
        }
    else:
        raise ValueError(f"Unknown model type for: {model_id}")

def test_capture_hooks(model_id, debug=False):
    config = get_model_config(model_id)
    
    print("=" * 60)
    print(f"Testing AttentionValueCapture for {config['model_type'].upper()}")
    print(f"Model: {config['model_id']}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    print("=" * 60)
    
    # Load model
    print(f"\n[1/5] Loading model...")
    model = config['model_class'].from_pretrained(
        config['model_id'],
        device_map="cpu"
    )
    processor = config['processor_class'].from_pretrained(config['model_id'])
    print("✓ Model loaded")
    
    # Create capture instance
    print("\n[2/5] Setting up AttentionValueCapture...")
    capture = AttentionValueCapture(model, model_type=config['model_type'], debug=debug)
    print("✓ Capture hooks registered")
    
    # Load test image
    print("\n[3/5] Loading test image...")
    image_path = "data/images/ILSVRC2012_val_00000073.JPEG"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Creating dummy image...")
        image = Image.new('RGB', (224, 224), color='red')
    else:
        image = Image.open(image_path).convert('RGB')
    print("✓ Image loaded")
    
    # Process image
    print("\n[4/5] Processing image through processor...")
    inputs = processor(images=image, text="Dummy text", return_tensors="pt")
    pixel_values = inputs['pixel_values']
    print(f"✓ Pixel values shape: {pixel_values.shape}")
    
    # Forward pass
    print("\n[5/5] Running forward pass through vision model...")
    capture.reset()
    with torch.set_grad_enabled(True):
        _ = capture.vision_model(pixel_values, output_attentions=True, output_hidden_states=True)
    print("✓ Forward pass complete")
    
    # Results
    print("\n" + "=" * 60)
    print("CAPTURE RESULTS")
    print("=" * 60)
    
    print(f"\nSummary:")
    print(f"  • Attention layers captured: {len(capture.attentions)}")
    print(f"  • Value layers captured: {len(capture.values)}")
    
    # Attention details
    if len(capture.attentions) > 0:
        print(f"\nAttention Details:")
        print(f"  • First layer shape: {capture.attentions[0].shape}")
        print(f"  • Last layer shape: {capture.attentions[-1].shape}")
        
        batch_size, num_heads, seq_len, _ = capture.attentions[0].shape
        print(f"  • Batch size: {batch_size}")
        print(f"  • Num heads: {num_heads}")
        print(f"  • Sequence length: {seq_len}")
        print(f"  • Num patches (excluding CLS): {seq_len - 1}")
        
        # Check gradient flow
        print(f"  • Requires grad: {capture.attentions[0].requires_grad}")
    else:
        print("\nWARNING: No attention weights captured!")
    
    # Value details
    if len(capture.values) > 0:
        print(f"\nValue Details:")
        print(f"  • First layer shape: {capture.values[0].shape}")
        print(f"  • Last layer shape: {capture.values[-1].shape}")
        
        batch_size, num_heads, seq_len, d_v = capture.values[0].shape
        print(f"  • Batch size: {batch_size}")
        print(f"  • Num heads: {num_heads}")
        print(f"  • Sequence length: {seq_len}")
        print(f"  • Value dimension per head: {d_v}")
        
        # Check gradient flow
        print(f"  • Requires grad: {capture.values[0].requires_grad}")
    else:
        print("\nWARNING: No values captured!")
        print("\nDebugging - Checking vision model structure...")
        
        attn_layers = []
        for name, module in capture.vision_model.named_modules():
            if 'self_attn' in name and not any(x in name for x in ['k_proj', 'v_proj', 'q_proj', 'out_proj', 'qkv', 'projection']):
                attn_layers.append((name, module))
        
        print(f"\nFound {len(attn_layers)} attention layers:")
        for i, (name, module) in enumerate(attn_layers[:3]):  # Show first 3
            print(f"\n  Layer {i}: {name}")
            print(f"    • Has v_proj: {hasattr(module, 'v_proj')}")
            print(f"    • Has value: {hasattr(module, 'value')}")
            print(f"    • Has qkv: {hasattr(module, 'qkv')}")
            
            if hasattr(module, 'qkv'):
                print(f"    • qkv type: {type(module.qkv)}")
                if hasattr(module.qkv, 'out_features'):
                    print(f"    • qkv output features: {module.qkv.out_features}")
    
    # Test attention rollout computation
    if len(capture.attentions) > 0:
        print("\n" + "=" * 60)
        print("TESTING ATTENTION ROLLOUT")
        print("=" * 60)
        
        from train_adversarial_image import attention_rollout_aggregation
        
        # Define some test ROI patches (center of image)
        patch_size = 14
        img_h, img_w = 224, 224
        n_patches_h = img_h // patch_size
        n_patches_w = img_w // patch_size
        
        # Center patches
        center_r, center_c = n_patches_h // 2, n_patches_w // 2
        roi_patches = [
            (center_r, center_c),
            (center_r - 1, center_c),
            (center_r + 1, center_c),
            (center_r, center_c - 1),
            (center_r, center_c + 1),
        ]
        
        print(f"\nTest ROI: {len(roi_patches)} patches around center")
        print(f"Patches grid size: {n_patches_h}x{n_patches_w}")
        
        try:
            rollout_value = attention_rollout_aggregation(
                capture.attentions,
                roi_patches,
                (n_patches_h, n_patches_w),
                num_layers=-1
            )
            print(f"\n✓ Rollout computed successfully")
            print(f"  • Rollout value: {rollout_value.item():.6f}")
            print(f"  • Requires grad: {rollout_value.requires_grad}")
        except Exception as e:
            print(f"\n❌ Error computing rollout: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Test AttentionValueCapture hooks for different VLM models")
    parser.add_argument("--model_id", type=str, default="blip2",
                        help="Model ID or shorthand (blip2, llava, instructblip) or full HF model ID")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug output from hooks")
    
    args = parser.parse_args()
    
    test_capture_hooks(args.model_id, debug=args.debug)

if __name__ == "__main__":
    main()
