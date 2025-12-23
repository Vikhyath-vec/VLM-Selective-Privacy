import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration, InstructBlipForConditionalGeneration

# Perceptual loss dependencies
import lpips
import pytorch_msssim
import torchvision.models as models

from utils import preprocess_image_and_boxes, get_patch_size, get_patch_indices
from attention_rollout import AttentionRollout


# ====================
# Attention Aggregation Methods
# ====================

def attention_sum(attentions, roi_patches, all_patches, num_layers=-1):
    """
    Sum attention values for ROI patches across specified layers and heads.
    
    Args:
        attentions: List of attention tensors [(batch, heads, seq, seq), ...] for each layer
        roi_patches: List of (row, col) tuples indicating ROI patch positions
        all_patches: Total number of patches (height, width)
        num_layers: Number of layers to use (-1 = all)
    
    Returns:
        Scalar loss value
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_attentions = attentions
    else:
        selected_attentions = attentions[:num_layers]
    
    total_attention = 0.0
    patch_h, patch_w = all_patches
    
    for attn in selected_attentions:
        batch_size, num_heads, seq_len, _ = attn.shape
        
        # Sum over all heads and all tokens attending TO roi patches
        for roi_r, roi_c in roi_patches:
            # Convert 2D patch position to sequence index (skip CLS token at 0)
            roi_token_idx = 1 + roi_r * patch_w + roi_c
            
            if roi_token_idx < seq_len:
                # Sum attention TO this ROI token from all other tokens
                total_attention += attn[:, :, :, roi_token_idx].sum()
    
    return total_attention


def attention_rollout_aggregation(attentions, roi_patches, all_patches, num_layers=-1, head_fusion='max', discard_ratio=0.5, background_patch_indices=None):
    """
    Use AttentionRollout to aggregate attention to ROI patches.
    Uses the existing AttentionRollout.rollout() method.
    
    Args:
        attentions: List of attention tensors for each layer
        roi_patches: List of (row, col) tuples indicating ROI patch positions  
        all_patches: Total number of patches (height, width)
        num_layers: Number of layers to use (-1 = all)
        head_fusion: How to fuse attention heads ('mean', 'max', 'min')
        discard_ratio: Ratio of attention to discard in rollout
        background_patch_indices: Optional list of flat indices for background patches
        
    Returns:
        If background_patch_indices is None:
            Scalar loss value (ROI rollout sum)
        Else:
            Tuple of (roi_rollout_scalar, bg_rollout_tensor)
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_attentions = attentions
    else:
        selected_attentions = attentions[:num_layers]
    
    if not selected_attentions:
        if background_patch_indices is None:
            return 0.0
        else:
            return 0.0, torch.tensor([])
    
    batch_size, num_heads, seq_len, _ = selected_attentions[0].shape
    device = selected_attentions[0].device
    patch_h, patch_w = all_patches
    
    # Use AttentionRollout logic (replicate key parts to maintain differentiability)
    # Initialize result as identity
    result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
    
    for attention in selected_attentions:
        # Fuse heads
        if head_fusion == "mean":
            attention = torch.mean(attention, dim=1)
        elif head_fusion == "max":
            attention = torch.max(attention, dim=1)[0]
        elif head_fusion == "min":
            attention = torch.min(attention, dim=1)[0]
        
        # Discard low attention values
        if discard_ratio > 0:
            flat = attention.view(batch_size, -1)
            num_discard = int(flat.size(-1) * discard_ratio)
            _, idx = torch.topk(flat, num_discard, dim=-1, largest=False)
            
            for b in range(batch_size):
                idx_b = idx[b]
                idx_b = idx_b[idx_b != 0]
                flat[b, idx_b] = 0.0
            attention = flat.view(batch_size, seq_len, seq_len)
        
        # Add identity and normalize
        I = torch.eye(seq_len, device=device).unsqueeze(0)
        a = (attention + I) / 2.0
        a = a / a.sum(dim=-1, keepdim=True)
        
        # Matrix multiplication
        result = torch.bmm(a, result)
    
    # Sum attention from CLS token to ROI patches
    total_rollout = 0.0
    for roi_r, roi_c in roi_patches:
        roi_token_idx = 1 + roi_r * patch_w + roi_c
        if roi_token_idx < seq_len:
            # Attention from CLS (index 0) to this ROI token
            total_rollout += result[:, 0, roi_token_idx].sum()
    
    # If background indices provided, also extract background rollout values
    if background_patch_indices is not None:
        # Extract CLS attention to all patches (excluding CLS itself)
        cls_to_patches = result[:, 0, 1:]  # (batch, num_patches)
        
        # Extract background patch values
        bg_rollout = cls_to_patches[:, background_patch_indices]  # (batch, num_bg_patches)
        
        return total_rollout, bg_rollout
    
    return total_rollout


# ====================
# Value Matrix Aggregation Methods
# ====================

def value_l2_norm(value_matrices, roi_patches, all_patches, num_layers=-1, normalize=True):
    """
    Compute L2 norm of value matrices for ROI patches.
    
    Args:
        value_matrices: List of value tensors [(batch, heads, seq, d_v), ...] for each layer
        roi_patches: List of (row, col) tuples
        all_patches: Total patches (height, width)
        num_layers: Number of layers to use (-1 = all)
        normalize: If True, divide by (num_layers * num_roi_patches) to balance with attention loss
        
    Returns:
        Scalar regularization value
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_values = value_matrices
    else:
        selected_values = value_matrices[:num_layers]
    
    total_norm = 0.0
    patch_h, patch_w = all_patches
    count = 0
    
    for values in selected_values:
        for roi_r, roi_c in roi_patches:
            roi_token_idx = 1 + roi_r * patch_w + roi_c
            
            if roi_token_idx < values.shape[2]:
                # Get value vector for this token
                v = values[:, :, roi_token_idx, :]  # (batch, heads, d_v)
                # L2 norm
                total_norm += torch.norm(v, p=2)
                count += 1
    
    # Normalize by number of elements if requested
    if normalize and count > 0:
        total_norm = total_norm / count
    
    return total_norm


def value_frobenius_norm(value_matrices, roi_patches, all_patches, num_layers=-1, normalize=True):
    """
    Compute Frobenius norm of value matrices for ROI patches.
    
    Args:
        normalize: If True, divide by (num_layers * num_roi_patches) to balance with attention loss
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_values = value_matrices
    else:
        selected_values = value_matrices[:num_layers]
    
    total_norm = 0.0
    patch_h, patch_w = all_patches
    count = 0
    
    for values in selected_values:
        for roi_r, roi_c in roi_patches:
            roi_token_idx = 1 + roi_r * patch_w + roi_c
            
            if roi_token_idx < values.shape[2]:
                v = values[:, :, roi_token_idx, :]
                total_norm += torch.norm(v, p='fro')
                count += 1
    
    # Normalize by number of elements if requested
    if normalize and count > 0:
        total_norm = total_norm / count
    
    return total_norm


# ====================
# Perceptual/Visual Loss Methods
# ====================

def perceptual_mse(x_adv, x_clean):
    """
    Simple MSE-based perceptual loss.
    
    Args:
        x_adv: Adversarial image (batch, 3, H, W)
        x_clean: Clean image (batch, 3, H, W)
    
    Returns:
        Scalar MSE loss
    """
    return torch.nn.functional.mse_loss(x_adv, x_clean)


def perceptual_lpips(x_adv, x_clean, lpips_fn=None):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) loss.
    Requires: pip install lpips
    
    Args:
        x_adv: Adversarial image (batch, 3, H, W) in range [0, 1]
        x_clean: Clean image (batch, 3, H, W) in range [0, 1]
        lpips_fn: Pre-initialized LPIPS model (passed from training loop)
    
    Returns:
        Scalar LPIPS loss
    """
    if lpips_fn is None:
        device = x_adv.device
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print("Warning: Creating LPIPS model on-the-fly. For efficiency, pass pre-initialized model.")
    
    return lpips_fn(x_adv, x_clean).mean()


def perceptual_vgg(x_adv, x_clean, vgg_model=None):
    """
    VGG-based perceptual loss using pretrained features.
    Uses multiple layers from VGG16 for perceptual comparison.
    
    Args:
        x_adv: Adversarial image (batch, 3, H, W)
        x_clean: Clean image (batch, 3, H, W)
        vgg_model: Pre-initialized VGG feature extractor (optional)
    
    Returns:
        Scalar perceptual loss
    """
    if vgg_model is None:
        # Create VGG16 feature extractor
        vgg = models.vgg16(pretrained=True).features
        # Extract features from multiple layers
        vgg_model = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:16],  # relu3_3
        ]).to(x_adv.device).eval()
        
        # Freeze parameters
        for param in vgg_model.parameters():
            param.requires_grad = False
            
        print("Warning: Creating VGG model on-the-fly. For efficiency, pass pre-initialized model.")
    
    # VGG expects ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_adv.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_adv.device).view(1, 3, 1, 1)
    
    x_adv_norm = (x_adv - mean) / std
    x_clean_norm = (x_clean - mean) / std
    
    loss = 0
    x_adv_feat = x_adv_norm
    x_clean_feat = x_clean_norm
    
    for layer in vgg_model:
        x_adv_feat = layer(x_adv_feat)
        x_clean_feat = layer(x_clean_feat)
        loss += torch.nn.functional.mse_loss(x_adv_feat, x_clean_feat)
    
    return loss


def perceptual_ssim(x_adv, x_clean):
    """
    Differentiable SSIM loss (1 - SSIM to minimize).
    Requires: pip install pytorch-msssim
    
    Args:
        x_adv: Adversarial image (batch, 3, H, W)
        x_clean: Clean image (batch, 3, H, W)
    
    Returns:
        Scalar SSIM loss (1 - SSIM)
    """
    ssim_val = pytorch_msssim.ssim(x_adv, x_clean, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


# Registry of perceptual loss methods
PERCEPTUAL_LOSS_METHODS = {
    'none': lambda x_adv, x_clean, **kwargs: 0.0,  # No perceptual loss
    'mse': perceptual_mse,
    'lpips': perceptual_lpips,
    'vgg': perceptual_vgg,
    'ssim': perceptual_ssim,
}


# ====================
# Aggregation Method Registry
# ====================

ATTENTION_AGGREGATION_METHODS = {
    'sum': attention_sum,
    'rollout': attention_rollout_aggregation,
}

VALUE_AGGREGATION_METHODS = {
    'l2': value_l2_norm,
    'frobenius': value_frobenius_norm,
}


# ====================
# Model Hooks for Capturing Attention and Values
# ====================

class AttentionValueCapture:
    """Captures attention weights and value matrices during forward pass."""
    
    def __init__(self, model, model_type='blip2', debug=False):
        self.model = model
        self.model_type = model_type
        self.debug = debug
        self.attentions = []
        self.values = []
        self.vision_model = self._get_vision_model()
        self._register_hooks()
    
    def _get_vision_model(self):
        """Extract vision encoder from VLM."""
        if hasattr(self.model, 'vision_tower'):
            # LLaVA
            if hasattr(self.model.vision_tower, 'set_attn_implementation'):
                self.model.vision_tower.set_attn_implementation('eager')
            if hasattr(self.model.vision_tower, 'vision_model'):
                return self.model.vision_tower.vision_model
            return self.model.vision_tower
        
        if hasattr(self.model, 'vision_model'):
            # BLIP-2 / InstructBLIP
            if hasattr(self.model.vision_model, 'set_attn_implementation'):
                self.model.vision_model.set_attn_implementation('eager')
            return self.model.vision_model
        
        return self.model
    
    def _register_hooks(self):
        """Register forward hooks to capture attention and values."""
        for name, module in self.vision_model.named_modules():
            if name.endswith('self_attn') or 'attention' in name.lower():
                module.register_forward_hook(self._attention_hook)
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights and value projections."""
        if self.debug:
            print(f"\n[HOOK] Called on module: {module.__class__.__name__}")
            print(f"[HOOK] Input type: {type(input)}, len: {len(input) if isinstance(input, tuple) else 'N/A'}")
            print(f"[HOOK] Output type: {type(output)}, len: {len(output) if isinstance(output, tuple) else 'N/A'}")
        
        # Different architectures return different outputs
        if isinstance(output, tuple):
            if self.debug:
                print(f"[HOOK] Output is tuple with {len(output)} elements")
            # Usually (hidden_states, attention_weights)
            if len(output) >= 2:
                attn_weights = output[1]
                if self.debug:
                    print(f"[HOOK] Attention weights: {type(attn_weights)}, shape: {attn_weights.shape if attn_weights is not None else 'None'}")
                
                if attn_weights is not None:
                    self.attentions.append(attn_weights)
                    if self.debug:
                        print(f"[HOOK] ✓ Captured attention (total: {len(self.attentions)})")
                    
                    # Capture values - handle different architectures
                    # Get hidden states from input
                    hidden_states = None
                    
                    if self.debug:
                        print(f"[HOOK] Looking for hidden states in input...")
                    
                    if isinstance(input, tuple):
                        if self.debug:
                            print(f"[HOOK] Input is tuple, searching for 3D tensor...")
                        # Try to find the first tensor in the tuple
                        for i, inp in enumerate(input):
                            if self.debug:
                                print(f"[HOOK]   Input[{i}]: type={type(inp)}, is_tensor={isinstance(inp, torch.Tensor)}, dim={inp.dim() if isinstance(inp, torch.Tensor) else 'N/A'}")
                            if isinstance(inp, torch.Tensor) and inp.dim() == 3:
                                hidden_states = inp
                                if self.debug:
                                    print(f"[HOOK]   ✓ Found hidden states at index {i}, shape: {hidden_states.shape}")
                                break
                    elif isinstance(input, torch.Tensor):
                        hidden_states = input
                        if self.debug:
                            print(f"[HOOK] Input is tensor directly, shape: {hidden_states.shape}")
                    
                    # If we couldn't get hidden states from input, use output hidden states
                    if hidden_states is None:
                        if self.debug:
                            print(f"[HOOK] Hidden states not found in input, checking output...")
                        if isinstance(output[0], torch.Tensor) and output[0].dim() == 3:
                            hidden_states = output[0]
                            if self.debug:
                                print(f"[HOOK] ✓ Using output hidden states, shape: {hidden_states.shape}")
                        else:
                            # Really can't get hidden states, skip value capture
                            if self.debug:
                                print(f"[HOOK] ✗ Cannot get hidden states, skipping value capture")
                            return
                    
                    # Extract dimensions
                    try:
                        batch_size, seq_len, embed_dim = hidden_states.shape
                        num_heads = attn_weights.shape[1]
                        if self.debug:
                            print(f"[HOOK] Dimensions: batch={batch_size}, seq={seq_len}, embed={embed_dim}, heads={num_heads}")
                    except (AttributeError, ValueError) as e:
                        # Can't extract dimensions, skip value capture
                        if self.debug:
                            print(f"[HOOK] ✗ Error extracting dimensions: {e}")
                        return
                    
                    values = None
                    
                    # Check which projection style
                    has_v_proj = hasattr(module, 'v_proj')
                    has_value = hasattr(module, 'value')
                    has_qkv = hasattr(module, 'qkv')
                    if self.debug:
                        print(f"[HOOK] Projection style: v_proj={has_v_proj}, value={has_value}, qkv={has_qkv}")
                    
                    # CLIP-style: separate q_proj, k_proj, v_proj
                    if has_v_proj:
                        if self.debug:
                            print(f"[HOOK] Using CLIP-style v_proj")
                        values = module.v_proj(hidden_states)
                        if self.debug:
                            print(f"[HOOK] ✓ Values from v_proj, shape: {values.shape}")
                    # Alternative naming
                    elif has_value:
                        if self.debug:
                            print(f"[HOOK] Using alternative value projection")
                        values = module.value(hidden_states)
                        if self.debug:
                            print(f"[HOOK] ✓ Values from value, shape: {values.shape}")
                    # BLIP-2/InstructBLIP-style: combined qkv projection
                    elif has_qkv:
                        if self.debug:
                            print(f"[HOOK] Using BLIP-2-style qkv")
                        # qkv projects to 3 * num_heads * head_dim
                        qkv = module.qkv(hidden_states)
                        if self.debug:
                            print(f"[HOOK] QKV output shape: {qkv.shape}")
                        
                        # Split into q, k, v
                        # qkv shape: (batch, seq, 3 * num_heads * head_dim)
                        head_dim = embed_dim // num_heads
                        if self.debug:
                            print(f"[HOOK] Head dim: {head_dim}, reshaping to (batch={batch_size}, seq={seq_len}, 3, heads={num_heads}, head_dim={head_dim})")
                        
                        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
                        # Extract only values (index 2)
                        values = qkv[:, :, 2, :, :]  # (batch, seq, num_heads, head_dim)
                        if self.debug:
                            print(f"[HOOK] Extracted values from qkv, shape: {values.shape}")
                        
                        # Reshape to (batch, seq, num_heads * head_dim)
                        values = values.reshape(batch_size, seq_len, num_heads * head_dim)
                        if self.debug:
                            print(f"[HOOK] Reshaped values: {values.shape}")
                    else:
                        if self.debug:
                            print(f"[HOOK] ✗ No recognized projection layer found")
                    
                    if values is not None:
                        # Reshape to (batch, heads, seq, d_v)
                        d_v = values.shape[-1] // num_heads
                        if self.debug:
                            print(f"[HOOK] Reshaping to (batch, heads, seq, d_v) with d_v={d_v}")
                        values = values.view(batch_size, seq_len, num_heads, d_v)
                        values = values.transpose(1, 2)  # (batch, heads, seq, d_v)
                        
                        self.values.append(values)
                        if self.debug:
                            print(f"[HOOK] ✓ Captured values (total: {len(self.values)}), final shape: {values.shape}")
                    else:
                        if self.debug:
                            print(f"[HOOK] ✗ Values is None, not captured")
                else:
                    if self.debug:
                        print(f"[HOOK] ✗ Attention weights is None, skipping")
            else:
                if self.debug:
                    print(f"[HOOK] ✗ Output tuple has fewer than 2 elements")
        else:
            if self.debug:
                print(f"[HOOK] ✗ Output is not a tuple, skipping")
    
    def reset(self):
        """Clear captured attentions and values."""
        self.attentions = []
        self.values = []


# ====================
# Evaluation Function
# ====================

def evaluate_adversarial_image(model, processor, clean_image, adv_image, objects, model_type='blip2', mapping=None):
    """
    Evaluate adversarial image by querying VLM about ROI objects.
    
    Args:
        model: VLM model
        processor: Model processor
        clean_image: PIL Image (clean)
        adv_image: PIL Image (adversarial)
        objects: List of dicts with 'name' key (may contain ImageNet IDs)
        model_type: Model type ('blip2', 'llava', 'instructblip')
        mapping: Dict mapping ImageNet IDs to readable labels (optional)
    
    Returns:
        Dict with evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATING ADVERSARIAL IMAGE")
    print("=" * 60)
    
    device = next(model.parameters()).device
    results = {
        'object_detection': [],
        'description': {}
    }
    
    # Get unique object names and convert to readable labels
    unique_objects = {}
    for obj in objects:
        obj_id = obj['name']
        
        # Convert ImageNet ID to readable label if mapping provided
        if mapping and obj_id in mapping:
            obj_name = mapping[obj_id]
        else:
            obj_name = obj_id
        
        # Deduplicate by name
        if obj_name not in unique_objects:
            unique_objects[obj_name] = obj_id
    
    print(f"\nFound {len(unique_objects)} unique object types (out of {len(objects)} total objects)")
    
    # Test 1: Object detection questions
    print("\nTesting object detection...")
    for obj_name, obj_id in unique_objects.items():
        question = f"Is there a {obj_name} in the image?"
        
        print(f"\nQuestion: {question}")
        
        # Query clean image
        clean_response = query_vlm(model, processor, clean_image, question, model_type, device)
        print(f"  Clean: {clean_response}")
        
        # Query adversarial image
        adv_response = query_vlm(model, processor, adv_image, question, model_type, device)
        print(f"  Adversarial: {adv_response}")
        
        # Check if attack successful (model says "no" or doesn't mention object)
        attack_success = is_negative_response(adv_response, obj_name)
        
        results['object_detection'].append({
            'object_id': obj_id,
            'object_name': obj_name,
            'question': question,
            'clean_response': clean_response,
            'adv_response': adv_response,
            'attack_success': attack_success
        })
    
    # Test 2: General description
    print("\nTesting general description...")
    desc_question = "Provide a short description of the image."
    print(f"\nQuestion: {desc_question}")
    
    clean_desc = query_vlm(model, processor, clean_image, desc_question, model_type, device)
    print(f"  Clean: {clean_desc}")
    
    adv_desc = query_vlm(model, processor, adv_image, desc_question, model_type, device)
    print(f"  Adversarial: {adv_desc}")
    
    # Check if any objects mentioned in adversarial description
    objects_mentioned = [name for name in unique_objects.keys() if name.lower() in adv_desc.lower()]
    
    results['description'] = {
        'question': desc_question,
        'clean_response': clean_desc,
        'adv_response': adv_desc,
        'objects_mentioned': objects_mentioned,
        'attack_success': len(objects_mentioned) == 0
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    obj_success_rate = sum(1 for r in results['object_detection'] if r['attack_success']) / len(results['object_detection']) if results['object_detection'] else 0
    print(f"\nObject Detection Attack Success Rate: {obj_success_rate * 100:.1f}%")
    print(f"   - Successfully hidden: {sum(1 for r in results['object_detection'] if r['attack_success'])}/{len(results['object_detection'])} objects")
    
    print(f"\nDescription Attack: {'✓ Success' if results['description']['attack_success'] else '✗ Failed'}")
    if objects_mentioned:
        print(f"   - Objects still mentioned: {', '.join(objects_mentioned)}")
    else:
        print(f"   - No protected objects mentioned")
    
    # Generate attention rollout visualizations for debugging
    print("\n" + "=" * 60)
    print("GENERATING ATTENTION ROLLOUT VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Initialize AttentionRollout
        rollout = AttentionRollout(model, head_fusion="max", discard_ratio=0.5)
        
        # Helper function for visualization
        def show_mask_on_image(img, mask):
            import cv2
            img = np.float32(img) / 255
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return np.uint8(255 * cam)
        
        # Process clean image
        print("\nGenerating clean image rollout...")
        inputs_clean = processor(images=clean_image, return_tensors="pt").to(device)
        mask_clean, _ = rollout(inputs_clean.pixel_values, layer_idx=-1)
        mask_clean = mask_clean[0]
        
        # Resize mask to image size
        import cv2
        np_img_clean = np.array(clean_image)
        mask_clean_resized = cv2.resize(mask_clean, (np_img_clean.shape[1], np_img_clean.shape[0]))
        vis_clean = show_mask_on_image(np_img_clean, mask_clean_resized)
        
        # Process adversarial image
        print("Generating adversarial image rollout...")
        inputs_adv = processor(images=adv_image, return_tensors="pt").to(device)
        mask_adv, _ = rollout(inputs_adv.pixel_values, layer_idx=-1)
        mask_adv = mask_adv[0]
        
        np_img_adv = np.array(adv_image)
        mask_adv_resized = cv2.resize(mask_adv, (np_img_adv.shape[1], np_img_adv.shape[0]))
        vis_adv = show_mask_on_image(np_img_adv, mask_adv_resized)
        
        # Save visualizations (will be saved by caller)
        results['visualizations'] = {
            'clean_rollout': vis_clean,
            'adv_rollout': vis_adv
        }
        
        print("✓ Attention rollout visualizations generated")
        
    except Exception as e:
        print(f"Warning: Could not generate attention rollout visualizations: {e}")
        results['visualizations'] = None
    
    return results


def query_vlm(model, processor, image, question, model_type, device):
    """Query VLM with an image and question."""
    # Prepare inputs based on model type
    if model_type == 'llava' or 'llava' in model_type.lower():
        # LLaVA uses conversation format
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    elif model_type == 'blip2' or 'blip2' in model_type.lower():
        # BLIP-2 uses prompt directly
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    elif 'instructblip' in model_type.lower():
        # InstructBLIP also uses prompt directly
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    else:
        # Default to BLIP-2 style
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # Decode
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean up response (remove input prompt for LLaVA)
    if model_type == 'llava' or 'llava' in model_type.lower():
        # LLaVA includes the prompt in output, remove it
        if 'ASSISTANT:' in response:
            response = response.split('ASSISTANT:')[-1].strip()
    
    return response.strip()


def is_negative_response(response, object_name):
    """
    Check if response indicates object is not present.
    
    Args:
        response: Model's text response
        object_name: Name of the object
    
    Returns:
        True if response is negative (attack successful), False otherwise
    """
    response_lower = response.lower()
    
    # Negative indicators
    negative_patterns = [
        'no', 'not', 'cannot', 'can\'t', 'unable', 
        'don\'t see', 'do not see', 'doesn\'t', 'does not',
        'there is no', 'there are no', 'i do not see'
    ]
    
    # Positive indicators
    positive_patterns = [
        'yes', 'there is', 'there are', 'i see', 'i can see',
        'visible', 'present', 'shows', 'contains'
    ]
    
    # Check if object name is mentioned
    object_mentioned = object_name.lower() in response_lower
    
    # Check for negative patterns
    has_negative = any(pattern in response_lower for pattern in negative_patterns)
    
    # Check for positive patterns
    has_positive = any(pattern in response_lower for pattern in positive_patterns)
    
    # Attack successful if: negative response OR object not mentioned
    return has_negative or (not object_mentioned and not has_positive)


# ====================
# Main Training Function
# ====================

def train_adversarial_image(args):
    """
    Train adversarial perturbation delta for a single image.
    Implements the VIP paper optimization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    if 'llava' in args.model_id.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map=device
        )
    elif 'blip2' in args.model_id.lower():
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            device_map=device
        )
    elif 'instructblip' in args.model_id.lower():
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map=device
        )
    else:
        raise ValueError(f"Unsupported model: {args.model_id}")
    
    model.eval()
    
    # Setup attention/value capture
    capture = AttentionValueCapture(model, model_type=args.model_id)
    
    # Get aggregation functions
    attn_agg_fn = ATTENTION_AGGREGATION_METHODS[args.attention_aggregation]
    value_agg_fn = VALUE_AGGREGATION_METHODS[args.value_aggregation]
    perceptual_loss_fn = PERCEPTUAL_LOSS_METHODS[args.perceptual_loss]
    
    # Initialize LPIPS model if needed
    lpips_model = None
    if args.perceptual_loss == 'lpips':
        lpips_model = lpips.LPIPS(net='alex').to(device)
        print("LPIPS model loaded successfully")
    
    # Initialize VGG model if needed
    vgg_model = None
    if args.perceptual_loss == 'vgg':
        vgg = models.vgg16(pretrained=True).features
        vgg_model = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:16],  # relu3_3
        ]).to(device).eval()
        
        for param in vgg_model.parameters():
            param.requires_grad = False
            
        print("VGG16 model loaded successfully")
    
    # Preprocess image and get ROI
    print(f"Preprocessing image: {args.image_path}")
    preprocessed_img, scaled_objects, target_size = preprocess_image_and_boxes(
        args.image_path, args.xml_path, processor
    )
    
    # Get patch size
    patch_size = get_patch_size(args.model_id)
    
    # Get ROI patch indices
    roi_patches = []
    for obj in scaled_objects:
        bbox = obj['bbox']
        patches = get_patch_indices(bbox, target_size, patch_size)
        roi_patches.extend(patches)
    roi_patches = list(set(roi_patches))  # Remove duplicates
    
    print(f"ROI contains {len(roi_patches)} patches")
    
    # Convert preprocessed image to tensor
    img_array = np.array(preprocessed_img)
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Compute clean rollout as reference for background preservation
    print("\nComputing clean rollout reference...")
    
    # Normalize clean image
    if hasattr(processor, 'image_processor'):
        config = processor.image_processor
    else:
        config = processor
    mean = torch.tensor(config.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std, device=device).view(1, 3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # Get clean attentions (no gradients needed)
    capture.reset()
    with torch.no_grad():
        _ = capture.vision_model(img_normalized, output_attentions=True, output_hidden_states=True)
    
    # Compute rollout using AttentionRollout class
    rollout_module = AttentionRollout(model, head_fusion="max", discard_ratio=0.5)
    attentions_to_use = capture.attentions if args.num_layers < 0 else capture.attentions[:args.num_layers]
    
    # Get rollout - returns (mask, result_matrix) when differentiable=False
    _, clean_rollout_matrix = rollout_module.rollout(attentions_to_use, differentiable=False)
    
    # Extract CLS token attention to all patches: result_matrix[:, 0, 1:]
    # Index 0 is CLS token, indices 1: are image patches
    clean_cls_to_patches = clean_rollout_matrix[:, 0, 1:]  # (batch, num_patches)
    
    # Identify background patches (non-ROI)
    patch_h = target_size[0] // patch_size
    patch_w = target_size[1] // patch_size
    total_patches = patch_h * patch_w
    
    # Convert ROI patch tuples (r, c) to flat indices
    roi_flat_indices = set(r * patch_w + c for r, c in roi_patches)
    background_patch_indices = [i for i in range(total_patches) if i not in roi_flat_indices]
    
    # Extract clean rollout values for background patches only
    clean_bg_rollout = clean_cls_to_patches[:, background_patch_indices]  # (batch, num_bg_patches)
    
    print(f"  Total patches: {total_patches}")
    print(f"  ROI patches: {len(roi_flat_indices)}")
    print(f"  Background patches: {len(background_patch_indices)} ({100*len(background_patch_indices)/total_patches:.1f}%)")
    print(f"  Stored clean rollout for {clean_bg_rollout.shape[1]} background patches")
    
    # Initialize delta (perturbation)
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    
    # Optimizer for delta
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)
    
    # Training loop
    print(f"\nStarting adversarial training for {args.num_iterations} iterations...")
    print(f"Using {args.num_layers if args.num_layers > 0 else 'all'} layers")
    
    for iteration in tqdm(range(args.num_iterations)):
        optimizer.zero_grad()
        
        # Apply perturbation
        x_adv = img_tensor + delta
        x_adv = torch.clamp(x_adv, 0, 1)  # Keep in valid range
        
        # Normalize using processor's normalization params
        if hasattr(processor, 'image_processor'):
            config = processor.image_processor
        else:
            config = processor
        
        mean = torch.tensor(config.image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(config.image_std, device=device).view(1, 3, 1, 1)
        x_normalized = (x_adv - mean) / std
        
        # Forward pass through vision model
        capture.reset()
        
        with torch.set_grad_enabled(True):
            _ = capture.vision_model(x_normalized, output_attentions=True, output_hidden_states=True)
        
        # Compute attention loss
        patch_h = target_size[0] // patch_size
        patch_w = target_size[1] // patch_size
        all_patches = (patch_h, patch_w)
        
        # Call attention aggregation with background indices to get both ROI and background rollout
        result = attn_agg_fn(
            capture.attentions,
            roi_patches,
            all_patches,
            num_layers=args.num_layers,
            background_patch_indices=background_patch_indices
        )
        
        # Unpack result: if background_patch_indices provided, get both values
        if isinstance(result, tuple):
            attn_loss, adv_bg_rollout = result
        else:
            attn_loss = result
            adv_bg_rollout = None
        
        # Compute value regularization
        if len(capture.values) > 0:
            value_reg = value_agg_fn(
                capture.values,
                roi_patches,
                all_patches,
                num_layers=args.num_layers
            )
        else:
            value_reg = 0.0
        
        # Compute perceptual loss (visual similarity to original)
        perceptual_loss = 0.0
        if args.lambda_perceptual > 0 and args.perceptual_loss != 'none':
            if args.perceptual_loss == 'lpips' and lpips_model is not None:
                perceptual_loss = perceptual_loss_fn(x_adv, img_tensor, lpips_fn=lpips_model)
            elif args.perceptual_loss == 'vgg' and vgg_model is not None:
                perceptual_loss = perceptual_loss_fn(x_adv, img_tensor, vgg_model=vgg_model)
            else:
                perceptual_loss = perceptual_loss_fn(x_adv, img_tensor)
        
        # Compute background preservation loss (match background attention to clean reference)
        bg_preservation_loss = 0.0
        if adv_bg_rollout is not None and clean_bg_rollout is not None:
            # MSE loss between adversarial and clean background rollout values
            bg_preservation_loss = F.mse_loss(adv_bg_rollout, clean_bg_rollout)
        
        # Total loss (minimize ROI attention, regularize values, maintain visual similarity, preserve background)
        loss = 10 * attn_loss + args.lambda_v * value_reg + args.lambda_perceptual * perceptual_loss + args.lambda_bg * bg_preservation_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Project delta to satisfy epsilon constraint
        with torch.no_grad():
            if args.epsilon > 0:
                if args.norm == 'linf':
                    # L-infinity: clamp each pixel to [-epsilon, epsilon]
                    delta.clamp_(-args.epsilon, args.epsilon)
                elif args.norm == 'l2':
                    # L2: scale delta if ||delta||_2 > epsilon
                    delta_norm = torch.norm(delta)
                    if delta_norm > args.epsilon:
                        delta.mul_(args.epsilon / delta_norm)
        
        # Log progress
        if (iteration + 1) % args.log_interval == 0:
            delta_linf = torch.max(torch.abs(delta)).item()
            delta_l2 = torch.norm(delta).item()
            print(f"\nIteration {iteration + 1}/{args.num_iterations}")
            print(f"  Attention Loss: {attn_loss.item():.6f}")
            print(f"  Value Reg: {value_reg if isinstance(value_reg, float) else value_reg.item():.6f}")
            if args.lambda_perceptual > 0 and args.perceptual_loss != 'none':
                print(f"  Perceptual Loss ({args.perceptual_loss}): {perceptual_loss if isinstance(perceptual_loss, float) else perceptual_loss.item():.6f}")
            if args.lambda_bg > 0:
                print(f"  Background Preservation: {bg_preservation_loss if isinstance(bg_preservation_loss, float) else bg_preservation_loss.item():.6f}")
            print(f"  Total Loss: {loss.item():.6f}")
            print(f"  Delta L∞: {delta_linf:.6f}, L2: {delta_l2:.6f}")
    
    # Save adversarial image
    os.makedirs(args.output_dir, exist_ok=True)
    
    x_adv_final = torch.clamp(img_tensor + delta, 0, 1)
    x_adv_np = (x_adv_final.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    adv_image = Image.fromarray(x_adv_np)
    
    output_path = os.path.join(args.output_dir, f"adversarial_{os.path.basename(args.image_path)}")
    adv_image.save(output_path)
    print(f"\nSaved adversarial image to: {output_path}")
    
    # Also save delta for analysis
    delta_np = (delta.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.int16)
    delta_path = os.path.join(args.output_dir, f"delta_{os.path.basename(args.image_path).replace('.JPEG', '.npy')}")
    np.save(delta_path, delta_np)
    print(f"Saved delta to: {delta_path}")
    
    # Evaluate adversarial image if requested
    if args.evaluate:
        print("\n" + "=" * 60)
        print("STARTING EVALUATION")
        print("=" * 60)
        
        # Load mapping file if provided
        mapping = None
        if args.mapping_path and os.path.exists(args.mapping_path):
            import json
            with open(args.mapping_path, 'r') as f:
                mapping = json.load(f)
            print(f"Loaded label mapping from: {args.mapping_path}")
        elif args.mapping_path:
            print(f"Warning: Mapping file not found: {args.mapping_path}")
            print("Will use ImageNet IDs as-is")
        
        # Load clean image
        clean_img = Image.open(args.image_path).convert('RGB')
        
        # Evaluate
        eval_results = evaluate_adversarial_image(
            model=model,
            processor=processor,
            clean_image=clean_img,
            adv_image=adv_image,
            objects=scaled_objects,
            model_type=args.model_id,
            mapping=mapping
        )
        
        # Save attention rollout visualizations if generated
        if eval_results.get('visualizations'):
            import cv2
            vis_clean = eval_results['visualizations']['clean_rollout']
            vis_adv = eval_results['visualizations']['adv_rollout']
            
            clean_vis_path = os.path.join(args.output_dir, f"rollout_clean_{os.path.basename(args.image_path)}")
            adv_vis_path = os.path.join(args.output_dir, f"rollout_adv_{os.path.basename(args.image_path)}")
            
            cv2.imwrite(clean_vis_path, cv2.cvtColor(vis_clean, cv2.COLOR_RGB2BGR))
            cv2.imwrite(adv_vis_path, cv2.cvtColor(vis_adv, cv2.COLOR_RGB2BGR))
            
            print(f"\nSaved attention rollout visualizations:")
            print(f"  Clean: {clean_vis_path}")
            print(f"  Adversarial: {adv_vis_path}")
            
            # Remove from results before JSON serialization
            del eval_results['visualizations']
        
        # Save evaluation results
        eval_path = os.path.join(args.output_dir, f"eval_{os.path.basename(args.image_path).replace('.JPEG', '.json')}")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved evaluation results to: {eval_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Train Adversarial Image using VIP method")
    
    # Data paths
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to XML annotation")
    parser.add_argument("--output_dir", type=str, default="adversarial_output", help="Output directory")
    
    # Model
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl",
                        choices=[
                            "Salesforce/blip2-flan-t5-xl",
                            "llava-hf/llava-1.5-7b-hf",
                            "Salesforce/instructblip-vicuna-7b",
                        ],
                        help="Model ID")
    
    # Training parameters
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for delta optimization")
    parser.add_argument("--lambda_v", type=float, default=1.0, help="Weight for value regularization")
    parser.add_argument("--lambda_bg", type=float, default=0.0,
                        help="Weight for background attention preservation (0=off). Recommended: 0.5-2.0 to preserve background understanding")
    
    # Perturbation constraints
    parser.add_argument("--epsilon", type=float, default=0.0, 
                        help="Maximum perturbation magnitude (0 = no constraint). Recommended: 0.03 for linf, 1.0 for l2")
    parser.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"],
                        help="Norm for epsilon constraint: linf (per-pixel) or l2 (total energy)")
    
    # Layer selection
    parser.add_argument("--num_layers", type=int, default=-1,
                        help="Number of layers to use for aggregation. -1 = all layers")
    
    # Aggregation methods
    parser.add_argument("--attention_aggregation", type=str, default="rollout",
                        choices=list(ATTENTION_AGGREGATION_METHODS.keys()),
                        help="Method to aggregate attention")
    parser.add_argument("--value_aggregation", type=str, default="l2",
                        choices=list(VALUE_AGGREGATION_METHODS.keys()),
                        help="Method to aggregate values")
    
    # Perceptual loss
    parser.add_argument("--perceptual_loss", type=str, default="none",
                        choices=list(PERCEPTUAL_LOSS_METHODS.keys()),
                        help="Perceptual loss method: none, mse, lpips, ssim")
    parser.add_argument("--lambda_perceptual", type=float, default=0.0,
                        help="Weight for perceptual loss. Recommended: mse=1-10, lpips=0.1-1.0, ssim=1-5")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate adversarial image by querying VLM about ROI objects")
    parser.add_argument("--mapping_path", type=str, default=None,
                        help="Path to ImageNet ID to label mapping JSON (e.g., imagenet_mapping.json)")
    
    args = parser.parse_args()
    
    train_adversarial_image(args)


if __name__ == "__main__":
    main()
