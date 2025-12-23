import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration, InstructBlipForConditionalGeneration, InstructBlipProcessor
from attention_flow import AttentionFlow

# Perceptual loss dependencies
import lpips
import pytorch_msssim
import torchvision.models as models

from utils import preprocess_image_and_boxes, get_patch_size, get_patch_indices
from attention_rollout import AttentionRollout
from lrp_relevance import lrp_aggregation_loss, compute_transformer_attribution, lrp_class_specific_loss

EPS = 1e-9

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
    total_elements = 0

    roi_token_indices = [
        1 + r * patch_w + c
        for r, c in roi_patches
    ]
    roi_token_indices = torch.tensor(roi_token_indices, device=selected_attentions[0].device)
    
    for attn in selected_attentions:
        seq_len = attn.shape[3]
        valid_mask = roi_token_indices < seq_len
        valid_indices = roi_token_indices[valid_mask]
        if len(valid_indices) == 0:
            continue
        attn_to_roi = attn[:, :, :, valid_indices]
        total_attention += attn_to_roi.sum()
        total_elements += attn_to_roi.numel()
    
    if total_elements > 0:
        total_attention = total_attention / total_elements
    return total_attention

def attention_rollout_aggregation(attentions, roi_patches, all_patches, num_layers=-1, head_fusion='max', discard_ratio=0.5):
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
        
    Returns:
        Scalar loss value
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_attentions = attentions
    else:
        selected_attentions = attentions[:num_layers]
    
    if not selected_attentions:
        return 0.0
    
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
    
    return total_rollout

# New added attention flow function
def attention_flow_aggregation(attentions, roi_patches, all_patches, num_layers=-1, head_fusion='max', discard_ratio=0.5):
    """
    Use Attention Flow logic (Recursive multiplication without strong residual bias).
    """
    # Determine which layers to use
    if num_layers < 0:
        selected_attentions = attentions
    else:
        selected_attentions = attentions[:num_layers]
    
    if not selected_attentions:
        return 0.0
    
    batch_size, num_heads, seq_len, _ = selected_attentions[0].shape
    device = selected_attentions[0].device
    patch_h, patch_w = all_patches
    
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
        
        # === FLOW LOGIC ===
        # Unlike Rollout, we do NOT mix with Identity (I) using 0.5/0.5
        # We use the raw attention probabilities to simulate pure flow.
        a = attention
        
        # Ensure normalization
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Matrix multiplication
        result = torch.bmm(a, result)
    
    # Sum attention from CLS token to ROI patches
    total_flow = 0.0
    for roi_r, roi_c in roi_patches:
        roi_token_idx = 1 + roi_r * patch_w + roi_c
        if roi_token_idx < seq_len:
            # Attention from CLS (index 0) to this ROI token
            total_flow += result[:, 0, roi_token_idx].sum()
    
    return total_flow

def softmin(a, b, tau=0.1):
    # soft approximation of min(a,b)
    # = -tau * log( exp(-a/tau) + exp(-b/tau) )
    return -tau * torch.log(torch.exp(-a / tau) + torch.exp(-b / tau) + 1e-12)

def softmax_max(x, dim=0, tau=0.02):
    # replaces max(x) ≈ tau * logsumexp(x/tau)
    return tau * torch.logsumexp(x / tau, dim=dim)

def fuse_heads_and_normalize_tensor(attentions_tensor, head_fusion='mean'):
    """
    attentions_tensor: (n_layers, batch, n_heads, L, L)
    returns: A tensor of shape (n_layers, batch, L, L) where each layer is fused and row-normalized.
    """
    n_layers, batch, n_heads, L, _ = attentions_tensor.shape
    device, dtype = attentions_tensor.device, attentions_tensor.dtype
    eye = torch.eye(L, device=device, dtype=dtype)[None, None, ...]

    if head_fusion == 'mean':
        fused = attentions_tensor.mean(dim=2)
    elif head_fusion == 'max':
        fused = attentions_tensor.max(dim=2)[0]
    elif head_fusion == 'min':
        fused = attentions_tensor.min(dim=2)[0]
    else:
        raise ValueError("head_fusion must be 'mean' | 'max' | 'min'")

    # fused = fused + eye
    fused = fused / (fused.sum(dim=-1, keepdim=True) + EPS)
    return fused


def _compute_dp_for_single_example(A_per_layer, source_indices, tau=0.02, per_layer_norm=True):
    """
    Soft differentiable version.
    """
    n_layers, L, _ = A_per_layer.shape
    device, dtype = A_per_layer.device, A_per_layer.dtype

    F_by_layer = [None] * (n_layers + 1)

    F_top = torch.zeros((L,), device=device, dtype=dtype)
    F_top[source_indices] = 1.0
    F_by_layer[n_layers] = F_top

    for l in reversed(range(n_layers)):
        F_next = F_by_layer[l + 1]
        A_l = A_per_layer[l]

        min_mat = softmin(F_next[:, None], A_l, tau=tau)
        F_curr = softmax_max(min_mat, dim=0, tau=tau)

        if per_layer_norm:
            F_curr = F_curr / (F_curr.max() + EPS)
        F_by_layer[l] = F_curr

    return F_by_layer


def get_flow_relevance_torch(attentions, input_tokens, layer, head_fusion='mean',
                             source_indices=None, device=None):
    """
    attentions: list or tensor of shape (n_layers, batch, n_heads, L, L)
                (function expects batch == 1)
    input_tokens: list of token labels length L (used to build labels_to_index)
    layer: selected layer index (0-based); if negative, caller should supply proper index beforehand.
    Returns:
      relevance_attention_raw: tensor (L,) relevance per input token (layer 0)
      final_layer_attention_raw: tensor (L_source, L_target) for requested layer (single example)
      labels_to_index: dict mapping label->index consistent with adjacency indexing
    """
    device = device or attentions.device
    # Convert input to tensor if list provided
    if isinstance(attentions, list):
        attentions_tensor = torch.stack(attentions, dim=0).to(device)
    else:
        attentions_tensor = attentions.to(device)

    n_layers, batch, n_heads, L, _ = attentions_tensor.shape
    if batch != 1:
        raise ValueError("This function expects batch == 1 (strip batch before calling if needed).")

    if layer < 0:
        layer_idx = n_layers - 1
    else:
        layer_idx = layer

    labels_to_index = {f"{k}_{input_tokens[k]}": k for k in range(L)}
    for i in range(1, n_layers + 1):
        for k in range(L):
            labels_to_index[f"L{i}_{k}"] = i * L + k

    fused = fuse_heads_and_normalize_tensor(attentions_tensor, head_fusion=head_fusion)

    A = fused[:, 0, :, :]

    if source_indices is None:
        source_indices = [0]

    F_by_layer = _compute_dp_for_single_example(A, source_indices, tau=0.02, per_layer_norm=True)

    relevance_attention_raw = F_by_layer[0]
    src_strengths = F_by_layer[layer_idx + 1]
    A_block = A[layer_idx]
    final_layer_attention_raw = softmin(src_strengths[:, None], A_block, tau=0.02)

    return relevance_attention_raw, final_layer_attention_raw, labels_to_index


def attention_flow_loss(attentions, roi_patches, all_patches, num_layers=-1,
                        head_fusion='mean', source_indices=None, device=None):
    """
    attentions: (n_layers, batch, n_heads, L, L)  (expects batch==1)
    roi_patches: list of (r,c) tuples
    all_patches: (patch_h, patch_w)
    num_layers: if <0 -> use last layer; else layer_idx = num_layers  (0-based)
    Returns:
      scalar loss (torch tensor) summing relevance at ROI patch tokens.
    """
    device = device or attentions[0].device

    if isinstance(attentions, list):
        attentions_tensor = torch.stack(attentions, dim=0).to(device)
    else:
        attentions_tensor = attentions.to(device)

    n_layers, batch, n_heads, L, _ = attentions_tensor.shape
    if batch != 1:
        raise ValueError("attention_flow_loss expects batch size 1 (per your request).")

    layer_idx = n_layers - 1 if num_layers < 0 else num_layers
    patch_h, patch_w = all_patches

    input_tokens = ["CLS"] + [f"P{i}" for i in range(L - 1)]

    relevance, _, _ = get_flow_relevance_torch(attentions_tensor, input_tokens, layer_idx,
                                               head_fusion=head_fusion, source_indices=source_indices,
                                               device=device)
    
    total_loss = torch.tensor(0.0, device=device, dtype=relevance.dtype)
    for (r, c) in roi_patches:
        roi_token_idx = 1 + r * patch_w + c
        if roi_token_idx < L:
            total_loss = total_loss + relevance[roi_token_idx]
    print(f"DEBUG: attention_flow_loss = {total_loss.item()}")
    return total_loss

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

def attention_lrp_aggregation(attentions, roi_patches, all_patches, num_layers=-1, values=None):
    """
    Use LRP (Layer-wise Relevance Propagation) to aggregate attention.

    This method tracks actual information flow through the network including
    residual connections, unlike raw attention or rollout.

    Args:
        attentions: List of attention tensors
        roi_patches: List of (row, col) tuples
        all_patches: Total patches (height, width)
        num_layers: Number of layers to use (-1 = all)
        values: List of value tensors (required for LRP)

    Returns:
        Scalar loss value
    """
    if values is None:
        raise ValueError("LRP aggregation requires values to be passed")
    # LRP expects both attentions and values
    return lrp_aggregation_loss(attentions, values, roi_patches, all_patches, num_layers)


def attention_transformer_attribution(attentions, roi_patches, all_patches, num_layers=-1):
    """
    Use Transformer Attribution (gradient-weighted attention rollout).

    NOTE: This requires attention tensors to have gradients attached.
    You must call loss.backward() before computing this.

    Args:
        attentions: List of attention tensors with gradients
        roi_patches: List of (row, col) tuples
        all_patches: Total patches (height, width)
        num_layers: Number of layers to use (-1 = all)

    Returns:
        Scalar loss value
    """
    return compute_transformer_attribution(attentions, roi_patches, all_patches, num_layers)


def attention_lrp_class_specific(attentions, roi_patches, all_patches, num_layers=-1, values=None,
                                  model=None, processor=None, image_tensor=None, target_classes=None, device=None):
    """
    Use class-specific LRP to aggregate attention for specific object classes.

    This targets hiding specific classes (e.g., "porcupine") rather than all information.

    Args:
        attentions: List of attention tensors
        roi_patches: List of (row, col) tuples
        all_patches: Total patches (height, width)
        num_layers: Number of layers to use (-1 = all)
        values: List of value tensors (required)
        model: VLM model (required for class-specific)
        processor: Model processor (required for class-specific)
        image_tensor: Preprocessed image (required for class-specific)
        target_classes: List of class names to hide (required for class-specific)
        device: Torch device (required for class-specific)

    Returns:
        Scalar loss value
    """
    if values is None:
        raise ValueError("Class-specific LRP requires values to be passed")
    if model is None or processor is None or image_tensor is None or target_classes is None or device is None:
        raise ValueError("Class-specific LRP requires: model, processor, image_tensor, target_classes, device")

    return lrp_class_specific_loss(
        model, processor, image_tensor, target_classes,
        attentions, values, roi_patches, all_patches, device, num_layers
    )


ATTENTION_AGGREGATION_METHODS = {
    'sum': attention_sum,
    'rollout': attention_rollout_aggregation,
    'flow': attention_flow_aggregation,
    'lrp': attention_lrp_aggregation,
    'lrp_class_specific': attention_lrp_class_specific,
    'transformer_attribution': attention_transformer_attribution,
    'flow_loss': attention_flow_loss,
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
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,   # Shorter = less hallucination
            do_sample=False,     # Greedy decoding = deterministic
            num_beams=3,         # Beam search = better quality
            early_stopping=True  # Stop when answer complete
        )
    
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
    # Select device: CUDA or CPU (MPS disabled due to memory issues with large models)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading model: {args.model_id}")
    if 'instructblip' in args.model_id.lower():
        processor = InstructBlipProcessor.from_pretrained(args.model_id)
    else:
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
    
    # Get ROI patch indices and extract class names
    roi_patches = []
    target_classes = []

    # Load mapping if available for class-specific LRP
    mapping = None
    if args.use_class_specific and args.mapping_path and os.path.exists(args.mapping_path):
        with open(args.mapping_path, 'r') as f:
            mapping = json.load(f)
        print(f"Loaded class mapping for class-specific LRP: {args.mapping_path}")

    for obj in scaled_objects:
        bbox = obj['bbox']
        patches = get_patch_indices(bbox, target_size, patch_size)
        roi_patches.extend(patches)

        # Extract class name for class-specific LRP
        if args.use_class_specific:
            obj_id = obj['name']
            if mapping and obj_id in mapping:
                class_name = mapping[obj_id]
                if class_name not in target_classes:
                    target_classes.append(class_name)
            else:
                print(f"Warning: No mapping found for object ID: {obj_id}")

    roi_patches = list(set(roi_patches))  # Remove duplicates

    print(f"ROI contains {len(roi_patches)} patches")
    if args.use_class_specific and target_classes:
        print(f"Target classes for hiding: {target_classes}")
    
    # Convert preprocessed image to tensor
    img_array = np.array(preprocessed_img)
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    
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

        # Pass values to aggregation function if using LRP
        if args.attention_aggregation == 'lrp_class_specific':
            # Class-specific LRP needs extra parameters
            attn_loss = attn_agg_fn(
                capture.attentions,
                roi_patches,
                all_patches,
                num_layers=args.num_layers,
                values=capture.values,
                model=model,
                processor=processor,
                image_tensor=x_adv,
                target_classes=target_classes,
                device=device
            )
        elif args.attention_aggregation in ['lrp', 'transformer_attribution']:
            attn_loss = attn_agg_fn(
                capture.attentions,
                roi_patches,
                all_patches,
                num_layers=args.num_layers,
                values=capture.values
            )
        else:
            attn_loss = attn_agg_fn(
                capture.attentions,
                roi_patches,
                all_patches,
                num_layers=args.num_layers
            )
        
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
        
        # Total loss (minimize attention to ROI, regularize values, maintain visual similarity)
        loss = attn_loss + args.lambda_v * value_reg + args.lambda_perceptual * perceptual_loss
        
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
        
        # Save evaluation results
        eval_path = os.path.join(args.output_dir, f"eval_{os.path.basename(args.image_path).replace('.JPEG', '.json')}")
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved evaluation results to: {eval_path}")

    # Visualize attention if requested
    if args.visualize_attention:
        print("\n" + "=" * 60)
        print("CREATING ATTENTION VISUALIZATIONS")
        print("=" * 60)

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend

            # Run one more forward pass to capture attention
            capture.reset()
            with torch.no_grad():
                x_final = torch.clamp(img_tensor + delta, 0, 1)
                if hasattr(processor, 'image_processor'):
                    config = processor.image_processor
                else:
                    config = processor
                mean = torch.tensor(config.image_mean, device=device).view(1, 3, 1, 1)
                std = torch.tensor(config.image_std, device=device).view(1, 3, 1, 1)
                x_normalized = (x_final - mean) / std
                _ = capture.vision_model(x_normalized, output_attentions=True, output_hidden_states=True)

            if len(capture.attentions) == 0:
                print("Warning: No attentions captured for visualization")
            else:
                # Compute attention maps
                batch_size, num_heads, seq_len, _ = capture.attentions[0].shape
                num_patches = seq_len - 1
                patch_size_val = int(np.sqrt(num_patches))

                # 1. Raw attention (average over layers and heads)
                raw_attention = []
                for attn in capture.attentions:
                    attn_mean = attn.mean(dim=1)  # Average over heads
                    cls_attn = attn_mean[0, 0, 1:]  # CLS to patches
                    raw_attention.append(cls_attn)
                raw_attention = torch.stack(raw_attention).mean(dim=0)  # Average over layers
                raw_attention_map = raw_attention.cpu().numpy().reshape(patch_size_val, patch_size_val)

                # 2. Attention Rollout
                result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
                for attention in capture.attentions:
                    attention_mean = torch.mean(attention, dim=1)
                    I = torch.eye(seq_len, device=device).unsqueeze(0)
                    a = (attention_mean + I) / 2.0
                    a = a / a.sum(dim=-1, keepdim=True)
                    result = torch.bmm(a, result)
                rollout_map = result[0, 0, 1:].cpu().numpy().reshape(patch_size_val, patch_size_val)

                # 3. LRP if values available
                if len(capture.values) > 0 and args.attention_aggregation == 'lrp':
                    from lrp_relevance import compute_lrp_rollout
                    lrp_relevance = compute_lrp_rollout(capture.attentions, capture.values)
                    lrp_map = lrp_relevance[0, 1:].cpu().numpy().reshape(patch_size_val, patch_size_val)
                else:
                    lrp_map = None

                # Normalize maps
                raw_attention_map = (raw_attention_map - raw_attention_map.min()) / (raw_attention_map.max() - raw_attention_map.min() + 1e-8)
                rollout_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)
                if lrp_map is not None:
                    lrp_map = (lrp_map - lrp_map.min()) / (lrp_map.max() - lrp_map.min() + 1e-8)

                # Create visualization
                num_plots = 3 if lrp_map is not None else 2
                fig, axes = plt.subplots(1, num_plots + 1, figsize=(5 * (num_plots + 1), 5))

                # Original image
                axes[0].imshow(adv_image)
                axes[0].set_title('Adversarial Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                # Raw attention
                im1 = axes[1].imshow(raw_attention_map, cmap='hot', interpolation='bilinear')
                axes[1].set_title(f'Raw Attention', fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                # Rollout
                im2 = axes[2].imshow(rollout_map, cmap='hot', interpolation='bilinear')
                axes[2].set_title(f'Attention Rollout', fontsize=12)
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)

                # LRP if available
                if lrp_map is not None:
                    im3 = axes[3].imshow(lrp_map, cmap='hot', interpolation='bilinear')
                    axes[3].set_title(f'LRP Relevance', fontsize=12, fontweight='bold')
                    axes[3].axis('off')
                    plt.colorbar(im3, ax=axes[3], fraction=0.046)

                plt.tight_layout()

                # Save figure
                viz_path = os.path.join(args.output_dir, f"attention_viz_{os.path.basename(args.image_path)}")
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"✓ Saved attention visualization to: {viz_path}")

        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")
        except Exception as e:
            print(f"Warning: Failed to create visualization: {e}")
            import traceback
            traceback.print_exc()


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

    # Class-specific LRP
    parser.add_argument("--use_class_specific", action="store_true",
                        help="Use class-specific LRP (only works with --attention_aggregation lrp_class_specific)")
    
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

    # Visualization
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Save attention heatmap visualizations after training")
    
    args = parser.parse_args()
    
    train_adversarial_image(args)


if __name__ == "__main__":
    main()
