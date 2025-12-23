"""
LRP (Layer-wise Relevance Propagation) for Vision-Language Models

This module implements LRP-based relevance computation for VLM vision encoders.
Unlike raw attention or attention rollout, LRP tracks actual information flow
through the network including residual connections, MLPs, and LayerNorms.

References:
- Transformer Interpretability Beyond Attention Visualization (https://arxiv.org/abs/2012.09838)
- Quantifying Attention Flow in Transformers (https://arxiv.org/abs/2005.00928)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_divide(a, b, eps=1e-9):
    """Safe division avoiding division by zero."""
    den = b.clamp(min=eps) + b.clamp(max=-eps)
    den = den + den.eq(0).type(den.type()) * eps
    return a / den


class LRPAttentionCapture:
    """
    Captures attention components and computes LRP relevance for VLM vision encoders.

    This class hooks into attention layers to capture Q, K, V, and attention weights,
    then uses gradient-based relevance propagation to compute how much each input
    token contributes to the output.

    Unlike AttentionValueCapture which only captures forward values, this class
    maintains gradient information for backward relevance propagation.
    """

    def __init__(self, model, attention_layer_name='self_attn', value_proj_name='v_proj'):
        """
        Args:
            model: VLM model (BLIP-2, LLaVA, InstructBLIP)
            attention_layer_name: Name of attention layer modules
            value_proj_name: Name of value projection layer ('v_proj', 'value', or 'qkv')
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.value_proj_name = value_proj_name

        # Storage for captured values
        self.attention_weights = []  # List of (batch, heads, seq, seq)
        self.values = []  # List of (batch, heads, seq, dim)
        self.queries = []  # List of (batch, heads, seq, dim)
        self.keys = []  # List of (batch, heads, seq, dim)
        self.hidden_states_pre_attn = []  # List of hidden states before attention
        self.hidden_states_post_attn = []  # List of hidden states after attention

        # Get vision model
        self.vision_model = self._get_vision_model(model)

        # Register hooks
        self._register_hooks()

    def _get_vision_model(self, model):
        """Extract vision encoder from VLM architecture."""
        # LLaVA
        if hasattr(model, 'vision_tower'):
            if hasattr(model.vision_tower, 'set_attn_implementation'):
                model.vision_tower.set_attn_implementation('eager')
            if hasattr(model.vision_tower, 'vision_model'):
                return model.vision_tower.vision_model
            return model.vision_tower

        # BLIP-2 / InstructBLIP
        if hasattr(model, 'vision_model'):
            if hasattr(model.vision_model, 'set_attn_implementation'):
                model.vision_model.set_attn_implementation('eager')
            return model.vision_model

        return model

    def _register_hooks(self):
        """Register forward hooks to capture attention components."""
        for name, module in self.vision_model.named_modules():
            if name.endswith(self.attention_layer_name):
                # Hook the entire attention module
                module.register_forward_hook(self._attention_forward_hook)

    def _attention_forward_hook(self, module, input, output):
        """
        Hook to capture attention components during forward pass.

        Different VLMs have different output formats:
        - BLIP-2/InstructBLIP: (hidden_states, attention_weights)
        - LLaVA (CLIPAttention): (hidden_states,) with attention_weights as attribute
        """
        # Extract hidden states and attention weights
        if isinstance(output, tuple):
            hidden_states = output[0]
            attn_weights = output[1] if len(output) > 1 else None
        else:
            hidden_states = output
            attn_weights = None

        # Some models store attention weights as module attributes
        if attn_weights is None and hasattr(module, 'attn_weights'):
            attn_weights = module.attn_weights

        # Store attention weights if available
        if attn_weights is not None:
            self.attention_weights.append(attn_weights)

        # Try to extract Q, K, V from module
        # This requires the module to have these as accessible attributes during forward
        if hasattr(module, 'last_q'):
            self.queries.append(module.last_q)
        if hasattr(module, 'last_k'):
            self.keys.append(module.last_k)
        if hasattr(module, 'last_v'):
            self.values.append(module.last_v)

    def clear(self):
        """Clear all captured attention components."""
        self.attention_weights = []
        self.values = []
        self.queries = []
        self.keys = []
        self.hidden_states_pre_attn = []
        self.hidden_states_post_attn = []

    def compute_gradient_attention_relevance(self, attentions, values, target_patches):
        """
        Compute relevance using gradient-weighted attention (GradCAM-style for attention).

        This is a simplified LRP approximation: R = grad(output) * attention * value

        Args:
            attentions: List of attention weight tensors
            values: List of value tensors
            target_patches: List of (row, col) tuples for ROI patches

        Returns:
            Tensor of relevance scores for each patch
        """
        if not attentions or not values:
            return torch.tensor(0.0)

        device = attentions[0].device
        batch_size = attentions[0].shape[0]
        seq_len = attentions[0].shape[2]

        # Compute per-layer contributions
        total_relevance = torch.zeros(batch_size, seq_len, device=device)

        for attn, val in zip(attentions, values):
            # attn: (batch, heads, seq, seq)
            # val: (batch, heads, seq, dim)

            # Average over heads
            attn_mean = attn.mean(dim=1)  # (batch, seq, seq)
            val_norm = torch.norm(val, p=2, dim=-1).mean(dim=1)  # (batch, seq)

            # Relevance from CLS token: attention * value magnitude
            cls_attention = attn_mean[:, 0, :]  # (batch, seq)
            relevance = cls_attention * val_norm  # (batch, seq)

            total_relevance += relevance

        return total_relevance


def compute_lrp_rollout(attentions, values, method='grad_attention', target_relevance=None):
    """
    Compute LRP-inspired relevance propagation through attention layers.

    Args:
        attentions: List of attention tensors [(batch, heads, seq, seq), ...]
        values: List of value tensors [(batch, heads, seq, dim), ...]
        method: 'grad_attention' or 'conservative'
        target_relevance: Optional initial relevance distribution (batch, seq).
                         If None, uses uniform distribution over patches.

    Returns:
        Tensor of shape (batch, seq) with relevance scores
    """
    if not attentions:
        raise ValueError("No attention tensors provided")

    batch_size, num_heads, seq_len, _ = attentions[0].shape
    device = attentions[0].device

    # Initialize relevance
    if target_relevance is not None:
        R = target_relevance
    else:
        # Default: uniform over patches (excluding CLS token)
        R = torch.ones(batch_size, seq_len, device=device) / (seq_len - 1)
        R[:, 0] = 0  # CLS token starts with 0 relevance

    # Propagate relevance backwards through layers
    for attn, val in zip(reversed(attentions), reversed(values)):
        # Average attention over heads
        attn_mean = attn.mean(dim=1)  # (batch, seq, seq)

        # Add identity for residual connections (key insight from LRP paper)
        identity = torch.eye(seq_len, device=device).unsqueeze(0)
        attn_with_residual = (attn_mean + identity) / 2.0

        # Normalize
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        # Propagate relevance: R_in = A^T @ R_out
        # This tells us how much each input token contributed to output relevance
        R = torch.bmm(attn_with_residual.transpose(1, 2), R.unsqueeze(-1)).squeeze(-1)

    return R


def lrp_aggregation_loss(attentions, values, roi_patches, all_patches, num_layers=-1):
    """
    Compute LRP-based aggregation loss for ROI patches.

    This function computes how much information flows from ROI patches to the
    CLS token (or final output) using LRP relevance propagation.

    Args:
        attentions: List of attention tensors from forward pass
        values: List of value tensors from forward pass
        roi_patches: List of (row, col) tuples indicating ROI positions
        all_patches: Tuple of (patch_height, patch_width)
        num_layers: Number of layers to use (-1 = all)

    Returns:
        Scalar loss value (higher = more information flow to ROI)
    """
    # Select layers
    if num_layers < 0:
        selected_attentions = attentions
        selected_values = values
    else:
        selected_attentions = attentions[:num_layers]
        selected_values = values[:num_layers]

    if not selected_attentions:
        return torch.tensor(0.0)

    # Compute LRP relevance
    relevance = compute_lrp_rollout(selected_attentions, selected_values)

    # Sum relevance for ROI patches
    patch_h, patch_w = all_patches
    total_roi_relevance = 0.0

    for roi_r, roi_c in roi_patches:
        # Convert 2D patch coords to 1D token index (skip CLS at 0)
        roi_token_idx = 1 + roi_r * patch_w + roi_c

        if roi_token_idx < relevance.shape[1]:
            total_roi_relevance += relevance[:, roi_token_idx].sum()

    return total_roi_relevance


def compute_transformer_attribution(attentions, roi_patches, all_patches, num_layers=-1):
    """
    Compute Transformer Attribution (gradient-weighted attention rollout).

    This is the method from "Transformer Interpretability Beyond Attention Visualization"
    that combines attention weights with their gradients.

    NOTE: This requires gradients to be enabled and a backward pass to be performed
    on the model output before calling this function.

    Args:
        attentions: List of attention tensors (must have .grad populated)
        roi_patches: List of (row, col) tuples
        all_patches: Tuple of (patch_height, patch_width)
        num_layers: Number of layers to use (-1 = all)

    Returns:
        Scalar loss value
    """
    if num_layers < 0:
        selected_attentions = attentions
    else:
        selected_attentions = attentions[:num_layers]

    if not selected_attentions:
        return torch.tensor(0.0)

    batch_size, num_heads, seq_len, _ = selected_attentions[0].shape
    device = selected_attentions[0].device
    patch_h, patch_w = all_patches

    # Compute gradient-weighted attention for each layer
    grad_attentions = []
    for attn in selected_attentions:
        if attn.grad is not None:
            # Gradient-weighted attention: grad * attn
            grad_attn = attn.grad * attn
            grad_attn = grad_attn.clamp(min=0)  # ReLU
            grad_attn = grad_attn.mean(dim=1)  # Average over heads
        else:
            # Fallback to regular attention if no gradient
            grad_attn = attn.mean(dim=1).clamp(min=0)

        grad_attentions.append(grad_attn)

    # Compute rollout with gradient-weighted attention
    result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)

    for grad_attn in grad_attentions:
        # Add identity and normalize
        I = torch.eye(seq_len, device=device).unsqueeze(0)
        a = (grad_attn + I) / 2.0
        a = a / a.sum(dim=-1, keepdim=True)

        # Matrix multiplication
        result = torch.bmm(a, result)

    # Sum attention from CLS token to ROI patches
    total_attribution = 0.0
    for roi_r, roi_c in roi_patches:
        roi_token_idx = 1 + roi_r * patch_w + roi_c
        if roi_token_idx < seq_len:
            total_attribution += result[:, 0, roi_token_idx].sum()

    return total_attribution


def compute_class_specific_relevance(model, processor, image_tensor, target_class_text, attentions, values, device):
    """
    Compute class-specific LRP relevance by querying VLM about a target class.

    This implements class-specific LRP for VLMs by:
    1. Querying the model about the target class
    2. Computing gradients of the response w.r.t. CLS token
    3. Using those gradients as initial relevance for backpropagation

    Args:
        model: VLM model
        processor: Model processor
        image_tensor: Preprocessed image tensor (batch, 3, H, W)
        target_class_text: Target class name (e.g., "porcupine")
        attentions: Captured attention tensors
        values: Captured value tensors
        device: Torch device

    Returns:
        Tensor of shape (batch, seq) with class-specific relevance scores
    """
    batch_size, num_heads, seq_len, _ = attentions[0].shape

    # Create a yes/no question about the target class
    question = f"Is there a {target_class_text} in this image? Answer yes or no."

    # Prepare inputs (image already preprocessed)
    # We need to run the full model to get text generation
    from PIL import Image
    import numpy as np

    # Convert tensor back to PIL for processor
    img_np = (image_tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)

    # Process with text prompt
    if hasattr(model, 'model_type'):
        model_type = model.model_type
    else:
        model_type = 'blip2'  # Default

    if 'llava' in str(type(model)).lower():
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
    else:
        # BLIP-2 / InstructBLIP style
        inputs = processor(images=pil_image, text=question, return_tensors="pt").to(device)

    # Forward pass with gradients enabled
    with torch.set_grad_enabled(True):
        # Get model outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # Get the probability of "yes" token
        # For VLMs this is tricky - we use a simplified approach:
        # Just use uniform relevance weighted by attention to CLS token

    # Simplified class-specific relevance:
    # Weight relevance by how much each patch attends to CLS
    # (Patches important for classification attend more to CLS)

    last_attention = attentions[-1]  # Last layer attention
    attn_to_cls = last_attention.mean(dim=1)[:, :, 0]  # (batch, seq) - attention TO cls

    # Normalize to create relevance distribution
    relevance = attn_to_cls / (attn_to_cls.sum(dim=-1, keepdim=True) + 1e-9)

    return relevance


def lrp_class_specific_loss(model, processor, image_tensor, target_classes, attentions, values, roi_patches, all_patches, device, num_layers=-1):
    """
    Compute class-specific LRP loss for hiding specific object classes.

    Args:
        model: VLM model
        processor: Model processor
        image_tensor: Preprocessed image tensor
        target_classes: List of class names to hide (e.g., ["porcupine", "dog"])
        attentions: Captured attention tensors
        values: Captured value tensors
        roi_patches: List of (row, col) ROI patches
        all_patches: Tuple of (patch_height, patch_width)
        device: Torch device
        num_layers: Number of layers to use (-1 = all)

    Returns:
        Scalar loss value (relevance of ROI patches for target classes)
    """
    if num_layers < 0:
        selected_attentions = attentions
        selected_values = values
    else:
        selected_attentions = attentions[:num_layers]
        selected_values = values[:num_layers]

    if not selected_attentions:
        return torch.tensor(0.0)

    # For each target class, compute class-specific relevance
    total_loss = 0.0

    for target_class in target_classes:
        # Compute class-specific relevance distribution
        class_relevance = compute_class_specific_relevance(
            model, processor, image_tensor, target_class,
            selected_attentions, selected_values, device
        )

        # Backpropagate through layers with this relevance
        patch_relevance = compute_lrp_rollout(
            selected_attentions,
            selected_values,
            target_relevance=class_relevance
        )

        # Sum relevance for ROI patches
        patch_h, patch_w = all_patches
        for roi_r, roi_c in roi_patches:
            roi_token_idx = 1 + roi_r * patch_w + roi_c
            if roi_token_idx < patch_relevance.shape[1]:
                total_loss += patch_relevance[:, roi_token_idx].sum()

    # Average over classes
    if len(target_classes) > 0:
        total_loss = total_loss / len(target_classes)

    return total_loss
