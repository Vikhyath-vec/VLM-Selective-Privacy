import torch
import numpy as np
import math

class AttentionFlow:
    """
    Class to compute Attention Flow for Vision Transformers.
    Unlike Rollout, Flow typically tracks raw attention probabilities without 
    averaging with the identity matrix (residual), or uses a custom residual weight.
    """
    def __init__(self, model, attention_layer_name='self_attn', head_fusion='mean', discard_ratio=0.9):
        """
        Initialize the AttentionFlow class.
        """
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.vision_model = self._get_vision_model(model)
        
        # Register hooks
        for name, module in self.vision_model.named_modules():
            if name.endswith(self.attention_layer_name):
                module.register_forward_hook(self.get_attention_hook)
        self.attentions = []
    
    def get_attention_hook(self, module, input, output):
        # Handle tuple output (hidden_states, attentions)
        if isinstance(output, tuple):
            attn_probs = output[1]
        else:
            attn_probs = output
            
        if attn_probs is not None:
            self.attentions.append(attn_probs.detach().cpu())        
    
    def _get_vision_model(self, model):
        # Same helper as in AttentionRollout
        if hasattr(model, 'vision_tower'):
            if hasattr(model.vision_tower, 'set_attn_implementation'):
                model.vision_tower.set_attn_implementation('eager')
            if hasattr(model.vision_tower, 'vision_model'):
                return model.vision_tower.vision_model
            return model.vision_tower
        
        if hasattr(model, 'vision_model'):
            if hasattr(model.vision_model, 'set_attn_implementation'):
                model.vision_model.set_attn_implementation('eager')
            return model.vision_model
            
        return model
    
    def compute_flow(self, attentions, differentiable=False, residual_weight=0.0):
        """
        Compute the attention flow.
        
        Args:
            attentions (list): List of attention matrices.
            differentiable (bool): Maintain gradients.
            residual_weight (float): Weight for residual connection (Identity). 
                                     Rollout uses 0.5. Flow typically uses 0.0 (raw) or learned weights.
        """
        if not attentions:
            raise ValueError("No attentions captured.")
            
        batch_size, num_heads, seq_len, _ = attentions[0].shape
        device = attentions[0].device
        
        # Initialize result as identity matrix
        result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        context = torch.enable_grad() if differentiable else torch.no_grad()
        
        with context:
            for attention in attentions:
                # Fuse heads
                if self.head_fusion == "mean":
                    attention = torch.mean(attention, dim=1)
                elif self.head_fusion == "max":
                    attention = torch.max(attention, dim=1)[0]
                elif self.head_fusion == "min":
                    attention = torch.min(attention, dim=1)[0]
                
                # Discard low attention values (Noise filtering)
                if self.discard_ratio > 0:
                    flat = attention.view(batch_size, -1)
                    num_discard = int(flat.size(-1) * self.discard_ratio)
                    _, idx = torch.topk(flat, num_discard, dim=-1, largest=False)
                    
                    for b in range(batch_size):
                        idx_b = idx[b]
                        idx_b = idx_b[idx_b != 0]
                        flat[b, idx_b] = 0.0
                    attention = flat.view(batch_size, seq_len, seq_len)
                
                # === KEY DIFFERENCE FROM ROLLOUT ===
                # Instead of hardcoded (A + I)/2, we use explicit residual weight.
                # If residual_weight is 0, this is pure Attention Flow.
                if residual_weight > 0:
                    I = torch.eye(seq_len, device=device).unsqueeze(0)
                    a = (1 - residual_weight) * attention + (residual_weight * I)
                else:
                    a = attention
                
                # Normalize row sum to 1 to maintain probability mass
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)
                
                # Matrix multiplication: Current_Layer x Accumulated_Flow
                result = torch.bmm(a, result)

        if differentiable:
            return result
        else:
            mask = result[:, 0, 1:]
            size = int(math.sqrt(seq_len - 1))
            mask = mask.view(batch_size, size, size)
            mask = mask.cpu().numpy()
            mask = mask / (mask.max(axis=(1, 2), keepdims=True) + 1e-9)
            return mask, result

    def __call__(self, input_tensor, layer_idx=-1):
        self.attentions = []
        self.vision_model.eval()
        with torch.no_grad():
            self.vision_model(input_tensor, output_attentions=True)
        
        if layer_idx < 0:
            attentions = self.attentions
        else:
            attentions = self.attentions[:layer_idx + 1]
            
        return self.compute_flow(attentions)