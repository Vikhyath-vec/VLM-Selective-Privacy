import torch
import numpy as np
import math

class AttentionRollout:
    """
    Class to compute Attention Rollout for Vision Transformers.
    """
    def __init__(self, model, attention_layer_name='self_attn', head_fusion='mean', discard_ratio=0.9):
        """
        Initialize the AttentionRollout class.

        Args:
            model: The model to extract attention from.
            attention_layer_name (str): The name of the attention layer to hook into.
            head_fusion (str): Strategy to fuse attention heads ('mean', 'max', 'min').
            discard_ratio (float): Ratio of attention values to discard (set to 0 to keep all).
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
        """
        Hook to capture attention probabilities.
        """
        # Handle tuple output (hidden_states, attentions)
        if isinstance(output, tuple):
            attn_probs = output[1]
        else:
            attn_probs = output
            
        if attn_probs is not None:
            self.attentions.append(attn_probs.detach().cpu())        
    
    def _get_vision_model(self, model):
        """
        Helper to extract the vision encoder from various VLM architectures.
        """
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
            
        # Fallback: assume the model itself is the vision model
        return model
    
    def rollout(self, attentions, differentiable=False):
        """
        Compute the attention rollout from the captured attentions.

        Args:
            attentions (list): List of attention matrices from the model layers.
            differentiable (bool): If True, maintain gradients for backprop. Default False.

        Returns:
            mask (np.ndarray or torch.Tensor): The computed attention mask.
            result (torch.Tensor): The full rollout matrix.
        """
        if not attentions:
            raise ValueError("No attentions captured. Ensure model is loaded with attn_implementation='eager' if using Transformers >= 4.36.")
            
        batch_size, num_heads, seq_len, _ = attentions[0].shape
        device = attentions[0].device
        
        # Initialize result as identity matrix
        result = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len, seq_len)
        
        # Context manager for gradient control
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
                else:
                    raise ValueError(f"Unknown head_fusion: {self.head_fusion}")
                
                # Discard low attention values
                if self.discard_ratio > 0:
                    flat = attention.view(batch_size, -1)
                    num_discard = int(flat.size(-1) * self.discard_ratio)
                    _, idx = torch.topk(flat, num_discard, dim=-1, largest=False)
                    
                    # Set smallest values to 0
                    # Iterate to handle batch indices correctly with flat view
                    for b in range(batch_size):
                        idx_b = idx[b]
                        idx_b = idx_b[idx_b != 0]
                        flat[b, idx_b] = 0.0
                    attention = flat.view(batch_size, seq_len, seq_len)
                
                # Add identity and normalize
                I = torch.eye(seq_len, device=device).unsqueeze(0)
                a = (attention + I) / 2.0
                a /= a.sum(dim=-1, keepdim=True)
                
                # Matrix multiplication
                result = torch.bmm(a, result)

        if differentiable:
            # Return tensor as-is for gradient flow
            return result
        else:
            # Extract mask for CLS token (index 0) to other tokens
            mask = result[:, 0, 1:]
            
            # Reshape to 2D image
            size = int(math.sqrt(seq_len - 1))
            mask = mask.view(batch_size, size, size)
            
            # Convert to numpy and normalize
            mask = mask.cpu().numpy()
            mask = mask / mask.max(axis=(1, 2), keepdims=True)
            
            return mask, result
    
    def __call__(self, input_tensor, layer_idx=-1):
        """
        Run the model and compute attention rollout.

        Args:
            input_tensor (torch.Tensor): Input images.
            layer_idx (int): The layer index to stop at.

        Returns:
            mask (np.ndarray): The attention rollout mask.
        """
        self.attentions = []
        self.vision_model.eval()
             
        with torch.no_grad():
            self.vision_model(input_tensor, output_attentions=True)
        
        if layer_idx < 0:
            attentions = self.attentions
        elif layer_idx >= len(self.attentions):
            raise ValueError(f"layer_idx {layer_idx} is out of bounds for model with {len(self.attentions)} layers.")
        else:
            attentions = self.attentions[:layer_idx + 1]
            
        return self.rollout(attentions)
