"""
Model definitions for ENTRep 2025 Track 2
Vector Field Model with Flow Matching for Entity Recognition
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Projection for time encoding
    Maps scalar time t to higher-dimensional periodic space using sinusoidal features
    """
    def __init__(self, embed_dim, scale=10.0):
        super().__init__()
        # Fixed random weights for projecting scalar t to higher frequency space
        self.W = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        """
        Forward pass for time encoding
        Args:
            t: Time tensor, shape can be [B] or [B, 1]
        Returns:
            Sinusoidal and cosinusoidal projections: [sin(tW), cos(tW)], shape [B, embed_dim]
        """
        # Ensure t has shape [B, 1]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        
        proj = t * self.W  # Shape: [B, embed_dim//2]
        
        # Return sinusoidal and cosinusoidal projection: [sin(tW), cos(tW)]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class VectorField(nn.Module):
    """
    Vector Field Model for Flow Matching
    
    This model learns a time-dependent vector field that maps input embeddings
    to flow directions at time t. It uses:
    - Time encoding via Gaussian Fourier Projection
    - Input normalization with LayerNorm
    - Multi-head MLPs for processing
    - Residual connections for stability
    """
    
    def __init__(self, dim, t_dim=32, hidden_dim=256, n_heads=4, dropout_prob=0.1):
        """
        Initialize Vector Field Model
        
        Args:
            dim: Input embedding dimension (512 for CLIP)
            t_dim: Time embedding dimension
            hidden_dim: Hidden layer dimension for MLPs
            n_heads: Number of independent MLP heads
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Input normalization
        self.x_norm = nn.LayerNorm(dim)
        
        # Time encoding module
        self.time_encoder = GaussianFourierProjection(t_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Create multiple independent heads (like lightweight transformer blocks)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + t_dim, hidden_dim),     # Project input + time
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),                               # Activation: SiLU (Swish)
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, dim)              # Back to original embedding dimension
            ) for _ in range(n_heads)
        ])

        # Learnable residual scaling weight
        self.res_weight = nn.Parameter(torch.tensor(1.0))
        
        # Final normalization layer (optional, not used in forward by default)
        self.out_norm = nn.LayerNorm(dim)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization (good for ReLU/SiLU)
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, t):
        """
        Forward pass of the Vector Field Model
        
        Args:
            x: Input embeddings, shape [B, dim]
            t: Time values, can be scalar, [B], or [B, 1]
            
        Returns:
            Vector field output: dx/dt at time t, shape [B, dim]
        """
        # Handle different time input formats → ensure shape [B, 1]
        if not isinstance(t, torch.Tensor):
            # Convert scalar to tensor
            t = torch.full((x.shape[0], 1), t, device=x.device)
        elif t.ndim == 0:
            # Convert 0D tensor to [B, 1]
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            # Convert [B] to [B, 1]
            t = t.unsqueeze(-1)

        # Normalize input embeddings
        x_normed = self.x_norm(x)
        
        # Encode time t into sinusoidal features
        t_encoded = self.time_encoder(t.to(x.device))
        
        # Concatenate normalized input with time encoding
        inp = torch.cat([x_normed, t_encoded], dim=-1)

        # Pass through each head and average their outputs
        head_outs = [head(inp) for head in self.heads]
        out = torch.mean(torch.stack(head_outs), dim=0)

        # Add residual connection scaled by learnable weight
        return out + self.res_weight * x

    def get_model_info(self):
        """Get model information for debugging"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'n_heads': len(self.heads),
            'embedding_dim': self.x_norm.normalized_shape[0],
            'time_dim': self.time_encoder.W.shape[1] * 2,
            'hidden_dim': self.heads[0][0].out_features,
            'residual_weight': self.res_weight.item()
        }

def create_vector_field_model(embed_dim=512, **kwargs):
    """
    Factory function to create VectorField model with default parameters
    
    Args:
        embed_dim: Embedding dimension (default: 512 for CLIP)
        **kwargs: Additional arguments to pass to VectorField
        
    Returns:
        VectorField model instance
    """
    return VectorField(embed_dim, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    batch_size = 4
    embed_dim = 512
    x = torch.randn(batch_size, embed_dim).to(device)
    t = torch.tensor([0.1, 0.3, 0.5, 0.7]).to(device)
    
    # Test GaussianFourierProjection
    print("Testing GaussianFourierProjection...")
    time_encoder = GaussianFourierProjection(32).to(device)
    t_encoded = time_encoder(t)
    print(f"Time encoding shape: {t_encoded.shape}")
    
    # Test VectorField
    print("Testing VectorField...")
    vf = VectorField(embed_dim).to(device)
    output = vf(x, t)
    print(f"VectorField output shape: {output.shape}")
    
    # Print model info
    print("\nModel Information:")
    info = vf.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\nModels tested successfully!")