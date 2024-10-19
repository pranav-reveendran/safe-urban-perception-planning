"""
Attention mechanisms for U-Net enhancement in LIDAR BEV semantic segmentation
Implements multiple attention types: Attention Gates, Self-Attention, and CBAM

Author: Graduate Student, SJSU AI/ML Program
Project: Woven by Toyota Perception - Pedestrian Segmentation Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AttentionGate(nn.Module):
    """
    Attention Gate module for U-Net skip connections
    
    Suppresses irrelevant regions in feature maps while highlighting
    salient features for the target task (pedestrian segmentation).
    
    Reference: Oktay et al. (2018) - "Attention U-Net: Learning Where to Look 
    for the Pancreas"
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int = None):
        """
        Initialize Attention Gate
        
        Args:
            gate_channels: Number of channels in gating signal (from decoder)
            skip_channels: Number of channels in skip connection (from encoder)
            inter_channels: Number of intermediate channels (default: skip_channels // 2)
        """
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
            
        # Gating signal transformation
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Skip connection transformation
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Attention coefficient generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention Gate
        
        Args:
            gate: Gating signal from decoder [B, gate_channels, H_g, W_g]
            skip: Skip connection from encoder [B, skip_channels, H_s, W_s]
            
        Returns:
            Attention-weighted skip connection [B, skip_channels, H_s, W_s]
        """
        # Get dimensions
        batch_size, skip_channels, H_s, W_s = skip.size()
        _, gate_channels, H_g, W_g = gate.size()
        
        # Upsample gate to match skip connection size if needed
        if H_g != H_s or W_g != W_s:
            gate_upsampled = F.interpolate(gate, size=(H_s, W_s), mode='bilinear', align_corners=False)
        else:
            gate_upsampled = gate
        
        # Transform gate and skip connections
        gate_transformed = self.gate_conv(gate_upsampled)  # [B, inter_channels, H_s, W_s]
        skip_transformed = self.skip_conv(skip)            # [B, inter_channels, H_s, W_s]
        
        # Combine and generate attention coefficients
        combined = self.relu(gate_transformed + skip_transformed)
        attention_coeffs = self.attention_conv(combined)  # [B, 1, H_s, W_s]
        
        # Apply attention to skip connection
        attended_skip = skip * attention_coeffs
        
        return attended_skip


class SelfAttention2D(nn.Module):
    """
    2D Self-Attention module for capturing global spatial dependencies
    
    Adapted for BEV LIDAR feature maps to capture long-range spatial relationships
    important for understanding scene context for pedestrian segmentation.
    """
    
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize Self-Attention module
        
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SelfAttention2D, self).__init__()
        
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Self-Attention
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Self-attended feature map [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Store input for residual connection
        residual = x
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, H*W, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(2, 3).contiguous().view(B, C, H, W)
        output = self.proj(attn_output)
        
        # Residual connection with layer normalization
        # Apply layer norm to flattened spatial dimensions
        output_flat = output.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        residual_flat = residual.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        normalized = self.norm(output_flat + residual_flat)
        result = normalized.permute(0, 2, 1).view(B, C, H, W)
        
        return result


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (part of CBAM)
    
    Focuses on 'what' is meaningful by exploiting inter-channel relationships.
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Initialize Channel Attention
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for bottleneck
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Channel Attention
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Channel attention weights [B, C, 1, 1]
        """
        # Average and max pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        
        return channel_attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (part of CBAM)
    
    Focuses on 'where' is meaningful by exploiting inter-spatial relationships.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Spatial Attention
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Spatial attention weights [B, 1, H, W]
        """
        # Channel-wise average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate and apply convolution
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        spatial_attention = self.sigmoid(self.conv(spatial_input))  # [B, 1, H, W]
        
        return spatial_attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Combines channel and spatial attention sequentially.
    
    Reference: Woo et al. (2018) - "CBAM: Convolutional Block Attention Module"
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        """
        Initialize CBAM
        
        Args:
            channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            spatial_kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CBAM
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Attention-refined feature map [B, C, H, W]
        """
        # Apply channel attention
        channel_weights = self.channel_attention(x)
        x_channel_refined = x * channel_weights
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(x_channel_refined)
        x_refined = x_channel_refined * spatial_weights
        
        return x_refined


class MixedAttention(nn.Module):
    """
    Mixed Attention Module combining multiple attention mechanisms
    
    Allows for experimentation with different attention combinations.
    """
    
    def __init__(self, channels: int, attention_types: list = ['channel', 'spatial'], 
                 reduction_ratio: int = 16, num_heads: int = 8):
        """
        Initialize Mixed Attention
        
        Args:
            channels: Number of input channels
            attention_types: List of attention types to use
            reduction_ratio: Reduction ratio for channel attention
            num_heads: Number of heads for self-attention
        """
        super(MixedAttention, self).__init__()
        
        self.attention_types = attention_types
        self.modules_dict = nn.ModuleDict()
        
        if 'channel' in attention_types:
            self.modules_dict['channel'] = ChannelAttention(channels, reduction_ratio)
        
        if 'spatial' in attention_types:
            self.modules_dict['spatial'] = SpatialAttention()
        
        if 'self_attention' in attention_types:
            self.modules_dict['self_attention'] = SelfAttention2D(channels, num_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mixed Attention
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Multi-attention refined feature map [B, C, H, W]
        """
        output = x
        
        # Apply channel attention if specified
        if 'channel' in self.attention_types:
            channel_weights = self.modules_dict['channel'](output)
            output = output * channel_weights
        
        # Apply spatial attention if specified
        if 'spatial' in self.attention_types:
            spatial_weights = self.modules_dict['spatial'](output)
            output = output * spatial_weights
        
        # Apply self-attention if specified
        if 'self_attention' in self.attention_types:
            output = self.modules_dict['self_attention'](output)
        
        return output


def get_attention_module(attention_type: str, channels: int, **kwargs) -> nn.Module:
    """
    Factory function to get attention modules
    
    Args:
        attention_type: Type of attention ('attention_gates', 'self_attention', 'cbam', 'mixed')
        channels: Number of channels
        **kwargs: Additional arguments for specific attention modules
        
    Returns:
        Attention module instance
    """
    if attention_type == 'attention_gates':
        # For attention gates, additional parameters needed during forward pass
        return None  # Will be handled in U-Net architecture
    
    elif attention_type == 'self_attention':
        num_heads = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.1)
        return SelfAttention2D(channels, num_heads, dropout)
    
    elif attention_type == 'cbam':
        reduction_ratio = kwargs.get('reduction_ratio', 16)
        spatial_kernel_size = kwargs.get('spatial_kernel_size', 7)
        return CBAM(channels, reduction_ratio, spatial_kernel_size)
    
    elif attention_type == 'mixed':
        attention_types = kwargs.get('attention_types', ['channel', 'spatial'])
        reduction_ratio = kwargs.get('reduction_ratio', 16)
        num_heads = kwargs.get('num_heads', 8)
        return MixedAttention(channels, attention_types, reduction_ratio, num_heads)
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Test attention modules
    batch_size, channels, height, width = 2, 64, 128, 128
    x = torch.randn(batch_size, channels, height, width)
    
    print("Testing Attention Modules:")
    print(f"Input shape: {x.shape}")
    
    # Test Attention Gate
    print("\n1. Attention Gate:")
    gate = torch.randn(batch_size, 128, 64, 64)  # Different size gate
    skip = x
    att_gate = AttentionGate(gate_channels=128, skip_channels=channels)
    out_ag = att_gate(gate, skip)
    print(f"   Output shape: {out_ag.shape}")
    
    # Test Self-Attention
    print("\n2. Self-Attention:")
    self_att = SelfAttention2D(channels, num_heads=8)
    out_sa = self_att(x)
    print(f"   Output shape: {out_sa.shape}")
    
    # Test CBAM
    print("\n3. CBAM:")
    cbam = CBAM(channels, reduction_ratio=16)
    out_cbam = cbam(x)
    print(f"   Output shape: {out_cbam.shape}")
    
    # Test Mixed Attention
    print("\n4. Mixed Attention:")
    mixed_att = MixedAttention(channels, attention_types=['channel', 'spatial'])
    out_mixed = mixed_att(x)
    print(f"   Output shape: {out_mixed.shape}")
    
    print("\n✓ All attention modules tested successfully!") 