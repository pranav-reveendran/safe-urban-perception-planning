"""
Attention-Enhanced U-Net implementation for improved pedestrian segmentation
Integrates various attention mechanisms into the baseline U-Net architecture.

Author: Graduate Student, SJSU AI/ML Program
Project: Woven by Toyota Perception - Pedestrian Segmentation Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .baseline_unet import ConvBlock, EncoderBlock
from .attention_modules import AttentionGate, get_attention_module


class AttentionDecoderBlock(nn.Module):
    """
    Decoder block with integrated attention mechanisms
    Supports attention gates and other attention types
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 depth: int = 2, activation: str = 'relu', use_batch_norm: bool = True, 
                 dropout: float = 0.0, upsample_mode: str = 'transpose',
                 use_attention_gate: bool = True, attention_type: str = None,
                 attention_kwargs: dict = None):
        """
        Initialize Attention Decoder Block
        
        Args:
            in_channels: Number of input channels from previous decoder level
            skip_channels: Number of channels in skip connection
            out_channels: Number of output channels
            depth: Number of convolutional layers in the block
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            upsample_mode: Upsampling method ('transpose' or 'interpolate')
            use_attention_gate: Whether to use attention gates on skip connections
            attention_type: Type of attention to apply to output features
            attention_kwargs: Additional arguments for attention modules
        """
        super(AttentionDecoderBlock, self).__init__()
        
        self.use_attention_gate = use_attention_gate
        self.upsample_mode = upsample_mode
        
        # Upsampling layer
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Attention gate for skip connections
        if use_attention_gate:
            self.attention_gate = AttentionGate(
                gate_channels=in_channels,
                skip_channels=skip_channels
            )
        
        # Convolutional layers after concatenation
        concat_channels = in_channels + skip_channels
        layers = []
        current_channels = concat_channels
        
        for i in range(depth):
            layers.append(ConvBlock(
                current_channels, out_channels,
                activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
            ))
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Additional attention mechanism for output features
        if attention_type and attention_type != 'attention_gates':
            attention_kwargs = attention_kwargs or {}
            self.feature_attention = get_attention_module(
                attention_type, out_channels, **attention_kwargs
            )
        else:
            self.feature_attention = None
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention Decoder Block
        
        Args:
            x: Input from previous decoder level
            skip: Skip connection from corresponding encoder level
            
        Returns:
            Decoder output with attention mechanisms applied
        """
        # Upsample input
        if self.upsample_mode == 'transpose':
            upsampled = self.upsample(x)
        else:
            upsampled = self.upsample(x)
        
        # Handle size mismatch between upsampled and skip connection
        if upsampled.size()[2:] != skip.size()[2:]:
            upsampled = F.interpolate(upsampled, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Apply attention gate to skip connection if enabled
        if self.use_attention_gate:
            attended_skip = self.attention_gate(upsampled, skip)
        else:
            attended_skip = skip
        
        # Concatenate with (possibly attended) skip connection
        concatenated = torch.cat([upsampled, attended_skip], dim=1)
        
        # Apply convolutional layers
        output = self.conv_layers(concatenated)
        
        # Apply feature attention if specified
        if self.feature_attention is not None:
            output = self.feature_attention(output)
        
        return output


class AttentionUNet(nn.Module):
    """
    Attention-Enhanced U-Net for BEV LIDAR semantic segmentation
    
    Integrates multiple attention mechanisms:
    1. Attention Gates in skip connections
    2. Self-attention or CBAM at the bottleneck
    3. Optional attention in decoder blocks
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 10, base_filters: int = 64,
                 encoder_depths: List[int] = [2, 2, 2, 2], decoder_depths: List[int] = [2, 2, 2],
                 activation: str = 'relu', use_batch_norm: bool = True, dropout: float = 0.1,
                 upsample_mode: str = 'transpose', attention_type: str = 'attention_gates',
                 bottleneck_attention: bool = True, decoder_attention: bool = False,
                 attention_kwargs: dict = None):
        """
        Initialize Attention U-Net
        
        Args:
            in_channels: Number of input channels (BEV features)
            out_channels: Number of output classes
            base_filters: Base number of filters (doubles at each encoder level)
            encoder_depths: Number of conv layers in each encoder block
            decoder_depths: Number of conv layers in each decoder block
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            upsample_mode: Upsampling method in decoder
            attention_type: Type of attention mechanism ('attention_gates', 'self_attention', 'cbam', 'mixed')
            bottleneck_attention: Whether to apply attention at bottleneck
            decoder_attention: Whether to apply attention in decoder blocks
            attention_kwargs: Additional arguments for attention modules
        """
        super(AttentionUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.attention_type = attention_type
        
        # Default attention kwargs
        if attention_kwargs is None:
            attention_kwargs = {}
        
        # Calculate filter sizes for each level
        num_levels = len(encoder_depths)
        encoder_filters = [base_filters * (2 ** i) for i in range(num_levels)]
        decoder_filters = encoder_filters[::-1][1:]  # Reverse and exclude bottleneck
        
        # Encoder path (reuse from baseline)
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        
        for i, depth in enumerate(encoder_depths):
            encoder = EncoderBlock(
                current_channels, encoder_filters[i], depth,
                activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
            )
            self.encoders.append(encoder)
            current_channels = encoder_filters[i]
        
        # Bottleneck with optional attention
        bottleneck_filters = encoder_filters[-1] * 2
        bottleneck_layers = []
        
        # Add conv layers
        for i in range(encoder_depths[-1]):
            bottleneck_layers.append(ConvBlock(
                encoder_filters[-1] if i == 0 else bottleneck_filters,
                bottleneck_filters,
                activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
            ))
        
        self.bottleneck_conv = nn.Sequential(*bottleneck_layers)
        
        # Bottleneck attention
        if bottleneck_attention and attention_type != 'attention_gates':
            self.bottleneck_attention = get_attention_module(
                attention_type, bottleneck_filters, **attention_kwargs
            )
        else:
            self.bottleneck_attention = None
        
        # Attention-enhanced decoder path
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_filters
        
        for i, depth in enumerate(decoder_depths):
            skip_channels = encoder_filters[-(i+1)]  # Corresponding encoder level
            
            # Determine whether to use attention gates
            use_attention_gate = attention_type == 'attention_gates' or attention_type == 'mixed'
            
            # Determine whether to use feature attention in this decoder block
            decoder_attn_type = attention_type if decoder_attention else None
            if decoder_attn_type == 'attention_gates':
                decoder_attn_type = None  # Already handled by attention gates
            
            decoder = AttentionDecoderBlock(
                current_channels, skip_channels, decoder_filters[i], depth,
                activation=activation, use_batch_norm=use_batch_norm, 
                dropout=dropout, upsample_mode=upsample_mode,
                use_attention_gate=use_attention_gate,
                attention_type=decoder_attn_type,
                attention_kwargs=attention_kwargs
            )
            self.decoders.append(decoder)
            current_channels = decoder_filters[i]
        
        # Final classification layer
        self.final_conv = nn.Conv2d(decoder_filters[-1], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Attention U-Net
        
        Args:
            x: Input BEV tensor [B, in_channels, H, W]
            
        Returns:
            Segmentation logits [B, out_channels, H, W]
        """
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        current = x
        for encoder in self.encoders:
            skip, current = encoder(current)
            skip_connections.append(skip)
        
        # Bottleneck with optional attention
        current = self.bottleneck_conv(current)
        if self.bottleneck_attention is not None:
            current = self.bottleneck_attention(current)
        
        # Attention-enhanced decoder path
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i+1)]  # Get corresponding skip connection
            current = decoder(current, skip)
        
        # Final classification
        output = self.final_conv(current)
        
        return output
    
    def get_model_size(self) -> dict:
        """
        Get model size information
        
        Returns:
            Dictionary with model size statistics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }
    
    def get_attention_info(self) -> dict:
        """
        Get information about attention mechanisms used
        
        Returns:
            Dictionary with attention configuration
        """
        info = {
            'attention_type': self.attention_type,
            'bottleneck_attention': self.bottleneck_attention is not None,
            'attention_gates_count': 0,
            'decoder_attention_count': 0
        }
        
        # Count attention gates and decoder attention modules
        for decoder in self.decoders:
            if hasattr(decoder, 'attention_gate') and decoder.attention_gate is not None:
                info['attention_gates_count'] += 1
            if hasattr(decoder, 'feature_attention') and decoder.feature_attention is not None:
                info['decoder_attention_count'] += 1
        
        return info


def create_attention_unet(config) -> AttentionUNet:
    """
    Factory function to create attention U-Net from configuration
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Configured AttentionUNet model
    """
    # Extract attention-specific parameters
    attention_kwargs = {}
    
    if config.model.attention_type == 'self_attention':
        attention_kwargs.update({
            'num_heads': config.model.self_attention_heads,
            'dropout': config.model.dropout_rate
        })
    elif config.model.attention_type == 'cbam':
        attention_kwargs.update({
            'reduction_ratio': config.model.cbam_reduction_ratio
        })
    elif config.model.attention_type == 'mixed':
        attention_kwargs.update({
            'attention_types': ['channel', 'spatial'],
            'reduction_ratio': config.model.cbam_reduction_ratio,
            'num_heads': config.model.self_attention_heads
        })
    
    model = AttentionUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_filters=config.model.base_filters,
        encoder_depths=config.model.encoder_depths,
        decoder_depths=config.model.decoder_depths,
        activation=config.model.activation,
        use_batch_norm=config.model.use_batch_norm,
        dropout=config.model.dropout_rate,
        attention_type=config.model.attention_type,
        bottleneck_attention=True,
        decoder_attention=False,  # Can be made configurable
        attention_kwargs=attention_kwargs
    )
    
    return model


class AttentionUNetEnsemble(nn.Module):
    """
    Ensemble of attention U-Nets with different attention mechanisms
    Useful for ablation studies and performance comparison
    """
    
    def __init__(self, models: List[AttentionUNet], ensemble_mode: str = 'average'):
        """
        Initialize ensemble of attention U-Nets
        
        Args:
            models: List of AttentionUNet models
            ensemble_mode: How to combine predictions ('average', 'voting', 'weighted')
        """
        super(AttentionUNetEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_mode = ensemble_mode
        self.num_models = len(models)
        
        if ensemble_mode == 'weighted':
            # Learnable weights for ensemble
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            x: Input BEV tensor [B, in_channels, H, W]
            
        Returns:
            Ensemble prediction [B, out_channels, H, W]
        """
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Combine predictions
        if self.ensemble_mode == 'average':
            output = torch.stack(predictions, dim=0).mean(dim=0)
        elif self.ensemble_mode == 'voting':
            # Take argmax of each prediction, then majority vote
            votes = torch.stack([torch.argmax(pred, dim=1) for pred in predictions], dim=0)
            output = torch.mode(votes, dim=0)[0]
            # Convert back to logits format (simplified)
            output = F.one_hot(output, num_classes=predictions[0].shape[1]).permute(0, 3, 1, 2).float()
        elif self.ensemble_mode == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            output = sum(w * pred for w, pred in zip(weights, predictions))
        
        return output


if __name__ == "__main__":
    # Test attention U-Net variants
    print("Testing Attention U-Net variants:")
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 512, 512)
    print(f"Input shape: {input_tensor.shape}")
    
    # Test different attention types
    attention_types = ['attention_gates', 'self_attention', 'cbam', 'mixed']
    
    for attn_type in attention_types:
        print(f"\n{attn_type.upper()}:")
        
        # Configure attention kwargs
        attention_kwargs = {}
        if attn_type == 'self_attention':
            attention_kwargs = {'num_heads': 8, 'dropout': 0.1}
        elif attn_type == 'cbam':
            attention_kwargs = {'reduction_ratio': 16}
        elif attn_type == 'mixed':
            attention_kwargs = {'attention_types': ['channel', 'spatial'], 'reduction_ratio': 16}
        
        # Create model
        model = AttentionUNet(
            in_channels=4,
            out_channels=10,
            base_filters=64,
            encoder_depths=[2, 2, 2],
            decoder_depths=[2, 2],
            attention_type=attn_type,
            attention_kwargs=attention_kwargs
        )
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"   Output shape: {output.shape}")
        
        # Model information
        model_info = model.get_model_size()
        attention_info = model.get_attention_info()
        
        print(f"   Parameters: {model_info['total_parameters']:,}")
        print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
        print(f"   Attention info: {attention_info}")
    
    print("\n✓ All attention U-Net variants tested successfully!") 