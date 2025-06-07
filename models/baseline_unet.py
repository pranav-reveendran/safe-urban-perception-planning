"""
Baseline U-Net implementation for BEV LIDAR semantic segmentation
This serves as the foundation for comparison with attention-enhanced variants.

Author: Graduate Student, SJSU AI/ML Program
Project: Woven by Toyota Perception - Pedestrian Segmentation Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block with Conv2d -> BatchNorm -> Activation
    Used as building block for encoder and decoder paths
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, activation: str = 'relu', 
                 use_batch_norm: bool = True, dropout: float = 0.0):
        """
        Initialize ConvBlock
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super(ConvBlock, self).__init__()
        
        layers = []
        
        # Convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm))
        
        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConvBlock"""
        return self.conv_block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block consisting of two ConvBlocks followed by downsampling
    """
    
    def __init__(self, in_channels: int, out_channels: int, depth: int = 2, 
                 activation: str = 'relu', use_batch_norm: bool = True, dropout: float = 0.0):
        """
        Initialize EncoderBlock
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            depth: Number of convolutional layers in the block
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super(EncoderBlock, self).__init__()
        
        layers = []
        current_channels = in_channels
        
        # Add convolutional layers
        for i in range(depth):
            layers.append(ConvBlock(
                current_channels, out_channels, 
                activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
            ))
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of EncoderBlock
        
        Returns:
            tuple: (skip_connection, downsampled_output)
        """
        # Apply convolutional layers
        skip_connection = self.conv_layers(x)
        
        # Downsample for next level
        downsampled = self.pool(skip_connection)
        
        return skip_connection, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder block consisting of upsampling, concatenation, and ConvBlocks
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, 
                 depth: int = 2, activation: str = 'relu', use_batch_norm: bool = True, 
                 dropout: float = 0.0, upsample_mode: str = 'transpose'):
        """
        Initialize DecoderBlock
        
        Args:
            in_channels: Number of input channels from previous decoder level
            skip_channels: Number of channels in skip connection
            out_channels: Number of output channels
            depth: Number of convolutional layers in the block
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            upsample_mode: Upsampling method ('transpose' or 'interpolate')
        """
        super(DecoderBlock, self).__init__()
        
        self.upsample_mode = upsample_mode
        
        # Upsampling layer
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
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
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DecoderBlock
        
        Args:
            x: Input from previous decoder level
            skip: Skip connection from corresponding encoder level
            
        Returns:
            Decoder output after upsampling, concatenation, and convolution
        """
        # Upsample input
        if self.upsample_mode == 'transpose':
            upsampled = self.upsample(x)
        else:
            upsampled = self.upsample(x)
        
        # Handle size mismatch between upsampled and skip connection
        if upsampled.size()[2:] != skip.size()[2:]:
            upsampled = F.interpolate(upsampled, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        concatenated = torch.cat([upsampled, skip], dim=1)
        
        # Apply convolutional layers
        output = self.conv_layers(concatenated)
        
        return output


class BaselineUNet(nn.Module):
    """
    Baseline U-Net implementation for BEV LIDAR semantic segmentation
    
    This implementation follows the standard U-Net architecture without attention mechanisms.
    It serves as the baseline for comparison with attention-enhanced variants.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 10, base_filters: int = 64,
                 encoder_depths: List[int] = [2, 2, 2, 2, 2], decoder_depths: List[int] = [2, 2, 2, 2],
                 activation: str = 'relu', use_batch_norm: bool = True, dropout: float = 0.1,
                 upsample_mode: str = 'transpose'):
        """
        Initialize Baseline U-Net
        
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
        """
        super(BaselineUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        
        # Calculate filter sizes for each level
        num_levels = len(encoder_depths)
        encoder_filters = [base_filters * (2 ** i) for i in range(num_levels)]
        decoder_filters = encoder_filters[::-1][1:]  # Reverse and exclude bottleneck
        
        # Encoder path
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        
        for i, depth in enumerate(encoder_depths):
            encoder = EncoderBlock(
                current_channels, encoder_filters[i], depth,
                activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
            )
            self.encoders.append(encoder)
            current_channels = encoder_filters[i]
        
        # Bottleneck (deepest level without pooling)
        bottleneck_filters = encoder_filters[-1] * 2
        self.bottleneck = ConvBlock(
            encoder_filters[-1], bottleneck_filters,
            activation=activation, use_batch_norm=use_batch_norm, dropout=dropout
        )
        
        # Decoder path
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_filters
        
        for i, depth in enumerate(decoder_depths):
            skip_channels = encoder_filters[-(i+1)]  # Corresponding encoder level
            decoder = DecoderBlock(
                current_channels, skip_channels, decoder_filters[i], depth,
                activation=activation, use_batch_norm=use_batch_norm, 
                dropout=dropout, upsample_mode=upsample_mode
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
        Forward pass of Baseline U-Net
        
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
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path
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
    
    def get_feature_maps_info(self, input_size: tuple = (512, 512)) -> dict:
        """
        Get information about feature map sizes at each level
        
        Args:
            input_size: Input spatial size (H, W)
            
        Returns:
            Dictionary with feature map information
        """
        info = {}
        h, w = input_size
        
        # Encoder levels
        for i, encoder in enumerate(self.encoders):
            filters = self.base_filters * (2 ** i)
            info[f'encoder_{i}'] = {'size': (h, w), 'channels': filters}
            h, w = h // 2, w // 2
        
        # Bottleneck
        bottleneck_filters = self.base_filters * (2 ** len(self.encoders))
        info['bottleneck'] = {'size': (h, w), 'channels': bottleneck_filters}
        
        # Decoder levels
        for i, decoder in enumerate(self.decoders):
            h, w = h * 2, w * 2
            filters = self.base_filters * (2 ** (len(self.encoders) - i - 1))
            info[f'decoder_{i}'] = {'size': (h, w), 'channels': filters}
        
        return info


def create_baseline_unet(config) -> BaselineUNet:
    """
    Factory function to create baseline U-Net from configuration
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Configured BaselineUNet model
    """
    model = BaselineUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_filters=config.model.base_filters,
        encoder_depths=config.model.encoder_depths,
        decoder_depths=config.model.decoder_depths,
        activation=config.model.activation,
        use_batch_norm=config.model.use_batch_norm,
        dropout=config.model.dropout_rate
    )
    
    return model


if __name__ == "__main__":
    # Test baseline U-Net
    print("Testing Baseline U-Net:")
    
    # Create model
    model = BaselineUNet(
        in_channels=4,
        out_channels=10,
        base_filters=64,
        encoder_depths=[2, 2, 2, 2],
        decoder_depths=[2, 2, 2]
    )
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 4, 512, 512)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    
    # Model information
    model_info = model.get_model_size()
    print(f"\nModel Information:")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Feature map information
    feature_info = model.get_feature_maps_info()
    print(f"\nFeature Map Sizes:")
    for level, info in feature_info.items():
        print(f"{level}: {info['size']} -> {info['channels']} channels")
    
    print("\n✓ Baseline U-Net tested successfully!") 