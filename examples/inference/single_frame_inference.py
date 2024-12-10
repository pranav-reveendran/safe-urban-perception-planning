#!/usr/bin/env python3
"""
Single Frame Inference Example
==============================

This script demonstrates how to run inference on a single frame using a trained model.

Usage:
    python single_frame_inference.py --model_path ../../results/checkpoints/best_model.pth --input sample_frame.npy

Output:
    - Segmentation mask (PNG)
    - Visualization overlay
    - Performance metrics
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from configs.base_config import Config
from models.unet import BaselineUNet
from models.attention_unet import AttentionUNet
from utils.preprocessing import LidarBEVProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Single Frame Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input BEV tensor (.npy file)')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--model_type', type=str, default='auto', choices=['baseline', 'attention', 'auto'], 
                       help='Model type (auto-detect from checkpoint)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization overlay')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model(model_path, model_type, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Auto-detect model type if needed
    if model_type == 'auto':
        if 'attention' in str(model_path).lower():
            model_type = 'attention'
        else:
            model_type = 'baseline'
    
    # Create model
    config = Config()
    if model_type == 'attention':
        model = AttentionUNet(
            in_channels=config.data.bev_channels,
            num_classes=len(config.data.class_names),
            attention_type='attention_gates'  # Default for inference
        )
    else:
        model = BaselineUNet(
            in_channels=config.data.bev_channels,
            num_classes=len(config.data.class_names)
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model type: {model_type}")
    print(f"Model loaded successfully")
    
    return model


def load_input_data(input_path):
    """Load and preprocess input BEV data."""
    print(f"Loading input data from: {input_path}")
    
    # Load BEV tensor
    bev_data = np.load(input_path)
    
    # Ensure correct shape [C, H, W]
    if bev_data.ndim == 2:
        # If 2D, assume it's a single channel and add channel dimension
        bev_data = np.expand_dims(bev_data, axis=0)
    elif bev_data.ndim == 3 and bev_data.shape[0] != 4:
        # If 3D but wrong channel dimension, try to fix
        if bev_data.shape[-1] == 4:
            bev_data = np.transpose(bev_data, (2, 0, 1))
    
    # Normalize if needed (assuming data is already preprocessed)
    # Add batch dimension
    bev_tensor = torch.FloatTensor(bev_data).unsqueeze(0)  # [1, C, H, W]
    
    print(f"Input shape: {bev_tensor.shape}")
    return bev_tensor


def run_inference(model, bev_tensor, device):
    """Run inference on the input tensor."""
    print("Running inference...")
    
    # Move to device
    bev_tensor = bev_tensor.to(device)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(bev_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
    
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.4f} seconds")
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()


def create_visualization(bev_input, predictions, probabilities, output_path):
    """Create visualization overlay."""
    print("Creating visualization...")
    
    # Define class colors
    class_colors = [
        [0, 0, 0],        # Background - Black
        [255, 0, 0],      # Vehicle - Red
        [0, 255, 0],      # Pedestrian - Green
        [0, 0, 255],      # Cyclist - Blue
        [255, 255, 0],    # Traffic Sign - Yellow
    ]
    
    # Create colormap
    cmap = ListedColormap(np.array(class_colors) / 255.0)
    
    # Get BEV visualization (use first channel as intensity)
    bev_vis = bev_input[0, 0] if bev_input.shape[1] > 0 else np.zeros((512, 512))
    bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # BEV input
    axes[0, 0].imshow(bev_vis, cmap='gray')
    axes[0, 0].set_title('BEV Input (Height Max)')
    axes[0, 0].axis('off')
    
    # Segmentation prediction
    pred_vis = predictions[0]
    axes[0, 1].imshow(pred_vis, cmap=cmap, vmin=0, vmax=4)
    axes[0, 1].set_title('Segmentation Prediction')
    axes[0, 1].axis('off')
    
    # Pedestrian probability
    ped_prob = probabilities[0, 2]  # Pedestrian class
    im = axes[1, 0].imshow(ped_prob, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Pedestrian Probability')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Overlay
    overlay = np.stack([bev_vis] * 3, axis=-1)
    mask = pred_vis == 2  # Pedestrian mask
    overlay[mask] = [0, 1, 0]  # Green for pedestrians
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('BEV + Pedestrian Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path}")


def save_results(predictions, probabilities, output_dir, input_name):
    """Save inference results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save segmentation mask
    pred_path = output_dir / f"{input_name}_segmentation.png"
    cv2.imwrite(str(pred_path), predictions[0].astype(np.uint8))
    
    # Save probabilities
    prob_path = output_dir / f"{input_name}_probabilities.npy"
    np.save(prob_path, probabilities[0])
    
    print(f"Segmentation mask saved: {pred_path}")
    print(f"Probabilities saved: {prob_path}")
    
    return pred_path, prob_path


def analyze_predictions(predictions, probabilities):
    """Analyze and print prediction statistics."""
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    pred = predictions[0]
    prob = probabilities[0]
    
    # Class distribution
    class_names = ['Background', 'Vehicle', 'Pedestrian', 'Cyclist', 'Traffic Sign']
    unique, counts = np.unique(pred, return_counts=True)
    total_pixels = pred.size
    
    print("\nClass Distribution:")
    for class_id, count in zip(unique, counts):
        if class_id < len(class_names):
            percentage = (count / total_pixels) * 100
            print(f"  {class_names[class_id]}: {count:,} pixels ({percentage:.2f}%)")
    
    # Pedestrian analysis
    pedestrian_mask = pred == 2
    if pedestrian_mask.any():
        ped_count = pedestrian_mask.sum()
        ped_percentage = (ped_count / total_pixels) * 100
        avg_confidence = prob[2][pedestrian_mask].mean()
        max_confidence = prob[2][pedestrian_mask].max()
        
        print(f"\nPedestrian Detection:")
        print(f"  Detected pixels: {ped_count:,}")
        print(f"  Coverage: {ped_percentage:.3f}% of frame")
        print(f"  Avg confidence: {avg_confidence:.3f}")
        print(f"  Max confidence: {max_confidence:.3f}")
        
        # Estimate number of pedestrian instances (simple blob counting)
        ped_binary = (pred == 2).astype(np.uint8)
        contours, _ = cv2.findContours(ped_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]  # Filter small blobs
        
        print(f"  Estimated pedestrians: {len(significant_contours)}")
    else:
        print("\nPedestrian Detection:")
        print("  No pedestrians detected in this frame")
    
    print("="*50)


def main():
    """Main inference function."""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.model_type, device)
    
    # Load input data
    bev_tensor = load_input_data(args.input)
    
    # Run inference
    predictions, probabilities = run_inference(model, bev_tensor, device)
    
    # Get input filename for naming outputs
    input_name = Path(args.input).stem
    
    # Save results
    pred_path, prob_path = save_results(predictions, probabilities, args.output_dir, input_name)
    
    # Create visualization if requested
    if args.visualize:
        vis_path = Path(args.output_dir) / f"{input_name}_visualization.png"
        create_visualization(bev_tensor.numpy(), predictions, probabilities, vis_path)
    
    # Analyze predictions
    analyze_predictions(predictions, probabilities)
    
    print(f"\nInference completed successfully!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main() 