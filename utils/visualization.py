"""
Visualization Utilities for Autonomous Vehicle Perception
========================================================

This module provides visualization functions for BEV data, segmentation results,
and attention mechanisms.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from matplotlib.colors import ListedColormap

# Set consistent style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def visualize_bev_features(bev_tensor: torch.Tensor, 
                          titles: Optional[List[str]] = None) -> plt.Figure:
    """Visualize 4-channel BEV features."""
    if isinstance(bev_tensor, torch.Tensor):
        bev_data = bev_tensor.cpu().numpy()
    else:
        bev_data = bev_tensor
    
    if len(bev_data.shape) == 4:  # Batch dimension
        bev_data = bev_data[0]  # Take first sample
    
    if titles is None:
        titles = ['Height Max', 'Height Mean', 'Intensity Max', 'Density']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        im = axes[i].imshow(bev_data[i], cmap='viridis', origin='lower')
        axes[i].set_title(titles[i], fontweight='bold')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    plt.suptitle('BEV Feature Channels', fontweight='bold', fontsize=16)
    plt.tight_layout()
    return fig

def visualize_segmentation_comparison(bev_input: np.ndarray,
                                    ground_truth: np.ndarray, 
                                    prediction: np.ndarray,
                                    class_names: List[str] = None) -> plt.Figure:
    """Compare segmentation results."""
    if class_names is None:
        class_names = ['Background', 'Vehicle', 'Pedestrian', 'Cyclist', 'Traffic Sign']
    
    # Define class colors
    class_colors = np.array([
        [0, 0, 0],        # Background - Black
        [255, 0, 0],      # Vehicle - Red
        [0, 255, 0],      # Pedestrian - Green
        [0, 0, 255],      # Cyclist - Blue
        [255, 255, 0],    # Traffic Sign - Yellow
    ]) / 255.0
    
    cmap = ListedColormap(class_colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # BEV input (height channel)
    if len(bev_input.shape) > 2:
        bev_display = bev_input[0] if len(bev_input.shape) == 3 else bev_input
    else:
        bev_display = bev_input
        
    axes[0].imshow(bev_display, cmap='viridis', origin='lower')
    axes[0].set_title('BEV Input (Height)', fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=4, origin='lower')
    axes[1].set_title('Ground Truth', fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=4, origin='lower')
    axes[2].set_title('Prediction', fontweight='bold')
    axes[2].axis('off')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=class_colors[i], 
                                   label=class_names[i]) for i in range(5)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    plt.suptitle('Segmentation Comparison', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

def visualize_attention_maps(attention_weights: torch.Tensor,
                           feature_map: torch.Tensor,
                           attention_type: str = "Mixed") -> plt.Figure:
    """Visualize attention mechanism weights."""
    if isinstance(attention_weights, torch.Tensor):
        attention_data = attention_weights.cpu().numpy()
    else:
        attention_data = attention_weights
        
    if isinstance(feature_map, torch.Tensor):
        feature_data = feature_map.cpu().numpy()
    else:
        feature_data = feature_map
    
    # Handle batch dimensions
    if len(attention_data.shape) > 2:
        attention_data = attention_data[0, 0] if len(attention_data.shape) == 4 else attention_data[0]
    if len(feature_data.shape) > 2:
        feature_data = feature_data[0, 0] if len(feature_data.shape) == 4 else feature_data[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original features
    im1 = axes[0].imshow(feature_data, cmap='viridis', origin='lower')
    axes[0].set_title('Original Features', fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Attention weights
    im2 = axes[1].imshow(attention_data, cmap='hot', origin='lower')
    axes[1].set_title(f'{attention_type} Attention Weights', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.suptitle(f'{attention_type} Attention Visualization', fontweight='bold', fontsize=16)
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics_history: Dict[str, List[float]]) -> plt.Figure:
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, metrics_history['val_loss'], 'r--', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU curves
    axes[0, 1].plot(epochs, metrics_history['train_miou'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, metrics_history['val_miou'], 'r--', label='Validation', linewidth=2)
    axes[0, 1].set_title('mIoU Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pedestrian mIoU
    if 'train_ped_miou' in metrics_history:
        axes[1, 0].plot(epochs, metrics_history['train_ped_miou'], 'g-', label='Training', linewidth=2)
        axes[1, 0].plot(epochs, metrics_history['val_ped_miou'], 'orange', linestyle='--', label='Validation', linewidth=2)
        axes[1, 0].set_title('Pedestrian mIoU', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Pedestrian mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in metrics_history:
        axes[1, 1].plot(epochs, metrics_history['learning_rate'], 'purple', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', fontweight='bold', fontsize=16)
    plt.tight_layout()
    return fig

def create_performance_summary(baseline_metrics: Dict[str, float],
                             enhanced_metrics: Dict[str, float]) -> plt.Figure:
    """Create performance comparison summary."""
    metrics = list(baseline_metrics.keys())
    baseline_values = list(baseline_metrics.values())
    enhanced_values = list(enhanced_metrics.values())
    improvements = [(e - b) / b * 100 for b, e in zip(baseline_values, enhanced_values)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', 
                   color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, enhanced_values, width, label='Enhanced', 
                   color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontweight='bold')
    ax1.set_ylabel('Performance', fontweight='bold')
    ax1.set_title('Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar1, bar2, val1, val2 in zip(bars1, bars2, baseline_values, enhanced_values):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                f'{val1:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                f'{val2:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement percentages
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars3 = ax2.bar(metrics, improvements, color=colors, alpha=0.8)
    ax2.set_xlabel('Metrics', fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontweight='bold')
    ax2.set_title('Performance Improvement', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars3, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (1 if val > 0 else -3),
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', 
                fontweight='bold')
    
    plt.tight_layout()
    return fig

def save_visualization(fig: plt.Figure, filename: str, dpi: int = 300):
    """Save visualization with high quality."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved visualization: {filename}")

# Example usage functions
def demo_visualizations():
    """Demonstrate visualization capabilities."""
    print("🎨 Autonomous Vehicle Perception Visualizations")
    print("=" * 50)
    
    # Create dummy data for demonstration
    dummy_bev = np.random.rand(4, 512, 512)
    dummy_gt = np.random.randint(0, 5, (512, 512))
    dummy_pred = np.random.randint(0, 5, (512, 512))
    
    # BEV features
    fig1 = visualize_bev_features(dummy_bev)
    save_visualization(fig1, 'demo_bev_features.png')
    plt.close(fig1)
    
    # Segmentation comparison
    fig2 = visualize_segmentation_comparison(dummy_bev[0], dummy_gt, dummy_pred)
    save_visualization(fig2, 'demo_segmentation.png')
    plt.close(fig2)
    
    print("✅ Demo visualizations created!")

if __name__ == '__main__':
    demo_visualizations()
