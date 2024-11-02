"""
Model Evaluation Script for Woven by Toyota Perception Dataset

Comprehensive evaluation of trained U-Net models on test dataset with
detailed performance analysis and visualization.

Author: Graduate Student, AI/ML Program, San Jose State University
Date: November 2024
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import Config
from models.baseline_unet import UNet
from models.attention_unet import AttentionUNet, create_attention_unet
from utils.data_loading import WovenDataset, create_data_loader, create_datasets
from utils.metrics import SegmentationMetrics, evaluate_model, calculate_pedestrian_metrics
from utils.preprocessing import visualize_bev_tensor


class ModelEvaluator:
    """
    Comprehensive model evaluation framework
    
    Provides methods for loading models, running evaluation, and generating
    detailed performance reports with visualizations.
    """
    
    def __init__(
        self,
        config: Config,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator
        
        Args:
            config: Configuration object
            device: Device to run evaluation on
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.results_dir = Path('results/evaluation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        print(f"Evaluator initialized on device: {self.device}")
    
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load trained model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        print(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model type from checkpoint
        model_config = checkpoint.get('model_config', {})
        model_type = model_config.get('model_type', 'baseline')
        
        if model_type == 'baseline':
            model = UNet(
                in_channels=self.config.data.bev_channels,
                out_channels=self.config.data.num_classes,
                base_filters=self.config.model.base_filters
            )
        else:
            attention_type = model_config.get('attention_type', 'attention_gates')
            model = create_attention_unet(
                in_channels=self.config.data.bev_channels,
                out_channels=self.config.data.num_classes,
                attention_type=attention_type,
                base_filters=self.config.model.base_filters
            )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully: {model_type}")
        if 'attention_type' in model_config:
            print(f"Attention type: {model_config['attention_type']}")
        
        return model
    
    def create_test_dataloader(self, subset_size: Optional[int] = None) -> DataLoader:
        """Create test dataset and dataloader"""
        
        print("Creating test dataset...")
        
        # Create test dataset
        test_dataset = WovenDataset(
            data_root=self.config.data.dataset_path,
            split='test',
            bev_params={
                'x_range': self.config.data.x_range,
                'y_range': self.config.data.y_range,
                'z_range': self.config.data.z_range,
                'resolution': self.config.data.resolution,
                'height': self.config.data.bev_height,
                'width': self.config.data.bev_width,
                'channels': self.config.data.bev_channels
            },
            augmentation=False,
            cache_data=True,
            subset_size=subset_size
        )
        
        # Create dataloader
        test_loader = create_data_loader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print(f"Test dataset created: {len(test_dataset)} samples")
        return test_loader
    
    def evaluate_model_comprehensive(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        save_prefix: str = "evaluation"
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_loader: Test data loader
            save_prefix: Prefix for saved files
            
        Returns:
            Dictionary with evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        # Initialize metrics tracker
        metrics = SegmentationMetrics(
            num_classes=self.config.data.num_classes,
            pedestrian_class_id=2  # Assuming pedestrian is class 2
        )
        
        # Track additional statistics
        sample_predictions = []
        inference_times = []
        memory_usage = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                inputs = batch['bev_tensor'].to(self.device)
                targets = batch['seg_labels'].to(self.device)
                
                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Forward pass
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    batch_time = start_time.elapsed_time(end_time)  # milliseconds
                    inference_times.append(batch_time / inputs.size(0))  # per sample
                
                # Update metrics
                metrics.update(predictions, targets)
                
                # Save sample predictions for visualization
                if batch_idx < 5:  # Save first 5 batches for visualization
                    sample_predictions.append({
                        'inputs': inputs[0].cpu().numpy(),
                        'targets': targets[0].cpu().numpy(),
                        'predictions': predictions[0].cpu().numpy(),
                        'metadata': batch['metadata']
                    })
                
                # Track memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches")
        
        # Calculate comprehensive metrics
        all_metrics = metrics.compute_all_metrics()
        
        # Add performance statistics
        if inference_times:
            all_metrics['avg_inference_time'] = np.mean(inference_times)
            all_metrics['std_inference_time'] = np.std(inference_times)
        
        if memory_usage:
            all_metrics['avg_memory_usage_gb'] = np.mean(memory_usage)
            all_metrics['max_memory_usage_gb'] = np.max(memory_usage)
        
        # Save detailed results
        self._save_detailed_results(all_metrics, save_prefix)
        
        # Generate visualizations
        self._generate_visualizations(sample_predictions, metrics, save_prefix)
        
        # Print summary
        self._print_evaluation_summary(all_metrics)
        
        return all_metrics
    
    def _save_detailed_results(self, metrics: Dict, save_prefix: str):
        """Save detailed evaluation results to JSON"""
        
        results_file = self.results_dir / f"{save_prefix}_detailed_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.int64, np.int32)):
                serializable_metrics[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, dict):
                serializable_metrics[key] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32)) else v
                    for k, v in value.items()
                }
            else:
                serializable_metrics[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
    
    def _generate_visualizations(
        self,
        sample_predictions: List[Dict],
        metrics: SegmentationMetrics,
        save_prefix: str
    ):
        """Generate evaluation visualizations"""
        
        import matplotlib.pyplot as plt
        
        # Plot confusion matrix
        confusion_path = self.figures_dir / f"{save_prefix}_confusion_matrix.png"
        metrics.plot_confusion_matrix(str(confusion_path), normalize=True)
        
        # Visualize sample predictions
        for i, sample in enumerate(sample_predictions[:3]):  # First 3 samples
            self._visualize_sample_prediction(sample, f"{save_prefix}_sample_{i}")
        
        # Plot class-wise performance
        self._plot_class_performance(metrics, save_prefix)
    
    def _visualize_sample_prediction(self, sample: Dict, save_prefix: str):
        """Visualize individual sample prediction"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # BEV channels visualization
        bev_tensor = sample['inputs']
        channel_names = ['Height Max', 'Height Mean', 'Intensity Max', 'Density']
        
        for i in range(min(4, bev_tensor.shape[0])):
            row = i // 2
            col = i % 2
            if i < 2:
                axes[row, col].imshow(bev_tensor[i], cmap='viridis')
                axes[row, col].set_title(f'BEV {channel_names[i]}')
                axes[row, col].axis('off')
        
        # Ground truth
        axes[0, 2].imshow(sample['targets'], cmap='tab10', vmin=0, vmax=9)
        axes[0, 2].set_title('Ground Truth')
        axes[0, 2].axis('off')
        
        # Predictions
        axes[1, 2].imshow(sample['predictions'], cmap='tab10', vmin=0, vmax=9)
        axes[1, 2].set_title('Predictions')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{save_prefix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_performance(self, metrics: SegmentationMetrics, save_prefix: str):
        """Plot class-wise performance metrics"""
        
        import matplotlib.pyplot as plt
        
        all_metrics = metrics.compute_all_metrics()
        
        # Extract class-wise metrics
        class_names = list(all_metrics['class_ious'].keys())
        ious = list(all_metrics['class_ious'].values())
        precisions = list(all_metrics['class_precisions'].values())
        recalls = list(all_metrics['class_recalls'].values())
        f1s = list(all_metrics['class_f1s'].values())
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # IoU
        axes[0, 0].bar(range(len(class_names)), ious, color='skyblue')
        axes[0, 0].set_title('Class-wise IoU')
        axes[0, 0].set_xticks(range(len(class_names)))
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].set_ylabel('IoU')
        
        # Precision
        axes[0, 1].bar(range(len(class_names)), precisions, color='lightgreen')
        axes[0, 1].set_title('Class-wise Precision')
        axes[0, 1].set_xticks(range(len(class_names)))
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        axes[0, 1].set_ylabel('Precision')
        
        # Recall
        axes[1, 0].bar(range(len(class_names)), recalls, color='lightcoral')
        axes[1, 0].set_title('Class-wise Recall')
        axes[1, 0].set_xticks(range(len(class_names)))
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        axes[1, 0].set_ylabel('Recall')
        
        # F1-Score
        axes[1, 1].bar(range(len(class_names)), f1s, color='lightyellow')
        axes[1, 1].set_title('Class-wise F1-Score')
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        axes[1, 1].set_ylabel('F1-Score')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{save_prefix}_class_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_evaluation_summary(self, metrics: Dict):
        """Print evaluation summary"""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n🎯 PRIMARY METRICS (Pedestrian Focus):")
        print(f"  Pedestrian mIoU:         {metrics['pedestrian_iou']:.4f}")
        print(f"  Pedestrian Precision:    {metrics['pedestrian_precision']:.4f}")
        print(f"  Pedestrian Recall:       {metrics['pedestrian_recall']:.4f}")
        print(f"  Pedestrian F1:           {metrics['pedestrian_f1']:.4f}")
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Mean IoU:                {metrics['mean_iou']:.4f}")
        print(f"  Overall Accuracy:        {metrics['overall_accuracy']:.4f}")
        print(f"  Macro Precision:         {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:            {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:                {metrics['macro_f1']:.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        if 'avg_inference_time' in metrics:
            print(f"  Avg Inference Time:      {metrics['avg_inference_time']:.2f} ms/sample")
        if 'avg_memory_usage_gb' in metrics:
            print(f"  Avg Memory Usage:        {metrics['avg_memory_usage_gb']:.2f} GB")
            print(f"  Max Memory Usage:        {metrics['max_memory_usage_gb']:.2f} GB")
        
        print("="*80)


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base_config.py',
                        help='Path to configuration file')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Evaluate on subset of test data')
    parser.add_argument('--save_prefix', type=str, default='evaluation',
                        help='Prefix for saved files')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    print(f"Configuration loaded from: {args.config}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config, device)
    
    # Load model
    model = evaluator.load_model(args.checkpoint)
    
    # Create test dataloader
    test_loader = evaluator.create_test_dataloader(args.subset_size)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_model_comprehensive(
        model, test_loader, args.save_prefix
    )
    
    print(f"\nEvaluation completed! Results saved in: {evaluator.results_dir}")


if __name__ == "__main__":
    main() 