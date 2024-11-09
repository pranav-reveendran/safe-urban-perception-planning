"""
Evaluation Metrics for Semantic Segmentation

This module provides comprehensive evaluation metrics for semantic segmentation
tasks with special focus on pedestrian detection performance in autonomous
vehicle perception systems.

Author: Graduate Student, AI/ML Program, San Jose State University
Date: October 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    
    Provides methods for calculating various metrics including IoU, precision,
    recall, F1-score with special attention to class-specific performance.
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        pedestrian_class_id: int = 2,
        ignore_index: int = 255
    ):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of semantic classes
            class_names: Names for each class
            pedestrian_class_id: ID for pedestrian class (primary metric)
            ignore_index: Class ID to ignore in calculations
        """
        self.num_classes = num_classes
        self.pedestrian_class_id = pedestrian_class_id
        self.ignore_index = ignore_index
        
        # Default class names for nuScenes-style dataset
        if class_names is None:
            self.class_names = [
                'background', 'vehicle', 'pedestrian', 'bicycle', 'motorcycle',
                'bus', 'truck', 'traffic_cone', 'barrier', 'vegetation'
            ][:num_classes]
        else:
            self.class_names = class_names
        
        # Initialize confusion matrix
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Predicted class labels [B, H, W] or [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        
        # Convert to numpy and flatten
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Handle logits input
        if len(predictions.shape) == 4:  # [B, C, H, W]
            predictions = np.argmax(predictions, axis=1)
        
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Remove ignore index
        valid_mask = target_flat != self.ignore_index
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            target_flat, 
            pred_flat, 
            labels=range(self.num_classes)
        )
        self.confusion_matrix += cm
        self.total_samples += len(pred_flat)
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute Intersection over Union (IoU) metrics
        
        Returns:
            Dictionary with IoU metrics
        """
        
        # Calculate IoU for each class
        ious = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            union = (self.confusion_matrix[i, :].sum() + 
                    self.confusion_matrix[:, i].sum() - 
                    intersection)
            
            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        
        # Calculate mean IoU
        valid_ious = [iou for iou in ious if iou > 0]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        
        # Prepare results
        results = {
            'mean_iou': mean_iou,
            'pedestrian_iou': ious[self.pedestrian_class_id],
            'class_ious': {
                self.class_names[i]: ious[i] 
                for i in range(len(ious))
            }
        }
        
        return results
    
    def compute_precision_recall(self) -> Dict[str, float]:
        """
        Compute precision and recall metrics
        
        Returns:
            Dictionary with precision/recall metrics
        """
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Calculate macro averages
        valid_indices = [i for i in range(len(precisions)) if precisions[i] > 0]
        macro_precision = np.mean([precisions[i] for i in valid_indices]) if valid_indices else 0.0
        macro_recall = np.mean([recalls[i] for i in valid_indices]) if valid_indices else 0.0
        macro_f1 = np.mean([f1_scores[i] for i in valid_indices]) if valid_indices else 0.0
        
        results = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'pedestrian_precision': precisions[self.pedestrian_class_id],
            'pedestrian_recall': recalls[self.pedestrian_class_id],
            'pedestrian_f1': f1_scores[self.pedestrian_class_id],
            'class_precisions': {
                self.class_names[i]: precisions[i] 
                for i in range(len(precisions))
            },
            'class_recalls': {
                self.class_names[i]: recalls[i] 
                for i in range(len(recalls))
            },
            'class_f1s': {
                self.class_names[i]: f1_scores[i] 
                for i in range(len(f1_scores))
            }
        }
        
        return results
    
    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy metrics
        
        Returns:
            Dictionary with accuracy metrics
        """
        
        # Overall accuracy
        correct = np.sum(np.diag(self.confusion_matrix))
        total = np.sum(self.confusion_matrix)
        overall_accuracy = correct / total if total > 0 else 0.0
        
        # Per-class accuracy
        class_accuracies = {}
        for i in range(self.num_classes):
            class_total = self.confusion_matrix[i, :].sum()
            class_correct = self.confusion_matrix[i, i]
            class_accuracy = class_correct / class_total if class_total > 0 else 0.0
            class_accuracies[self.class_names[i]] = class_accuracy
        
        results = {
            'overall_accuracy': overall_accuracy,
            'pedestrian_accuracy': class_accuracies[self.class_names[self.pedestrian_class_id]],
            'class_accuracies': class_accuracies
        }
        
        return results
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics and return comprehensive results
        
        Returns:
            Dictionary with all metrics
        """
        
        results = {}
        
        # IoU metrics
        iou_results = self.compute_iou()
        results.update(iou_results)
        
        # Precision/Recall metrics
        pr_results = self.compute_precision_recall()
        results.update(pr_results)
        
        # Accuracy metrics
        acc_results = self.compute_accuracy()
        results.update(acc_results)
        
        # Add summary metrics
        results['pedestrian_score'] = (
            results['pedestrian_iou'] + 
            results['pedestrian_f1']
        ) / 2
        
        return results
    
    def print_summary(self):
        """Print summary of metrics"""
        
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("SEGMENTATION METRICS SUMMARY")
        print("="*60)
        
        print(f"\n🎯 PRIMARY METRICS (Pedestrian Focus):")
        print(f"  Pedestrian mIoU:     {metrics['pedestrian_iou']:.4f}")
        print(f"  Pedestrian F1:       {metrics['pedestrian_f1']:.4f}")
        print(f"  Pedestrian Score:    {metrics['pedestrian_score']:.4f}")
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Mean IoU:            {metrics['mean_iou']:.4f}")
        print(f"  Overall Accuracy:    {metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
        
        print(f"\n📋 CLASS-WISE IoU:")
        for class_name, iou in metrics['class_ious'].items():
            print(f"  {class_name:15s}: {iou:.4f}")
        
        print("="*60)
    
    def plot_confusion_matrix(
        self, 
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save plot
            normalize: Whether to normalize by row (recall)
        """
        
        plt.figure(figsize=(10, 8))
        
        cm = self.confusion_matrix.copy()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.3f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]]
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def calculate_miou(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int = 10,
    ignore_index: int = 255
) -> float:
    """
    Calculate mean Intersection over Union (mIoU)
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        num_classes: Number of classes
        ignore_index: Class ID to ignore
        
    Returns:
        Mean IoU value
    """
    
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    metrics.update(predictions, targets)
    results = metrics.compute_iou()
    
    return results['mean_iou']


def calculate_pedestrian_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    pedestrian_class_id: int = 2,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    Calculate pedestrian-specific metrics
    
    Args:
        predictions: Predicted class labels
        targets: Ground truth labels
        pedestrian_class_id: ID for pedestrian class
        ignore_index: Class ID to ignore
        
    Returns:
        Dictionary with pedestrian metrics
    """
    
    metrics = SegmentationMetrics(
        pedestrian_class_id=pedestrian_class_id,
        ignore_index=ignore_index
    )
    metrics.update(predictions, targets)
    
    all_metrics = metrics.compute_all_metrics()
    
    return {
        'pedestrian_iou': all_metrics['pedestrian_iou'],
        'pedestrian_precision': all_metrics['pedestrian_precision'],
        'pedestrian_recall': all_metrics['pedestrian_recall'],
        'pedestrian_f1': all_metrics['pedestrian_f1'],
        'pedestrian_accuracy': all_metrics['pedestrian_accuracy']
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
    pedestrian_class_id: int = 2
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        pedestrian_class_id: ID for pedestrian class
        
    Returns:
        Dictionary with evaluation metrics
    """
    
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        pedestrian_class_id=pedestrian_class_id
    )
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['bev_tensor'].to(device)
            targets = batch['seg_labels'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(predictions, targets)
    
    return metrics.compute_all_metrics()


class MetricTracker:
    """
    Track metrics during training
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize metric tracker
        
        Args:
            metric_names: Names of metrics to track
        """
        self.metric_names = metric_names
        self.history = {name: [] for name in metric_names}
        self.current_epoch = 0
    
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics"""
        for name in self.metric_names:
            if name in metrics:
                self.history[name].append(metrics[name])
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[int, float]:
        """
        Get best value for a metric
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (epoch, value)
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return -1, 0.0
        
        values = self.history[metric_name]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return best_idx, values[best_idx]
    
    def plot_history(
        self, 
        metric_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot training history"""
        
        if metric_names is None:
            metric_names = self.metric_names
        
        fig, axes = plt.subplots(
            len(metric_names), 1, 
            figsize=(10, 3 * len(metric_names))
        )
        
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            if metric_name in self.history and self.history[metric_name]:
                axes[i].plot(self.history[metric_name])
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test metrics calculation
    print("Testing Segmentation Metrics...")
    
    # Create dummy data
    batch_size, height, width = 2, 256, 256
    num_classes = 10
    
    # Random predictions and targets
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Update with dummy data
    metrics.update(predictions, targets)
    
    # Calculate metrics
    results = metrics.compute_all_metrics()
    
    # Print results
    metrics.print_summary()
    
    # Test pedestrian metrics
    print("\nTesting pedestrian-specific metrics...")
    ped_metrics = calculate_pedestrian_metrics(predictions, targets)
    
    for metric, value in ped_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test metric tracker
    print("\nTesting metric tracker...")
    tracker = MetricTracker(['loss', 'pedestrian_iou', 'mean_iou'])
    
    # Simulate training history
    for epoch in range(10):
        dummy_metrics = {
            'loss': np.random.rand() * 0.5 + 0.1,
            'pedestrian_iou': np.random.rand() * 0.3 + 0.6,
            'mean_iou': np.random.rand() * 0.2 + 0.7
        }
        tracker.update(dummy_metrics)
    
    # Get best metrics
    best_epoch, best_value = tracker.get_best('pedestrian_iou')
    print(f"Best pedestrian IoU: {best_value:.4f} at epoch {best_epoch}")
    
    print("\nMetrics testing completed successfully!") 