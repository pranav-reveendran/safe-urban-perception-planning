"""
Training script for Attention-Enhanced U-Net on Woven by Toyota dataset
Supports both baseline and attention-enhanced models with comprehensive logging.

Author: Graduate Student, SJSU AI/ML Program
Project: Woven by Toyota Perception - Pedestrian Segmentation Enhancement
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import yaml
import json
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from configs.base_config import Config
from models.baseline_unet import create_baseline_unet
from models.attention_unet import create_attention_unet
from utils.metrics import SegmentationMetrics
from utils.data_loading import WovenDataset, create_data_loaders
from utils.preprocessing import BEVProcessor


class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and Dice Loss for pedestrian-focused segmentation"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 dice_weight: float = 0.5, focal_weight: float = 0.5,
                 num_classes: int = 10, pedestrian_class: int = 2,
                 class_weights: torch.Tensor = None):
        super(FocalDiceLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes
        self.pedestrian_class = pedestrian_class
        
        # Set up class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss"""
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.class_weights, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss for pedestrian class"""
        # Convert inputs to probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Focus on pedestrian class
        pred_ped = probs[:, self.pedestrian_class]
        target_ped = targets_one_hot[:, self.pedestrian_class]
        
        # Compute Dice coefficient
        intersection = (pred_ped * target_ped).sum()
        dice_coeff = (2 * intersection + 1e-8) / (pred_ped.sum() + target_ped.sum() + 1e-8)
        
        return 1 - dice_coeff
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass combining Focal and Dice losses"""
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


class Trainer:
    """Training manager for U-Net models"""
    
    def __init__(self, config: Config, model_type: str = 'attention'):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
            model_type: 'baseline' or 'attention'
        """
        self.config = config
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self.set_random_seeds(config.experiment.random_seed)
        
        # Initialize components
        self.model = self._create_model()
        self.criterion = self._create_loss_function()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics = SegmentationMetrics(
            num_classes=config.model.out_channels,
            pedestrian_class=config.data.pedestrian_class_id
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Setup logging
        self.setup_logging()
        
        print(f"✓ Trainer initialized for {model_type} model")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        if self.model_type == 'baseline':
            model = create_baseline_unet(self.config)
        elif self.model_type == 'attention':
            model = create_attention_unet(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model = model.to(self.device)
        
        # Enable mixed precision training if configured
        if self.config.training.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        return model
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration"""
        # Create class weights if specified
        class_weights = None
        if self.config.training.use_class_weights:
            weights = torch.ones(self.config.model.out_channels)
            weights[self.config.data.pedestrian_class_id] = self.config.training.pedestrian_weight
            class_weights = weights.to(self.device)
        
        if self.config.training.loss_type == 'ce':
            return nn.CrossEntropyLoss(weight=class_weights)
        elif self.config.training.loss_type == 'focal':
            return FocalDiceLoss(
                alpha=self.config.training.focal_alpha,
                gamma=self.config.training.focal_gamma,
                dice_weight=0.0, focal_weight=1.0,
                num_classes=self.config.model.out_channels,
                pedestrian_class=self.config.data.pedestrian_class_id,
                class_weights=class_weights
            )
        elif self.config.training.loss_type == 'combined':
            return FocalDiceLoss(
                alpha=self.config.training.focal_alpha,
                gamma=self.config.training.focal_gamma,
                dice_weight=self.config.training.dice_weight,
                focal_weight=self.config.training.focal_weight,
                num_classes=self.config.model.out_channels,
                pedestrian_class=self.config.data.pedestrian_class_id,
                class_weights=class_weights
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config.training.loss_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.training.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.initial_lr,
                weight_decay=self.config.training.weight_decay,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.initial_lr,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.min_lr
            )
        elif self.config.training.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.gamma,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create experiment directory
        exp_name = f"{self.model_type}_{self.config.experiment.experiment_name}"
        self.exp_dir = Path(self.config.experiment.log_dir) / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard logging
        self.writer = SummaryWriter(self.exp_dir)
        
        # Save configuration
        config_path = self.exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config.__dict__, f, default_flow_style=False)
        
        print(f"✓ Logging setup complete: {self.exp_dir}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.training.epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.max_grad_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.experiment.log_frequency == 0:
                self.writer.add_scalar(
                    'Train/BatchLoss', 
                    loss.item(), 
                    self.current_epoch * num_batches + batch_idx
                )
            
            # Clear CUDA cache periodically
            if batch_idx % self.config.system.empty_cache_frequency == 0:
                torch.cuda.empty_cache()
        
        return epoch_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        self.metrics.reset()
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                self.metrics.update(predictions.cpu(), targets.cpu())
        
        # Compute final metrics
        metrics_dict = self.metrics.compute()
        metrics_dict['val_loss'] = val_loss / len(val_loader)
        
        return metrics_dict
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.exp_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.exp_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved: {metrics[self.config.evaluation.primary_metric]:.4f}")
        
        # Keep only the best N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the best N checkpoints"""
        checkpoint_files = list(self.exp_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoint_files) > self.config.experiment.keep_n_checkpoints:
            # Sort by epoch number and remove oldest
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for file_to_remove in checkpoint_files[:-self.config.experiment.keep_n_checkpoints]:
                file_to_remove.unlink()
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"🚀 Starting training for {self.config.training.epochs} epochs")
        
        early_stopping_counter = 0
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            if epoch % self.config.evaluation.eval_frequency == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['val_loss'])
                self.val_metrics.append(val_metrics)
                
                # Check for improvement
                current_metric = val_metrics[self.config.evaluation.primary_metric]
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Log metrics
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['val_loss'], epoch)
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != 'val_loss':
                        self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
                
                # Save checkpoint
                if epoch % self.config.experiment.save_frequency == 0:
                    self.save_checkpoint(val_metrics, is_best)
                
                # Print progress
                print(f"Epoch {epoch+1}/{self.config.training.epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  {self.config.evaluation.primary_metric}: {current_metric:.4f}")
                if is_best:
                    print(f"  🎯 New best model!")
                
                # Early stopping
                if early_stopping_counter >= self.config.training.patience:
                    print(f"⏹️  Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()
        
        # Save final checkpoint
        final_metrics = self.validate(val_loader)
        self.save_checkpoint(final_metrics, False)
        
        print(f"✅ Training completed!")
        print(f"Best {self.config.evaluation.primary_metric}: {self.best_metric:.4f}")
        
        # Close writer
        self.writer.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train U-Net models on Woven by Toyota dataset')
    parser.add_argument('--model', type=str, choices=['baseline', 'attention'], 
                       default='attention', help='Model type to train')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load custom config (implementation would load from file)
        config = Config()  # Placeholder
    else:
        config = Config()
    
    print(f"🎯 Training {args.model} model")
    print(f"Target: {config.evaluation.primary_metric} improvement")
    
    # Create data loaders (placeholder - implement in utils/data_loading.py)
    print("📥 Creating data loaders...")
    # train_loader, val_loader = create_data_loaders(config)
    
    # For now, create dummy loaders
    print("⚠️  Using dummy data loaders - implement actual data loading")
    from torch.utils.data import TensorDataset
    
    # Dummy data
    dummy_inputs = torch.randn(100, config.model.in_channels, 
                              config.data.bev_height, config.data.bev_width)
    dummy_targets = torch.randint(0, config.model.out_channels, 
                                 (100, config.data.bev_height, config.data.bev_width))
    
    train_dataset = TensorDataset(dummy_inputs[:80], dummy_targets[:80])
    val_dataset = TensorDataset(dummy_inputs[80:], dummy_targets[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False)
    
    # Create trainer
    trainer = Trainer(config, model_type=args.model)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("🎉 Training script completed successfully!")


if __name__ == '__main__':
    main() 