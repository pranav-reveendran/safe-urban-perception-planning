#!/usr/bin/env python3
"""
Baseline U-Net Training Example
==============================

This script demonstrates how to train a baseline U-Net model for pedestrian segmentation
on the Woven by Toyota Perception Dataset.

Usage:
    python train_baseline.py --epochs 100 --batch_size 8 --lr 1e-3

Expected Results:
    - Pedestrian mIoU: ~55.4%
    - Overall mIoU: ~61.1%
    - Training Time: ~24 hours on H100 GPU
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from configs.base_config import Config
from models.unet import BaselineUNet
from utils.data_loading import WovenDataset
from utils.metrics import SegmentationMetrics
from utils.losses import FocalDiceLoss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Baseline U-Net Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_root', type=str, default='data/woven_dataset', help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='results/logs', help='Log directory for TensorBoard')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def create_data_loaders(data_root, batch_size, num_workers=4):
    """Create training and validation data loaders."""
    print("Creating data loaders...")
    
    # Training dataset
    train_dataset = WovenDataset(
        data_root=data_root,
        split='train',
        augmentation=True,
        num_sweeps=5
    )
    
    # Validation dataset
    val_dataset = WovenDataset(
        data_root=data_root,
        split='val',
        augmentation=False,
        num_sweeps=5
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_model(device):
    """Create and initialize the baseline U-Net model."""
    config = Config()
    
    model = BaselineUNet(
        in_channels=config.data.bev_channels,
        num_classes=len(config.data.class_names),
        base_channels=64
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: BaselineUNet")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes=5)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        bev_features = batch['bev_features'].to(device, non_blocking=True)
        segmentation = batch['segmentation'].to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(bev_features)
        loss = criterion(outputs, segmentation)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        metrics.update(predictions.cpu().numpy(), segmentation.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
        })
        
        # Log to TensorBoard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    avg_loss = running_loss / len(train_loader)
    
    # Log epoch metrics
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Pedestrian_mIoU', epoch_metrics['pedestrian_iou'], epoch)
    writer.add_scalar('Train/Overall_mIoU', epoch_metrics['mean_iou'], epoch)
    
    return avg_loss, epoch_metrics


def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes=5)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            bev_features = batch['bev_features'].to(device, non_blocking=True)
            segmentation = batch['segmentation'].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(bev_features)
            loss = criterion(outputs, segmentation)
            
            # Update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions.cpu().numpy(), segmentation.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
    
    # Compute epoch metrics
    epoch_metrics = metrics.compute()
    avg_loss = running_loss / len(val_loader)
    
    # Log epoch metrics
    writer.add_scalar('Val/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Val/Pedestrian_mIoU', epoch_metrics['pedestrian_iou'], epoch)
    writer.add_scalar('Val/Overall_mIoU', epoch_metrics['mean_iou'], epoch)
    
    return avg_loss, epoch_metrics


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'best_baseline_model.pth'
        torch.save(checkpoint, best_path)
        print(f"New best model saved: {best_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.data_root, args.batch_size)
    
    # Create model
    model = create_model(device)
    
    # Setup training components
    config = Config()
    criterion = FocalDiceLoss(
        focal_weight=config.training.focal_loss_weight,
        dice_weight=config.training.dice_loss_weight,
        class_weights=config.training.class_weights
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Setup logging
    writer = SummaryWriter(log_dir=f"{args.log_dir}/baseline_training")
    
    # Training loop
    best_miou = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint['metrics'].get('pedestrian_iou', 0.0)
    
    print(f"Starting training from epoch {start_epoch}")
    print(f"Training for {args.epochs} epochs")
    print("-" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step(val_metrics['pedestrian_iou'])
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Ped mIoU: {train_metrics['pedestrian_iou']:.4f} | Val Ped mIoU: {val_metrics['pedestrian_iou']:.4f}")
        print(f"Train Overall mIoU: {train_metrics['mean_iou']:.4f} | Val Overall mIoU: {val_metrics['mean_iou']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['pedestrian_iou'] > best_miou
        if is_best:
            best_miou = val_metrics['pedestrian_iou']
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint_path = Path(args.checkpoint_dir) / f'baseline_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, is_best)
        
        print("-" * 60)
    
    # Final results
    print(f"\nTraining completed!")
    print(f"Best validation pedestrian mIoU: {best_miou:.4f}")
    print(f"Expected baseline performance: ~55.4%")
    
    writer.close()


if __name__ == '__main__':
    main() 