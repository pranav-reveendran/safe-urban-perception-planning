"""
Data Loading Utilities for Woven by Toyota Perception Dataset

This module provides dataset classes and data loading utilities for processing
the Woven by Toyota Perception Dataset in nuScenes format for BEV LIDAR
segmentation tasks.

Author: Graduate Student, AI/ML Program, San Jose State University
Date: October 2024
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .preprocessing import LidarBEVProcessor
from .metrics import calculate_miou


class WovenDataset(Dataset):
    """
    PyTorch Dataset class for Woven by Toyota Perception Dataset
    
    Handles loading and preprocessing of LIDAR point clouds and semantic
    annotations for BEV segmentation tasks.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        bev_params: Optional[Dict] = None,
        num_sweeps: int = 10,
        augmentation: bool = True,
        cache_data: bool = False,
        subset_size: Optional[int] = None
    ):
        """
        Initialize Woven Dataset
        
        Args:
            data_root: Path to dataset root directory
            split: Dataset split ('train', 'val', 'test')
            bev_params: BEV rasterization parameters
            num_sweeps: Number of LIDAR sweeps to accumulate
            augmentation: Whether to apply data augmentation
            cache_data: Whether to cache preprocessed data
            subset_size: Limit dataset size for development
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_sweeps = num_sweeps
        self.augmentation = augmentation and (split == 'train')
        self.cache_data = cache_data
        
        # Default BEV parameters
        self.bev_params = bev_params or {
            'x_range': [-50.0, 50.0],
            'y_range': [-50.0, 50.0],
            'z_range': [-3.0, 5.0],
            'resolution': 0.1,
            'height': 512,
            'width': 512,
            'channels': 4  # height_max, height_mean, intensity_max, density
        }
        
        # Initialize BEV processor
        self.bev_processor = LidarBEVProcessor(self.bev_params)
        
        # Load dataset splits
        self.samples = self._load_dataset_splits()
        
        # Apply subset if specified
        if subset_size is not None:
            self.samples = self.samples[:subset_size]
            
        # Cache for preprocessed data
        self.cache = {} if cache_data else None
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_dataset_splits(self) -> List[Dict]:
        """Load dataset splits from metadata files"""
        splits_file = self.data_root / f'splits/{self.split}.json'
        
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                samples = json.load(f)
        else:
            # Create splits if not exist
            samples = self._create_dataset_splits()
            
        return samples
    
    def _create_dataset_splits(self) -> List[Dict]:
        """Create dataset splits from available data"""
        # This would be implemented based on the actual dataset structure
        # For now, return placeholder structure
        
        samples = []
        scenes_dir = self.data_root / 'scenes'
        
        if scenes_dir.exists():
            for scene_file in scenes_dir.glob('*.json'):
                with open(scene_file, 'r') as f:
                    scene_data = json.load(f)
                
                # Extract samples from scene
                for sample in scene_data.get('samples', []):
                    samples.append({
                        'scene_token': scene_data['token'],
                        'sample_token': sample['token'],
                        'lidar_token': sample['data']['LIDAR_TOP'],
                        'timestamp': sample['timestamp'],
                        'scene_name': scene_data['name']
                    })
        
        # Split data into train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        if self.split == 'train':
            indices = indices[:int(0.7 * len(indices))]
        elif self.split == 'val':
            indices = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
        else:  # test
            indices = indices[int(0.85 * len(indices)):]
            
        return [samples[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item
        
        Returns:
            Dictionary containing:
                - bev_tensor: BEV representation [4, H, W]
                - seg_labels: Segmentation labels [H, W]
                - metadata: Sample metadata
        """
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        
        # Load LIDAR point cloud
        point_cloud = self._load_lidar_data(sample['lidar_token'])
        
        # Load segmentation annotations
        seg_labels = self._load_segmentation_labels(sample['sample_token'])
        
        # Convert to BEV representation
        bev_tensor = self.bev_processor.process_point_cloud(
            point_cloud, 
            num_sweeps=self.num_sweeps
        )
        
        # Apply augmentation if enabled
        if self.augmentation:
            bev_tensor, seg_labels = self._apply_augmentation(bev_tensor, seg_labels)
        
        # Convert to torch tensors
        bev_tensor = torch.from_numpy(bev_tensor).float()
        seg_labels = torch.from_numpy(seg_labels).long()
        
        item = {
            'bev_tensor': bev_tensor,
            'seg_labels': seg_labels,
            'metadata': {
                'sample_token': sample['sample_token'],
                'scene_name': sample['scene_name'],
                'timestamp': sample['timestamp']
            }
        }
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = item
            
        return item
    
    def _load_lidar_data(self, lidar_token: str) -> np.ndarray:
        """Load LIDAR point cloud data"""
        # Placeholder implementation - would load actual LIDAR data
        # from the dataset based on the token
        
        lidar_file = self.data_root / f'lidar/{lidar_token}.bin'
        
        if lidar_file.exists():
            # Load binary LIDAR data (typical format: x, y, z, intensity)
            points = np.fromfile(str(lidar_file), dtype=np.float32)
            points = points.reshape(-1, 4)  # Assuming 4 channels
        else:
            # Generate dummy data for development
            points = np.random.randn(10000, 4).astype(np.float32)
            points[:, :3] *= 50  # Scale to reasonable range
            points[:, 3] = np.random.rand(10000)  # Intensity
            
        return points
    
    def _load_segmentation_labels(self, sample_token: str) -> np.ndarray:
        """Load segmentation labels"""
        # Placeholder implementation - would load actual labels
        
        labels_file = self.data_root / f'labels/{sample_token}.png'
        
        if labels_file.exists():
            # Load actual labels
            import cv2
            labels = cv2.imread(str(labels_file), cv2.IMREAD_GRAYSCALE)
        else:
            # Generate dummy labels for development
            labels = np.random.randint(
                0, 10, 
                size=(self.bev_params['height'], self.bev_params['width'])
            ).astype(np.uint8)
            
        return labels
    
    def _apply_augmentation(
        self, 
        bev_tensor: np.ndarray, 
        seg_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        
        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)  # degrees
            bev_tensor, seg_labels = self._rotate_bev(bev_tensor, seg_labels, angle)
        
        # Random flip
        if np.random.rand() < 0.5:
            bev_tensor = np.flip(bev_tensor, axis=2)  # Flip width
            seg_labels = np.flip(seg_labels, axis=1)
        
        # Random noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, bev_tensor.shape)
            bev_tensor = bev_tensor + noise
        
        return bev_tensor, seg_labels
    
    def _rotate_bev(
        self, 
        bev_tensor: np.ndarray, 
        seg_labels: np.ndarray, 
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate BEV tensor and labels"""
        import cv2
        
        h, w = bev_tensor.shape[1], bev_tensor.shape[2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate each channel
        rotated_bev = np.zeros_like(bev_tensor)
        for i in range(bev_tensor.shape[0]):
            rotated_bev[i] = cv2.warpAffine(
                bev_tensor[i], M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        # Rotate labels
        rotated_labels = cv2.warpAffine(
            seg_labels, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return rotated_bev, rotated_labels


def create_data_loader(
    dataset: WovenDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for WovenDataset
    
    Args:
        dataset: WovenDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched data dictionary
    """
    bev_tensors = torch.stack([item['bev_tensor'] for item in batch])
    seg_labels = torch.stack([item['seg_labels'] for item in batch])
    
    # Collect metadata
    metadata = {
        'sample_tokens': [item['metadata']['sample_token'] for item in batch],
        'scene_names': [item['metadata']['scene_name'] for item in batch],
        'timestamps': [item['metadata']['timestamp'] for item in batch]
    }
    
    return {
        'bev_tensor': bev_tensors,
        'seg_labels': seg_labels,
        'metadata': metadata
    }


def get_class_weights(dataset: WovenDataset) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        dataset: WovenDataset instance
        
    Returns:
        Class weights tensor
    """
    print("Calculating class weights...")
    
    class_counts = np.zeros(10)  # Assuming 10 classes
    
    for i in range(min(len(dataset), 1000)):  # Sample subset for efficiency
        item = dataset[i]
        labels = item['seg_labels'].numpy()
        
        for class_id in range(10):
            class_counts[class_id] += np.sum(labels == class_id)
    
    # Calculate inverse frequency weights
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (10 * class_counts + 1e-8)
    
    # Apply pedestrian boost (assuming pedestrian is class 2)
    class_weights[2] *= 3.0  # 3x weight for pedestrians
    
    return torch.FloatTensor(class_weights)


def create_datasets(
    data_root: str,
    config: Dict,
    subset_size: Optional[int] = None
) -> Tuple[WovenDataset, WovenDataset, WovenDataset]:
    """
    Create train, validation, and test datasets
    
    Args:
        data_root: Path to dataset root
        config: Configuration dictionary
        subset_size: Limit dataset size for development
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Extract BEV parameters from config
    bev_params = {
        'x_range': config.get('x_range', [-50.0, 50.0]),
        'y_range': config.get('y_range', [-50.0, 50.0]),
        'z_range': config.get('z_range', [-3.0, 5.0]),
        'resolution': config.get('resolution', 0.1),
        'height': config.get('bev_height', 512),
        'width': config.get('bev_width', 512),
        'channels': config.get('bev_channels', 4)
    }
    
    # Create datasets
    train_dataset = WovenDataset(
        data_root=data_root,
        split='train',
        bev_params=bev_params,
        augmentation=True,
        cache_data=False,
        subset_size=subset_size
    )
    
    val_dataset = WovenDataset(
        data_root=data_root,
        split='val',
        bev_params=bev_params,
        augmentation=False,
        cache_data=True,
        subset_size=subset_size // 4 if subset_size else None
    )
    
    test_dataset = WovenDataset(
        data_root=data_root,
        split='test',
        bev_params=bev_params,
        augmentation=False,
        cache_data=True,
        subset_size=subset_size // 4 if subset_size else None
    )
    
    return train_dataset, val_dataset, test_dataset


# Example usage and testing
if __name__ == "__main__":
    # Test dataset creation
    data_root = "data/woven_dataset"
    
    # Create test dataset
    dataset = WovenDataset(
        data_root=data_root,
        split='train',
        subset_size=10  # Small subset for testing
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test data loading
    if len(dataset) > 0:
        item = dataset[0]
        print(f"BEV tensor shape: {item['bev_tensor'].shape}")
        print(f"Segmentation labels shape: {item['seg_labels'].shape}")
        print(f"Metadata: {item['metadata']}")
    
    # Test data loader
    dataloader = create_data_loader(dataset, batch_size=2)
    
    for batch in dataloader:
        print(f"Batch BEV shape: {batch['bev_tensor'].shape}")
        print(f"Batch labels shape: {batch['seg_labels'].shape}")
        break
    
    print("Data loading test completed successfully!") 