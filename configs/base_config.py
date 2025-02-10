"""
Base configuration file for the Woven by Toyota Perception Project
Author: Graduate Student, SJSU AI/ML Program
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass
class DataConfig:
    """Data-related configuration"""
    # Dataset paths
    dataset_root: str = "data/woven_dataset"
    processed_data_root: str = "data/processed"
    splits_dir: str = "data/splits"
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data subset for development (use fraction of full dataset)
    use_data_fraction: float = 0.25  # Start with 25% for development
    
    # BEV (Bird's Eye View) parameters
    bev_width: int = 512
    bev_height: int = 512
    bev_resolution: float = 0.1  # meters per pixel
    
    # LIDAR processing parameters
    point_cloud_range: List[float] = field(default_factory=lambda: [-50.0, -50.0, -3.0, 50.0, 50.0, 5.0])  # [x_min, y_min, z_min, x_max, y_max, z_max]
    max_sweeps: int = 5  # Number of LIDAR sweeps to accumulate
    
    # BEV features to encode
    bev_channels: List[str] = field(default_factory=lambda: ['height_max', 'height_mean', 'intensity_max', 'density'])
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 0.78539816  # ±45 degrees in radians
    scale_range: Tuple[float, float] = (0.9, 1.1)
    flip_probability: float = 0.5
    
    # Class information
    pedestrian_class_id: int = 2  # Assuming pedestrian is class 2 (adjust based on actual dataset)
    num_classes: int = 10  # Total number of semantic classes
    
    # Dataloader parameters
    batch_size: int = 8  # Adjust based on GPU memory
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Base U-Net parameters
    in_channels: int = 4  # Based on BEV channels (height_max, height_mean, intensity_max, density)
    out_channels: int = 10  # Number of semantic classes
    base_filters: int = 64
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])  # Depth of each encoder block
    decoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])     # Depth of each decoder block
    
    # Attention mechanism configuration
    use_attention: bool = True
    attention_type: str = "attention_gates"  # Options: "attention_gates", "self_attention", "cbam", "mixed"
    
    # Attention Gates parameters
    attention_gate_filters: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Self-attention parameters
    self_attention_heads: int = 8
    self_attention_dim: int = 512
    
    # CBAM parameters
    cbam_reduction_ratio: int = 16
    
    # Dropout and regularization
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    activation: str = "relu"  # Options: "relu", "leaky_relu", "gelu"


@dataclass
class TrainingConfig:
    """Training-related configuration"""
    # Training parameters
    epochs: int = 100
    initial_lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # Options: "adam", "adamw", "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # Options: "cosine", "step", "plateau", "linear"
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1   # For step scheduler
    min_lr: float = 1e-6  # For cosine scheduler
    
    # Loss function
    loss_type: str = "combined"  # Options: "ce", "focal", "dice", "combined"
    focal_alpha: float = 1.0  # Focal loss alpha
    focal_gamma: float = 2.0  # Focal loss gamma
    dice_weight: float = 0.5  # Weight for dice loss in combined loss
    focal_weight: float = 0.5  # Weight for focal loss in combined loss
    
    # Class weighting (for pedestrian focus)
    use_class_weights: bool = True
    pedestrian_weight: float = 3.0  # Higher weight for pedestrian class
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1  # Increase if batch size needs to be larger than GPU memory allows


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration"""
    # Primary metrics
    primary_metric: str = "pedestrian_miou"
    
    # Metrics to compute
    compute_metrics: List[str] = field(default_factory=lambda: [
        "miou_overall", "miou_pedestrian", "precision_pedestrian", 
        "recall_pedestrian", "f1_pedestrian", "accuracy_overall"
    ])
    
    # Evaluation frequency
    eval_frequency: int = 5  # Evaluate every N epochs
    save_best_model: bool = True
    
    # Visualization
    save_predictions: bool = True
    num_visualizations: int = 10  # Number of prediction visualizations to save
    
    # Statistical validation
    num_seeds: int = 3  # Number of random seeds for statistical validation
    confidence_level: float = 0.95


@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration"""
    # Experiment identification
    experiment_name: str = "attention_unet_pedestrian_segmentation"
    project_name: str = "woven_toyota_perception"
    
    # Logging
    log_dir: str = "results/logs"
    checkpoint_dir: str = "results/checkpoints"
    figures_dir: str = "results/figures"
    
    # Checkpointing
    save_frequency: int = 10  # Save checkpoint every N epochs
    keep_n_checkpoints: int = 5  # Keep only the best N checkpoints
    
    # Logging frequency
    log_frequency: int = 10  # Log every N batches
    
    # Reproducibility
    random_seed: int = 42
    
    # Weights & Biases (optional)
    use_wandb: bool = False  # Set to True if using W&B
    wandb_project: str = "woven-toyota-perception"
    wandb_entity: str = ""  # Your W&B username/entity


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Device configuration
    device: str = "cuda"  # Will be automatically set to "cuda" if available, else "cpu"
    
    # Memory management
    max_memory_usage: float = 0.9  # Maximum fraction of GPU memory to use
    empty_cache_frequency: int = 100  # Empty CUDA cache every N batches
    
    # Parallel processing
    num_gpus: int = 1  # Number of GPUs to use (for multi-GPU training)
    distributed: bool = False
    
    # Debugging
    debug_mode: bool = False
    profile_model: bool = False


@dataclass
class Config:
    """Main configuration class combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Create directories
        os.makedirs(self.data.processed_data_root, exist_ok=True)
        os.makedirs(self.data.splits_dir, exist_ok=True)
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.checkpoint_dir, exist_ok=True)
        os.makedirs(self.experiment.figures_dir, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Data validation
        assert 0 < self.data.use_data_fraction <= 1.0, "Data fraction must be between 0 and 1"
        assert abs(self.data.train_split + self.data.val_split + self.data.test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Model validation
        assert self.model.in_channels == len(self.data.bev_channels), "Model input channels must match BEV channels"
        assert self.model.out_channels == self.data.num_classes, "Model output channels must match number of classes"
        
        # Training validation
        assert self.training.epochs > 0, "Number of epochs must be positive"
        assert self.training.initial_lr > 0, "Learning rate must be positive"
        
        print("✓ Configuration validation passed")
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import yaml
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# Default configuration instance
default_config = Config()

if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Configuration created successfully!")
    print(f"Experiment name: {config.experiment.experiment_name}")
    print(f"BEV resolution: {config.data.bev_width}x{config.data.bev_height}")
    print(f"Attention type: {config.model.attention_type}")
    print(f"Target improvement: 15-40% in pedestrian mIoU") # Configuration optimization for Colab
