# Examples Directory

This directory contains practical examples and sample code for using the Attention-Enhanced U-Net for pedestrian segmentation.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── basic_usage/                 # Basic usage examples
│   ├── train_baseline.py        # Train baseline U-Net
│   ├── train_attention.py       # Train attention-enhanced model
│   └── evaluate_model.py        # Model evaluation
├── data_processing/             # Data handling examples
│   ├── load_woven_dataset.py    # Dataset loading
│   ├── bev_preprocessing.py     # BEV conversion
│   └── visualize_data.py        # Data visualization
├── model_examples/              # Model architecture examples
│   ├── attention_mechanisms.py  # Attention components
│   ├── custom_models.py         # Custom model variants
│   └── model_comparison.py      # Compare different models
├── inference/                   # Inference and deployment
│   ├── single_frame_inference.py
│   ├── batch_inference.py
│   └── real_time_demo.py
└── notebooks/                   # Jupyter notebook examples
    ├── 01_dataset_exploration.ipynb
    ├── 02_model_training.ipynb
    ├── 03_results_analysis.ipynb
    └── 04_visualization_demo.ipynb
```

## 🚀 Quick Start Examples

### 1. Train Baseline Model
```bash
cd examples/basic_usage
python train_baseline.py --epochs 50 --batch_size 8
```

### 2. Train Attention-Enhanced Model
```bash
cd examples/basic_usage
python train_attention.py --attention_type attention_gates --epochs 100
```

### 3. Evaluate Trained Model
```bash
cd examples/basic_usage
python evaluate_model.py --checkpoint ../../results/checkpoints/best_model.pth
```

### 4. Process New Data
```bash
cd examples/data_processing
python bev_preprocessing.py --input_dir /path/to/lidar --output_dir /path/to/bev
```

### 5. Run Inference
```bash
cd examples/inference
python single_frame_inference.py --model_path ../../results/checkpoints/best_model.pth --input sample_frame.npy
```

## 📖 Example Categories

### Basic Usage
- **train_baseline.py**: Complete training script for baseline U-Net
- **train_attention.py**: Training script with attention mechanisms
- **evaluate_model.py**: Comprehensive model evaluation with metrics

### Data Processing
- **load_woven_dataset.py**: Load and process Woven by Toyota dataset
- **bev_preprocessing.py**: Convert LIDAR point clouds to BEV representation
- **visualize_data.py**: Visualize dataset samples and annotations

### Model Examples
- **attention_mechanisms.py**: Implement and test different attention types
- **custom_models.py**: Create custom model architectures
- **model_comparison.py**: Compare baseline vs attention-enhanced models

### Inference
- **single_frame_inference.py**: Process individual frames
- **batch_inference.py**: Process multiple frames efficiently
- **real_time_demo.py**: Real-time pedestrian detection demo

### Notebooks
- **01_dataset_exploration.ipynb**: Interactive dataset exploration
- **02_model_training.ipynb**: Step-by-step model training
- **03_results_analysis.ipynb**: Analyze and visualize results
- **04_visualization_demo.ipynb**: Create publication-quality figures

## 🎯 Performance Results from Examples

When you run these examples, you should expect to achieve:

### Baseline Model (train_baseline.py)
- **Pedestrian mIoU**: ~55.4%
- **Overall mIoU**: ~61.1%
- **Training Time**: ~24 hours on H100

### Attention-Enhanced Model (train_attention.py)
- **Pedestrian mIoU**: ~71.4% (+28.9% improvement)
- **Overall mIoU**: ~68.2% (+11.8% improvement)
- **Training Time**: ~48 hours on H100

## 💡 Tips for Using Examples

### For Beginners
1. Start with `notebooks/01_dataset_exploration.ipynb`
2. Run `basic_usage/train_baseline.py` first
3. Try different attention types with `basic_usage/train_attention.py`

### For Researchers
1. Modify `model_examples/custom_models.py` for new architectures
2. Use `model_examples/model_comparison.py` for ablation studies
3. Adapt `inference/batch_inference.py` for your evaluation pipeline

### For Practitioners
1. Use `inference/real_time_demo.py` for deployment testing
2. Optimize `data_processing/bev_preprocessing.py` for your data
3. Extend `inference/single_frame_inference.py` for production use

## 🔧 Configuration

All examples use the centralized configuration system:
```python
from configs.base_config import Config
config = Config()
# Modify parameters as needed
config.model.attention_type = 'attention_gates'
config.training.epochs = 100
```

## 📊 Expected Outputs

### Training Scripts
- Model checkpoints in `results/checkpoints/`
- Training logs in `results/logs/`
- TensorBoard logs for visualization

### Evaluation Scripts
- Metrics reports (JSON/CSV format)
- Confusion matrices and performance plots
- Qualitative results with visualizations

### Inference Scripts
- Segmentation masks (PNG format)
- Performance metrics (latency, accuracy)
- Visualization overlays

## 🤝 Contributing

To add new examples:
1. Follow the existing code style
2. Include comprehensive docstrings
3. Add configuration options
4. Provide expected output examples
5. Update this README with your example

## 📄 License

All examples are provided under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Need Help?** Check out the main [README](../README.md) or open an issue for support. 