# Woven by Toyota Perception Project - Implementation Summary

## 🎯 Project Overview

**Title**: Improving Pedestrian Segmentation in Cluttered Urban Scenes using Attention-Enhanced U-Net on the Woven by Toyota Perception Dataset

**Objective**: Achieve 15-40% improvement in pedestrian mIoU over baseline U-Net through attention mechanism integration

**Timeline**: 10 weeks (2-3 months)

**Author**: Graduate Student, AI/ML Program, San Jose State University

## 📊 Current Implementation Status

### ✅ Completed Components

#### 1. Project Structure & Configuration
- [x] Complete project directory structure
- [x] Comprehensive configuration system (`configs/base_config.py`)
- [x] Requirements and dependencies (`requirements.txt`)
- [x] Professional README with project overview

#### 2. Model Architectures
- [x] **Baseline U-Net** (`models/baseline_unet.py`)
  - Standard encoder-decoder architecture
  - Configurable depths and filters
  - Skip connections and proper initialization
  
- [x] **Attention Mechanisms** (`models/attention_modules.py`)
  - Attention Gates for skip connections
  - Self-Attention (2D adapted for BEV)
  - CBAM (Channel and Spatial Attention)
  - Mixed Attention combining multiple types
  
- [x] **Attention-Enhanced U-Net** (`models/attention_unet.py`)
  - Integration of attention mechanisms
  - Flexible attention placement (bottleneck, decoder)
  - Factory functions for easy model creation

#### 3. Training Infrastructure
- [x] **Training Script** (`experiments/train.py`)
  - Combined Focal + Dice Loss for pedestrian focus
  - Mixed precision training support
  - Comprehensive logging with TensorBoard
  - Early stopping and checkpointing
  - Statistical validation with multiple seeds

#### 4. Data Exploration Framework
- [x] **Jupyter Notebook** (`notebooks/data_exploration.ipynb`)
  - Dataset structure analysis framework
  - BEV rasterization parameter visualization
  - Pedestrian analysis preparation
  - Model architecture comparison

### ⏳ Pending Implementation (Week 1-2 Tasks)

#### 1. Dataset Integration
- [ ] Download Woven by Toyota Perception Dataset
- [ ] Install Lyft's modified nuScenes SDK
- [ ] Implement dataset loading (`utils/data_loading.py`)
- [ ] BEV preprocessing pipeline (`utils/preprocessing.py`)

#### 2. Evaluation System
- [ ] Metrics computation (`utils/metrics.py`)
- [ ] Visualization utilities (`utils/visualization.py`)
- [ ] Ablation study framework

#### 3. Data Analysis
- [ ] Complete dataset exploration with real data
- [ ] Pedestrian class distribution analysis
- [ ] Scene characteristics and challenges identification

## 🏗️ Technical Architecture

### Model Design Philosophy

**Incremental Improvement Approach**: Build upon proven U-Net baseline with targeted attention enhancements

**Key Innovations**:
1. **Attention Gates**: Suppress irrelevant background features in skip connections
2. **Bottleneck Self-Attention**: Capture global scene context for pedestrian understanding
3. **Class-Weighted Loss**: Focus learning on pedestrian class with 3x weighting
4. **Combined Loss Function**: Focal Loss (hard examples) + Dice Loss (boundary accuracy)

### BEV Processing Pipeline

```
LIDAR Point Cloud → Multi-sweep Accumulation → BEV Rasterization → Feature Encoding
                                                      ↓
Height Statistics + Intensity + Density → 4-Channel BEV Tensor → U-Net Input
```

**Configuration**:
- BEV Resolution: 512×512 pixels
- Spatial Range: 100m × 100m (-50 to +50 meters)
- Point Cloud Range: [-50, -50, -3, 50, 50, 5] meters
- Features: [height_max, height_mean, intensity_max, density]

### Attention Mechanism Strategy

1. **Attention Gates in Skip Connections**:
   - Gating signal from decoder suppresses irrelevant encoder features
   - Particularly effective for cluttered urban scenes

2. **Self-Attention at Bottleneck**:
   - Captures long-range spatial dependencies
   - Helps understand pedestrian context in complex scenes

3. **Optional CBAM in Decoder**:
   - Channel attention: "what" features are important
   - Spatial attention: "where" to focus attention

## 📈 Expected Performance Improvements

### Baseline Performance (Reference)
- **U-Net variants on nuScenes**: ~0.69-0.70 mIoU (PolarNet, DB-Unet)
- **Target Baseline**: Woven by Toyota provided U-Net solution

### Projected Improvements
- **Conservative**: 15% improvement in pedestrian mIoU
- **Target**: 25% improvement in pedestrian mIoU  
- **Optimistic**: 40% improvement in pedestrian mIoU

### Success Metrics
- Primary: Pedestrian class mIoU improvement
- Secondary: Overall mIoU, precision, recall, F1-score
- Statistical: Significance testing across multiple seeds
- Practical: Training efficiency within H100 GPU constraints

## 🛠️ Development Environment

### Hardware Requirements
- **Primary**: Google Colab Pro with H100 GPU
- **Estimated GPU Hours**: 100-150 hours total
- **Memory**: Mixed precision + gradient accumulation for optimization

### Software Stack
- **Framework**: PyTorch 2.x with CUDA support
- **Data**: Lyft's modified nuScenes SDK
- **Logging**: TensorBoard + optional Weights & Biases
- **Development**: Jupyter notebooks for exploration

### Resource Management
- **Dataset Subset**: 25-50% initially for development
- **Memory Optimization**: AMP, gradient accumulation, efficient data loading
- **Storage**: 20-40 GB for dataset subset and model checkpoints

## 📅 Week-by-Week Timeline

### Week 1-2: Setup & Familiarization ✅→⏳
- ✅ Project structure and configuration
- ✅ Model architectures implementation  
- ⏳ Dataset download and SDK installation
- ⏳ Data exploration and preprocessing pipeline
- ⏳ Baseline model validation

### Week 3-5: Core Implementation
- [ ] Attention mechanism integration testing
- [ ] Training pipeline debugging and optimization
- [ ] Initial model comparison experiments
- [ ] Hyperparameter tuning framework

### Week 6-7: Experimentation & Validation
- [ ] Comprehensive ablation studies
- [ ] Statistical validation with multiple seeds
- [ ] Performance optimization and scaling
- [ ] Baseline vs attention model comparison

### Week 8: Full Evaluation
- [ ] Final model training on larger dataset
- [ ] Comprehensive metrics evaluation
- [ ] Qualitative analysis and failure case study
- [ ] Results interpretation and analysis

### Week 9-10: Documentation & Presentation
- [ ] Technical report writing
- [ ] Code documentation and cleanup
- [ ] Results visualization and presentation
- [ ] Reproducibility verification

## 🎯 Next Immediate Actions

### 1. Dataset Setup (Priority 1)
```bash
# Download Woven by Toyota Perception Dataset
# 1. Sample dataset (573 MB) for initial development
# 2. Training subset (~25% of 58 GB) for model development
# 3. Validation dataset for final evaluation

# Install modified nuScenes SDK
pip install <woven-nuscenes-sdk>
```

### 2. Data Pipeline Implementation (Priority 2)
- Complete `utils/data_loading.py` with WovenDataset class
- Implement BEV rasterization in `utils/preprocessing.py`
- Add metrics computation in `utils/metrics.py`

### 3. Initial Training (Priority 3)
- Test baseline U-Net with real data
- Validate attention mechanisms integration
- Establish performance baseline

## 🔬 Research Contributions

### Academic Value
1. **Methodological**: Systematic evaluation of attention mechanisms for BEV segmentation
2. **Empirical**: Performance analysis on real-world urban driving data
3. **Practical**: Efficient attention integration for resource-constrained environments

### Industry Relevance
1. **Safety**: Improved pedestrian detection for AV deployment
2. **Efficiency**: Lightweight attention for real-time processing
3. **Robustness**: Better performance in cluttered urban scenarios

## 📚 References & Related Work

### Core Dataset
- Woven by Toyota Perception Dataset (nuScenes format)
- nuScenes Dataset and SDK (Lyft/Caesar)

### Attention Mechanisms
- Oktay et al. (2018): Attention U-Net
- Woo et al. (2018): CBAM
- Vaswani et al. (2017): Transformer attention

### LIDAR Perception
- Recent surveys on 3D object detection (2022-2024)
- BEV representation methods
- Urban scene understanding

## 🚀 Getting Started

### Quick Start
1. **Clone/Setup**: Project structure is ready
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Download Dataset**: Woven by Toyota Perception Dataset
4. **Explore Data**: Run `notebooks/data_exploration.ipynb`
5. **Train Models**: `python experiments/train.py --model attention`

### Configuration Customization
Edit `configs/base_config.py` to modify:
- Model architecture parameters
- Training hyperparameters  
- Data processing settings
- Evaluation metrics

---

**Project Status**: Foundation Complete ✅ | Ready for Dataset Integration ⏳

**Next Milestone**: Complete data pipeline and begin baseline training by end of Week 2 