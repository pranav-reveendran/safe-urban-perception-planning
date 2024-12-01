# 🚗 Autonomous Vehicle Perception: Attention-Enhanced Pedestrian Segmentation

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)

**Advancing Autonomous Vehicle Safety Through Deep Learning-Enhanced Pedestrian Detection**

*A comprehensive research project leveraging attention mechanisms in U-Net architectures for improved pedestrian segmentation in cluttered urban environments using the Woven by Toyota Perception Dataset.*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Research Motivation](#-research-motivation)
- [Dataset Information](#-dataset-information)
- [Technical Architecture](#-technical-architecture)
- [Implementation Details](#-implementation-details)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Results & Performance](#-results--performance)
- [Project Timeline](#-project-timeline)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 Project Overview

### **Objective**
Develop and evaluate attention-enhanced U-Net architectures to achieve **15-40% improvement in pedestrian mIoU** over baseline models for semantic segmentation in Bird's Eye View (BEV) LIDAR representations of urban driving scenarios.

### **Key Innovation**
Integration of multiple attention mechanisms (Attention Gates, Self-Attention, CBAM) into U-Net architectures specifically optimized for pedestrian detection in cluttered urban environments characteristic of Bay Area driving conditions.

### **Research Contribution**
- **Methodological**: Systematic evaluation of attention mechanisms for BEV LIDAR segmentation
- **Empirical**: Performance analysis on real-world urban driving data from Woven by Toyota
- **Practical**: Efficient attention integration suitable for autonomous vehicle deployment

## 🔬 Research Motivation

### **The Challenge**
Autonomous vehicles face critical safety challenges in detecting Vulnerable Road Users (VRUs), particularly pedestrians, in complex urban environments. Current perception systems struggle with:

- **Occlusion Handling**: Pedestrians partially hidden by vehicles, infrastructure
- **Cluttered Scenes**: Dense urban environments with multiple objects
- **Scale Variation**: Pedestrians at various distances from the ego vehicle
- **Real-time Constraints**: Computational efficiency requirements for deployment

### **The Opportunity**
Recent advances in attention mechanisms offer promising solutions for focusing neural networks on relevant spatial and feature dimensions, potentially improving pedestrian detection accuracy while maintaining computational efficiency.

### **Local Relevance**
As a Bay Area resident and San Jose State University graduate student, this research directly addresses local transportation challenges including high pedestrian density, complex urban infrastructure, and the increasing presence of autonomous vehicle testing fleets.

## 📊 Dataset Information

### **Woven by Toyota Perception Dataset**

**Overview**: 
- **Format**: nuScenes-compatible structure for seamless integration
- **Scale**: 350+ scenes with 60-90 minute collection periods
- **Coverage**: Bounded geographic area (likely Palo Alto/Bay Area)
- **License**: CC-BY-NC-SA-4.0 (suitable for academic research)

**Technical Specifications**:
```yaml
LIDAR Sensor: Velodyne HDL-32E (inferred)
  - Beams: 32 channels
  - Range: Up to 100m (effective ~70-80m)
  - Frequency: 20 Hz
  - Field of View: 360° horizontal, ~41° vertical
  - Accuracy: ±2 cm

Dataset Structure:
  - Sample: 573 MB (development subset)
  - Training: 58 GB
  - Testing/Validation: 58 GB
  - Total: ~116 GB

Annotations:
  - Provider: Scale AI
  - Format: 3D bounding boxes + semantic labels
  - Classes: ~23 categories (nuScenes taxonomy)
  - HD Maps: OpenStreetMap-derived semantic maps
```

**BEV Processing Configuration**:
- **Resolution**: 512×512 pixels
- **Spatial Range**: 100m × 100m (-50 to +50 meters)
- **Height Range**: 8m (-3 to +5 meters)
- **Features**: Height statistics, intensity, point density
- **Temporal**: 10-sweep accumulation for density enhancement

## 🏗️ Technical Architecture

### **Model Design Philosophy**

**Incremental Improvement Approach**: Build upon proven U-Net baseline with targeted attention enhancements rather than revolutionary architectural changes.

### **Core Architectures**

#### 1. **Baseline U-Net**
```python
Input: BEV Tensor [B, 4, 512, 512]
├── Encoder Path (5 levels)
│   ├── ConvBlock(4→64) → MaxPool
│   ├── ConvBlock(64→128) → MaxPool  
│   ├── ConvBlock(128→256) → MaxPool
│   ├── ConvBlock(256→512) → MaxPool
│   └── ConvBlock(512→1024)
├── Bottleneck: ConvBlock(1024→2048)
├── Decoder Path (4 levels)
│   ├── Upsample + Skip + ConvBlock(2048+512→512)
│   ├── Upsample + Skip + ConvBlock(512+256→256)
│   ├── Upsample + Skip + ConvBlock(256+128→128)
│   └── Upsample + Skip + ConvBlock(128+64→64)
└── Output: Conv1x1(64→10) → [B, 10, 512, 512]
```

#### 2. **Attention-Enhanced U-Net**
```python
Attention Integration Points:
├── Skip Connections: Attention Gates
│   └── Suppress irrelevant background features
├── Bottleneck: Self-Attention/CBAM
│   └── Global context understanding  
└── Decoder Blocks: Optional feature attention
    └── Refine pedestrian-specific features

Attention Types Supported:
├── Attention Gates: Skip connection gating
├── Self-Attention: Global spatial dependencies
├── CBAM: Channel + Spatial attention
└── Mixed: Combination of multiple mechanisms
```

### **Data Processing Pipeline**

```mermaid
graph LR
    A[LIDAR Point Cloud] --> B[Multi-sweep Accumulation]
    B --> C[BEV Rasterization]
    C --> D[Feature Encoding]
    D --> E[4-Channel BEV Tensor]
    E --> F[U-Net Processing]
    F --> G[Segmentation Output]
    
    C --> H[Height Statistics]
    C --> I[Intensity Values]  
    C --> J[Point Density]
    C --> K[HD Map Features]
```

### **Loss Function Strategy**

**Combined Focal + Dice Loss**:
```python
Loss = α × FocalLoss + β × DiceLoss

FocalLoss: Handles class imbalance, focuses on hard examples
DiceLoss: Optimizes boundary accuracy for pedestrian shapes
Class Weighting: 3× weight for pedestrian class
```

## 💻 Implementation Details

### **Technology Stack**

#### **Core Framework**
- **Deep Learning**: PyTorch 2.0+ with CUDA support
- **Data Processing**: NumPy, Pandas, OpenCV
- **3D Processing**: Lyft's modified nuScenes SDK
- **Visualization**: Matplotlib, Seaborn, Plotly

#### **Training Infrastructure**
- **Optimization**: Mixed Precision Training (AMP)
- **Logging**: TensorBoard + optional Weights & Biases
- **Hardware**: Google Colab Pro H100 GPU optimization
- **Memory**: Gradient accumulation, efficient data loading

#### **Development Tools**
- **Environment**: Jupyter notebooks for exploration
- **Configuration**: YAML-based parameter management
- **Version Control**: Git with detailed commit history
- **Documentation**: Comprehensive inline documentation

### **Model Configurations**

#### **Attention Mechanism Options**
```yaml
attention_gates:
  description: "Skip connection gating for background suppression"
  parameters:
    - gate_channels: Variable by level
    - skip_channels: Corresponding encoder channels
    - inter_channels: Computed as skip_channels // 2

self_attention:
  description: "Global spatial dependency modeling"
  parameters:
    - num_heads: 8
    - dropout: 0.1
    - channels: Level-dependent

cbam:
  description: "Channel and spatial attention combination"
  parameters:
    - reduction_ratio: 16
    - spatial_kernel_size: 7

mixed:
  description: "Combination of multiple attention types"
  parameters:
    - attention_types: ['channel', 'spatial']
    - configurable_combinations: true
```

### **Training Configuration**
```yaml
Hardware Requirements:
  - GPU: H100 (Google Colab Pro)
  - Memory: ~24GB VRAM with optimization
  - Storage: 20-40 GB for dataset subset
  - Compute Hours: 100-150 estimated

Optimization Strategy:
  - Mixed Precision: torch.cuda.amp
  - Gradient Accumulation: Effective batch size scaling
  - Memory Management: Periodic cache clearing
  - Efficiency: Sub-dataset training for development
```

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8+ 
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 40GB+ available storage

### **Environment Setup**

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/woven-toyota-perception.git
cd woven-toyota-perception
```

2. **Install Dependencies**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

3. **Dataset Setup**
```bash
# Create data directory structure
mkdir -p data/woven_dataset
mkdir -p data/processed
mkdir -p data/splits

# Download Woven by Toyota Perception Dataset
# Extract to data/woven_dataset/
# Install modified nuScenes SDK (provided with dataset)
```

4. **Configuration Validation**
```bash
python configs/base_config.py
```

### **Google Colab Setup**
```python
# For Colab users
!git clone https://github.com/yourusername/woven-toyota-perception.git
%cd woven-toyota-perception
!pip install -r requirements.txt

# Mount Google Drive for dataset storage
from google.colab import drive
drive.mount('/content/drive')
```

## 📚 Usage Guide

### **Data Exploration**
```bash
# Start Jupyter notebook for data exploration
jupyter notebook notebooks/data_exploration.ipynb

# Explore dataset structure and characteristics
# Analyze BEV parameters and pedestrian distribution
# Visualize sample data and annotations
```

### **Model Training**

#### **Baseline Model**
```bash
python experiments/train.py \
  --model baseline \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3
```

#### **Attention-Enhanced Model**
```bash
python experiments/train.py \
  --model attention \
  --attention_type attention_gates \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3
```

#### **Ablation Studies**
```bash
# Test different attention mechanisms
for attention in attention_gates self_attention cbam mixed; do
  python experiments/train.py \
    --model attention \
    --attention_type $attention \
    --experiment_name "ablation_${attention}"
done
```

### **Model Evaluation**
```bash
python experiments/evaluate.py \
  --checkpoint results/checkpoints/best_model.pth \
  --dataset_split test \
  --metrics pedestrian_miou overall_miou precision recall
```

### **Custom Configuration**
```python
# Modify configs/base_config.py for custom settings
from configs.base_config import Config

config = Config()
config.model.attention_type = 'cbam'
config.training.epochs = 150
config.data.batch_size = 16

# Save custom configuration
config.save_config('configs/my_experiment.yaml')
```

## 📊 Results & Performance

### **Expected Performance Targets**

| Metric | Baseline | Target | Optimistic |
|--------|----------|---------|------------|
| Pedestrian mIoU | 0.65 | 0.75-0.85 | 0.90+ |
| Overall mIoU | 0.70 | 0.75-0.80 | 0.85+ |
| Improvement | - | 15-25% | 40%+ |

### **Evaluation Metrics**
- **Primary**: Pedestrian class mean IoU (mIoU)
- **Secondary**: Overall mIoU, Precision, Recall, F1-Score
- **Statistical**: Significance testing across multiple random seeds
- **Practical**: Training time, memory usage, inference speed

### **Baseline Comparisons**
- **Woven Toyota U-Net**: Official baseline solution
- **Literature Benchmarks**: PolarNet (~0.69 mIoU), DB-Unet (~0.70 mIoU)
- **Attention Variants**: Systematic ablation across attention types

## 📅 Project Timeline

### **Development History (Sep 2024 - Feb 2025)**

```
Sep 2024: Project Initiation & Literature Review
├── Research proposal development
├── Literature survey on attention mechanisms
└── Dataset access and initial exploration

Oct 2024: Foundation Implementation  
├── Project structure and configuration system
├── Baseline U-Net implementation
└── Attention mechanism development

Nov 2024: Core Development
├── Attention-enhanced U-Net integration
├── Training pipeline implementation  
└── Data preprocessing optimization

Dec 2024: Experimentation & Validation
├── Comprehensive ablation studies
├── Hyperparameter optimization
└── Performance benchmarking

Jan 2025: Analysis & Optimization
├── Results analysis and interpretation
├── Model optimization and efficiency tuning
└── Documentation and visualization

Feb 2025: Finalization & Presentation
├── Final report and presentation preparation
├── Code cleanup and documentation
└── Open-source release preparation
```

### **Key Milestones**
- ✅ **Sep 2024**: Project proposal and initial setup
- ✅ **Oct 2024**: Baseline implementation complete
- ✅ **Nov 2024**: Attention mechanisms integrated
- 🎯 **Dec 2024**: First performance improvements demonstrated
- 🎯 **Jan 2025**: 15-40% improvement target achieved
- 🎯 **Feb 2025**: Final evaluation and documentation complete

## 🔬 Research Contributions

### **Academic Impact**
1. **Novel Application**: First systematic study of attention mechanisms for BEV LIDAR pedestrian segmentation
2. **Methodological Innovation**: Efficient attention integration for resource-constrained AV applications  
3. **Empirical Analysis**: Comprehensive evaluation on real-world urban driving data

### **Industry Relevance**
1. **Safety Enhancement**: Improved pedestrian detection for autonomous vehicle deployment
2. **Efficiency Optimization**: Lightweight attention mechanisms for real-time processing
3. **Urban Deployment**: Solutions for complex urban driving scenarios

### **Open Source Contribution**
- Complete, reproducible implementation
- Comprehensive documentation and tutorials
- Modular design for easy extension and adaptation

## 🤝 Contributing

We welcome contributions to improve pedestrian segmentation in autonomous vehicles! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Ways to Contribute**
- **Research**: Novel attention mechanisms or architectural improvements
- **Implementation**: Code optimization, bug fixes, feature additions
- **Documentation**: Tutorials, examples, improved documentation
- **Evaluation**: Testing on additional datasets or scenarios

### **Development Setup**
```bash
# Fork the repository
git clone https://github.com/yourusername/woven-toyota-perception.git
cd woven-toyota-perception

# Create feature branch
git checkout -b feature/your-improvement

# Make changes and commit
git commit -m "Add your improvement description"

# Submit pull request
git push origin feature/your-improvement
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Dataset License**: The Woven by Toyota Perception Dataset is licensed under CC-BY-NC-SA-4.0, restricting use to non-commercial academic research.

## 🙏 Acknowledgments

### **Dataset & Infrastructure**
- **Woven by Toyota** for providing the Perception Dataset
- **Lyft/nuScenes team** for the foundational SDK and data format
- **Scale AI** for high-quality 3D annotations

### **Academic Support**
- **San Jose State University** AI/ML Graduate Program
- **Bay Area** autonomous vehicle research community
- **Open source community** for foundational tools and libraries

### **Technical Foundation**
- **PyTorch team** for the deep learning framework
- **Google Colab** for accessible GPU compute resources
- **GitHub** for version control and collaboration platform

---

<div align="center">

**🚗 Advancing Autonomous Vehicle Safety Through AI Research 🚗**

*A graduate research project by a San Jose State University AI/ML student*  
*Contributing to safer autonomous vehicle deployment in urban environments*

**[Documentation](docs/) | [Examples](examples/) | [Results](results/) | [Issues](https://github.com/yourusername/woven-toyota-perception/issues)**

</div> 