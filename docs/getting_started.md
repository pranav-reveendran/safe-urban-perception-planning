# Getting Started

Welcome to the Attention-Enhanced U-Net for Pedestrian Segmentation project! This guide will help you get up and running quickly.

## 🎯 Project Overview

This project implements an attention-enhanced U-Net architecture for improving pedestrian segmentation in autonomous vehicle perception systems. We achieved a **28.9% improvement** in pedestrian mIoU over baseline models using the Woven by Toyota Perception Dataset.

### Key Features
- **Attention-Enhanced U-Net** with multiple attention mechanisms
- **Synergistic training strategy** combining attention + class-weighted loss
- **Production-ready implementation** optimized for autonomous vehicles
- **Comprehensive evaluation** on urban driving scenarios

## 🚀 Quick Start (5 minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/pranav-reveendran/safe-urban-perception-planning.git
cd safe-urban-perception-planning
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 3. Test Configuration
```bash
python configs/base_config.py
```
Expected output:
```
✓ Configuration loaded successfully
✓ All paths validated
Target improvement: 15-40% in pedestrian mIoU
```

### 4. Run a Quick Example
```bash
# Train a small model on sample data (if available)
python examples/basic_usage/train_baseline.py --epochs 5 --batch_size 4
```

## 📊 Expected Results

After following this guide, you should be able to:

### Training Results
- **Baseline Model**: ~55.4% pedestrian mIoU
- **Attention-Enhanced Model**: ~71.4% pedestrian mIoU (+28.9% improvement)
- **Training Time**: 24-48 hours on H100 GPU

### Inference Performance
- **Latency**: ~23ms per frame (512×512 BEV)
- **Memory**: ~2.8GB VRAM
- **Accuracy**: 71.4% pedestrian mIoU

## 🏗️ Project Structure

```
safe-urban-perception-planning/
├── 📁 configs/           # Configuration files
├── 📁 data/             # Dataset and preprocessing
├── 📁 docs/             # Documentation (you are here)
├── 📁 examples/         # Usage examples and tutorials
├── 📁 experiments/      # Training and evaluation scripts
├── 📁 models/           # Model architectures
├── 📁 notebooks/        # Jupyter notebooks for exploration
├── 📁 results/          # Training results and analysis
├── 📁 utils/            # Utility functions
├── 📄 README.md         # Main project documentation
└── 📄 requirements.txt  # Python dependencies
```

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **GPU Memory**: 8GB VRAM (for training)
- **Storage**: 50GB free space
- **RAM**: 16GB system memory

### Recommended Setup
- **GPU**: NVIDIA H100 or A100 (Google Colab Pro)
- **GPU Memory**: 24GB+ VRAM
- **Storage**: 250GB+ SSD
- **RAM**: 32GB+ system memory

### Software Dependencies
- **PyTorch**: 2.0+ with CUDA support
- **CUDA**: 11.8+ for GPU acceleration
- **OpenCV**: For image processing
- **NumPy/Pandas**: For data manipulation

## 📚 Next Steps

Depending on your goal, follow these learning paths:

### 🔬 For Researchers
1. **[Model Architecture](model_architecture.md)** - Understand the technical details
2. **[Attention Mechanisms](research/attention_mechanisms.md)** - Deep dive into attention
3. **[Ablation Studies](research/ablation_studies.md)** - Analyze component contributions

### 👨‍💻 For Practitioners
1. **[Data Preparation](data_preparation.md)** - Set up your dataset
2. **[Training Guide](training_guide.md)** - Train models effectively
3. **[Inference Deployment](inference_deployment.md)** - Deploy in production

### 🎓 For Students
1. **[Examples](../examples/)** - Hands-on code examples
2. **[Notebooks](../notebooks/)** - Interactive tutorials
3. **[Configuration](configuration.md)** - Customize the system

## 💡 Key Concepts

### Attention Mechanisms
The project implements multiple attention types:
- **Attention Gates**: Suppress irrelevant background features
- **Self-Attention**: Capture global spatial dependencies  
- **CBAM**: Channel and spatial attention combination
- **Mixed Attention**: Combination of multiple mechanisms

### BEV (Bird's Eye View) Processing
- **Input**: LIDAR point clouds from autonomous vehicles
- **Processing**: 4-channel BEV rasterization (height, intensity, density)
- **Resolution**: 512×512 pixels covering 100m×100m area
- **Output**: Semantic segmentation masks

### Training Strategy
- **Base Model**: U-Net with skip connections
- **Loss Function**: Combined Focal + Dice loss
- **Class Weighting**: 3× weight for pedestrian class
- **Optimization**: AdamW with learning rate scheduling

## 🎯 Performance Targets

### Primary Metrics
- **Pedestrian mIoU**: Target >70% (achieved 71.4%)
- **Overall mIoU**: Target >65% (achieved 68.2%)
- **False Negative Rate**: <20% (achieved 17.9%)

### Secondary Metrics
- **Precision**: >85% (achieved 88.3%)
- **Recall**: >80% (achieved 82.1%)
- **F1-Score**: >82% (achieved 85.1%)

### Computational Efficiency
- **Inference Time**: <30ms per frame
- **Memory Usage**: <4GB VRAM
- **Model Size**: <100M parameters

## 🛠️ Common Setup Issues

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvidia-smi
```

### Memory Issues
- Reduce batch size in configuration
- Use gradient accumulation for effective larger batches
- Enable mixed precision training (AMP)

### Dependencies
```bash
# If torch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If OpenCV issues
pip install opencv-python-headless
```

## 📞 Getting Help

### Documentation
1. **[Troubleshooting Guide](troubleshooting.md)** - Common issues and solutions
2. **[API Reference](api_reference/)** - Detailed code documentation
3. **[Configuration Guide](configuration.md)** - System configuration

### Community
1. **GitHub Issues**: Report bugs or ask questions
2. **Examples**: Check working code examples
3. **Notebooks**: Interactive learning materials

### Academic Resources
1. **Research Papers**: Original attention mechanism papers
2. **Dataset Documentation**: Woven by Toyota dataset details
3. **Related Work**: nuScenes, KITTI, and other AV datasets

## 🎉 Success Indicators

You'll know you're ready to proceed when:

✅ Configuration test passes  
✅ Dependencies install without errors  
✅ Example scripts run successfully  
✅ GPU is detected and functional  
✅ Small training run completes  

## 🚀 Ready to Go Deeper?

Choose your next step:

- **[Installation Guide](installation.md)** - Detailed setup instructions
- **[Data Preparation](data_preparation.md)** - Prepare your dataset
- **[Training Guide](training_guide.md)** - Start training models
- **[Examples](../examples/)** - Try hands-on examples

---

**Estimated Time to Complete**: 30-60 minutes  
**Difficulty Level**: Beginner to Intermediate  
**Prerequisites**: Basic Python, PyTorch familiarity 