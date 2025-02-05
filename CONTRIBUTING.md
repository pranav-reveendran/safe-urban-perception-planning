# Contributing to Attention-Enhanced U-Net

Thank you for your interest in contributing to the Attention-Enhanced U-Net for Pedestrian Segmentation project! 🚗🤖

This document provides guidelines for contributing to this autonomous vehicle perception research project.

## 🎯 Project Overview

This project focuses on improving pedestrian segmentation in autonomous vehicles using attention-enhanced U-Net architectures. We achieved a **28.9% improvement** in pedestrian mIoU over baseline models, and we welcome contributions to further advance the state-of-the-art.

## 🚀 Ways to Contribute

### 1. 🐛 Bug Reports
- Report bugs or unexpected behavior
- Provide reproduction steps and environment details
- Include relevant error messages and logs

### 2. 💡 Feature Requests
- Suggest new attention mechanisms or architectures
- Propose evaluation metrics or datasets
- Request documentation improvements

### 3. 🔬 Research Contributions
- New attention mechanism implementations
- Performance optimizations
- Ablation studies and analysis
- Multi-modal fusion approaches

### 4. 📚 Documentation
- Improve existing documentation
- Add tutorials and examples
- Translate documentation
- Fix typos and clarify explanations

### 5. 🧪 Testing
- Add unit tests for models and utilities
- Improve test coverage
- Add integration tests
- Performance benchmarking

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git for version control

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/safe-urban-perception-planning.git
cd safe-urban-perception-planning

# Add the original repository as upstream
git remote add upstream https://github.com/pranav-reveendran/safe-urban-perception-planning.git
```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
pip install pytest black flake8 mypy pre-commit
```

### 3. Verify Setup
```bash
# Test configuration
python configs/base_config.py

# Run basic tests (if available)
pytest tests/ -v

# Check code style
black --check .
flake8 .
```

## 📝 Contribution Guidelines

### Code Style

#### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use `black` for code formatting: `black .`
- Maximum line length: 100 characters
- Use type hints for function signatures

#### Example:
```python
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """Train the model and return metrics.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        epochs: Number of training epochs
        device: Device to use for training
        
    Returns:
        Dictionary containing training metrics
    """
    # Implementation here
    pass
```

#### Documentation Style
- Use Google-style docstrings for all functions and classes
- Include comprehensive examples in docstrings
- Document all parameters and return values
- Add type hints for better IDE support

#### Model Implementation
- Inherit from `torch.nn.Module` for all models
- Implement `forward()` method with clear input/output shapes
- Include model parameter count and FLOPs calculation
- Add comprehensive unit tests

### Commit Guidelines

#### Commit Message Format
```
<type>(<scope>): <description>

<body>

<footer>
```

#### Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

#### Examples
```bash
feat(attention): add CBAM attention mechanism

Implement Convolutional Block Attention Module (CBAM) for channel 
and spatial attention. Includes both channel and spatial attention 
components with configurable reduction ratios.

- Add CBAM class in models/attention.py
- Update AttentionUNet to support CBAM
- Add unit tests for CBAM functionality
- Update configuration to include CBAM options

Closes #123
```

```bash
fix(data): resolve BEV preprocessing memory leak

Fix memory accumulation in LidarBEVProcessor when processing 
large batches. The issue was caused by tensors not being 
properly moved to CPU after GPU processing.

- Add explicit .cpu() calls after GPU processing
- Implement proper tensor cleanup in __del__
- Add memory monitoring in unit tests

Fixes #456
```

### Pull Request Process

#### Before Submitting
1. **Create an Issue**: Discuss major changes in an issue first
2. **Branch Naming**: Use descriptive branch names
   - `feature/attention-transformer`
   - `fix/memory-leak-preprocessing`
   - `docs/improve-readme`
3. **Test Locally**: Ensure all tests pass
4. **Update Documentation**: Update relevant docs

#### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement (include benchmark)
- [ ] Performance regression (justified and documented)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

#### Review Process
1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Comprehensive testing for new features
4. **Documentation**: Ensure docs are updated

## 🧪 Testing Guidelines

### Unit Tests
```python
import pytest
import torch
from models.attention import AttentionGate

class TestAttentionGate:
    def test_attention_gate_output_shape(self):
        """Test that attention gate outputs correct shape."""
        attention = AttentionGate(in_channels=256, gate_channels=128)
        x = torch.randn(2, 256, 32, 32)
        g = torch.randn(2, 128, 16, 16)
        
        output, attention_weights = attention(x, g)
        
        assert output.shape == x.shape
        assert attention_weights.shape == (2, 1, 32, 32)
    
    def test_attention_gate_gradient_flow(self):
        """Test that gradients flow through attention gate."""
        attention = AttentionGate(in_channels=256, gate_channels=128)
        x = torch.randn(2, 256, 32, 32, requires_grad=True)
        g = torch.randn(2, 128, 16, 16, requires_grad=True)
        
        output, _ = attention(x, g)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert g.grad is not None
```

### Integration Tests
```python
def test_full_training_pipeline():
    """Test complete training pipeline with small model."""
    config = Config()
    config.training.epochs = 2
    config.training.batch_size = 2
    
    # Create small dataset
    dataset = create_test_dataset(num_samples=10)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Create model
    model = AttentionUNet(in_channels=4, num_classes=5)
    
    # Train for 2 epochs
    trainer = Trainer(model, config)
    metrics = trainer.train(dataloader, dataloader)
    
    # Verify training completed
    assert 'loss' in metrics
    assert metrics['loss'] > 0
```

### Performance Tests
```python
def test_inference_speed():
    """Test that inference meets real-time requirements."""
    model = AttentionUNet(in_channels=4, num_classes=5)
    model.eval()
    
    input_tensor = torch.randn(1, 4, 512, 512)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 100
    assert avg_inference_time < 0.05  # 50ms threshold
```

## 🔬 Research Contributions

### Adding New Attention Mechanisms

1. **Implementation Location**: `models/attention.py`
2. **Base Class**: Inherit from `torch.nn.Module`
3. **Required Methods**: `forward()` method
4. **Documentation**: Include mathematical formulation
5. **Testing**: Add comprehensive unit tests

#### Example Template:
```python
class NewAttentionMechanism(nn.Module):
    """New attention mechanism for feature enhancement.
    
    Mathematical formulation:
    attention_weights = softmax(Q @ K^T / sqrt(d_k))
    output = attention_weights @ V
    
    Args:
        in_channels: Number of input channels
        attention_dim: Attention dimension
        
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
    """
    
    def __init__(self, in_channels: int, attention_dim: int = 64):
        super().__init__()
        # Implementation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of attention mechanism."""
        # Implementation
        return output
```

### Adding New Datasets

1. **Dataset Class**: Create in `utils/data_loading.py`
2. **Preprocessing**: Add to `utils/preprocessing.py`
3. **Configuration**: Update `configs/base_config.py`
4. **Documentation**: Add dataset description and citation

### Adding New Evaluation Metrics

1. **Metrics Class**: Extend `utils/metrics.py`
2. **Integration**: Update evaluation pipeline
3. **Documentation**: Describe metric purpose and interpretation
4. **Testing**: Add unit tests with known ground truth

## 📊 Performance Benchmarking

### Adding Benchmarks
```python
def benchmark_attention_mechanisms():
    """Benchmark different attention mechanisms."""
    mechanisms = [
        ('attention_gates', AttentionGate),
        ('self_attention', SelfAttention),
        ('cbam', CBAM)
    ]
    
    results = {}
    for name, mechanism_class in mechanisms:
        # Benchmark implementation
        results[name] = benchmark_single_mechanism(mechanism_class)
    
    return results
```

### Performance Requirements
- **Inference Speed**: Must maintain >30 FPS for real-time deployment
- **Memory Usage**: Should not exceed 4GB VRAM for training
- **Accuracy**: New contributions should maintain or improve mIoU
- **Model Size**: Consider deployment constraints (<500MB)

## 📚 Documentation Standards

### Code Documentation
- **Docstrings**: Use Google-style for all public functions
- **Type Hints**: Include for all function parameters and returns
- **Examples**: Provide usage examples in docstrings
- **Mathematical Notation**: Use LaTeX for mathematical formulations

### Research Documentation
- **Methodology**: Clearly explain the approach and motivation
- **Experiments**: Document experimental setup and hyperparameters
- **Results**: Include quantitative results and statistical significance
- **Comparison**: Compare with existing methods

### User Documentation
- **Getting Started**: Clear setup and installation instructions
- **Tutorials**: Step-by-step guides with code examples
- **API Reference**: Complete function and class documentation
- **Troubleshooting**: Common issues and solutions

## 🔄 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks completed
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Git tag created

## 🌟 Recognition

### Contributor Types
- **Code Contributors**: Listed in README.md
- **Research Contributors**: Co-authorship opportunities for significant contributions
- **Documentation Contributors**: Recognition in documentation credits
- **Community Contributors**: Special mention for helping others

### Hall of Fame
We maintain a contributor hall of fame for significant contributions:
- **Major Feature Contributors**: New attention mechanisms, architectures
- **Performance Optimizers**: Significant speed/memory improvements
- **Research Pioneers**: Novel insights and methodological advances

## 📞 Getting Help

### Communication Channels
1. **GitHub Issues**: Bug reports and feature requests
2. **GitHub Discussions**: General questions and community discussions
3. **Pull Request Reviews**: Code-specific discussions

### Maintainers
- **Primary Maintainer**: San Jose State University AI/ML Graduate Student
- **Research Advisor**: Faculty supervisor (when applicable)
- **Community Moderators**: Active contributors with commit access

### Response Times
- **Bug Reports**: 24-48 hours for acknowledgment
- **Feature Requests**: 1 week for initial feedback
- **Pull Requests**: 1-2 weeks for review
- **Security Issues**: 24 hours for response

## 🛡️ Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior
- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide helpful and constructive feedback
- **Be Patient**: Help newcomers learn and grow
- **Be Professional**: Maintain professional communication

### Unacceptable Behavior
- Harassment or discrimination of any kind
- Inappropriate or offensive language
- Personal attacks or inflammatory comments
- Sharing private information without permission

### Enforcement
Violations of the code of conduct will result in warnings, temporary bans, or permanent bans depending on severity.

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## 🎉 Thank You!

Thank you for contributing to advancing autonomous vehicle safety through improved pedestrian segmentation! Your contributions help make our roads safer for everyone. 🚗👥

---

**Questions?** Open an issue or start a discussion. We're here to help! 💬

**Latest Update**: February 2025  
**Maintainer**: pranav-reveendran  
**Project Status**: Active Development ✅ 