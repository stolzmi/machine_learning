## MNIST CNN with Explainable AI (XAI)

A Flax-based Convolutional Neural Network for MNIST digit classification with integrated Explainable AI (XAI) techniques including **GradCAM** and **Saliency Maps**.

## Overview

This project implements a CNN using JAX and Flax to classify MNIST handwritten digits, with a strong focus on model interpretability through XAI patterns. The implementation allows you to visualize what the model "looks at" when making predictions on grayscale images.

### Features

- **Flax CNN Architecture**: Modern CNN built with JAX/Flax framework
- **GradCAM Visualization**: Gradient-weighted Class Activation Mapping for grayscale images
- **Saliency Maps**: Gradient-based visualization showing pixel-level importance
- **Smooth Saliency**: Noise-reduced saliency maps using SmoothGrad
- **Batch Normalization**: For stable and faster training
- **Comprehensive Training Pipeline**: Full training loop with validation
- **Visualization Suite**: Rich visualization tools optimized for grayscale images

## Project Structure

```
advanced_machine_learning/
├── mnist_cnn_model.py               # CNN model architecture
├── mnist_xai_visualizations.py      # GradCAM and Saliency Map implementations
├── train_mnist.py                   # Training script
├── demo_mnist_xai.py                # Visualization demo script
├── simple_mnist_xai_demo.py         # Quick demo (no training)
└── README_MNIST_XAI.md              # This file
```

## Installation

### Dependencies

Same as CIFAR-10 project - see `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Model Architecture

The CNN consists of:

- **3 Convolutional Blocks**:
  - Block 1: 2x Conv(32) + MaxPool → 28x28 → 14x14
  - Block 2: 2x Conv(64) + MaxPool → 14x14 → 7x7
  - Block 3: 2x Conv(128) (no pooling) → 7x7

- **Classification Head**:
  - Global Average Pooling
  - FC(256) + ReLU + Dropout
  - FC(10) for digit predictions

Each convolutional layer uses:
- Batch Normalization
- ReLU activation
- SAME padding

**Total parameters**: ~850K (lighter than CIFAR-10 model)
**Expected accuracy**: 98-99% on MNIST

## Usage

### 1. Quick Demo (No Training Required)

```bash
python simple_mnist_xai_demo.py
```

See XAI methods in action with a randomly initialized model (takes ~30 seconds).

### 2. Train the Model

```bash
python train_mnist.py
```

**Training Parameters**:
- Epochs: 20
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam

**Output**:
- `mnist_model.pkl`: Trained model checkpoint
- `mnist_training_history.png`: Training/validation curves

**Expected Performance**:
- Training accuracy: ~99%
- Validation accuracy: ~98-99%
- Test accuracy: ~98-99%
- Training time: 5-10 minutes on GPU, 20-30 minutes on CPU

### 3. Run XAI Visualizations

```bash
python demo_mnist_xai.py
```

This script will:
1. Load the trained model
2. Generate detailed XAI visualizations for individual digits
3. Create batch visualizations

**Output Directory**: `mnist_xai_results/`

### 4. Custom XAI Analysis

```python
from mnist_cnn_model import MNISTCNN
from mnist_xai_visualizations import GradCAM, SaliencyMap
import pickle
import jax.numpy as jnp

# Load model
with open('mnist_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
params = checkpoint['params']
batch_stats = checkpoint['batch_stats']
model = MNISTCNN(num_classes=10)

# Load your image (28x28x1, normalized to [0, 1])
image = your_image  # shape: (28, 28, 1) or (28, 28)

# GradCAM
gradcam = GradCAM(model, target_layer='conv3')
heatmap, pred_class, score = gradcam.compute_gradcam(
    params, batch_stats, image
)
overlay = gradcam.visualize(image, heatmap)

# Saliency Map
saliency = SaliencyMap(model)
saliency_map, pred_class, score = saliency.compute_saliency(
    params, batch_stats, image
)

# Smooth Saliency
smooth_sal, _, _ = saliency.compute_smooth_saliency(
    params, batch_stats, image, n_samples=50
)
```

## Key Differences from CIFAR-10

### 1. Grayscale Images
- MNIST: 28x28x1 (grayscale)
- CIFAR-10: 32x32x3 (RGB)

### 2. Visualization Adaptations
- GradCAM overlays convert grayscale to RGB for heatmap overlay
- Saliency maps use single channel (no max across channels needed)
- Cleaner visualizations due to simpler images

### 3. Model Size
- MNIST model is smaller (~850K parameters vs ~2.5M)
- Fewer channels: 32→64→128 (vs 64→128→256)
- Easier problem, so simpler architecture works well

### 4. Training Time
- MNIST trains much faster (5-10 min GPU vs 30-60 min)
- Higher accuracy achievable (98-99% vs 75-80%)

## XAI Techniques Explained

### 1. GradCAM for Grayscale Images

**How it works**:
- Same principle as CIFAR-10, but adapted for single-channel images
- Produces heatmaps showing important regions for digit recognition
- Typically highlights the strokes and curves of the digit

**What to expect**:
- For digit "1": Highlights vertical stroke
- For digit "8": Highlights both loops
- For digit "7": Highlights horizontal bar and diagonal

### 2. Saliency Maps for MNIST

**How it works**:
- Computes gradient of class score w.r.t. input pixels
- For grayscale, directly uses single channel gradient
- Shows fine-grained pixel importance

**What to expect**:
- Highlights edges and boundaries of digits
- Shows critical strokes (e.g., curves, intersections)
- More detailed than GradCAM

### 3. Interpreting Results

**Good predictions**:
- GradCAM highlights the digit shape
- Saliency highlights important strokes
- Both focus on relevant features

**Misclassifications**:
- Often occur on ambiguous digits (e.g., 4 vs 9, 3 vs 8)
- XAI shows what confused the model
- Helpful for understanding failure modes

## Visualization Examples

The demo script generates:

1. **Detailed Single Digit Analysis**:
   - Original grayscale image
   - GradCAM overlay (colored heatmap on grayscale)
   - Saliency map overlay
   - Smooth saliency overlay
   - Individual heatmaps with colorbars
   - Top-3 predictions

2. **Batch Visualization**:
   - Grid view of multiple digits
   - Side-by-side original, GradCAM, and heatmap
   - Prediction accuracy indicators

## Performance Tips

1. **CPU Training**: MNIST trains reasonably fast even on CPU
2. **Batch Size**: Can reduce to 64 if memory constrained
3. **Fewer Epochs**: 10 epochs often enough for 98%+ accuracy
4. **Quick Experiments**: Use simple_mnist_xai_demo.py for fast iteration

## Comparison with CIFAR-10

| Aspect | MNIST | CIFAR-10 |
|--------|-------|----------|
| **Images** | 28x28x1 grayscale | 32x32x3 RGB |
| **Classes** | 10 digits | 10 objects |
| **Difficulty** | Easy | Moderate |
| **Accuracy** | 98-99% | 75-80% |
| **Train Time** | 5-10 min GPU | 30-60 min GPU |
| **Parameters** | ~850K | ~2.5M |
| **Epochs** | 20 | 50 |
| **XAI Clarity** | Very clear | Moderate |

## Common Patterns in MNIST XAI

### GradCAM Patterns

- **Digit 0**: Highlights the circular loop
- **Digit 1**: Highlights vertical stroke
- **Digit 2**: Highlights curve and horizontal base
- **Digit 3**: Highlights curves, especially top and bottom
- **Digit 4**: Highlights intersection of vertical and horizontal
- **Digit 5**: Highlights top horizontal and bottom curve
- **Digit 6**: Highlights loop and top curve
- **Digit 7**: Highlights horizontal bar and diagonal
- **Digit 8**: Highlights both loops equally
- **Digit 9**: Highlights top loop and stem

### Saliency Patterns

- **Edges and boundaries** are always highlighted
- **Intersections** and **corners** show high saliency
- **Smooth regions** show low saliency
- **Pen strokes** are clearly visible

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size to 64 or 32

### Issue: Low Accuracy
**Possible causes**:
- Too few epochs (train for at least 10)
- Data not normalized properly (should be in [0, 1])
- Check learning rate

### Issue: Visualizations Look Random
**Solution**: Make sure you've trained the model first! Use `train_mnist.py` before `demo_mnist_xai.py`

## Extensions and Experiments

Try these experiments:

1. **Different Architectures**:
   - Add more convolutional blocks
   - Try different channel sizes
   - Experiment with residual connections

2. **Data Augmentation**:
   - Random rotations (small angles)
   - Random translations
   - Elastic deformations

3. **Additional XAI Methods**:
   - Integrated Gradients
   - Attention maps
   - Layer-wise relevance propagation

4. **Adversarial Examples**:
   - Generate adversarial digits
   - Visualize what fools the model
   - Study robustness

5. **Confusion Analysis**:
   - Find commonly confused pairs (e.g., 4 vs 9)
   - Analyze with XAI why they're confused
   - Improve model based on insights

## Educational Value

MNIST is perfect for learning XAI because:

1. **Simple and Fast**: Train in minutes, experiment rapidly
2. **Clear Visualizations**: Grayscale images make XAI easier to interpret
3. **High Accuracy**: Model works well, so you see meaningful patterns
4. **Well-Understood**: Lots of literature and examples to compare with
5. **Foundation**: Skills transfer to more complex problems

## Use Cases

1. **Learning XAI**: Start here before tackling CIFAR-10
2. **Teaching**: Great for demonstrating concepts in class
3. **Research**: Baseline for new XAI techniques
4. **Prototyping**: Quick experiments with new ideas
5. **Debugging**: Understanding neural network behavior

## References

### Papers
1. **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. **SmoothGrad**: Smilkov et al., "SmoothGrad: removing noise by adding noise", arXiv 2017
3. **MNIST**: LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998

### Resources
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## Quick Start Summary

```bash
# 1. Quick demo (30 seconds)
python simple_mnist_xai_demo.py

# 2. Train model (5-10 minutes on GPU)
python train_mnist.py

# 3. Generate XAI visualizations
python demo_mnist_xai.py

# 4. View results
# Check mnist_xai_results/ folder
```

## Expected Results

### After Training
- **Validation accuracy**: 98-99%
- **Test accuracy**: 98-99%
- **Training curves**: Smooth convergence in ~20 epochs

### XAI Visualizations
- **Clear digit outlines** in GradCAM
- **Sharp stroke highlights** in saliency maps
- **Focused attention** on relevant features
- **Meaningful explanations** for predictions

## Conclusion

MNIST with XAI is an excellent starting point for:
- Learning explainable AI techniques
- Understanding CNN behavior
- Prototyping new XAI methods
- Teaching machine learning concepts

The simpler nature of MNIST (grayscale, high accuracy) makes it ideal for understanding XAI before moving to more complex datasets like CIFAR-10 or ImageNet.

**Happy Explaining! 🔍**

---

**Project Status**: Complete and tested ✓
**Difficulty**: Beginner-friendly
**Time to Complete**: 30 minutes to full understanding
