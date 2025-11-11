# CIFAR-10 CNN with Explainable AI (XAI)

A Flax-based Convolutional Neural Network for CIFAR-10 classification with integrated Explainable AI (XAI) techniques including **GradCAM** and **Saliency Maps**.

## Overview

This project implements a CNN using JAX and Flax to classify CIFAR-10 images, with a strong focus on model interpretability through XAI patterns. The implementation allows you to visualize what the model "looks at" when making predictions.

### Features

- **Flax CNN Architecture**: Modern CNN built with JAX/Flax framework
- **GradCAM Visualization**: Gradient-weighted Class Activation Mapping to highlight important image regions
- **Saliency Maps**: Gradient-based visualization showing pixel-level importance
- **Smooth Saliency**: Noise-reduced saliency maps using SmoothGrad technique
- **Batch Normalization**: For stable and faster training
- **Comprehensive Training Pipeline**: Full training loop with validation
- **Visualization Suite**: Rich visualization tools for model interpretation

## Project Structure

```
advanced_machine_learning/
├── cifar10_cnn_model.py      # CNN model architecture
├── xai_visualizations.py     # GradCAM and Saliency Map implementations
├── train_cifar10.py          # Training script
├── demo_xai.py               # Visualization demo script
├── requirements.txt          # Python dependencies
└── README_CIFAR10_XAI.md    # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows/CPU users**: If you're using CPU-only JAX, install with:
```bash
pip install jax[cpu]
```

**Note for GPU users**: For CUDA support, follow the [JAX GPU installation guide](https://github.com/google/jax#installation).

### 2. Verify Installation

```python
import jax
import flax
print(f"JAX version: {jax.__version__}")
print(f"Flax version: {flax.__version__}")
```

## Model Architecture

The CNN consists of:

- **3 Convolutional Blocks**:
  - Block 1: 2x Conv(64) + MaxPool → 32x32 → 16x16
  - Block 2: 2x Conv(128) + MaxPool → 16x16 → 8x8
  - Block 3: 3x Conv(256) + MaxPool → 8x8 → 4x4

- **Classification Head**:
  - Global Average Pooling
  - FC(512) + ReLU + Dropout
  - FC(10) for class predictions

Each convolutional layer uses:
- Batch Normalization
- ReLU activation
- SAME padding

The architecture is designed to capture hierarchical features while maintaining interpretability through intermediate activation access.

## Usage

### 1. Train the Model

```bash
python train_cifar10.py
```

**Training Parameters**:
- Epochs: 50
- Batch size: 128
- Learning rate: 0.001
- Optimizer: Adam
- Train/Val split: 90/10

**Output**:
- `cifar10_model.pkl`: Trained model checkpoint
- `training_history.png`: Training/validation curves

**Expected Performance**:
- Training accuracy: ~85-90%
- Validation accuracy: ~75-80%
- Test accuracy: ~75-80%

### 2. Run XAI Visualizations

```bash
python demo_xai.py
```

This script will:
1. Load the trained model
2. Generate detailed XAI visualizations for individual images
3. Create batch visualizations
4. Analyze and visualize misclassified examples

**Output Directory**: `xai_results/`

### 3. Custom XAI Analysis

```python
from cifar10_cnn_model import CIFAR10CNN
from xai_visualizations import GradCAM, SaliencyMap
import pickle
import jax.numpy as jnp

# Load model
with open('cifar10_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
params = checkpoint['params']
batch_stats = checkpoint['batch_stats']
model = CIFAR10CNN(num_classes=10)

# Load your image (32x32x3, normalized to [0, 1])
image = your_image  # shape: (32, 32, 3)

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

## XAI Techniques Explained

### 1. GradCAM (Gradient-weighted Class Activation Mapping)

**How it works**:
- Computes gradients of the target class score with respect to the final convolutional layer
- Weights each activation map by the gradient importance
- Produces a coarse localization map highlighting important regions

**Advantages**:
- Class-discriminative (shows regions specific to the predicted class)
- Resolution-independent
- Works with any CNN architecture

**Use case**: Understanding which spatial regions influenced the prediction

### 2. Saliency Maps

**How it works**:
- Computes gradient of class score with respect to input pixels
- Shows which pixels, if changed, would most affect the prediction
- Pixel-level importance visualization

**Advantages**:
- High resolution (pixel-level)
- Fast to compute
- Shows fine-grained details

**Use case**: Understanding pixel-level feature importance

### 3. Smooth Saliency (SmoothGrad)

**How it works**:
- Averages saliency maps computed over multiple noisy versions of the input
- Reduces noise and visual artifacts
- More stable and interpretable visualizations

**Advantages**:
- Cleaner visualizations
- More robust to noise
- Better highlights actual important features

**Use case**: Production-quality explanations with reduced noise

## Visualization Examples

The demo script generates several types of visualizations:

1. **Detailed Single Image Analysis**:
   - Original image
   - GradCAM overlay
   - Saliency map overlay
   - Smooth saliency overlay
   - Individual heatmaps with colorbars
   - Class-specific GradCAM (for different target classes)
   - Top-3 predictions with probabilities

2. **Batch Visualization**:
   - Grid view of multiple images
   - Side-by-side original, GradCAM, and heatmap
   - Prediction accuracy indicators

3. **Misclassification Analysis**:
   - Focused analysis on incorrect predictions
   - Understanding why the model failed
   - Comparing predicted vs. true class visualizations

## Understanding the Results

### Interpreting GradCAM

- **Red/Yellow regions**: High importance for the prediction
- **Blue/Purple regions**: Low importance
- **Focus areas**: Where the model is "looking" to make its decision

**Good predictions**: Heatmap highlights semantically relevant regions (e.g., animal's head/body)

**Poor predictions**: Heatmap may highlight background or irrelevant features

### Interpreting Saliency Maps

- **Bright pixels**: High gradient magnitude (important for prediction)
- **Dark pixels**: Low gradient magnitude (less important)
- **Edge highlighting**: Often shows object boundaries and distinctive features

### Common Patterns

1. **Correct classifications**: Usually show focused attention on the main object
2. **Misclassifications**: Often show:
   - Attention to background
   - Focus on ambiguous features
   - Confusion between similar classes (e.g., cat vs. dog)

## CIFAR-10 Classes

The model classifies images into 10 categories:

0. Airplane
1. Automobile
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Technical Details

### JAX/Flax Advantages

- **JIT Compilation**: Fast execution through XLA
- **Automatic Differentiation**: Easy gradient computation for XAI
- **Functional Programming**: Explicit parameter handling
- **Scalability**: Easy to scale to TPU/GPU

### XAI Implementation Notes

- GradCAM targets the last convolutional layer (`conv3`) by default
- Activations are captured via `return_activations=True` flag
- All gradient computations use JAX's automatic differentiation
- Visualizations use bilinear interpolation for upsampling

## Performance Tips

1. **GPU Acceleration**: Use GPU for faster training
2. **Batch Size**: Adjust based on available memory
3. **Data Augmentation**: Can improve model robustness (not implemented, but recommended)
4. **Learning Rate Scheduling**: Could improve convergence
5. **Model Size**: Increase channels for better accuracy (at cost of speed)

## Troubleshooting

### Common Issues

**Issue**: `Out of memory` during training
- **Solution**: Reduce batch size in [train_cifar10.py](train_cifar10.py:224)

**Issue**: JAX not using GPU
- **Solution**: Check CUDA installation, reinstall jaxlib with GPU support

**Issue**: TensorFlow datasets not downloading
- **Solution**: Check internet connection, set `TFDS_DATA_DIR` environment variable

**Issue**: Slow XAI visualization
- **Solution**: Reduce `n_samples` in smooth saliency computation

## Extensions and Future Work

Potential improvements:

1. **Data Augmentation**: Random crops, flips, color jittering
2. **Advanced Architectures**: ResNet, EfficientNet, Vision Transformer
3. **Additional XAI Methods**:
   - Integrated Gradients
   - Layer-wise Relevance Propagation (LRP)
   - LIME (Local Interpretable Model-agnostic Explanations)
4. **Interactive Visualization**: Web-based interface with Streamlit
5. **Model Compression**: Quantization, pruning for deployment
6. **Adversarial Analysis**: Robustness testing with adversarial examples

## References

### Papers

1. **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
2. **Saliency Maps**: Simonyan et al., "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014
3. **SmoothGrad**: Smilkov et al., "SmoothGrad: removing noise by adding noise", arXiv 2017

### Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Explainable AI Book](https://christophm.github.io/interpretable-ml-book/)

## License

This project is for educational purposes as part of the Advanced Machine Learning course.

## Author

Created for the Advanced Machine Learning course at University.

---

**Happy Explaining!** If you have questions or improvements, feel free to experiment with the code.
