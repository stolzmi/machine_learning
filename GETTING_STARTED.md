# Getting Started with CIFAR-10 XAI

## What Was Fixed

The initial training script had two issues that have been resolved:

1. **Data Pipeline Issue**: The normalize function was trying to use `jnp.array()` (JAX) inside TensorFlow's data pipeline, which isn't supported. Fixed by using TensorFlow operations (`tf.cast()`) in the pipeline.

2. **Dropout RNG Issue**: The Dropout layer requires a random number generator (RNG) key, which wasn't being provided. Fixed by:
   - Adding `dropout_rng` to the `TrainState` class
   - Splitting and updating the RNG key at each training step
   - Passing the RNG to the model via `rngs={'dropout': dropout_rng}`

## Quick Start

### Option 1: Simple Demo (No Training Required)

See how XAI works with a randomly initialized model:

```bash
python simple_xai_demo.py
```

This will:
- Load a CIFAR-10 image
- Compute GradCAM and Saliency Maps
- Show visualizations

**Note**: Since the model isn't trained, visualizations will show random patterns. This is just to demonstrate the XAI methods work!

### Option 2: Quick Test (2 epochs)

Test that everything works with minimal training:

```bash
python quick_test.py
```

This trains for just 2 epochs to verify the setup is correct.

### Option 3: Full Training

Train the complete model (takes time!):

```bash
python train_cifar10.py
```

**Expected time**:
- GPU: 30-60 minutes
- CPU: 2-4 hours

**Output**:
- `cifar10_model.pkl` - Trained model
- `training_history.png` - Training curves

**Expected accuracy**: 75-80% on test set

### Option 4: XAI Visualization Demo

After training, run the comprehensive XAI demo:

```bash
python demo_xai.py
```

This will generate extensive visualizations in the `xai_results/` folder showing:
- GradCAM overlays
- Saliency maps
- Smooth saliency
- Misclassification analysis

## Installation

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### For CPU-only JAX (Windows):

```bash
pip install jax[cpu]
```

### For GPU JAX:

Follow the [JAX GPU installation guide](https://github.com/google/jax#installation).

## Files Overview

### Core Implementation

- **`cifar10_cnn_model.py`** - CNN architecture with XAI support
- **`xai_visualizations.py`** - GradCAM and Saliency Map implementations
- **`train_cifar10.py`** - Full training pipeline
- **`demo_xai.py`** - Comprehensive XAI visualization demo

### Quick Start Scripts

- **`simple_xai_demo.py`** - Demo with untrained model (fast!)
- **`quick_test.py`** - 2-epoch training test (moderate)

### Documentation

- **`README_CIFAR10_XAI.md`** - Complete documentation
- **`GETTING_STARTED.md`** - This file

## Understanding the Code

### Model Architecture

The CNN has 3 convolutional blocks with increasing channels (64→128→256):

```python
# Block structure
Conv(64) → BatchNorm → ReLU → Conv(64) → BatchNorm → ReLU → MaxPool
Conv(128) → BatchNorm → ReLU → Conv(128) → BatchNorm → ReLU → MaxPool
Conv(256) → BatchNorm → ReLU → Conv(256) → BatchNorm → ReLU → Conv(256) → MaxPool
GlobalAvgPool → FC(512) → Dropout → FC(10)
```

### XAI Methods

#### GradCAM (Gradient-weighted Class Activation Mapping)

Shows **WHERE** the model is looking:

```python
from xai_visualizations import GradCAM

gradcam = GradCAM(model, target_layer='conv3')
heatmap, pred_class, score = gradcam.compute_gradcam(
    params, batch_stats, image
)
```

#### Saliency Maps

Shows **WHICH PIXELS** matter:

```python
from xai_visualizations import SaliencyMap

saliency = SaliencyMap(model)
saliency_map, pred_class, score = saliency.compute_saliency(
    params, batch_stats, image
)
```

#### Smooth Saliency

Cleaner version using SmoothGrad:

```python
smooth_saliency, _, _ = saliency.compute_smooth_saliency(
    params, batch_stats, image, n_samples=50
)
```

## Custom Usage Example

```python
import jax.numpy as jnp
import pickle
from cifar10_cnn_model import CIFAR10CNN
from xai_visualizations import GradCAM, SaliencyMap

# Load trained model
with open('cifar10_model.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

params = checkpoint['params']
batch_stats = checkpoint['batch_stats']
model = CIFAR10CNN(num_classes=10)

# Your image (32x32x3, normalized to [0,1])
# image = load_your_image()

# Compute XAI
gradcam = GradCAM(model)
heatmap, pred, score = gradcam.compute_gradcam(params, batch_stats, image)

# Visualize
import matplotlib.pyplot as plt
overlay = gradcam.visualize(image, heatmap)
plt.imshow(overlay)
plt.show()
```

## Troubleshooting

### Out of Memory

Reduce batch size in [train_cifar10.py](train_cifar10.py:256):

```python
train_model(batch_size=64)  # instead of 128
```

### TensorFlow Dataset Download Fails

Set data directory:

```bash
export TFDS_DATA_DIR=/path/to/data  # Linux/Mac
set TFDS_DATA_DIR=C:\path\to\data   # Windows
```

### Slow Training

- Use GPU if available
- Reduce number of epochs
- Use smaller batch size
- Train with `quick_test.py` first

### Visualizations Look Random

This is normal if the model isn't trained! XAI methods need a trained model to produce meaningful explanations.

## Next Steps

1. **Run `simple_xai_demo.py`** to verify installation
2. **Run `quick_test.py`** to verify training works
3. **Run `train_cifar10.py`** for full training (optional: reduce epochs)
4. **Run `demo_xai.py`** to see XAI in action
5. **Experiment** with your own images!

## Key Concepts

### Why XAI Matters

- **Trust**: Understand model decisions
- **Debugging**: Find what the model learned
- **Compliance**: Explain predictions to stakeholders
- **Science**: Discover patterns in data

### GradCAM vs Saliency

| GradCAM | Saliency |
|---------|----------|
| Coarse (spatial regions) | Fine (pixel-level) |
| Class-discriminative | Input-gradient based |
| Conv layer specific | Works on input |
| Better for objects | Better for details |

### Best Practices

1. Always visualize multiple examples
2. Compare correct vs incorrect predictions
3. Check different target classes
4. Use smooth saliency for cleaner results
5. Combine multiple XAI methods

## Resources

- Full documentation: [README_CIFAR10_XAI.md](README_CIFAR10_XAI.md)
- JAX docs: https://jax.readthedocs.io/
- Flax docs: https://flax.readthedocs.io/
- GradCAM paper: https://arxiv.org/abs/1610.02391

## Questions?

Check the full README for detailed explanations of:
- Architecture design
- XAI theory
- Implementation details
- Advanced usage

Happy explaining! 🔍
