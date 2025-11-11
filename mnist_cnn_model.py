"""
Flax CNN Model for MNIST with XAI capabilities
This model is designed to support GradCAM and Saliency Map visualizations
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Callable, Optional


class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BatchNorm -> Activation"""
    features: int
    kernel_size: Sequence[int] = (3, 3)
    strides: Sequence[int] = (1, 1)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(features=self.features,
                   kernel_size=self.kernel_size,
                   strides=self.strides,
                   padding='SAME',
                   use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = self.activation(x)
        return x


class MNISTCNN(nn.Module):
    """
    CNN for MNIST classification with intermediate activation access for XAI

    Architecture:
    - 3 Convolutional blocks with increasing channels (32, 64, 128)
    - Max pooling after first two blocks
    - Global average pooling
    - Fully connected layer for classification

    Simpler than CIFAR-10 model since MNIST is easier
    """
    num_classes: int = 10
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = True, return_activations: bool = False):
        """
        Forward pass with optional activation return for XAI

        Args:
            x: Input images [batch, height, width, channels]
            training: Whether in training mode
            return_activations: If True, returns intermediate activations for XAI

        Returns:
            logits: Class predictions [batch, num_classes]
            activations: (optional) Dictionary of intermediate activations
        """
        activations = {}

        # Block 1: 28x28 -> 14x14
        x = ConvBlock(features=32, name='conv1_1')(x, training)
        x = ConvBlock(features=32, name='conv1_2')(x, training)
        activations['conv1'] = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2: 14x14 -> 7x7
        x = ConvBlock(features=64, name='conv2_1')(x, training)
        x = ConvBlock(features=64, name='conv2_2')(x, training)
        activations['conv2'] = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3: 7x7 (no pooling)
        x = ConvBlock(features=128, name='conv3_1')(x, training)
        x = ConvBlock(features=128, name='conv3_2')(x, training)
        activations['conv3'] = x  # This is the target layer for GradCAM

        # Classification head
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        activations['gap'] = x

        x = nn.Dense(features=256, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        activations['fc1'] = x

        logits = nn.Dense(features=self.num_classes, name='fc_out')(x)

        if return_activations:
            return logits, activations
        return logits


def create_model(num_classes: int = 10):
    """Create a MNISTCNN model instance"""
    return MNISTCNN(num_classes=num_classes)


def initialize_model(rng_key, input_shape=(1, 28, 28, 1), num_classes=10):
    """
    Initialize model parameters

    Args:
        rng_key: JAX random key
        input_shape: Shape of input tensor (MNIST is 28x28x1)
        num_classes: Number of output classes

    Returns:
        params: Model parameters
        batch_stats: Batch statistics
        model: Model instance
    """
    model = create_model(num_classes=num_classes)
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    variables = model.init(rng_key, dummy_input, training=False)
    return variables['params'], variables.get('batch_stats'), model
