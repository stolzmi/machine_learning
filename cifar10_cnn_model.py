"""
Flax CNN Model for CIFAR-10 with XAI capabilities
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


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification with intermediate activation access for XAI

    Architecture:
    - 3 Convolutional blocks with increasing channels (64, 128, 256)
    - Max pooling after each block
    - Global average pooling
    - Fully connected layer for classification
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

        # Block 1: 32x32 -> 16x16
        x = ConvBlock(features=64, name='conv1_1')(x, training)
        x = ConvBlock(features=64, name='conv1_2')(x, training)
        activations['conv1'] = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2: 16x16 -> 8x8
        x = ConvBlock(features=128, name='conv2_1')(x, training)
        x = ConvBlock(features=128, name='conv2_2')(x, training)
        activations['conv2'] = x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3: 8x8 -> 4x4
        x = ConvBlock(features=256, name='conv3_1')(x, training)
        x = ConvBlock(features=256, name='conv3_2')(x, training)
        x = ConvBlock(features=256, name='conv3_3')(x, training)
        activations['conv3'] = x  # This is the target layer for GradCAM
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Classification head
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        activations['gap'] = x

        x = nn.Dense(features=512, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        activations['fc1'] = x

        logits = nn.Dense(features=self.num_classes, name='fc_out')(x)

        if return_activations:
            return logits, activations
        return logits


def create_model(num_classes: int = 10):
    """Create a CIFAR10CNN model instance"""
    return CIFAR10CNN(num_classes=num_classes)


def initialize_model(rng_key, input_shape=(1, 32, 32, 3), num_classes=10):
    """
    Initialize model parameters

    Args:
        rng_key: JAX random key
        input_shape: Shape of input tensor
        num_classes: Number of output classes

    Returns:
        params: Model parameters
        model: Model instance
    """
    model = create_model(num_classes=num_classes)
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    variables = model.init(rng_key, dummy_input, training=False)
    return variables['params'], variables.get('batch_stats'), model
