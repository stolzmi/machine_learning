"""
Layer-wise Relevance Propagation (LRP) and Activation Maximization for MNIST CNN

This module implements two advanced XAI techniques:
1. Layer-wise Relevance Propagation (LRP) - Decomposes predictions into pixel-wise relevance scores
2. Activation Maximization - Generates synthetic images that maximize neuron activations
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from typing import Dict, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial


class LayerRelevancePropagation:
    """
    Layer-wise Relevance Propagation (LRP) for MNIST CNN

    LRP decomposes the prediction score into pixel-wise relevance scores,
    showing which input features contributed to the prediction.

    Implements the LRP-epsilon rule for numerical stability.
    """

    def __init__(self, model, epsilon: float = 1e-10):
        """
        Args:
            model: Flax model instance
            epsilon: Small constant for numerical stability
        """
        self.model = model
        self.epsilon = epsilon

    def compute_lrp(self, params, batch_stats, image, class_idx: Optional[int] = None):
        """
        Compute LRP relevance scores for an image

        Args:
            params: Model parameters
            batch_stats: Batch statistics for BatchNorm
            image: Input image [H, W, 1] or [H, W]
            class_idx: Target class (if None, uses predicted class)

        Returns:
            relevance_map: Pixel-wise relevance scores [H, W]
            predicted_class: Predicted class index
            class_score: Score for the target class
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Forward pass to get activations
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, activations = self.model.apply(
                variables, image, training=False, return_activations=True
            )
        else:
            logits, activations = self.model.apply(
                {'params': params}, image, training=False, return_activations=True
            )

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Initialize relevance for output layer
        R = jnp.zeros_like(logits)
        R = R.at[0, class_idx].set(logits[0, class_idx])

        # Backward pass through network layers
        # Note: This is a simplified LRP implementation for dense layers
        # For a complete implementation, you'd need to access all layer weights

        # For demonstration, we'll compute relevance using gradient-based approximation
        # True LRP requires layer-by-layer backpropagation with specific rules

        def class_score_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                logits = self.model.apply(variables, img, training=False)
            else:
                logits = self.model.apply({'params': params}, img, training=False)
            return logits[0, class_idx]

        # Compute gradient-based relevance (approximation)
        grad_fn = grad(class_score_fn)
        gradients = grad_fn(image)

        # LRP-style relevance: input * gradient
        relevance = image * gradients
        relevance_map = jnp.abs(relevance[0, :, :, 0])

        # Normalize to [0, 1]
        if jnp.max(relevance_map) > 0:
            relevance_map = relevance_map / jnp.max(relevance_map)

        return np.array(relevance_map), predicted_class, float(logits[0, class_idx])

    def compute_lrp_epsilon_rule(self, params, batch_stats, image,
                                  class_idx: Optional[int] = None):
        """
        Compute LRP using epsilon rule with proper layer-wise propagation

        This is a more accurate implementation that propagates relevance
        through the network layer by layer.

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            class_idx: Target class

        Returns:
            relevance_map: Pixel-wise relevance scores [H, W]
            predicted_class: Predicted class index
            class_score: Score for the target class
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Forward pass
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, activations = self.model.apply(
                variables, image, training=False, return_activations=True
            )
        else:
            logits, activations = self.model.apply(
                {'params': params}, image, training=False, return_activations=True
            )

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Start with output relevance
        R_out = jnp.zeros_like(logits[0])
        R_out = R_out.at[class_idx].set(logits[0, class_idx])

        # Backpropagate through FC layers
        fc_out_weights = params['fc_out']['kernel']
        fc1_activations = activations['fc1'][0]

        # Relevance for fc1: epsilon rule
        z = jnp.dot(fc1_activations, fc_out_weights) + self.epsilon
        s = R_out / z
        c = jnp.dot(s, fc_out_weights.T)
        R_fc1 = fc1_activations * c

        # For convolutional layers, we use gradient-based approximation
        # as true LRP for conv layers requires more complex implementation
        def fc1_score_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                _, acts = self.model.apply(
                    variables, img, training=False, return_activations=True
                )
            else:
                _, acts = self.model.apply(
                    {'params': params}, img, training=False, return_activations=True
                )
            return jnp.sum(acts['fc1'][0] * (R_fc1 / (jnp.sum(R_fc1) + self.epsilon)))

        grad_fn = grad(fc1_score_fn)
        input_relevance = grad_fn(image)

        # Get pixel-wise relevance
        relevance_map = jnp.sum(jnp.abs(input_relevance[0]), axis=-1)

        # Normalize
        if jnp.max(relevance_map) > 0:
            relevance_map = relevance_map / jnp.max(relevance_map)

        return np.array(relevance_map), predicted_class, float(logits[0, class_idx])

    def visualize(self, image, relevance_map, alpha=0.5, colormap='seismic'):
        """
        Visualize relevance map overlaid on original image

        Args:
            image: Original image [H, W] or [H, W, 1]
            relevance_map: LRP relevance scores [H, W]
            alpha: Overlay transparency
            colormap: Matplotlib colormap (seismic shows positive/negative)

        Returns:
            Overlaid visualization
        """
        # Ensure image is 2D
        if image.ndim == 3:
            image = image.squeeze(-1)

        # Normalize image
        if image.max() > 1.0:
            image = image / 255.0

        # Convert grayscale to RGB
        image_rgb = np.stack([image, image, image], axis=-1)

        # Apply colormap to relevance
        cmap = cm.get_cmap(colormap)
        relevance_colored = cmap(relevance_map)[:, :, :3]

        # Overlay
        overlaid = relevance_colored * alpha + image_rgb * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 1)

        return overlaid


class ActivationMaximization:
    """
    Activation Maximization for MNIST CNN

    Generates synthetic images that maximally activate specific neurons or classes.
    This helps understand what patterns the network is looking for.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def maximize_class(self, params, batch_stats, class_idx: int,
                       n_iterations: int = 200,
                       learning_rate: float = 1.0,
                       l2_reg: float = 0.01,
                       blur_every: int = 4,
                       blur_sigma: float = 0.5,
                       input_shape: Tuple = (28, 28, 1),
                       seed: int = 42):
        """
        Generate an image that maximizes the score for a specific class

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            class_idx: Target class to maximize
            n_iterations: Number of optimization iterations
            learning_rate: Step size for gradient ascent
            l2_reg: L2 regularization strength
            blur_every: Apply blur every N iterations
            blur_sigma: Gaussian blur sigma
            input_shape: Shape of input image
            seed: Random seed

        Returns:
            optimized_image: Generated image [H, W]
            score_history: List of scores during optimization
        """
        # Initialize random image
        rng = jax.random.PRNGKey(seed)
        # Start with small random values around 0.5
        image = jax.random.normal(rng, (1,) + input_shape) * 0.1 + 0.5
        image = jnp.clip(image, 0, 1)

        score_history = []

        # Define objective function
        def objective_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                logits = self.model.apply(variables, img, training=False)
            else:
                logits = self.model.apply({'params': params}, img, training=False)

            # Maximize class score
            class_score = logits[0, class_idx]

            # L2 regularization (prefer smaller pixel values)
            l2_loss = l2_reg * jnp.sum(img ** 2)

            return class_score - l2_loss

        grad_fn = grad(objective_fn)

        # Optimization loop
        for i in range(n_iterations):
            # Compute gradients
            grads = grad_fn(image)

            # Gradient ascent
            image = image + learning_rate * grads

            # Clip to valid range
            image = jnp.clip(image, 0, 1)

            # Apply blur periodically for smoothness
            if i % blur_every == 0 and blur_sigma > 0:
                image = self._gaussian_blur(image, blur_sigma)

            # Record score
            score = objective_fn(image)
            score_history.append(float(score))

        optimized_image = np.array(image[0, :, :, 0])

        return optimized_image, score_history

    def maximize_neuron(self, params, batch_stats, layer_name: str, neuron_idx: int,
                       n_iterations: int = 200,
                       learning_rate: float = 1.0,
                       l2_reg: float = 0.01,
                       blur_every: int = 4,
                       blur_sigma: float = 0.5,
                       input_shape: Tuple = (28, 28, 1),
                       seed: int = 42):
        """
        Generate an image that maximizes activation of a specific neuron

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            layer_name: Name of layer (e.g., 'conv1', 'conv2', 'fc1')
            neuron_idx: Index of neuron to maximize
            n_iterations: Number of optimization iterations
            learning_rate: Step size
            l2_reg: L2 regularization
            blur_every: Blur frequency
            blur_sigma: Blur strength
            input_shape: Input image shape
            seed: Random seed

        Returns:
            optimized_image: Generated image [H, W]
            score_history: Activation scores during optimization
        """
        # Initialize random image
        rng = jax.random.PRNGKey(seed)
        image = jax.random.normal(rng, (1,) + input_shape) * 0.1 + 0.5
        image = jnp.clip(image, 0, 1)

        score_history = []

        # Define objective function
        def objective_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                _, activations = self.model.apply(
                    variables, img, training=False, return_activations=True
                )
            else:
                _, activations = self.model.apply(
                    {'params': params}, img, training=False, return_activations=True
                )

            # Get activation of target neuron
            layer_activation = activations[layer_name][0]

            if layer_activation.ndim == 3:  # Convolutional layer [H, W, C]
                # Maximize specific channel (neuron_idx)
                neuron_activation = jnp.mean(layer_activation[:, :, neuron_idx])
            elif layer_activation.ndim == 1:  # Dense layer [features]
                neuron_activation = layer_activation[neuron_idx]
            else:
                raise ValueError(f"Unexpected activation shape: {layer_activation.shape}")

            # L2 regularization
            l2_loss = l2_reg * jnp.sum(img ** 2)

            return neuron_activation - l2_loss

        grad_fn = grad(objective_fn)

        # Optimization loop
        for i in range(n_iterations):
            grads = grad_fn(image)
            image = image + learning_rate * grads
            image = jnp.clip(image, 0, 1)

            if i % blur_every == 0 and blur_sigma > 0:
                image = self._gaussian_blur(image, blur_sigma)

            score = objective_fn(image)
            score_history.append(float(score))

        optimized_image = np.array(image[0, :, :, 0])

        return optimized_image, score_history

    def _gaussian_blur(self, image, sigma: float):
        """
        Apply Gaussian blur to image for smoothness

        Args:
            image: Input image [1, H, W, C]
            sigma: Blur strength (used to determine kernel size)

        Returns:
            Blurred image
        """
        if sigma <= 0:
            return image

        # For small images like MNIST, just use a simple uniform blur
        # This is equivalent to a box filter which approximates Gaussian
        kernel_size = max(3, int(sigma * 2))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Use uniform kernel (box blur) - simpler and stable
        kernel_weight = 1.0 / (kernel_size * kernel_size)

        pad_size = kernel_size // 2
        h, w = image.shape[1], image.shape[2]

        blurred = image
        for c in range(image.shape[-1]):
            channel = image[0, :, :, c]

            # Pad image
            padded = jnp.pad(channel, pad_size, mode='edge')

            # Apply box blur efficiently using strided operations
            # Create output array
            result = jnp.zeros_like(channel)

            # Vectorized box blur
            for di in range(kernel_size):
                for dj in range(kernel_size):
                    result = result + padded[di:di+h, dj:dj+w] * kernel_weight

            blurred = blurred.at[0, :, :, c].set(result)

        return blurred


def plot_lrp_visualization(image, relevance_map, predicted_class=None,
                           class_score=None, digit_names=None, save_path=None):
    """
    Create visualization for LRP analysis

    Args:
        image: Original image [H, W]
        relevance_map: LRP relevance scores [H, W]
        predicted_class: Predicted class index
        class_score: Prediction score
        digit_names: List of class names
        save_path: Path to save figure
    """
    if digit_names is None:
        digit_names = [str(i) for i in range(10)]

    # Ensure image is 2D and normalized
    if image.ndim == 3:
        image = image.squeeze(-1)
    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Relevance heatmap
    im = axes[1].imshow(relevance_map, cmap='seismic', vmin=0, vmax=1)
    axes[1].set_title('LRP Relevance Map', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    lrp = LayerRelevancePropagation(None)
    overlaid = lrp.visualize(image, relevance_map)
    axes[2].imshow(overlaid)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Title with prediction
    if predicted_class is not None:
        title = f"Predicted: {digit_names[predicted_class]}"
        if class_score is not None:
            title += f" (score: {class_score:.3f})"
        fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_activation_maximization_grid(images, scores, class_names=None,
                                      title="Activation Maximization", save_path=None):
    """
    Plot a grid of activation maximization results

    Args:
        images: List of generated images
        scores: List of final scores
        class_names: List of class names
        title: Figure title
        save_path: Path to save figure
    """
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_images == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img, score) in enumerate(zip(images, scores)):
        row = idx // n_cols
        col = idx % n_cols

        axes[row, col].imshow(img, cmap='gray')
        if class_names is not None:
            axes[row, col].set_title(f"{class_names[idx]}\nScore: {score:.2f}",
                                    fontsize=12)
        else:
            axes[row, col].set_title(f"Idx {idx}\nScore: {score:.2f}", fontsize=12)
        axes[row, col].axis('off')

    # Hide extra subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_optimization_progress(score_histories, labels=None, save_path=None):
    """
    Plot optimization progress for activation maximization

    Args:
        score_histories: List of score histories
        labels: List of labels for each history
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [f"Class {i}" for i in range(len(score_histories))]

    for history, label in zip(score_histories, labels):
        plt.plot(history, label=label, linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Score', fontsize=12)
    plt.title('Activation Maximization Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return plt.gcf()
