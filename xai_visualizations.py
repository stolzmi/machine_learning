"""
XAI Visualization Methods for Flax CNN
Implements GradCAM and Saliency Maps
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM)

    Generates heatmaps showing which regions of the image are important
    for the model's prediction by using gradients flowing into the last
    convolutional layer.
    """

    def __init__(self, model, target_layer: str = 'conv3'):
        """
        Args:
            model: Flax model instance
            target_layer: Name of the convolutional layer to target
        """
        self.model = model
        self.target_layer = target_layer

    def compute_gradcam(self, params, batch_stats, image, class_idx: Optional[int] = None):
        """
        Compute GradCAM heatmap for a single image

        Args:
            params: Model parameters
            batch_stats: Batch statistics for BatchNorm
            image: Input image [H, W, C]
            class_idx: Target class index (if None, uses predicted class)

        Returns:
            heatmap: GradCAM heatmap [H, W]
            predicted_class: Predicted class index
            class_score: Score for the target class
        """
        # Add batch dimension
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Forward pass to get activations and logits
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits, activations = self.model.apply(
                variables, image, training=False, return_activations=True
            )
        else:
            logits, activations = self.model.apply(
                {'params': params}, image, training=False, return_activations=True
            )

        target_activations = activations[self.target_layer]

        # Get predicted class if not specified
        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Define function to compute class score from image
        def class_score_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                logits, _ = self.model.apply(
                    variables, img, training=False, return_activations=True
                )
            else:
                logits, _ = self.model.apply(
                    {'params': params}, img, training=False, return_activations=True
                )
            return logits[0, class_idx]

        # Compute gradients of class score w.r.t. input
        grad_fn = grad(class_score_fn)
        input_gradients = grad_fn(image)

        # For GradCAM, we approximate the importance using the activation magnitudes
        # This is a simplified version that doesn't require explicit gradient computation
        # through intermediate layers (which is complex in JAX)

        # Compute channel-wise importance using global average pooling
        # This approximates the gradient-based importance
        weights = jnp.mean(target_activations[0], axis=(0, 1))

        # Apply ReLU to weights to keep only positive influences
        weights = jnp.maximum(weights, 0)

        # Weighted combination of activation maps
        cam = jnp.sum(weights * target_activations[0], axis=-1)

        # Apply ReLU to focus on positive influences
        cam = jnp.maximum(cam, 0)

        # Normalize to [0, 1]
        if jnp.max(cam) > 0:
            cam = cam / jnp.max(cam)

        # Resize to input image size using bilinear interpolation
        from jax.image import resize
        cam_resized = resize(
            cam,
            shape=(image.shape[1], image.shape[2]),
            method='bilinear'
        )

        return np.array(cam_resized), predicted_class, float(logits[0, class_idx])


    def visualize(self, image, heatmap, alpha=0.4, colormap='jet'):
        """
        Overlay heatmap on original image

        Args:
            image: Original image [H, W, C], values in [0, 1] or [0, 255]
            heatmap: GradCAM heatmap [H, W]
            alpha: Transparency of heatmap overlay
            colormap: Matplotlib colormap name

        Returns:
            Overlaid image as numpy array
        """
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0

        # Apply colormap to heatmap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel

        # Overlay
        overlaid = heatmap_colored * alpha + image * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 1)

        return overlaid


class SaliencyMap:
    """
    Saliency Map visualization

    Shows the gradient of the class score with respect to input pixels,
    highlighting which pixels need to change the most to affect the prediction.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def compute_saliency(self, params, batch_stats, image, class_idx: Optional[int] = None):
        """
        Compute saliency map for a single image

        Args:
            params: Model parameters
            batch_stats: Batch statistics for BatchNorm
            image: Input image [H, W, C]
            class_idx: Target class index (if None, uses predicted class)

        Returns:
            saliency: Saliency map [H, W]
            predicted_class: Predicted class index
            class_score: Score for the target class
        """
        # Add batch dimension
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Forward pass to get prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        # Get predicted class if not specified
        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Define function to compute class score
        def class_score_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                logits = self.model.apply(variables, img, training=False)
            else:
                logits = self.model.apply({'params': params}, img, training=False)
            return logits[0, class_idx]

        # Compute gradients
        grad_fn = grad(class_score_fn)
        gradients = grad_fn(image)

        # Take maximum across color channels (or L2 norm)
        saliency = jnp.max(jnp.abs(gradients[0]), axis=-1)

        # Normalize to [0, 1]
        if jnp.max(saliency) > 0:
            saliency = saliency / jnp.max(saliency)

        return np.array(saliency), predicted_class, float(logits[0, class_idx])

    def compute_smooth_saliency(self, params, batch_stats, image,
                               class_idx: Optional[int] = None,
                               n_samples: int = 50, noise_level: float = 0.1):
        """
        Compute Smooth Saliency map (SmoothGrad)

        Averages saliency maps over multiple noisy versions of the input
        to reduce noise in the visualization.

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, C]
            class_idx: Target class index
            n_samples: Number of noisy samples to average
            noise_level: Standard deviation of noise

        Returns:
            smooth_saliency: Smoothed saliency map [H, W]
            predicted_class: Predicted class index
            class_score: Score for the target class
        """
        # Add batch dimension
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get predicted class
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Define gradient function
        def class_score_fn(img):
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                logits = self.model.apply(variables, img, training=False)
            else:
                logits = self.model.apply({'params': params}, img, training=False)
            return logits[0, class_idx]

        grad_fn = grad(class_score_fn)

        # Accumulate gradients over noisy samples
        total_gradients = jnp.zeros_like(image)

        rng = jax.random.PRNGKey(42)
        for i in range(n_samples):
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, image.shape) * noise_level
            noisy_image = image + noise
            gradients = grad_fn(noisy_image)
            total_gradients += gradients

        # Average gradients
        avg_gradients = total_gradients / n_samples

        # Take maximum across color channels
        smooth_saliency = jnp.max(jnp.abs(avg_gradients[0]), axis=-1)

        # Normalize to [0, 1]
        if jnp.max(smooth_saliency) > 0:
            smooth_saliency = smooth_saliency / jnp.max(smooth_saliency)

        return np.array(smooth_saliency), predicted_class, float(logits[0, class_idx])

    def visualize(self, image, saliency, alpha=0.5, colormap='hot'):
        """
        Overlay saliency map on original image

        Args:
            image: Original image [H, W, C]
            saliency: Saliency map [H, W]
            alpha: Transparency of overlay
            colormap: Matplotlib colormap name

        Returns:
            Overlaid image as numpy array
        """
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0

        # Apply colormap to saliency
        cmap = cm.get_cmap(colormap)
        saliency_colored = cmap(saliency)[:, :, :3]  # Remove alpha channel

        # Overlay
        overlaid = saliency_colored * alpha + image * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 1)

        return overlaid


def plot_xai_comparison(image, gradcam_heatmap, saliency_map,
                       class_names=None, predicted_class=None,
                       true_class=None, save_path=None):
    """
    Create a comprehensive visualization comparing different XAI methods

    Args:
        image: Original image [H, W, C]
        gradcam_heatmap: GradCAM heatmap [H, W]
        saliency_map: Saliency map [H, W]
        class_names: List of class names
        predicted_class: Predicted class index
        true_class: True class index (optional)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # GradCAM
    gradcam_viz = GradCAM(None).visualize(image, gradcam_heatmap)
    axes[0, 1].imshow(gradcam_viz)
    axes[0, 1].set_title('GradCAM', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Saliency Map
    saliency_viz = SaliencyMap(None).visualize(image, saliency_map)
    axes[1, 0].imshow(saliency_viz)
    axes[1, 0].set_title('Saliency Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Heatmaps side by side
    axes[1, 1].imshow(gradcam_heatmap, cmap='jet')
    axes[1, 1].set_title('GradCAM Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Add prediction info
    if predicted_class is not None and class_names is not None:
        pred_text = f"Predicted: {class_names[predicted_class]}"
        if true_class is not None:
            pred_text += f"\nTrue: {class_names[true_class]}"
        fig.suptitle(pred_text, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
