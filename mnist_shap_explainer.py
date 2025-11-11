"""
SHAP (SHapley Additive exPlanations) for MNIST
================================================

SHAP uses game theory to explain model predictions by computing
the contribution of each feature to the prediction.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List
import jax
import jax.numpy as jnp


class MNISTShapExplainer:
    """SHAP explainer for MNIST digit classification"""

    def __init__(
        self,
        predict_fn: Callable,
        background_data: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize SHAP explainer

        Args:
            predict_fn: Function that takes batch of images and returns class probabilities
            background_data: Background dataset for DeepExplainer/KernelExplainer
            class_names: List of class names (default: ['0', '1', ..., '9'])
        """
        self.predict_fn = predict_fn
        self.class_names = class_names or [str(i) for i in range(10)]
        self.background_data = background_data

        # Initialize explainer (will be set based on method)
        self.explainer = None
        self.explainer_type = None

    def create_gradient_explainer(
        self,
        model,
        model_params,
        batch_stats=None
    ):
        """
        Create GradientExplainer (fast, uses gradients)

        Args:
            model: JAX/Flax model
            model_params: Model parameters
            batch_stats: Batch statistics (if using BatchNorm)
        """
        # GradientExplainer for JAX is not directly supported
        # We'll use a custom gradient-based approach
        self.explainer_type = 'gradient'
        self.model = model
        self.model_params = model_params
        self.batch_stats = batch_stats

    def create_kernel_explainer(
        self,
        background_samples: Optional[np.ndarray] = None,
        n_background: int = 50
    ):
        """
        Create KernelExplainer (model-agnostic, slower but works for any model)

        Args:
            background_samples: Background data samples [n_samples, H, W, C]
            n_background: Number of background samples to use
        """
        if background_samples is None:
            # Create a simple background of zeros (black images)
            background_samples = np.zeros((n_background, 28, 28, 1))

        # Limit background samples for efficiency
        if len(background_samples) > n_background:
            indices = np.random.choice(len(background_samples), n_background, replace=False)
            background_samples = background_samples[indices]

        # Create KernelExplainer
        self.explainer = shap.KernelExplainer(self.predict_fn, background_samples)
        self.explainer_type = 'kernel'

    def explain_instance(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        nsamples: int = 100
    ) -> np.ndarray:
        """
        Explain a single prediction using SHAP

        Args:
            image: Input image [H, W] or [H, W, 1]
            target_class: Target class to explain (None = predicted class)
            nsamples: Number of samples for KernelExplainer

        Returns:
            SHAP values array [H, W] or [H, W, C]
        """
        # Ensure image has proper shape
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        # Convert to numpy if JAX array
        if isinstance(image, jnp.ndarray):
            image = np.array(image)

        if self.explainer_type == 'kernel':
            # Use KernelExplainer
            shap_values = self.explainer.shap_values(
                image.reshape(1, -1),
                nsamples=nsamples
            )

            # shap_values is a list of arrays, one per class
            # Each array has shape [1, n_features]
            if target_class is None:
                # Get predicted class
                pred = self.predict_fn(image[np.newaxis, :])
                target_class = np.argmax(pred[0])

            # Get SHAP values for target class and reshape
            shap_image = shap_values[target_class].reshape(image.shape)

        elif self.explainer_type == 'gradient':
            # Use gradient-based SHAP approximation
            shap_image = self._gradient_shap(image, target_class)

        else:
            raise ValueError("Explainer not initialized. Call create_*_explainer first.")

        return shap_image

    def _gradient_shap(
        self,
        image: np.ndarray,
        target_class: Optional[int] = None,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Compute SHAP values using gradient-based approximation (Integrated Gradients)

        Args:
            image: Input image [H, W, 1]
            target_class: Target class to explain
            n_samples: Number of interpolation steps

        Returns:
            SHAP values [H, W, 1]
        """
        # Convert to JAX array
        image_jax = jnp.array(image[np.newaxis, :], dtype=jnp.float32)

        # Get prediction
        if self.batch_stats is not None:
            variables = {'params': self.model_params, 'batch_stats': self.batch_stats}
            logits = self.model.apply(variables, image_jax, training=False)
        else:
            logits = self.model.apply({'params': self.model_params}, image_jax, training=False)

        # Handle CBN model that returns (logits, concepts)
        if isinstance(logits, tuple):
            logits = logits[0]

        if target_class is None:
            target_class = int(jnp.argmax(logits[0]))

        # Compute integrated gradients
        # Create interpolated images from baseline (zeros) to input
        baseline = jnp.zeros_like(image_jax)
        alphas = jnp.linspace(0, 1, n_samples)

        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (image_jax - baseline)

            # Define function to compute gradient
            def class_score_fn(x):
                if self.batch_stats is not None:
                    variables = {'params': self.model_params, 'batch_stats': self.batch_stats}
                    logits = self.model.apply(variables, x, training=False)
                else:
                    logits = self.model.apply({'params': self.model_params}, x, training=False)

                # Handle CBN model
                if isinstance(logits, tuple):
                    logits = logits[0]

                return logits[0, target_class]

            # Compute gradient
            grad = jax.grad(class_score_fn)(interpolated)
            gradients.append(grad)

        # Average gradients
        avg_gradients = jnp.mean(jnp.stack(gradients), axis=0)

        # Integrated gradients = (image - baseline) * avg_gradients
        integrated_grads = (image_jax - baseline) * avg_gradients

        # Convert to numpy and remove batch dimension
        shap_values = np.array(integrated_grads[0])

        return shap_values

    def visualize_shap(
        self,
        image: np.ndarray,
        shap_values: np.ndarray,
        predicted_class: int,
        show_colorbar: bool = True
    ) -> Tuple[np.ndarray, plt.Figure]:
        """
        Visualize SHAP values as a heatmap

        Args:
            image: Original image [H, W] or [H, W, 1]
            shap_values: SHAP values [H, W] or [H, W, 1]
            predicted_class: Predicted class
            show_colorbar: Whether to show colorbar

        Returns:
            Tuple of (heatmap array, matplotlib figure)
        """
        # Ensure 2D arrays
        if len(image.shape) == 3:
            image = image.squeeze(-1)
        if len(shap_values.shape) == 3:
            shap_values = shap_values.squeeze(-1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 2. SHAP heatmap
        im = axes[1].imshow(shap_values, cmap='RdBu_r', vmin=-np.abs(shap_values).max(),
                           vmax=np.abs(shap_values).max())
        axes[1].set_title(f'SHAP Values (Class {predicted_class})')
        axes[1].axis('off')
        if show_colorbar:
            plt.colorbar(im, ax=axes[1])

        # 3. Overlay
        # Normalize SHAP values to [0, 1]
        shap_normalized = np.abs(shap_values)
        shap_normalized = shap_normalized / (shap_normalized.max() + 1e-10)

        # Create overlay
        overlay = image.copy()
        axes[2].imshow(overlay, cmap='gray', alpha=0.7)
        axes[2].imshow(shap_normalized, cmap='hot', alpha=0.5)
        axes[2].set_title('SHAP Overlay')
        axes[2].axis('off')

        fig.suptitle(f'SHAP Explanation for Class {predicted_class}')
        plt.tight_layout()

        return shap_normalized, fig

    def get_top_features(
        self,
        shap_values: np.ndarray,
        image: np.ndarray,
        n_top: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        Get top contributing pixels

        Args:
            shap_values: SHAP values [H, W] or [H, W, 1]
            image: Original image [H, W] or [H, W, 1]
            n_top: Number of top features to return

        Returns:
            List of (row, col, importance) tuples
        """
        # Ensure 2D
        if len(shap_values.shape) == 3:
            shap_values = shap_values.squeeze(-1)

        # Get absolute SHAP values
        abs_shap = np.abs(shap_values)

        # Flatten and get top indices
        flat_shap = abs_shap.flatten()
        top_indices = np.argsort(flat_shap)[::-1][:n_top]

        # Convert to (row, col) coordinates
        h, w = shap_values.shape
        top_features = []
        for idx in top_indices:
            row = idx // w
            col = idx % w
            importance = float(shap_values[row, col])
            top_features.append((int(row), int(col), importance))

        return top_features


def create_shap_explainer(
    model,
    model_params,
    batch_stats=None,
    explainer_type: str = 'gradient',
    background_data: Optional[np.ndarray] = None
):
    """
    Create SHAP explainer for a JAX/Flax model

    Args:
        model: Flax model
        model_params: Model parameters
        batch_stats: Batch statistics (if using BatchNorm)
        explainer_type: Type of explainer ('gradient' or 'kernel')
        background_data: Background data for kernel explainer

    Returns:
        MNISTShapExplainer instance
    """
    def predict_fn(images):
        """Prediction function for SHAP"""
        # Handle flattened input from KernelExplainer
        if len(images.shape) == 2:
            # Reshape from [batch, n_features] to [batch, H, W, C]
            images = images.reshape(-1, 28, 28, 1)

        # Ensure proper shape
        if len(images.shape) == 3:
            images = images[:, :, :, np.newaxis]

        # Convert to JAX array
        images_jax = jnp.array(images, dtype=jnp.float32)

        # Forward pass
        if batch_stats is not None:
            variables = {'params': model_params, 'batch_stats': batch_stats}
            logits = model.apply(variables, images_jax, training=False)
        else:
            logits = model.apply({'params': model_params}, images_jax, training=False)

        # Handle CBN model that returns (logits, concepts)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)

        return np.array(probs)

    explainer = MNISTShapExplainer(predict_fn, background_data)

    if explainer_type == 'gradient':
        explainer.create_gradient_explainer(model, model_params, batch_stats)
    elif explainer_type == 'kernel':
        explainer.create_kernel_explainer(background_data)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

    return explainer


def compare_shap_methods(
    image: np.ndarray,
    gradient_shap_values: np.ndarray,
    kernel_shap_values: Optional[np.ndarray] = None,
    predicted_class: int = 0
):
    """
    Compare different SHAP computation methods

    Args:
        image: Original image
        gradient_shap_values: SHAP values from gradient method
        kernel_shap_values: SHAP values from kernel method (optional)
        predicted_class: Predicted class
    """
    n_plots = 3 if kernel_shap_values is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Ensure 2D
    if len(image.shape) == 3:
        image = image.squeeze(-1)

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Gradient SHAP
    if len(gradient_shap_values.shape) == 3:
        gradient_shap_values = gradient_shap_values.squeeze(-1)

    axes[1].imshow(gradient_shap_values, cmap='RdBu_r',
                   vmin=-np.abs(gradient_shap_values).max(),
                   vmax=np.abs(gradient_shap_values).max())
    axes[1].set_title('Gradient SHAP')
    axes[1].axis('off')

    # Kernel SHAP (if provided)
    if kernel_shap_values is not None:
        if len(kernel_shap_values.shape) == 3:
            kernel_shap_values = kernel_shap_values.squeeze(-1)

        axes[2].imshow(kernel_shap_values, cmap='RdBu_r',
                       vmin=-np.abs(kernel_shap_values).max(),
                       vmax=np.abs(kernel_shap_values).max())
        axes[2].set_title('Kernel SHAP')
        axes[2].axis('off')

    fig.suptitle(f'SHAP Methods Comparison (Class {predicted_class})')
    plt.tight_layout()

    return fig
