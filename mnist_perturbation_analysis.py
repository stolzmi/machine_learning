"""
Perturbation-Based Analytics for MNIST CNN

This module implements various perturbation-based XAI techniques that analyze
how model predictions change when input features are systematically modified.

Techniques included:
1. Occlusion Sensitivity - Systematically occlude regions and measure impact
2. RISE (Randomized Input Sampling for Explanation) - Random masking approach
3. Meaningful Perturbation - Find minimal perturbations that change predictions
4. Feature Ablation - Remove specific features and measure impact
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from functools import partial


class OcclusionSensitivity:
    """
    Occlusion Sensitivity Analysis

    Systematically occludes (masks) different regions of the input image
    and measures how the prediction changes. Regions that cause large
    changes when occluded are considered important.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def compute_occlusion_sensitivity(self, params, batch_stats, image,
                                     window_size: int = 4,
                                     stride: int = 2,
                                     class_idx: Optional[int] = None):
        """
        Compute occlusion sensitivity map

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            window_size: Size of occlusion window
            stride: Stride for sliding window
            class_idx: Target class (if None, uses predicted class)

        Returns:
            sensitivity_map: Map showing importance of each region [H, W]
            predicted_class: Predicted class index
            baseline_score: Original prediction score
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get baseline prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        baseline_score = float(logits[0, class_idx])

        # Initialize sensitivity map
        h, w = image.shape[1], image.shape[2]
        sensitivity_map = np.zeros((h, w))
        count_map = np.zeros((h, w))  # Track how many times each pixel was occluded

        # Slide window across image
        for i in range(0, h - window_size + 1, stride):
            for j in range(0, w - window_size + 1, stride):
                # Create occluded image
                occluded_image = image.copy()
                occluded_image = occluded_image.at[0, i:i+window_size, j:j+window_size, :].set(0)

                # Get prediction for occluded image
                if batch_stats is not None:
                    variables = {'params': params, 'batch_stats': batch_stats}
                    occluded_logits = self.model.apply(variables, occluded_image, training=False)
                else:
                    occluded_logits = self.model.apply({'params': params}, occluded_image, training=False)

                occluded_score = float(occluded_logits[0, class_idx])

                # Compute sensitivity (drop in confidence)
                sensitivity = baseline_score - occluded_score

                # Add to sensitivity map
                sensitivity_map[i:i+window_size, j:j+window_size] += sensitivity
                count_map[i:i+window_size, j:j+window_size] += 1

        # Average sensitivity where pixels were covered multiple times
        sensitivity_map = np.divide(sensitivity_map, count_map,
                                    where=count_map > 0,
                                    out=np.zeros_like(sensitivity_map))

        # Normalize to [0, 1]
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        return sensitivity_map, predicted_class, baseline_score

    def visualize(self, image, sensitivity_map, alpha=0.5, colormap='hot'):
        """
        Overlay sensitivity map on original image

        Args:
            image: Original image [H, W] or [H, W, 1]
            sensitivity_map: Occlusion sensitivity map [H, W]
            alpha: Transparency of overlay
            colormap: Matplotlib colormap

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

        # Apply colormap to sensitivity
        cmap = cm.get_cmap(colormap)
        sensitivity_colored = cmap(sensitivity_map)[:, :, :3]

        # Overlay
        overlaid = sensitivity_colored * alpha + image_rgb * (1 - alpha)
        overlaid = np.clip(overlaid, 0, 1)

        return overlaid


class RISE:
    """
    RISE (Randomized Input Sampling for Explanation)

    Generates random binary masks, measures prediction with each mask,
    and computes importance map as weighted average of masks.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def compute_rise_map(self, params, batch_stats, image,
                        n_masks: int = 1000,
                        mask_prob: float = 0.5,
                        mask_size: int = 8,
                        class_idx: Optional[int] = None,
                        seed: int = 42):
        """
        Compute RISE importance map

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            n_masks: Number of random masks to generate
            mask_prob: Probability of keeping each cell in mask
            mask_size: Size of upsampling for mask (smaller = coarser masks)
            class_idx: Target class
            seed: Random seed

        Returns:
            importance_map: RISE importance map [H, W]
            predicted_class: Predicted class index
            baseline_score: Original prediction score
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get baseline prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        baseline_score = float(logits[0, class_idx])

        h, w = image.shape[1], image.shape[2]

        # Initialize importance map
        importance_map = np.zeros((h, w))

        rng = np.random.RandomState(seed)

        # Generate random masks and compute predictions
        for _ in range(n_masks):
            # Generate random mask at low resolution
            small_mask = rng.rand(mask_size, mask_size) < mask_prob

            # Upsample mask to image size
            from scipy.ndimage import zoom
            mask = zoom(small_mask.astype(float),
                       (h / mask_size, w / mask_size),
                       order=1)
            mask = (mask > 0.5).astype(float)

            # Apply mask to image
            masked_image = image * mask[None, :, :, None]

            # Get prediction
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                masked_logits = self.model.apply(variables, masked_image, training=False)
            else:
                masked_logits = self.model.apply({'params': params}, masked_image, training=False)

            score = float(masked_logits[0, class_idx])

            # Weight mask by prediction score
            importance_map += mask * score

        # Normalize by number of masks
        importance_map = importance_map / n_masks

        # Normalize to [0, 1]
        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()

        return importance_map, predicted_class, baseline_score


class MeaningfulPerturbation:
    """
    Meaningful Perturbation

    Finds the smallest perturbation (mask) that causes the prediction
    to drop below a threshold. This identifies the minimal set of
    important features.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def find_minimal_mask(self, params, batch_stats, image,
                         n_iterations: int = 100,
                         learning_rate: float = 0.1,
                         tv_weight: float = 0.01,
                         l1_weight: float = 0.01,
                         class_idx: Optional[int] = None,
                         seed: int = 42):
        """
        Find minimal mask that removes important features

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent
            tv_weight: Total variation regularization weight
            l1_weight: L1 sparsity regularization weight
            class_idx: Target class
            seed: Random seed

        Returns:
            mask: Binary mask showing important regions [H, W]
            predicted_class: Predicted class index
            scores: Score history during optimization
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get baseline prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        # Initialize mask (in logit space for optimization)
        rng = jax.random.PRNGKey(seed)
        h, w = image.shape[1], image.shape[2]
        mask_logits = jax.random.normal(rng, (h, w)) * 0.1

        scores = []

        # Define objective function
        def objective_fn(mask_logits_param):
            # Convert to probability via sigmoid
            mask = jax.nn.sigmoid(mask_logits_param)
            mask_expanded = mask[None, :, :, None]

            # Apply mask (1 = keep, 0 = remove)
            masked_image = image * mask_expanded

            # Get prediction
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                masked_logits = self.model.apply(variables, masked_image, training=False)
            else:
                masked_logits = self.model.apply({'params': params}, masked_image, training=False)

            # Loss: maximize drop in target class score
            target_score = masked_logits[0, class_idx]

            # Regularization
            # L1: encourage sparse mask (remove as few pixels as possible)
            l1_loss = jnp.mean(mask)

            # Total variation: encourage smooth mask
            tv_loss = jnp.sum(jnp.abs(mask[:-1, :] - mask[1:, :])) + \
                     jnp.sum(jnp.abs(mask[:, :-1] - mask[:, 1:]))

            # Total loss: reduce target score while keeping mask sparse and smooth
            loss = target_score + l1_weight * l1_loss + tv_weight * tv_loss

            return loss

        grad_fn = jax.grad(objective_fn)

        # Optimization loop
        for i in range(n_iterations):
            # Compute gradients
            grads = grad_fn(mask_logits)

            # Gradient descent
            mask_logits = mask_logits - learning_rate * grads

            # Compute current score
            score = objective_fn(mask_logits)
            scores.append(float(score))

        # Convert final mask to binary
        final_mask = jax.nn.sigmoid(mask_logits)
        final_mask = (final_mask > 0.5).astype(float)

        return np.array(final_mask), predicted_class, scores


class FeatureAblation:
    """
    Feature Ablation Analysis

    Systematically removes different features or regions and measures
    the impact on predictions. Useful for understanding feature importance.
    """

    def __init__(self, model):
        """
        Args:
            model: Flax model instance
        """
        self.model = model

    def ablate_regions(self, params, batch_stats, image,
                      regions: Dict[str, Tuple[int, int, int, int]],
                      class_idx: Optional[int] = None):
        """
        Ablate specific regions and measure impact

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            regions: Dict of region_name -> (y_start, y_end, x_start, x_end)
            class_idx: Target class

        Returns:
            results: Dict of region_name -> score_drop
            predicted_class: Predicted class index
            baseline_score: Original prediction score
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get baseline prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        baseline_score = float(logits[0, class_idx])

        results = {}

        # Ablate each region
        for region_name, (y_start, y_end, x_start, x_end) in regions.items():
            # Create ablated image
            ablated_image = image.copy()
            ablated_image = ablated_image.at[0, y_start:y_end, x_start:x_end, :].set(0)

            # Get prediction
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                ablated_logits = self.model.apply(variables, ablated_image, training=False)
            else:
                ablated_logits = self.model.apply({'params': params}, ablated_image, training=False)

            ablated_score = float(ablated_logits[0, class_idx])
            score_drop = baseline_score - ablated_score

            results[region_name] = {
                'score': ablated_score,
                'score_drop': score_drop,
                'relative_drop': score_drop / baseline_score if baseline_score > 0 else 0
            }

        return results, predicted_class, baseline_score

    def progressive_ablation(self, params, batch_stats, image,
                           ablation_order: str = 'top_to_bottom',
                           step_size: int = 4,
                           class_idx: Optional[int] = None):
        """
        Progressively ablate image and track score degradation

        Args:
            params: Model parameters
            batch_stats: Batch statistics
            image: Input image [H, W, 1] or [H, W]
            ablation_order: 'top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left'
            step_size: Number of rows/columns to ablate at each step
            class_idx: Target class

        Returns:
            scores: List of scores after each ablation step
            ablation_fractions: Fraction of image ablated at each step
            predicted_class: Predicted class index
        """
        # Ensure proper shape
        if image.ndim == 2:
            image = jnp.expand_dims(image, -1)
        if image.ndim == 3:
            image = jnp.expand_dims(image, 0)

        # Get baseline prediction
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            logits = self.model.apply(variables, image, training=False)
        else:
            logits = self.model.apply({'params': params}, image, training=False)

        predicted_class = int(jnp.argmax(logits[0]))
        if class_idx is None:
            class_idx = predicted_class

        h, w = image.shape[1], image.shape[2]
        ablated_image = image.copy()

        scores = []
        ablation_fractions = []

        # Determine ablation sequence
        if ablation_order == 'top_to_bottom':
            steps = range(0, h, step_size)
            ablate_fn = lambda img, i: img.at[0, i:min(i+step_size, h), :, :].set(0)
        elif ablation_order == 'bottom_to_top':
            steps = range(h - step_size, -1, -step_size)
            ablate_fn = lambda img, i: img.at[0, max(0, i):i+step_size, :, :].set(0)
        elif ablation_order == 'left_to_right':
            steps = range(0, w, step_size)
            ablate_fn = lambda img, i: img.at[0, :, i:min(i+step_size, w), :].set(0)
        elif ablation_order == 'right_to_left':
            steps = range(w - step_size, -1, -step_size)
            ablate_fn = lambda img, i: img.at[0, :, max(0, i):i+step_size, :].set(0)
        else:
            raise ValueError(f"Unknown ablation order: {ablation_order}")

        # Progressive ablation
        for i in steps:
            ablated_image = ablate_fn(ablated_image, i)

            # Get prediction
            if batch_stats is not None:
                variables = {'params': params, 'batch_stats': batch_stats}
                ablated_logits = self.model.apply(variables, ablated_image, training=False)
            else:
                ablated_logits = self.model.apply({'params': params}, ablated_image, training=False)

            score = float(ablated_logits[0, class_idx])
            scores.append(score)

            # Calculate fraction ablated
            if ablation_order in ['top_to_bottom', 'bottom_to_top']:
                fraction = min((len(scores) * step_size) / h, 1.0)
            else:
                fraction = min((len(scores) * step_size) / w, 1.0)
            ablation_fractions.append(fraction)

        return scores, ablation_fractions, predicted_class


def plot_perturbation_comparison(image, occlusion_map, rise_map,
                                 predicted_class=None, save_path=None):
    """
    Compare different perturbation-based methods

    Args:
        image: Original image [H, W]
        occlusion_map: Occlusion sensitivity map [H, W]
        rise_map: RISE importance map [H, W]
        predicted_class: Predicted class index
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ensure image is 2D and normalized
    if image.ndim == 3:
        image = image.squeeze(-1)
    if image.max() > 1.0:
        image = image / 255.0

    # Original
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Occlusion sensitivity
    im1 = axes[1].imshow(occlusion_map, cmap='hot')
    axes[1].set_title('Occlusion Sensitivity', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # RISE
    im2 = axes[2].imshow(rise_map, cmap='hot')
    axes[2].set_title('RISE Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if predicted_class is not None:
        fig.suptitle(f'Perturbation-Based Analysis: Digit {predicted_class}',
                    fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_progressive_ablation(ablation_fractions, scores, ablation_order,
                              predicted_class=None, save_path=None):
    """
    Plot progressive ablation results

    Args:
        ablation_fractions: List of ablation fractions
        scores: List of prediction scores
        ablation_order: Type of ablation performed
        predicted_class: Predicted class index
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ablation_fractions, scores, linewidth=2, color='steelblue', marker='o')
    ax.set_xlabel('Fraction of Image Ablated', fontsize=12)
    ax.set_ylabel('Prediction Score', fontsize=12)
    ax.set_title(f'Progressive Ablation ({ablation_order})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    if predicted_class is not None:
        ax.text(0.5, 0.95, f'Predicted Class: {predicted_class}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
