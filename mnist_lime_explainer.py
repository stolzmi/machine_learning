"""
LIME (Local Interpretable Model-agnostic Explanations) for MNIST
==================================================================

LIME explains individual predictions by learning an interpretable model
locally around the prediction.
"""

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp


class MNISTLimeExplainer:
    """LIME explainer for MNIST digit classification"""

    def __init__(self, predict_fn: Callable, class_names: Optional[list] = None):
        """
        Initialize LIME explainer

        Args:
            predict_fn: Function that takes image and returns class probabilities
            class_names: List of class names (default: ['0', '1', ..., '9'])
        """
        self.predict_fn = predict_fn
        self.class_names = class_names or [str(i) for i in range(10)]

        # Initialize LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()

    def explain_instance(
        self,
        image: np.ndarray,
        top_labels: int = 2,
        num_samples: int = 1000,
        num_features: int = 10,
        hide_color: int = 0
    ) -> Tuple[object, np.ndarray]:
        """
        Explain a single prediction using LIME

        Args:
            image: Input image [H, W] or [H, W, 1]
            top_labels: Number of top classes to explain
            num_samples: Number of samples for LIME (more = better but slower)
            num_features: Number of superpixels in explanation
            hide_color: Color to use for hidden regions (0 = black)

        Returns:
            Tuple of (explanation object, visualization array)
        """
        # Ensure image is 2D for LIME
        if len(image.shape) == 3:
            image = image.squeeze(-1)

        # Convert to numpy if JAX array
        if isinstance(image, jnp.ndarray):
            image = np.array(image)

        # LIME expects images with values in a certain range
        # Make sure it's in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0

        # Create a wrapper for the predict function
        def predict_wrapper(images):
            """Wrapper that handles batch prediction"""
            batch_size = len(images)
            predictions = []

            for img in images:
                # Handle RGB images from LIME (convert to grayscale)
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    # Convert RGB to grayscale
                    img = np.mean(img, axis=-1, keepdims=True)
                elif len(img.shape) == 2:
                    # Add channel dimension for grayscale
                    img = img[:, :, np.newaxis]
                # If already [H, W, 1], keep as is

                # Get prediction
                pred = self.predict_fn(img)

                # Ensure it's a numpy array with shape [10]
                if isinstance(pred, jnp.ndarray):
                    pred = np.array(pred)

                if len(pred.shape) == 2:
                    pred = pred[0]  # Remove batch dimension

                predictions.append(pred)

            return np.array(predictions)

        # Generate explanation
        explanation = self.explainer.explain_instance(
            image,
            predict_wrapper,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples,
            segmentation_fn=lambda img: self._segment_image(img, num_features)
        )

        return explanation

    def _segment_image(self, image: np.ndarray, num_segments: int = 10) -> np.ndarray:
        """
        Segment image into superpixels

        For MNIST, we use a simple grid-based segmentation since
        quickshift/felzenszwalb might be overkill for 28x28 images

        Args:
            image: Input image [H, W]
            num_segments: Target number of segments

        Returns:
            Segmentation map [H, W] with integer labels
        """
        h, w = image.shape[:2]

        # Calculate grid size
        n_rows = int(np.sqrt(num_segments))
        n_cols = int(np.ceil(num_segments / n_rows))

        # Create grid segments
        row_size = h // n_rows
        col_size = w // n_cols

        segments = np.zeros((h, w), dtype=np.int32)

        for i in range(n_rows):
            for j in range(n_cols):
                segment_id = i * n_cols + j
                row_start = i * row_size
                row_end = (i + 1) * row_size if i < n_rows - 1 else h
                col_start = j * col_size
                col_end = (j + 1) * col_size if j < n_cols - 1 else w

                segments[row_start:row_end, col_start:col_end] = segment_id

        return segments

    def visualize_explanation(
        self,
        image: np.ndarray,
        explanation: object,
        label: int,
        positive_only: bool = False,
        num_features: int = 10,
        hide_rest: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize LIME explanation as a weighted heatmap

        Args:
            image: Original input image
            explanation: LIME explanation object
            label: Class label to explain
            positive_only: Show only positive contributions
            num_features: Number of features to show
            hide_rest: Hide rest of image

        Returns:
            Tuple of (heatmap, boundaries_image)
        """
        # Ensure image is 2D
        if len(image.shape) == 3:
            image = image.squeeze(-1)

        # Convert to numpy if JAX array
        if isinstance(image, jnp.ndarray):
            image = np.array(image)

        # Get the segments used in the explanation
        segments = explanation.segments

        # Get local explanation (feature importances)
        if label not in explanation.local_exp:
            # Fallback to first available label
            label = list(explanation.local_exp.keys())[0]

        local_exp = explanation.local_exp[label]

        # Create weighted heatmap based on feature importances
        heatmap = np.zeros(image.shape, dtype=np.float32)

        for feature_id, weight in local_exp:
            # Get the mask for this feature (superpixel)
            feature_mask = (segments == feature_id)
            # Add the weighted contribution to the heatmap
            heatmap[feature_mask] = weight

        # Normalize heatmap to [-1, 1] or [0, 1] depending on positive_only
        if positive_only:
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        else:
            # Keep both positive and negative contributions
            max_abs = np.abs(heatmap).max()
            if max_abs > 0:
                heatmap = heatmap / max_abs  # Range: [-1, 1]

        # Create boundaries visualization
        boundaries = mark_boundaries(image, segments)

        return heatmap, boundaries

    def get_explanation_dict(
        self,
        explanation: object,
        label: int,
        num_features: int = 10
    ) -> dict:
        """
        Get explanation as a dictionary with feature importances

        Args:
            explanation: LIME explanation object
            label: Class label to explain
            num_features: Number of top features

        Returns:
            Dictionary with explanation details
        """
        # Get local explanation
        local_exp = explanation.local_exp[label]

        # Sort by absolute importance
        local_exp_sorted = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)

        # Get top features
        top_features = local_exp_sorted[:num_features]

        # Get prediction probabilities
        pred_proba = explanation.predict_proba

        result = {
            'label': label,
            'class_name': self.class_names[label],
            'probability': float(pred_proba[label]),
            'top_features': [
                {
                    'feature_id': feat_id,
                    'importance': float(importance),
                    'direction': 'positive' if importance > 0 else 'negative'
                }
                for feat_id, importance in top_features
            ],
            'intercept': float(explanation.intercept[label]) if hasattr(explanation, 'intercept') else 0.0,
            'score': float(explanation.score) if hasattr(explanation, 'score') else 0.0
        }

        return result


def create_lime_explainer(model, model_params, batch_stats=None):
    """
    Create LIME explainer for a JAX/Flax model

    Args:
        model: Flax model
        model_params: Model parameters
        batch_stats: Batch statistics (if using BatchNorm)

    Returns:
        MNISTLimeExplainer instance
    """
    def predict_fn(image):
        """Prediction function for LIME"""
        # Ensure proper shape
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        if len(image.shape) == 3 and image.shape[0] != 1:
            image = image[np.newaxis, :]  # Add batch dimension

        # Convert to JAX array
        image_jax = jnp.array(image, dtype=jnp.float32)

        # Forward pass
        if batch_stats is not None:
            variables = {'params': model_params, 'batch_stats': batch_stats}
            logits = model.apply(variables, image_jax, training=False)
        else:
            logits = model.apply({'params': model_params}, image_jax, training=False)

        # Handle CBN model that returns (logits, concepts)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)

        return np.array(probs[0] if len(probs.shape) > 1 else probs)

    return MNISTLimeExplainer(predict_fn)


def plot_lime_explanation(
    image: np.ndarray,
    explanation: object,
    predicted_class: int,
    true_class: Optional[int] = None,
    num_features: int = 5,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot LIME explanation with multiple views

    Args:
        image: Original input image
        explanation: LIME explanation object
        predicted_class: Predicted class
        true_class: True class (optional)
        num_features: Number of features to visualize
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Ensure image is 2D
    if len(image.shape) == 3:
        image = image.squeeze(-1)

    # 1. Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 2. Positive contributions
    temp_pos, mask_pos = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    axes[1].imshow(mark_boundaries(temp_pos, mask_pos))
    axes[1].set_title(f'Positive Features (Class {predicted_class})')
    axes[1].axis('off')

    # 3. All contributions
    temp_all, mask_all = explanation.get_image_and_mask(
        predicted_class,
        positive_only=False,
        num_features=num_features,
        hide_rest=False
    )
    axes[2].imshow(mark_boundaries(temp_all, mask_all))
    axes[2].set_title('All Features')
    axes[2].axis('off')

    if true_class is not None:
        fig.suptitle(f'LIME Explanation (Predicted: {predicted_class}, True: {true_class})')
    else:
        fig.suptitle(f'LIME Explanation (Predicted: {predicted_class})')

    plt.tight_layout()
    return fig
