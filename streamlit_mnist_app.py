"""
Interactive MNIST XAI App with Drawing Canvas
Draw a digit and see real-time prediction with XAI analysis!
"""

import streamlit as st
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io

# For drawing canvas
from streamlit_drawable_canvas import st_canvas

from mnist_cnn_model import MNISTCNN
from mnist_xai_visualizations import GradCAM, SaliencyMap
from mnist_shape_analysis import (
    analyze_shape_importance,
    identify_key_regions,
    analyze_stroke_features,
    generate_interpretation
)
from mnist_lrp_activation_max import LayerRelevancePropagation

# Try to import CBN, LIME, and SHAP (may not be available if not trained/installed)
try:
    from mnist_cbn_model import (
        create_cbn_model,
        get_concept_names,
        interpret_concepts,
        explain_prediction_with_concepts,
        get_concept_importance_for_class,
        CONCEPT_NAMES
    )
    CBN_AVAILABLE = True
except ImportError:
    CBN_AVAILABLE = False

try:
    from mnist_lime_explainer import create_lime_explainer, MNISTLimeExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from mnist_shap_explainer import create_shap_explainer, MNISTShapExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="MNIST XAI Drawing App",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .interpretation-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffcc00;
        margin: 1rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


MNIST_DIGITS = [str(i) for i in range(10)]


@st.cache_resource
def load_model():
    """Load trained MNIST model"""
    model_path = 'mnist_model.pkl'

    if not Path(model_path).exists():
        st.error(f"❌ Model file '{model_path}' not found!")
        st.info("Please ensure mnist_model.pkl is in the repository.")
        st.stop()

    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    params = checkpoint['params']
    batch_stats = checkpoint['batch_stats']
    model = MNISTCNN(num_classes=10)

    return params, batch_stats, model


@st.cache_resource
def load_xai_analyzers(_model):
    """Load XAI analyzers (LRP)"""
    lrp = LayerRelevancePropagation(_model, epsilon=1e-10)
    return lrp


@st.cache_resource
def load_cbn_model():
    """Load trained CBN model (if available)"""
    if not CBN_AVAILABLE:
        return None, None, None

    model_path = 'mnist_cbn_model.pkl'

    if not Path(model_path).exists():
        return None, None, None

    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        params = checkpoint['params']
        batch_stats = checkpoint['batch_stats']
        model = create_cbn_model(n_concepts=len(CONCEPT_NAMES), n_classes=10)

        return params, batch_stats, model
    except Exception as e:
        st.warning(f"Could not load CBN model: {e}")
        return None, None, None


@st.cache_resource
def load_lime_shap_explainers(_model, _params, _batch_stats):
    """Load LIME and SHAP explainers"""
    lime_explainer = None
    shap_explainer = None

    if LIME_AVAILABLE:
        try:
            lime_explainer = create_lime_explainer(_model, _params, _batch_stats)
        except Exception as e:
            st.warning(f"Could not create LIME explainer: {e}")

    if SHAP_AVAILABLE:
        try:
            shap_explainer = create_shap_explainer(
                _model, _params, _batch_stats,
                explainer_type='gradient'
            )
        except Exception as e:
            st.warning(f"Could not create SHAP explainer: {e}")

    return lime_explainer, shap_explainer


def preprocess_canvas_image(canvas_data):
    """
    Preprocess drawn image to match MNIST format

    Args:
        canvas_data: Canvas image data from streamlit-drawable-canvas

    Returns:
        Preprocessed image ready for model [28, 28, 1]
    """
    if canvas_data is None:
        return None

    # Get image data (RGBA)
    img = canvas_data.astype(np.uint8)

    # Extract RGB channels (ignore alpha for now)
    # White drawing on black background
    img_rgb = img[:, :, :3]

    # Convert to grayscale
    img_gray = np.mean(img_rgb, axis=2).astype(np.uint8)

    # Find bounding box of the drawing
    coords = np.column_stack(np.where(img_gray > 10))

    if len(coords) == 0:
        # No drawing detected, return blank image
        return np.zeros((28, 28, 1), dtype=np.float32)

    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding (20% of size)
    height = y_max - y_min
    width = x_max - x_min
    pad_h = int(height * 0.2)
    pad_w = int(width * 0.2)

    y_min = max(0, y_min - pad_h)
    y_max = min(img_gray.shape[0], y_max + pad_h)
    x_min = max(0, x_min - pad_w)
    x_max = min(img_gray.shape[1], x_max + pad_w)

    # Crop to bounding box
    img_cropped = img_gray[y_min:y_max, x_min:x_max]

    # Resize maintaining aspect ratio to fit in 20x20 (MNIST digits are ~20x20 in center of 28x28)
    height, width = img_cropped.shape

    # Calculate scale to fit within 20x20 while maintaining aspect ratio
    max_size = 20
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))

    # Ensure dimensions are at least 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    # Resize maintaining aspect ratio
    img_pil = Image.fromarray(img_cropped)
    img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_aspect = np.array(img_resized)

    # Center in 28x28 image
    img_28x28 = np.zeros((28, 28), dtype=np.uint8)

    # Calculate offsets to center the resized image
    offset_y = (28 - new_height) // 2
    offset_x = (28 - new_width) // 2

    img_28x28[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = img_aspect

    # Normalize to [0, 1]
    img_normalized = img_28x28.astype(np.float32) / 255.0

    # Add channel dimension
    img_final = np.expand_dims(img_normalized, -1)

    return img_final


def predict_and_analyze(params, batch_stats, model, image, lrp, use_lrp=False,
                       cbn_model=None, cbn_params=None, cbn_batch_stats=None, use_cbn=False,
                       lime_explainer=None, use_lime=False,
                       shap_explainer=None, use_shap=False):
    """
    Make prediction and perform XAI analysis

    Returns:
        Dictionary with all analysis results
    """
    # Basic prediction
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, jnp.expand_dims(image, 0), training=False)
    probs = jax.nn.softmax(logits[0])
    predicted_class = int(jnp.argmax(probs))

    # XAI analysis
    analysis = analyze_shape_importance(params, batch_stats, model, image)
    region_scores = identify_key_regions(image, analysis['saliency'])
    stroke_features = analyze_stroke_features(image, analysis['saliency'])

    results = {
        'predicted_class': predicted_class,
        'probabilities': np.array(probs),
        'logits': np.array(logits[0]),
        'gradcam': analysis['gradcam'],
        'saliency': analysis['saliency'],
        'region_scores': region_scores,
        'stroke_features': stroke_features
    }

    # Add LRP analysis if requested
    if use_lrp:
        lrp_relevance, _, _ = lrp.compute_lrp(params, batch_stats, image)
        lrp_epsilon, _, _ = lrp.compute_lrp_epsilon_rule(params, batch_stats, image)
        results['lrp_relevance'] = lrp_relevance
        results['lrp_epsilon'] = lrp_epsilon

    # Add CBN analysis if requested
    if use_cbn and cbn_model is not None:
        cbn_variables = {'params': cbn_params, 'batch_stats': cbn_batch_stats}
        cbn_logits, concepts = cbn_model.apply(cbn_variables, jnp.expand_dims(image, 0), training=False)
        results['cbn_concepts'] = np.array(concepts[0])
        results['cbn_logits'] = np.array(cbn_logits[0])
        results['cbn_probs'] = np.array(jax.nn.softmax(cbn_logits[0]))

    # Add LIME (Activation Maximization) analysis if requested
    if use_lime and lime_explainer is not None:
        # Use activation maximization to generate the "ideal" image for this class
        from mnist_lrp_activation_max import ActivationMaximization
        actmax = ActivationMaximization(model)

        # Generate image that maximizes the predicted class
        ideal_image, scores = actmax.maximize_class(
            params, batch_stats, predicted_class,
            n_iterations=150,
            learning_rate=1.0,
            l2_reg=0.005,
            blur_every=10,
            blur_sigma=0.5,
            seed=42
        )

        results['lime_heatmap'] = ideal_image.squeeze()
        results['lime_score'] = scores[-1]

    # Add SHAP analysis if requested
    if use_shap and shap_explainer is not None:
        shap_values = shap_explainer.explain_instance(image, predicted_class)
        results['shap_values'] = shap_values

    return results


def plot_xai_visualizations(image, gradcam, saliency):
    """Create XAI visualization plots"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Ensure 2D image
    if image.ndim == 3:
        img_2d = image.squeeze(-1)
    else:
        img_2d = image

    # Original
    axes[0].imshow(img_2d, cmap='gray')
    axes[0].set_title('Your Drawing', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # GradCAM
    im1 = axes[1].imshow(gradcam, cmap='jet')
    axes[1].set_title('GradCAM\n(Spatial Importance)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Saliency
    im2 = axes[2].imshow(saliency, cmap='hot')
    axes[2].set_title('Saliency Map\n(Pixel Importance)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    return fig


def plot_region_importance(region_scores):
    """Plot region importance chart"""
    fig, ax = plt.subplots(figsize=(8, 5))

    regions = ['top', 'bottom', 'left', 'right', 'center']
    scores = [region_scores[r]['avg_importance'] for r in regions]

    bars = ax.barh(regions, scores, color='steelblue')
    ax.set_xlabel('Average Importance', fontsize=11)
    ax.set_title('Region Importance Analysis', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_stroke_features(stroke_features):
    """Plot stroke feature importance"""
    fig, ax = plt.subplots(figsize=(8, 5))

    features = ['Horizontal\nStrokes', 'Vertical\nStrokes', 'Curves', 'Intersections']
    scores = [
        stroke_features['horizontal_strokes'],
        stroke_features['vertical_strokes'],
        stroke_features['curves'],
        stroke_features['intersections']
    ]

    bars = ax.bar(features, scores, color='coral')
    ax.set_ylabel('Importance Score', fontsize=11)
    ax.set_title('Stroke Feature Importance', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def plot_probability_bars(probabilities):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(8, 6))

    digits = [str(i) for i in range(10)]

    bars = ax.barh(digits, probabilities, color='lightgreen')
    bars[np.argmax(probabilities)].set_color('darkgreen')

    ax.set_xlabel('Probability', fontsize=11)
    ax.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_lrp_analysis(image, lrp_relevance, lrp_epsilon):
    """Plot LRP analysis comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ensure 2D image
    if image.ndim == 3:
        img_2d = image.squeeze(-1)
    else:
        img_2d = image

    # Original
    axes[0].imshow(img_2d, cmap='gray')
    axes[0].set_title('Your Drawing', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # LRP standard
    im1 = axes[1].imshow(lrp_relevance, cmap='seismic', vmin=0, vmax=1)
    axes[1].set_title('LRP: Relevance Map\n(Gradient-based)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # LRP epsilon
    im2 = axes[2].imshow(lrp_epsilon, cmap='seismic', vmin=0, vmax=1)
    axes[2].set_title('LRP: Relevance Map\n(Epsilon Rule)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    return fig


def plot_cbn_concepts(concepts, predicted_class):
    """Plot CBN concept activations"""
    fig, ax = plt.subplots(figsize=(12, 6))

    concept_names = get_concept_names()
    x_pos = np.arange(len(concepts))

    colors = ['green' if c >= 0.5 else 'lightgray' for c in concepts]
    bars = ax.barh(x_pos, concepts, color=colors)

    # Add threshold line
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Activation Threshold')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('Concept Activation', fontsize=12, fontweight='bold')
    ax.set_title(f'Concept Activations for Digit {predicted_class}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lime_explanation(image, lime_heatmap, predicted_class):
    """Plot LIME (Activation Maximization) explanation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ensure 2D image
    if image.ndim == 3:
        img_2d = image.squeeze(-1)
    else:
        img_2d = image

    # Original drawing
    axes[0].imshow(img_2d, cmap='gray')
    axes[0].set_title('Your Drawing', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Generated "ideal" digit
    axes[1].imshow(lime_heatmap, cmap='gray')
    axes[1].set_title(f'LIME: Ideal Digit {predicted_class}', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Difference/comparison overlay
    diff = np.abs(img_2d - lime_heatmap)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference Map', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    return fig


def plot_shap_explanation(image, shap_values, predicted_class):
    """Plot SHAP explanation"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ensure 2D arrays
    if image.ndim == 3:
        img_2d = image.squeeze(-1)
    else:
        img_2d = image

    if shap_values.ndim == 3:
        shap_2d = shap_values.squeeze(-1)
    else:
        shap_2d = shap_values

    # Original image
    axes[0].imshow(img_2d, cmap='gray')
    axes[0].set_title('Your Drawing', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # SHAP heatmap
    vmax = np.abs(shap_2d).max()
    im = axes[1].imshow(shap_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'SHAP Values\n(Class {predicted_class})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    shap_abs = np.abs(shap_2d)
    shap_norm = shap_abs / (shap_abs.max() + 1e-10)
    axes[2].imshow(img_2d, cmap='gray', alpha=0.7)
    axes[2].imshow(shap_norm, cmap='hot', alpha=0.5)
    axes[2].set_title('SHAP Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<div class="main-header">✏️ MNIST XAI Drawing App</div>', unsafe_allow_html=True)
    st.markdown("Draw a digit and see real-time prediction with explainable AI analysis!")

    # Load models and analyzers
    with st.spinner("Loading models..."):
        # Load main CNN model
        params, batch_stats, model = load_model()
        lrp = load_xai_analyzers(model)

        # Load CBN model (if available)
        cbn_params, cbn_batch_stats, cbn_model = load_cbn_model()
        cbn_enabled = cbn_params is not None

        # Load LIME and SHAP explainers
        lime_explainer, shap_explainer = load_lime_shap_explainers(model, params, batch_stats)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    canvas_size = st.sidebar.slider("Canvas Size", 200, 400, 280, 20)
    stroke_width = st.sidebar.slider("Brush Size", 10, 50, 20, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Advanced XAI Techniques")

    enable_lrp = st.sidebar.checkbox("Enable Layer Relevance Propagation (LRP)", value=False,
                                     help="Shows pixel-wise relevance scores for predictions")

    enable_cbn = st.sidebar.checkbox("Enable Concept Bottleneck Network", value=False,
                                     help="Shows interpretable concept activations",
                                     disabled=not cbn_enabled)

    enable_lime = st.sidebar.checkbox("Enable LIME Explanations", value=False,
                                      help="Local Interpretable Model-agnostic Explanations",
                                      disabled=lime_explainer is None)

    enable_shap = st.sidebar.checkbox("Enable SHAP Explanations", value=False,
                                      help="SHapley Additive exPlanations",
                                      disabled=shap_explainer is None)

    if enable_lrp or enable_cbn or enable_lime or enable_shap:
        st.sidebar.info("⚠️ Advanced techniques may take longer to compute")

    if not cbn_enabled:
        st.sidebar.warning("⚠️ CBN model not available. Train the model first.")
    if lime_explainer is None and LIME_AVAILABLE:
        st.sidebar.warning("⚠️ LIME not available. Check dependencies.")
    if shap_explainer is None and SHAP_AVAILABLE:
        st.sidebar.warning("⚠️ SHAP not available. Check dependencies.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Instructions")
    st.sidebar.markdown("""
    1. Draw a digit (0-9) in the canvas
    2. Click 'Predict' button
    3. See prediction and XAI analysis
    4. Click 'Clear' to draw again
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Tips for Best Results")
    st.sidebar.markdown("""
    - Draw larger digits
    - Center your drawing
    - Make strokes clear
    - Similar to handwritten style
    """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="sub-header">🎨 Draw Here</div>', unsafe_allow_html=True)

        # Drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key="canvas",
        )

        # Buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

        with btn_col1:
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

        with btn_col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        with btn_col3:
            example_button = st.button("📝 Load Example", use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)

        # Placeholder for results
        result_placeholder = st.empty()

    # Handle example loading
    if example_button:
        st.info("Draw a digit in the canvas to see predictions!")

    # Handle prediction
    if predict_button:
        if canvas_result.image_data is not None:
            # Check if canvas is empty
            if np.sum(canvas_result.image_data[:, :, 3]) < 100:
                st.warning("⚠️ Canvas appears empty! Please draw a digit first.")
            else:
                # Preprocess image
                with st.spinner("Processing image..."):
                    processed_image = preprocess_canvas_image(canvas_result.image_data)

                if processed_image is not None:
                    # Make prediction and analyze
                    with st.spinner("Analyzing..."):
                        results = predict_and_analyze(
                            params, batch_stats, model, processed_image,
                            lrp,
                            use_lrp=enable_lrp,
                            cbn_model=cbn_model,
                            cbn_params=cbn_params,
                            cbn_batch_stats=cbn_batch_stats,
                            use_cbn=enable_cbn,
                            lime_explainer=lime_explainer,
                            use_lime=enable_lime,
                            shap_explainer=shap_explainer,
                            use_shap=enable_shap
                        )

                    # Display results
                    with result_placeholder.container():
                        # Prediction result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="text-align: center; color: #1f77b4;">
                                Predicted Digit: <span style="font-size: 4rem;">{results['predicted_class']}</span>
                            </h2>
                            <h3 style="text-align: center;">
                                Confidence: {results['probabilities'][results['predicted_class']]:.1%}
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)

                        # Tabs for different analyses
                        tab_names = ["🔍 XAI Visualization"]

                        # Add advanced tabs if enabled
                        if enable_lrp:
                            tab_names.append("🧬 LRP Analysis")
                        if enable_cbn:
                            tab_names.append("🧠 CBN Concepts")
                        if enable_lime:
                            tab_names.append("🔦 LIME Analysis")
                        if enable_shap:
                            tab_names.append("🎯 SHAP Analysis")

                        tab_names.extend([
                            "📍 Regional Analysis",
                            "✏️ Stroke Features",
                            "📊 All Probabilities",
                            "💡 Interpretation"
                        ])

                        tabs = st.tabs(tab_names)
                        current_tab = 0

                        # Tab 1: XAI Visualization
                        with tabs[current_tab]:
                            current_tab += 1
                            st.markdown("### Visual Explanations")

                            # Show preprocessed image for debugging
                            st.markdown("**Preprocessed Image (28×28):**")
                            col_a, col_b, col_c = st.columns([1, 1, 2])
                            with col_a:
                                fig_debug = plt.figure(figsize=(3, 3))
                                plt.imshow(processed_image.squeeze(), cmap='gray')
                                plt.title('Model Input', fontsize=10)
                                plt.axis('off')
                                st.pyplot(fig_debug)
                                plt.close()
                            with col_b:
                                st.markdown("""
                                This is what the model sees:
                                - 28×28 pixels
                                - Grayscale
                                - White digit on black background
                                - Centered
                                """)

                            st.markdown("---")

                            fig_xai = plot_xai_visualizations(
                                processed_image,
                                results['gradcam'],
                                results['saliency']
                            )
                            st.pyplot(fig_xai)
                            plt.close()

                            st.markdown("""
                            **How to read:**
                            - **Your Drawing**: Processed to 28×28 MNIST format
                            - **GradCAM** (middle): Shows which regions are important (red = high, blue = low)
                            - **Saliency Map** (right): Shows which pixels matter most (bright = important)
                            """)

                        # LRP Analysis tab (conditional)
                        if enable_lrp:
                            with tabs[current_tab]:
                                current_tab += 1
                                st.markdown("### Layer-wise Relevance Propagation")
                                st.markdown("""
                                LRP decomposes the prediction into pixel-wise relevance scores,
                                showing exactly which pixels contributed to the model's decision.
                                """)

                                fig_lrp = plot_lrp_analysis(
                                    processed_image,
                                    results['lrp_relevance'],
                                    results['lrp_epsilon']
                                )
                                st.pyplot(fig_lrp)
                                plt.close()

                                st.markdown("""
                                **How to read:**
                                - **Warmer colors (red)**: Higher positive relevance (contributed to prediction)
                                - **Cooler colors (blue)**: Lower relevance
                                - **Gradient-based**: Fast approximation using gradients
                                - **Epsilon Rule**: More precise layer-by-layer propagation

                                **Interpretation:** Bright red pixels had the strongest influence on
                                the model predicting this as digit '{}'.
                                """.format(results['predicted_class']))

                        # CBN Concepts tab (conditional)
                        if enable_cbn:
                            with tabs[current_tab]:
                                current_tab += 1
                                st.markdown("### Concept Bottleneck Network Analysis")
                                st.markdown("""
                                CBN learns interpretable visual concepts as intermediate representations.
                                The model makes predictions based on these human-understandable concepts.
                                """)

                                # Plot CBN concepts
                                fig_cbn = plot_cbn_concepts(results['cbn_concepts'], results['predicted_class'])
                                st.pyplot(fig_cbn)
                                plt.close()

                                st.markdown("---")
                                st.markdown("### Active Concepts (>50% activation)")

                                # Show active concepts
                                active_concepts_data = interpret_concepts(results['cbn_concepts'], threshold=0.5)
                                active_concepts_list = [(name, val) for name, val in active_concepts_data.items() if val >= 0.5]

                                if active_concepts_list:
                                    for name, value in sorted(active_concepts_list, key=lambda x: x[1], reverse=True):
                                        st.markdown(f"- **{name}**: {value:.3f} ({'✅ Active' if value >= 0.5 else ''})")
                                else:
                                    st.info("No concepts strongly activated (all below 0.5 threshold)")

                                st.markdown("---")
                                st.markdown("### Top 5 Concepts by Activation")

                                # Show top 5 concepts
                                top_concepts = sorted(active_concepts_data.items(), key=lambda x: x[1], reverse=True)[:5]
                                for i, (name, value) in enumerate(top_concepts, 1):
                                    st.markdown(f"{i}. **{name}**: {value:.3f}")

                                st.markdown("---")
                                st.markdown("### CBN Prediction Comparison")

                                col_cbn_a, col_cbn_b = st.columns(2)

                                with col_cbn_a:
                                    st.markdown("**Standard CNN:**")
                                    st.markdown(f"- Predicted: **{results['predicted_class']}**")
                                    st.markdown(f"- Confidence: **{results['probabilities'][results['predicted_class']]:.2%}**")

                                with col_cbn_b:
                                    cbn_predicted = int(np.argmax(results['cbn_probs']))
                                    st.markdown("**CBN (via concepts):**")
                                    st.markdown(f"- Predicted: **{cbn_predicted}**")
                                    st.markdown(f"- Confidence: **{results['cbn_probs'][cbn_predicted]:.2%}**")

                                if results['predicted_class'] == cbn_predicted:
                                    st.success("✅ Both models agree on the prediction!")
                                else:
                                    st.warning(f"⚠️ Models disagree: CNN says {results['predicted_class']}, CBN says {cbn_predicted}")

                                st.markdown("""
                                **Why CBN is Interpretable:**
                                - Shows which visual concepts (curves, lines, etc.) the model detects
                                - Predictions are based on these interpretable concepts
                                - Helps understand what features the model uses for classification
                                """)

                        # LIME Analysis tab (conditional)
                        if enable_lime:
                            with tabs[current_tab]:
                                current_tab += 1
                                st.markdown("### LIME (Feature Visualization)")
                                st.markdown("""
                                This visualization uses activation maximization to generate an "ideal" version
                                of the predicted digit according to the model. It helps you understand what
                                features the model expects to see for this digit class.
                                """)

                                # Plot LIME explanation
                                fig_lime = plot_lime_explanation(
                                    processed_image,
                                    results['lime_heatmap'],
                                    results['predicted_class']
                                )
                                st.pyplot(fig_lime)
                                plt.close()

                                st.markdown("""
                                **How to read:**
                                - **Your Drawing**: Your input processed to 28×28 MNIST format
                                - **Ideal Digit {}**: What the model considers the "perfect" version of this digit
                                - **Difference Map**: Shows where your drawing differs from the model's ideal (brighter = more different)

                                **Interpretation:** By comparing your drawing to the model's ideal representation,
                                you can see which features the model expects for digit '{}' and how your input
                                matches or differs from those expectations.
                                """.format(results['predicted_class'], results['predicted_class']))

                                # Display optimization score
                                st.markdown("---")
                                st.markdown("### Generation Quality")
                                if 'lime_score' in results:
                                    st.markdown(f"**Activation Score**: {results['lime_score']:.3f}")
                                    st.markdown("""
                                    This score indicates how confidently the model classifies the generated ideal image.
                                    Higher scores mean the model is very confident this is what the digit should look like.
                                    """)

                        # SHAP Analysis tab (conditional)
                        if enable_shap:
                            with tabs[current_tab]:
                                current_tab += 1
                                st.markdown("### SHAP (SHapley Additive exPlanations)")
                                st.markdown("""
                                SHAP values represent the contribution of each pixel to the prediction,
                                based on game theory. They provide a unified measure of feature importance.
                                """)

                                # Plot SHAP explanation
                                fig_shap = plot_shap_explanation(
                                    processed_image,
                                    results['shap_values'],
                                    results['predicted_class']
                                )
                                st.pyplot(fig_shap)
                                plt.close()

                                st.markdown("""
                                **How to read:**
                                - **Red pixels**: Positive contribution (increase prediction probability)
                                - **Blue pixels**: Negative contribution (decrease prediction probability)
                                - **White pixels**: No significant contribution

                                **Interpretation:** SHAP values show the exact contribution of each pixel
                                to predicting digit '{}'. Brighter red means stronger positive contribution.
                                """.format(results['predicted_class']))

                                # SHAP statistics
                                st.markdown("---")
                                st.markdown("### SHAP Value Statistics")

                                shap_array = results['shap_values'].flatten()
                                col_shap_a, col_shap_b, col_shap_c = st.columns(3)

                                with col_shap_a:
                                    st.metric("Max Positive", f"{shap_array.max():.4f}")
                                with col_shap_b:
                                    st.metric("Max Negative", f"{shap_array.min():.4f}")
                                with col_shap_c:
                                    st.metric("Mean Absolute", f"{np.abs(shap_array).mean():.4f}")

                                st.markdown("""
                                **Why SHAP is Useful:**
                                - Theoretically grounded in game theory
                                - Consistent and locally accurate
                                - Provides both positive and negative contributions
                                - Works with any machine learning model
                                """)

                        # Regional Analysis tab
                        with tabs[current_tab]:
                            current_tab += 1
                            st.markdown("### Which Regions Matter Most?")
                            fig_regions = plot_region_importance(results['region_scores'])
                            st.pyplot(fig_regions)
                            plt.close()

                            # Show top regions
                            sorted_regions = sorted(
                                results['region_scores'].items(),
                                key=lambda x: x[1]['avg_importance'],
                                reverse=True
                            )

                            st.markdown("**Top 3 Important Regions:**")
                            for i, (region, scores) in enumerate(sorted_regions[:3], 1):
                                st.markdown(f"{i}. **{region.upper()}**: {scores['avg_importance']:.3f}")

                        # Stroke Features tab
                        with tabs[current_tab]:
                            current_tab += 1
                            st.markdown("### Which Shapes/Strokes Are Important?")
                            fig_strokes = plot_stroke_features(results['stroke_features'])
                            st.pyplot(fig_strokes)
                            plt.close()

                            # Explain dominant feature
                            stroke_names = {
                                'horizontal_strokes': 'Horizontal Strokes',
                                'vertical_strokes': 'Vertical Strokes',
                                'curves': 'Curves/Loops',
                                'intersections': 'Intersections'
                            }

                            top_stroke = max(
                                results['stroke_features'].items(),
                                key=lambda x: x[1]
                            )

                            st.markdown(f"""
                            **Dominant Feature:** {stroke_names[top_stroke[0]]}
                            **Importance:** {top_stroke[1]:.3f}
                            """)

                            # Expected patterns for predicted digit
                            digit_patterns = {
                                0: "Expected: Curves (circular loop)",
                                1: "Expected: Vertical strokes",
                                2: "Expected: Curves + Horizontal strokes",
                                3: "Expected: Curves (stacked)",
                                4: "Expected: Intersections + Vertical strokes",
                                5: "Expected: Horizontal strokes + Curves",
                                6: "Expected: Curves (loop with stem)",
                                7: "Expected: Horizontal + Vertical strokes",
                                8: "Expected: Curves + Intersections (two loops)",
                                9: "Expected: Curves (top loop with stem)"
                            }

                            st.info(f"ℹ️ {digit_patterns[results['predicted_class']]}")

                        # All Probabilities tab
                        with tabs[current_tab]:
                            current_tab += 1
                            st.markdown("### Prediction Confidence for All Digits")
                            fig_probs = plot_probability_bars(results['probabilities'])
                            st.pyplot(fig_probs)
                            plt.close()

                            # Show top 3
                            top3_idx = np.argsort(results['probabilities'])[-3:][::-1]

                            st.markdown("**Top 3 Predictions:**")
                            for i, idx in enumerate(top3_idx, 1):
                                st.markdown(
                                    f"{i}. Digit **{idx}**: {results['probabilities'][idx]:.2%}"
                                )

                        # Interpretation tab
                        with tabs[current_tab]:
                            current_tab += 1
                            st.markdown("### Shape Interpretation")

                            # Generate interpretation
                            interpretation = generate_interpretation(
                                results['predicted_class'],
                                results['region_scores'],
                                results['stroke_features'],
                                results['probabilities']
                            )

                            st.markdown(f'<div class="interpretation-box">{interpretation}</div>',
                                      unsafe_allow_html=True)

                            # Additional insights
                            st.markdown("---")
                            st.markdown("### 🔬 Detailed Insights")

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.markdown("**Spatial Focus:**")
                                top_region = max(
                                    results['region_scores'].items(),
                                    key=lambda x: x[1]['avg_importance']
                                )
                                st.markdown(f"- Primary region: **{top_region[0]}**")
                                st.markdown(f"- Importance: **{top_region[1]['avg_importance']:.3f}**")

                            with col_b:
                                st.markdown("**Shape Features:**")
                                top_stroke = max(
                                    results['stroke_features'].items(),
                                    key=lambda x: x[1]
                                )
                                stroke_names = {
                                    'horizontal_strokes': 'Horizontal lines',
                                    'vertical_strokes': 'Vertical lines',
                                    'curves': 'Curves/loops',
                                    'intersections': 'Crossing points'
                                }
                                st.markdown(f"- Key feature: **{stroke_names[top_stroke[0]]}**")
                                st.markdown(f"- Importance: **{top_stroke[1]:.3f}**")

                            # Why this prediction?
                            st.markdown("---")
                            st.markdown("### 🎯 Why This Prediction?")

                            confidence = results['probabilities'][results['predicted_class']]

                            if confidence > 0.9:
                                st.success(f"""
                                ✅ **High Confidence ({confidence:.1%})**
                                The model found clear, distinctive features of digit '{results['predicted_class']}'.
                                The shape matches expected patterns very well.
                                """)
                            elif confidence > 0.7:
                                st.info(f"""
                                ℹ️ **Good Confidence ({confidence:.1%})**
                                The model identified key features of digit '{results['predicted_class']}'.
                                Some features may be ambiguous or similar to other digits.
                                """)
                            else:
                                st.warning(f"""
                                ⚠️ **Lower Confidence ({confidence:.1%})**
                                The model is uncertain. The digit may be:
                                - Ambiguously drawn
                                - Similar to multiple digits
                                - Missing key distinctive features

                                Try drawing more clearly or check top alternatives above.
                                """)

                        # Download results
                        st.markdown("---")
                        if st.button("💾 Save Analysis Report"):
                            # Create a simple text report
                            report = f"""
MNIST XAI Analysis Report
========================

Predicted Digit: {results['predicted_class']}
Confidence: {results['probabilities'][results['predicted_class']]:.2%}

Top 3 Predictions:
"""
                            top3_idx = np.argsort(results['probabilities'])[-3:][::-1]
                            for i, idx in enumerate(top3_idx, 1):
                                report += f"{i}. Digit {idx}: {results['probabilities'][idx]:.2%}\n"

                            report += f"\n{interpretation}"

                            st.download_button(
                                label="📄 Download Report",
                                data=report,
                                file_name=f"mnist_analysis_digit_{results['predicted_class']}.txt",
                                mime="text/plain"
                            )
        else:
            st.warning("⚠️ No drawing detected. Please draw a digit first!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using JAX, Flax, and Streamlit</p>
        <p>🔍 Explainable AI powered by GradCAM, Saliency Maps, LRP, Concept Bottleneck Networks, LIME, and SHAP</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
