"""
Training script for MNIST CNN with Flax
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
import numpy as np
from flax.training import train_state
from flax import struct
import tensorflow_datasets as tfds
from typing import Any, Tuple
import pickle
from tqdm import tqdm
import tensorflow as tf

from mnist_cnn_model import MNISTCNN, initialize_model


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics for BatchNorm and RNG key"""
    batch_stats: Any = None
    dropout_rng: Any = None


def create_train_state(rng, learning_rate: float = 0.001):
    """
    Create initial training state

    Args:
        rng: JAX random key
        learning_rate: Learning rate for optimizer

    Returns:
        state: Training state with parameters, optimizer, and batch stats
    """
    rng, init_rng, dropout_rng = random.split(rng, 3)
    params, batch_stats, model = initialize_model(init_rng, input_shape=(1, 28, 28, 1))

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dropout_rng=dropout_rng
    ), model


def load_mnist_data(batch_size: int = 128, train_split: float = 0.9):
    """
    Load and preprocess MNIST dataset

    Args:
        batch_size: Batch size for training
        train_split: Fraction of training data to use for training (rest for validation)

    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        info: Dataset info
    """
    def normalize_img(data):
        """Normalize images to [0, 1] range"""
        image = tf.cast(data['image'], tf.float32) / 255.0
        label = data['label']
        return {'image': image, 'label': label}

    # Load dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()

    # Split training data into train and validation
    train_split_str = f'train[:{int(train_split * 100)}%]'
    val_split_str = f'train[{int(train_split * 100)}%:]'

    train_ds = tfds.load(
        'mnist',
        split=train_split_str,
        as_supervised=False,
        shuffle_files=True
    )

    val_ds = tfds.load(
        'mnist',
        split=val_split_str,
        as_supervised=False,
        shuffle_files=False
    )

    test_ds = tfds.load(
        'mnist',
        split='test',
        as_supervised=False,
        shuffle_files=False
    )

    # Prepare datasets
    train_ds = train_ds.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(10000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        normalize_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=False)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    info = ds_builder.info

    return train_ds, val_ds, test_ds, info


def compute_metrics(logits, labels):
    """Compute accuracy and loss"""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


@jax.jit
def train_step(state, batch):
    """
    Single training step

    Args:
        state: Training state
        batch: Batch of data

    Returns:
        state: Updated training state
        metrics: Training metrics
    """
    # Split RNG for dropout
    dropout_rng, new_dropout_rng = random.split(state.dropout_rng)

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch['image'],
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
        dropout_rng=new_dropout_rng
    )

    metrics = compute_metrics(logits, batch['label'])
    return state, metrics


@jax.jit
def eval_step(state, batch):
    """
    Single evaluation step

    Args:
        state: Training state
        batch: Batch of data

    Returns:
        metrics: Evaluation metrics
    """
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['image'], training=False)
    return compute_metrics(logits, batch['label'])


def train_epoch(state, train_ds, epoch):
    """Train for a single epoch"""
    batch_metrics = []

    pbar = tqdm(train_ds.as_numpy_iterator(), desc=f'Epoch {epoch}')
    for batch in pbar:
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['accuracy']:.4f}"
        })

    # Compute mean metrics
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics


def evaluate_model(state, test_ds):
    """Evaluate model on test set"""
    batch_metrics = []

    for batch in test_ds.as_numpy_iterator():
        metrics = eval_step(state, batch)
        batch_metrics.append(metrics)

    # Compute mean metrics
    batch_metrics_np = jax.device_get(batch_metrics)
    eval_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return eval_metrics


def train_model(num_epochs: int = 20, batch_size: int = 128, learning_rate: float = 0.001,
                save_path: str = 'mnist_model.pkl'):
    """
    Full training loop

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save trained model

    Returns:
        state: Final training state
        history: Training history
    """
    # Initialize
    rng = random.PRNGKey(0)
    state, model = create_train_state(rng, learning_rate)

    # Load data
    print("Loading MNIST dataset...")
    train_ds, val_ds, test_ds, info = load_mnist_data(batch_size)

    print(f"Training samples: {info.splits['train'].num_examples}")
    print(f"Test samples: {info.splits['test'].num_examples}")
    print(f"Classes: {info.features['label'].num_classes}")

    # Training loop
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        state, train_metrics = train_epoch(state, train_ds, epoch)

        # Validate
        val_metrics = evaluate_model(state, val_ds)

        # Log metrics
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_accuracy'].append(float(train_metrics['accuracy']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_accuracy'].append(float(val_metrics['accuracy']))

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"  New best validation accuracy: {best_val_acc:.4f}")

            # Save model
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'batch_stats': state.batch_stats,
                    'epoch': epoch,
                    'val_accuracy': best_val_acc
                }, f)

    # Final test evaluation
    print("\n" + "="*50)
    print("Final Test Evaluation:")
    test_metrics = evaluate_model(state, test_ds)
    print(f"  Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print("="*50)

    return state, history, model


if __name__ == '__main__':
    # Train model
    state, history, model = train_model(
        num_epochs=20,
        batch_size=128,
        learning_rate=0.001,
        save_path='mnist_model.pkl'
    )

    # Plot training history
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_accuracy'], label='Train')
    ax2.plot(history['val_accuracy'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('mnist_training_history.png', dpi=150)
    plt.show()

    print("\nTraining complete! Model saved to 'mnist_model.pkl'")
    print("Training history plot saved to 'mnist_training_history.png'")
