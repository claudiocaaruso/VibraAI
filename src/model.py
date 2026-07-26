import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Hidden-layer widths per architecture. Kept in sync with `ann_classification`
# below so `count_params` can compute trainable-parameter counts analytically,
# without building a model — used for grid-search complexity-vs-performance plots.
ARCHITECTURE_LAYERS = {'S': [16, 8], 'M': [64, 32], 'L': [128, 64, 16]}


def count_params(size, num_components):
    """Trainable-parameter count for `ann_classification(num_components, size)`."""
    sizes = [num_components] + ARCHITECTURE_LAYERS[size] + [1]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def ann_classification(num_components, size):
    """
    Artificial Neural Network for binary classification.

    Parameters
    ----------
    num_components : int
        Number of input PCA components.
    size : str
        Architecture size: 'S' (small), 'M' (medium), or 'L' (large).
    """
    model = Sequential()

    if size == 'S':
        model.add(Dense(16, activation='relu', input_shape=(num_components,)))
        model.add(Dense(8,  activation='relu'))
    elif size == 'M':
        model.add(Dense(64, activation='relu', input_shape=(num_components,)))
        model.add(Dense(32, activation='relu'))
    elif size == 'L':
        model.add(Dense(128, activation='relu', input_shape=(num_components,)))
        model.add(Dense(64,  activation='relu'))
        model.add(Dense(16,  activation='relu'))
    else:
        raise ValueError("size must be 'S', 'M', or 'L'")

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.AUC(name='auc'),
        ],
    )
    return model
