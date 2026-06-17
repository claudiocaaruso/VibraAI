import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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
