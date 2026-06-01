import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#%%  ### ARTIFICIAL NEURAL NETWORK FOR BINARY CLASSIFICATION MODEL ###

def ann_classification (num_components, size):
    """
     Artificial Neural Network for binary classification:
         - num_components: the number of input bands 
         - size: 'S', 'M' or 'L' 
    """
    
    model = Sequential()
    
    if size=='S':
        # SMALL NETWORK 
        model.add(Dense(16, activation='relu', input_shape=(num_components,)))
        model.add(Dense(8, activation='relu')) 
    
    elif size=='M':
        # MEDIUM NETWORK 
        model.add(Dense(64, activation='relu', input_shape=(num_components,)))
        model.add(Dense(32, activation='relu'))
    
    elif size=='L':
        # LARGE NETWORK 
        model.add(Dense(128, activation='relu', input_shape=(num_components,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(16, activation='relu'))
    
    else:
        raise ValueError("Size must be 'S', 'M', or 'L'")
        
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=[
                      'accuracy', 
                      tf.keras.metrics.Recall(name='sensitivity'),
                      tf.keras.metrics.AUC(name='auc')
                  ])
    return model

# %%

my_model = ann_classification(40, "M")

my_model.summary()