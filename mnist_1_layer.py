import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import numpy as np
import time

tf.debugging.set_log_device_placement(True)


BATCH_SIZE = 2048
EPOCHS = 1
NUM_BATCHES = 1

# Random input data
num_samples = BATCH_SIZE * NUM_BATCHES
input_shape = (28, 28)
num_classes = 10

with tf.device('/CPU:0'):
    x_train = np.random.rand(num_samples, *input_shape)
    y_train = np.random.randint(0, num_classes, num_samples)

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)

    input_layer = InputLayer(shape=input_shape)


# Set device for each layer
with tf.device('/GPU:0'):
    print('-' * 100)
    print(f'x_train device: {x_train.device}')
    flatten = Flatten()
    dense1 = Dense(702400, activation='relu', name='dense1')
with tf.device('/CPU:0'):
    init = tf.keras.initializers.RandomNormal()
    def kernel_init(shape, dtype=None):
        with tf.device('/CPU:0'):
            return init(shape, dtype=dtype)
    dense2 = Dense(
        5120,
        activation='relu',
        name='dense2',
        kernel_initializer=kernel_init
    )
    output_layer = Dense(num_classes, activation='softmax', name='output_layer')

# Build the model
print('S-'*100)
model = Sequential([input_layer, flatten, dense1, dense2, output_layer])
print('C-'*100)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print('-'*100)

model.evaluate(x_train, y_train, batch_size=BATCH_SIZE)




