import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import numpy as np
import time

print("TensorFlow version:", tf.__version__)

# tf.debugging.set_log_device_placement(True)
tf.config.set_soft_device_placement(False)

BATCH_SIZE = 64
EPOCHS = 1
NUM_BATCHES = 100

# Random input data
num_samples = BATCH_SIZE * NUM_BATCHES
input_shape = (28, 28, 1)  # Updated to include channel dimension
num_classes = 10

x_train = np.random.rand(num_samples, *input_shape).astype(np.float32)
y_train = np.random.randint(0, num_classes, num_samples).astype(np.int32)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_train_tensor = tf.convert_to_tensor(x_train)
y_train_tensor = tf.convert_to_tensor(y_train)

print(f'x_train device: {x_train_tensor.device}')
print(f'y_train device: {y_train_tensor.device}')

# with tf.device('/GPU:0'):
#
#     # Generate random data directly in TensorFlow
#     x_train_tensor = tf.random.uniform(
#         shape=(num_samples, *input_shape),
#         minval=0.0,
#         maxval=1.0,
#         dtype=tf.float32
#     )
#     y_train_tensor = tf.random.uniform(
#         shape=(num_samples,),
#         minval=0,
#         maxval=num_classes,
#         dtype=tf.int32
#     )

print(x_train_tensor.shape)  # (x, 28, 28, 1)
print(y_train_tensor.shape)  # (x, 10)
# print(f'x_train device: {x_train_tensor.device}')
# print(f'y_train device: {y_train_tensor.device}')


def train_with_config(config):
    print('\n'* 10)
    print(f"Testing configuration: {config}")
    total_time_this_config = 0
    count_batch = 0

    # Custom callback to log batch times
    class BatchTimeLogger(Callback):
        def on_train_batch_begin(self, batch, logs=None):
            self.batch_start_time = time.time()

        def on_train_batch_end(self, batch, logs=None):
            nonlocal total_time_this_config, count_batch
            batch_time = time.time() - self.batch_start_time
            if count_batch > 0:  # Skip the first batch to avoid warm-up effects
                total_time_this_config += batch_time
            count_batch += 1

    # Set device for each layer
    with tf.device(config[0]):
        conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
    with tf.device(config[1]):
        conv2 = Conv2D(64, (3, 3), activation='relu')
    with tf.device(config[2]):
        flatten = Flatten()
        dense1 = Dense(2048, activation='relu')
        dense2 = Dense(4096, activation='relu')
    with tf.device(config[3]):
        dense3 = Dense(1024, activation='relu')
    with tf.device(config[4]):
        dense4 = Dense(512, activation='relu')
        output_layer = Dense(num_classes, activation='softmax')

    # Build the model
    model = Sequential([conv1, conv2, flatten, dense1, dense2, dense3, dense4, output_layer])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # Define the model
    # model = models.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(2048, activation='relu'),
    #     layers.Dense(4096, activation='relu'),
    #     layers.Dense(1024, activation='relu'),
    #     layers.Dense(512, activation='relu'),
    #     layers.Dense(10, activation='softmax')
    # ])
    #
    #
    # # Compile the model
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # Measure time taken for training
    start_time = time.time()
    model.fit(x_train_tensor, y_train_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[BatchTimeLogger()])
    end_time = time.time()

    time_taken = end_time - start_time
    print(
        f"Time taken for this configuration: {time_taken:.6f} seconds, Total time taken for batches: {total_time_this_config:.6f} seconds\n")
    return config, total_time_this_config, time_taken


# List of configurations
devices = ['/CPU:0', '/GPU:0']
configurations = [
    ('/GPU:0', '/GPU:0', '/GPU:0', '/GPU:0', '/GPU:0'),
    ('/CPU:0', '/CPU:0', '/CPU:0', '/CPU:0', '/CPU:0'),
    ('/CPU:0', '/CPU:0', '/GPU:0', '/GPU:0', '/GPU:0'),
    ('/CPU:0', '/GPU:0', '/GPU:0', '/GPU:0', '/CPU:0'),
    ('/GPU:0', '/GPU:0', '/CPU:0', '/CPU:0', '/GPU:0')
]

# Run tests
results = []
for config in configurations:
    results.append(train_with_config(config))

# Print results
for config, total_time_this_config, time_taken in results:
    print(
        f"Config: {config}, Time for {EPOCHS} epochs: {total_time_this_config:.6f} seconds, Total time: {time_taken:.6f} seconds")
