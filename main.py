import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import numpy as np
import time

print("TensorFlow version:", tf.__version__)

BATCH_SIZE = 2048
EPOCHS = 1
NUM_BATCHES = 100

# Random input data
num_samples = BATCH_SIZE * NUM_BATCHES
input_shape = (28, 28)
num_classes = 10

x_train = np.random.rand(num_samples, *input_shape)
y_train = np.random.randint(0, num_classes, num_samples)

devices = ['/CPU:0', '/GPU:0']
configurations = [
    ('/GPU:0', '/GPU:0', '/GPU:0'),
    ('/CPU:0', '/CPU:0', '/CPU:0'),
    ('/CPU:0', '/CPU:0', '/GPU:0'),
    ('/CPU:0', '/GPU:0', '/GPU:0'),
    ('/GPU:0', '/GPU:0', '/CPU:0')
]

results = []


# Custom callback to log batch times
class BatchTimeLogger(Callback):
    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        global total_time_this_config, count_batch
        if count_batch == 0:
            count_batch += 1
        else:
            total_time_this_config += batch_time
        # print(f"Batch {batch + 1} - Time taken: {batch_time:.6f} seconds")


for config in configurations:
    print(f"Testing configuration: {config}")
    total_time_this_config = 0
    count_batch = 0

    # Set device for each layer
    with tf.device(config[0]):
        flatten = Flatten(input_shape=input_shape)
        dense1 = Dense(1024, activation='relu')
    with tf.device(config[1]):
        dense2 = Dense(512, activation='relu')
    with tf.device(config[2]):
        output_layer = Dense(num_classes, activation='softmax')

    # Build the model
    model = Sequential([flatten, dense1, dense2, output_layer])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Measure time taken for training
    start_time = time.time()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[BatchTimeLogger()])
    end_time = time.time()

    # Record results
    time_taken = end_time - start_time
    results.append((config, total_time_this_config, time_taken))
    print(f"Time taken for this configuration: {time_taken:.6f} seconds, Total time taken: {total_time_this_config:.6f} seconds\n")

# Print results
for config, total_time_this_config, time_taken in results:
    print(f"Config: {config}, Time for {EPOCHS} epochs: {total_time_this_config:.6f} seconds, Total time: {time_taken:.6f} seconds")
