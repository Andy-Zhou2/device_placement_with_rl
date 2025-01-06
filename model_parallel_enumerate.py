import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

print("TensorFlow version:", tf.__version__)
# tf.debugging.set_log_device_placement(True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

import time

devices = ['/CPU:0', '/GPU:0']
configurations = [
    ('/GPU:0', '/GPU:0', '/GPU:0'),
    ('/CPU:0', '/CPU:0', '/CPU:0'),
    ('/CPU:0', '/CPU:0', '/GPU:0'),
    ('/CPU:0', '/GPU:0', '/GPU:0'),
    ('/GPU:0', '/GPU:0', '/CPU:0')
]

results = []

for config in configurations:
    with tf.device(config[0]):
        flatten = Flatten(input_shape=(28, 28))
    with tf.device(config[1]):
        dense1 = Dense(128, activation='relu')
    with tf.device(config[2]):
        dense2 = Dense(64, activation='relu')
        output_layer = Dense(10, activation='softmax')

    model = Sequential([flatten, dense1, dense2, output_layer])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    model.fit(x_train, y_train, epochs=2, verbose=1)
    end_time = time.time()

    results.append((config, end_time - start_time))

# Print results
for config, time_taken in results:
    print(f"Config: {config}, Time: {time_taken:.2f} seconds")

