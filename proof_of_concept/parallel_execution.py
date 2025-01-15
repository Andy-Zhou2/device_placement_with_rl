import tensorflow as tf
import time

# Define the random matrix
random_matrix = tf.random.uniform((10000, 10000), dtype=tf.float32)

# Define Dense layers
dense1 = tf.keras.layers.Dense(10000, activation='relu')
dense2 = tf.keras.layers.Dense(10000, activation='relu')

# @tf.function
def parallel_execution_test(x):
    with tf.device('/GPU:0'):
        x1 = dense1(x)
    with tf.device('/GPU:0'):
        x2 = dense2(x)
    return x1 + x2


# Input tensor
x = tf.random.uniform((1, 10000), dtype=tf.float32)

# Number of iterations
N = 1000

parallel_execution_test(x)  # Warm up

time_start = time.time()
for _ in range(N):
    result = parallel_execution_test(x)
    tf.test.experimental.sync_devices()
time_end = time.time()
print(f"Time taken for {N} iterations: {time_end - time_start} seconds")

# no f: 1.1718361377716064 seconds
# f: 1.090980052947998 seconds