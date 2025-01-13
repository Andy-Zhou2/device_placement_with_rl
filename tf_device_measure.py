import time
import numpy as np
import tensorflow as tf


DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    "/GPU:1",
]
NUM_DEVICE_CHOICES = len(DEVICE_OPTIONS)


class ConfigurableModel(tf.Module):
    """
    A TF model with 2 dense layers. The device placements
    come from a 3-token output: (device_x, device_dense1, device_dense2).
    """

    def __init__(self, device_dense1, device_dense2):
        super().__init__()
        # We'll fix D for demonstration
        self.D = 300000

        # Create variables on the chosen devices
        with tf.device(device_dense1):
            self.w1 = tf.Variable(tf.random.normal([10, self.D]), name='w1')
            self.b1 = tf.Variable(tf.zeros([self.D]), name='b1')

        with tf.device(device_dense2):
            self.w2 = tf.Variable(tf.random.normal([self.D, 1]), name='w2')
            self.b2 = tf.Variable(tf.zeros([1]), name='b2')

    def __call__(self, x, device_dense1, device_dense2):
        # Dense1 on device_dense1
        with tf.device(device_dense1):
            x = tf.matmul(x, self.w1) + self.b1
            x = tf.nn.relu(x)

        # Dense2 on device_dense2
        with tf.device(device_dense2):
            x = tf.matmul(x, self.w2) + self.b2

        return x


def create_dataset(num_samples=1000):
    """
    Create a random dataset (e.g., on /GPU:0).
    """
    return tf.random.normal([num_samples, 10])


def measure_inference_time_3devices(devices, n_warmup=1, n_iters=10, batch_size=1000):
    """
    Given a triple of devices: (device_x, device_dense1, device_dense2),
    measure the average inference time.
    """
    devices = [DEVICE_OPTIONS[d] for d in devices]
    device_x, device_d1, device_d2 = devices
    with tf.device(device_x):
        X = create_dataset(num_samples=batch_size)
    model = ConfigurableModel(device_d1, device_d2)

    times = []
    for i in range(n_iters):
        start = time.time()
        _ = model(X, device_d1, device_d2)
        tf.test.experimental.sync_devices()  # For GPU timing
        end = time.time()
        if i >= n_warmup:
            times.append(end - start)

    if len(times) == 0:
        return 0.0
    return sum(times) / len(times)


# We'll define 3 operations: [X, dense1, dense2].
# - Adjacency: X->dense1->dense2
ADJ_FORWARD = np.array([
    [0, 1, 0],  # X feeds dense1
    [0, 0, 1],  # dense1 feeds dense2
    [0, 0, 0],  # dense2 feeds nothing
], dtype=np.float32)

ADJ_BACKWARD = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float32)

# Operation types (just for demonstration)
OP_TYPES = [0, 1, 1]  # X is type 0, dense1 is type 1, dense2 is type 1

# Output shapes (toy example)
#   X: (batch, 10)
#   dense1: (batch, D)
#   dense2: (batch, 1)
# We'll zero-pad them to length 4 for demonstration: e.g. [batch, 10, 0, 0]
SHAPES = [
    [1000, 10, 0, 0],  # X
    [1000, 300000, 0, 0],  # dense1
    [1000, 1, 0, 0],  # dense2
]