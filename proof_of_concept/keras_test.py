import tensorflow as tf
import time
import matplotlib.pyplot as plt


# Define the model with layers on different devices
class ConfigurableModel(tf.keras.Model):
    def __init__(self, device_w1, device_w2, device_dense1, device_dense2):
        super().__init__()
        self.device_w1 = device_w1
        self.device_w2 = device_w2
        self.device_dense1 = device_dense1
        self.device_dense2 = device_dense2

        D = 300000

        with tf.device(self.device_w1):
            self.dense1 = tf.keras.layers.Dense(D, activation=None)

        with tf.device(self.device_w2):
            self.dense2 = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        with tf.device(self.device_dense1):
            x = self.dense1(x)
            x = tf.nn.relu(x)

        with tf.device(self.device_dense2):
            x = self.dense2(x)

        return x


# Create the artificial dataset
def create_dataset(num_samples=1000):
    return tf.random.normal([num_samples, 10])


# Define configurations as a list of device placements
configs = [
    # ["/CPU:0", "/CPU:0", "/CPU:0", "/CPU:0", "/CPU:0"],
    # ["/CPU:0", "/CPU:0", "/GPU:0", "/CPU:0", "/GPU:0"],
    ["/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0"],
    ["/GPU:0", "/GPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    ["/CPU:0", "/CPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    # ["/CPU:0", "/GPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    # ["/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0", "/GPU:0"],
    # ["/GPU:0", "/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0"],
]

results = []

for config in configs:
    device_x, device_w1, device_w2, device_dense1, device_dense2 = config
    model = ConfigurableModel(device_w1, device_w2, device_dense1, device_dense2)

    with tf.device(device_x):
        X = create_dataset(num_samples=1000)

    try:
        times = []
        for i in range(100):
            start_time = time.time()
            _ = model(X)
            tf.test.experimental.sync_devices()
            end_time = time.time()

            if i > 0:  # Discard the first measurement
                times.append(end_time - start_time)

        average_time = sum(times) / len(times)
        results.append((config, average_time))
        print(f"Config: {config} -> Average Time: {average_time} seconds")

        del model, X
    except Exception as e:
        print(f"Config: {config} -> Error: {e}")
