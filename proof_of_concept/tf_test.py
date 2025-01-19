import tensorflow as tf
import time
import matplotlib.pyplot as plt
import multiprocessing


# Define the model with layers on different devices
class ConfigurableModel(tf.Module):
    def __init__(self, device_w1, device_w2, device_dense1, device_dense2):
        super().__init__()
        self.device_w1 = device_w1
        self.device_w2 = device_w2
        self.device_dense1 = device_dense1
        self.device_dense2 = device_dense2

        D = 300000

        with tf.device(self.device_w1):
            self.w1 = tf.Variable(tf.random.normal([10, D]), name='w1')
            self.b1 = tf.Variable(tf.zeros([D]), name='b1')

        with tf.device(self.device_w2):
            self.w2 = tf.Variable(tf.random.normal([D, 1]), name='w2')
            self.b2 = tf.Variable(tf.zeros([1]), name='b2')

    # @tf.function
    def __call__(self, x):
        with tf.device(self.device_dense1):
            x = tf.matmul(x, self.w1) + self.b1
            x = tf.nn.relu(x)

        with tf.device(self.device_dense2):
            x = tf.matmul(x, self.w2) + self.b2

        return x


# Create the artificial dataset
def create_dataset(num_samples=1000):
    return tf.random.normal([num_samples, 10])


# Define configurations as a list of device placements
configs = [
    ["/CPU:0", "/GPU:0", "/GPU:0", "/CPU:0", "/CPU:0"],
    ["/CPU:0", "/CPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    # ["/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0"],
    # ["/CPU:0", "/GPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    # ["/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0", "/GPU:0"],
    # ["/GPU:0", "/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0"],
    # ["/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0"],
    # ["/GPU:0", "/GPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    # ["/CPU:0", "/CPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
]


def run_with_config(config, queue):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=16_000)]  # Limit to 1024MB
            )
        except RuntimeError as e:
            print(e)

    print(f'testing config: {config}')
    device_x, device_w1, device_w2, device_dense1, device_dense2 = config
    model = ConfigurableModel(device_w1, device_w2, device_dense1, device_dense2)

    with tf.device(device_x):
        X = create_dataset(num_samples=1000)

    times = []
    for i in range(1000):
        start_time = time.time()
        _ = model(X)
        tf.test.experimental.sync_devices()
        end_time = time.time()

        if i > 0:  # Discard the first measurement
            times.append(end_time - start_time)

    average_time = sum(times) / len(times)
    queue.put((config, average_time))
    print(f"Config: {config} -> Average Time: {average_time} seconds")


if __name__ == "__main__":
    results = []
    queue = multiprocessing.Queue()

    for config in configs:
        p = multiprocessing.Process(target=run_with_config, args=(config, queue))
        p.start()

        # Wait for the process to complete or timeout after 10 seconds
        p.join(timeout=1000)

        if p.is_alive():
            print(f"Process for config {config} timed out. Terminating...")
            p.terminate()
            p.join()
            queue.put((config, 10))  # Assign value of 10 for timeout
        elif p.exitcode != 0:
            print(f"Process for config {config} failed with exit code {p.exitcode}. Assigning value 10.")
            queue.put((config, 10))  # Assign value of 10 for failure

    # Collect results from the queue
    while not queue.empty():
        results.append(queue.get())

    print("Final Results:", results)

