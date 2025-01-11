import tensorflow as tf
import time
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8000),
         tf.config.LogicalDeviceConfiguration(memory_limit=8000),]
    )
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


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
    def __call__(self, x, device_x):
        with tf.device(device_x):
            x = tf.identity(x)  # Ensure input starts on the specified device

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
    ["/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0", "/GPU:0"],
    ["/GPU:1", "/GPU:1", "/GPU:1", "/GPU:1", "/GPU:1"],
    ["/GPU:1", "/GPU:1", "/GPU:0", "/GPU:1", "/GPU:0"],
    ["/CPU:0", "/CPU:0", "/CPU:0", "/CPU:0", "/CPU:0"],
    ["/CPU:0", "/GPU:0", "/CPU:0", "/GPU:0", "/GPU:0"],
    ["/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0", "/GPU:0"],
    ["/GPU:0", "/CPU:0", "/GPU:0", "/CPU:0", "/CPU:0"],
]

# Test each configuration and measure the time
with tf.device('/GPU:1'):
    X = create_dataset(num_samples=1000)
print('X:', X.device)
results = []

for config in configs:
    device_x, device_w1, device_w2, device_dense1, device_dense2 = config
    model = ConfigurableModel(device_w1, device_w2, device_dense1, device_dense2)

    times = []
    for i in range(100):
        start_time = time.time()
        _ = model(X, device_x)
        tf.test.experimental.sync_devices()
        end_time = time.time()

        if i > 0:  # Discard the first measurement
            times.append(end_time - start_time)

    # plot times
    plt.plot(times)
    # plt.title(f"Configuration: {config}")
    # plt.xlabel("Iteration")
    # plt.ylabel("Time (s)")
    plt.savefig(f"config.png")

    average_time = sum(times) / len(times)
    results.append((config, average_time))
    print(f"Config: {config} -> Average Time: {average_time:.4f} seconds")

# Display all results
# for config, avg_time in results:
#     print(f"Configuration {config} took {avg_time:.4f} seconds on average for inference.")
