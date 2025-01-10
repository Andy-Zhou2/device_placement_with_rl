import tensorflow as tf

# Enable logging of device placement
tf.debugging.set_log_device_placement(True)

# Create the TensorBoard log directory
log_dir = "./logs/device_placement"
writer = tf.summary.create_file_writer(log_dir)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Create a simple model
input_dim = 10
output_dim = 1

with tf.device('/GPU:0'):
    inputs = tf.keras.Input(shape=(input_dim,), name='input_layer')
    x = tf.keras.layers.Dense(32, activation='relu', name='dense_on_gpu0')(inputs)

with tf.device('/GPU:1'):
    x = tf.keras.layers.Dense(16, activation='relu', name='dense_on_gpu1')(x)
    outputs = tf.keras.layers.Dense(output_dim, name='output_layer')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Create some synthetic data
num_samples = 100
x_data = tf.random.normal(shape=(num_samples, input_dim))
y_data = tf.random.normal(shape=(num_samples, output_dim))

# Wrap training in a function to log the graph
@tf.function
def train_step(x, y):
    model.fit(x, y, batch_size=32, epochs=1, steps_per_epoch=1)

# Log the graph to TensorBoard
with writer.as_default():
    tf.summary.graph(tf.compat.v1.get_default_graph())

# Run a single training step
train_step(x_data, y_data)

print("Single training iteration complete. Check TensorBoard for device placement visualization.")
