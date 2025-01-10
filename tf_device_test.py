import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# Print TensorFlow version and list of physical devices for clarity
print("TensorFlow version:", tf.__version__)
print("Available devices:")
for device in tf.config.list_physical_devices():
    print("  -", device)

# Create two constant tensors
a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)

print("a device:", a.device)
print("b device:", b.device)

# Compute a + b on the CPU
with tf.device('/CPU:0'):
    sum_ab = a + b

# Compute a - b on the GPU
# Note: Make sure a GPU device is available.
with tf.device('/GPU:0'):
    diff_ab = a - b

print("a + b (computed on CPU):", sum_ab.numpy())
print("a - b (computed on GPU):", diff_ab.numpy())
