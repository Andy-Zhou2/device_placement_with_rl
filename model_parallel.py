import tensorflow as tf

print("TensorFlow version:", tf.__version__)
tf.debugging.set_log_device_placement(True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

with tf.device('/CPU:0'):  # Assign to CPU
    flatten = tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten')
    dense_1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')

with tf.device('/GPU:0'):  # Assign to GPU
    dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')

with tf.device('/CPU:0'):  # Assign output layer back to CPU
    output_layer = tf.keras.layers.Dense(10, activation='softmax', name='output')

# Build the model
model = tf.keras.Sequential([flatten, dense_1, dense_2, output_layer])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test,  y_test, verbose=2)
