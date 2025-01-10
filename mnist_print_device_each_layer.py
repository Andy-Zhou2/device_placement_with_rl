import tensorflow as tf

# fail


tf.config.optimizer.set_jit(False)  # Disable XLA JIT

# ----------------------------
# 1. Define a custom layer to print the device
# ----------------------------
class PrintDeviceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Prints the device of the input tensor.
        # On GPU it might say: /job:localhost/replica:0/task:0/device:GPU:0
        # On CPU it might say: /job:localhost/replica:0/task:0/device:CPU:0
        tf.print("Device of x:", inputs.device)
        return inputs


# ----------------------------
# 2. Load and preprocess the MNIST dataset
# ----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshape data to (N, 28, 28, 1) and normalize to [0, 1]
x_train = x_train[..., tf.newaxis] / 255.0
x_test  = x_test[..., tf.newaxis]  / 255.0

# Convert labels to one-hot encoding if needed, but SparseCategoricalCrossentropy
# in Keras can work directly with integer labels. We’ll leave them as is.
y_train = y_train.astype("int64")
y_test  = y_test.astype("int64")


# ----------------------------
# 3. Build the model
# ----------------------------
with tf.device('/CPU:0'):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),

        # Insert layer that prints the device
        PrintDeviceLayer(),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        PrintDeviceLayer(),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        PrintDeviceLayer(),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        PrintDeviceLayer(),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        PrintDeviceLayer(),

        tf.keras.layers.Flatten(),
        PrintDeviceLayer(),

        tf.keras.layers.Dense(64, activation='relu'),
        PrintDeviceLayer(),

        tf.keras.layers.Dense(10),   # logits
        PrintDeviceLayer()
    ])


    # ----------------------------
    # 4. Compile the model
    # ----------------------------
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy']
    # )

    model(x_train)

    # ----------------------------
    # 5. Train the model
    # ----------------------------
    # model.fit(
    #     x_train,
    #     y_train,
    #     epochs=2,               # keep epochs small for demonstration
    #     batch_size=64,          # you can adjust this as needed
    #     validation_data=(x_test, y_test)
    # )
