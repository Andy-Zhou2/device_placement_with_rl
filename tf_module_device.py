import tensorflow as tf
import time


# Define the model with layers on different devices
class TwoDeviceModel(tf.Module):
    def __init__(self):
        super().__init__()
        D = 2000000
        with tf.device('/CPU:0'):  # Layer 1 on GPU
            self.w1 = tf.Variable(tf.random.normal([10, D]), name='w1')
            self.b1 = tf.Variable(tf.zeros([D]), name='b1')
        with tf.device('/GPU:0'):  # Layer 2 on CPU
            self.w2 = tf.Variable(tf.random.normal([D, 1]), name='w2')
            self.b2 = tf.Variable(tf.zeros([1]), name='b2')

    def __call__(self, x):
        with tf.device('/CPU:0'):
            x = tf.matmul(x, self.w1) + self.b1
            x = tf.nn.relu(x)
        with tf.device('/GPU:0'):
            x = tf.matmul(x, self.w2) + self.b2
        return x


# Create the artificial dataset
def create_dataset(num_samples=1000):
    X = tf.random.normal([num_samples, 10])
    y = tf.random.normal([num_samples, 1])
    return X, y


# Training step
def train_step(model, X, y, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = tf.reduce_mean(tf.square(predictions - y))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Initialize model, dataset, and optimizer
model = TwoDeviceModel()
optimizer = tf.optimizers.SGD(learning_rate=0.01)
X, y = create_dataset()

# Measure time for a forward and backward pass
start_time = time.time()
loss = train_step(model, X, y, optimizer)
end_time = time.time()

print(f"Time for one forward and backward pass: {end_time - start_time:.4f} seconds")
