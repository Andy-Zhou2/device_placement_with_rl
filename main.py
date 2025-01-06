import tensorflow as tf
import numpy as np
import argparse
import os


# In TF2, we typically don't need placeholders or sessions.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (keep probability is 0.9 => rate=0.1)')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Data directory')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                        help='Log directory')
    return parser.parse_args()


def load_mnist(data_dir):
    """Loads MNIST from tf.keras.datasets or from the data_dir if specified."""
    # TF2 provides built-in MNIST loading, no need for separate input_data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Flatten 28x28 into 784
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    return (x_train, y_train), (x_test, y_test)


def build_model(learning_rate=0.001, dropout_rate=0.1):
    """Build a simple 2-layer feedforward network using tf.keras."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(500, activation='relu', name='layer1'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax', name='layer2')  # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print(f'model summary: {model.summary()}')

    for idx, layer in enumerate(model.layers):
        print(idx, layer.name, layer)


    return model


def main():
    args = parse_args()

    # Prepare data
    (x_train, y_train), (x_test, y_test) = load_mnist(args.data_dir)

    # Build model
    # Note: If dropout keep probability = 0.9 => dropout rate = 1 - 0.9 = 0.1
    dropout_rate = 1.0 - args.dropout
    model = build_model(learning_rate=args.learning_rate, dropout_rate=dropout_rate)

    # Define TensorBoard callback to log training info
    # This will save logs for TensorBoard in args.log_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.log_dir, histogram_freq=1
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,  # 10% of training data for validation
        callbacks=[tensorboard_callback]
    )

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()
