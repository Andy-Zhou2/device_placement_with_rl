import numpy as np
from transformers import AutoConfig
import tensorflow as tf

# Device options
DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    "/GPU:1",
]
NUM_DEVICE_CHOICES = len(DEVICE_OPTIONS)

# Model options
MODEL_NAME = 'facebook/opt-125m'
INPUT_TOKEN_IDS = np.array([[2, 100, 657, 30581, 3923, 12346, 328]])
OPERATION_VOCAB_SIZE = 3


def _get_model_task_info():
    # returns the adjacency matrix, operation types, and output shapes
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    num_operations = num_layers + 2

    adj_forward = np.zeros((num_operations, num_operations), dtype=np.float32)
    for i in range(num_operations - 1):
        adj_forward[i, i + 1] = 1.0
    adj_backward = adj_forward.T

    op_types = [0] + [1] * num_layers + [2]

    shapes = np.array([1, len(INPUT_TOKEN_IDS), hidden_dim, 0], dtype=np.int32)
    shapes = np.tile(shapes, (num_operations, 1))

    return adj_forward, adj_backward, op_types, shapes, num_operations


ADJ_FORWARD, ADJ_BACKWARD, OP_TYPES, SHAPES, NUM_OPERATIONS = _get_model_task_info()
