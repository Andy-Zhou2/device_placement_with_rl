import numpy as np
from transformers import AutoConfig

# Device options
DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    # "/GPU:1",
]
NUM_DEVICE_CHOICES = len(DEVICE_OPTIONS)

# RL algorithm options
ALGO = "PPO"  # "REINFORCE"
PROCESS_WAIT_TIME = 10  # the process waits for this time before killing the process
FAILURE_LOG_TIME = 0.5  # if the process fails, it returns this time
NUM_BATCHES = 250
BATCH_SIZE = 10
BASELINE_DECAY = 0.5
PPO_CLIP = 0.2
PPO_EPOCHS = 5

# measurement options
MEASURE_REPETITIONS = 10

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
