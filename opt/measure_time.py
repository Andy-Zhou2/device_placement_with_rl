import time
import numpy as np
import multiprocessing
import tensorflow as tf
from transformers import AutoTokenizer, AutoConfig, TFAutoModelForCausalLM
import logging

from task_config import DEVICE_OPTIONS, MODEL_NAME, INPUT_TOKEN_IDS, PROCESS_WAIT_TIME, FAILURE_LOG_TIME


def measure_time_with_process(action, queue, n_warmup=1, n_iters=100, batch_size=1000):
    # Set up logical GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=600),
                 tf.config.LogicalDeviceConfiguration(memory_limit=400),
                 ]
            )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    devices = [DEVICE_OPTIONS[d] for d in action]  # translate device id to device name

    # create model
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.device_placement = devices  # Inject device placement
    model = TFAutoModelForCausalLM.from_config(config)

    with tf.device(devices[0]):
        generated_ids = tf.constant(INPUT_TOKEN_IDS, dtype=tf.int32)
    times = []
    for i in range(n_iters):
        start = time.time()
        _ = model(input_ids=generated_ids)
        tf.test.experimental.sync_devices()  # For GPU timing
        end = time.time()
        if i >= n_warmup:
            times.append(end - start)

    if len(times) == 0:
        return 0.0
    avg_time = sum(times) / len(times)
    queue.put(avg_time)


def spawn_time_measurement_process(action, n_iters):
    """
    Spawns a process to measure the inference time for a given action.

    :param action: The action to measure.
    :param n_iters: The number of iterations to perform.
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=measure_time_with_process,
        kwargs={"action": action, "n_iters": n_iters, "queue": queue}
    )
    p.start()
    p.join(timeout=PROCESS_WAIT_TIME)

    if p.is_alive():
        logging.info(f"Process for action {action} did not complete within 10 seconds. Terminating.")
        p.terminate()
        p.join()
        return FAILURE_LOG_TIME
    elif p.exitcode != 0:
        logging.info(f"Process for action {action} exited with code {p.exitcode}. Assume OOM.")
        return FAILURE_LOG_TIME
    else:  # Process completed successfully
        t = queue.get()
        logging.info(f"Process for action {action} completed successfully. Time: {t}")
        return t


if __name__ == '__main__':
    print(spawn_time_measurement_process([0] * 14, 10))
