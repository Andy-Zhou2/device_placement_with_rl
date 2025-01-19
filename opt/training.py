import torch
import numpy as np
from torch import optim
from policy_network import AutoRegressiveTransformerPolicy
from measure_time import spawn_time_measurement_process
import logging
from logging_utils import copy_files_to_next_experiment_folder

from task_config import (OP_TYPES, ADJ_FORWARD, ADJ_BACKWARD, SHAPES, NUM_DEVICE_CHOICES,
                         NUM_OPERATIONS, OPERATION_VOCAB_SIZE, ALGO,
                         NUM_BATCHES, BATCH_SIZE, BASELINE_DECAY,
                         MEASURE_REPETITIONS)


def set_seed(seed):
    """
    Set seed for reproducibility.

    :param seed: The random seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def collect_data(policy, batch_size, baseline, inference_fn, n_iters, baseline_decay):
    log_probs_buffer = []
    actions_buffer = []
    rewards_buffer = []

    best_config = (None, float('inf'))

    for i in range(batch_size):
        action, log_prob, _ = policy.sample_action_and_logprob()
        action = action[0]
        log_prob = log_prob[0]

        t = torch.tensor(inference_fn(action, n_iters=n_iters))
        if t < best_config[1]:
            best_config = (action, t)
        reward = -torch.sqrt(t)  # Example reward function

        logging.info(f"Iteration: {i}, Reward: {reward.item()}")

        log_probs_buffer.append(log_prob)
        actions_buffer.append(action)
        rewards_buffer.append(reward)

    rewards_tensor = torch.stack(rewards_buffer)
    advantages = rewards_tensor - baseline
    batch_mean_reward = rewards_tensor.mean().item()
    updated_baseline = baseline * baseline_decay + batch_mean_reward * (1 - baseline_decay)

    return (
        torch.stack(log_probs_buffer),
        torch.stack(actions_buffer),
        rewards_tensor,
        advantages,
        updated_baseline,
        best_config
    )


def ppo_loss_update(policy, optimizer, log_probs, actions, advantages, ppo_clip, ppo_epochs):
    """
    Performs PPO loss update.

    :param policy: Policy network.
    :param optimizer: Optimizer.
    :param log_probs: Log probabilities under old policy.
    :param actions: Actions taken.
    :param advantages: Calculated advantages.
    :param ppo_clip: Clipping value for PPO.
    :param ppo_epochs: Number of epochs for PPO update.
    """
    log_probs = log_probs.detach()

    for _ in range(ppo_epochs):
        new_log_probs, _ = policy.get_log_prob(actions)
        ratio = torch.exp(new_log_probs - log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


def reinforce_loss_update(policy, optimizer, log_probs, actions, advantages):
    """
    Performs REINFORCE loss update.

    :param policy: Policy network.
    :param optimizer: Optimizer.
    :param log_probs: Log probabilities of actions.
    :param advantages: Calculated advantages.
    """
    loss = -torch.mean(log_probs * advantages)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(
        policy,
        optimizer,
        num_baches,
        batch_size,
        baseline_decay,
        update_fn,
        **update_kwargs
):
    """
    Unified training loop for both PPO and REINFORCE.

    :param policy: Policy network.
    :param optimizer: Optimizer.
    :param num_baches: Total number of iterations.
    :param batch_size: Number of steps per batch.
    :param baseline_decay: Decay factor for baseline.
    :param update_fn: Function to update the policy.
    :param update_kwargs: Additional arguments for the update function.
    """
    baseline = 0.0
    best_config = (None, float('inf'))

    logits = policy()
    logit_display = logits.detach().cpu().numpy()
    prob_display = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    logging.info(f"Logits: {logit_display}, \n"
                 f"Probs: {prob_display}")

    for iteration in range(num_baches):
        log_probs, actions, rewards, advantages, baseline, batch_best_config = collect_data(
            policy,
            batch_size,
            baseline,
            spawn_time_measurement_process,
            MEASURE_REPETITIONS,
            baseline_decay,
        )
        if batch_best_config[1] < best_config[1]:
            best_config = batch_best_config

        update_fn(policy, optimizer, log_probs, actions, advantages, **update_kwargs)

        logging.info(
            f"Batch: {iteration + 1}/{num_baches}, "
            f"Mean Reward: {rewards.mean().item()}, "
            f"Baseline: {baseline}, "
            f"Best Time: {best_config[1]}, "
            f"Best Config: {best_config[0]}"
        )

        logits = policy()
        logit_display = logits.detach().cpu().numpy()
        prob_display = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        logging.info(f"Logits: {logit_display}, \n"
                     f"Probs: {prob_display}")


if __name__ == "__main__":
    logging_dir = copy_files_to_next_experiment_folder(r'../experiment_logs')  # Returns the logging directory

    # Configure logging to write to a specific file with a desired format
    logging.basicConfig(
        filename=logging_dir / f'{ALGO}.log',  # Replace with your log file path
        filemode='a',  # Append mode; change to 'w' for overwrite mode
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    set_seed(42)

    op_types_tensor = torch.tensor(OP_TYPES, dtype=torch.long)  # shape [NUM_OPERATIONS]
    adj_forward_tensor = torch.tensor(ADJ_FORWARD, dtype=torch.float32)  # shape [NUM_OPERATIONS, NUM_OPERATIONS]
    adj_backward_tensor = torch.tensor(ADJ_BACKWARD, dtype=torch.float32)  # shape [NUM_OPERATIONS, NUM_OPERATIONS]
    shape_tensor = torch.tensor(SHAPES, dtype=torch.float32)  # shape [NUM_OPERATIONS, 4]

    policy = AutoRegressiveTransformerPolicy(
        type_vocab_size=OPERATION_VOCAB_SIZE,
        num_operations=NUM_OPERATIONS,
        num_device_choices=NUM_DEVICE_CHOICES,
        op_types_tensor=op_types_tensor,
        adj_forward_tensor=adj_forward_tensor,
        adj_backward_tensor=adj_backward_tensor,
        shape_tensor=shape_tensor,
    )
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    if ALGO == "PPO":
        from task_config import PPO_CLIP, PPO_EPOCHS
        train(
            policy=policy,
            optimizer=optimizer,
            num_baches=NUM_BATCHES,
            batch_size=BATCH_SIZE,
            baseline_decay=BASELINE_DECAY,
            update_fn=ppo_loss_update,
            ppo_clip=PPO_CLIP,
            ppo_epochs=PPO_EPOCHS,
        )
    elif ALGO == "REINFORCE":
        train(
            policy=policy,
            optimizer=optimizer,
            num_baches=NUM_BATCHES,
            batch_size=BATCH_SIZE,
            baseline_decay=BASELINE_DECAY,
            update_fn=reinforce_loss_update,
        )
    else:
        raise ValueError("Unknown algorithm")
