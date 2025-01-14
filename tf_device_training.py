import torch
import numpy as np
from torch import optim
from tf_device_network import AutoRegressiveTransformerPolicy
from tf_device_measure import measure_inference_time_3devices
import logging


def set_seed(seed):
    """
    Set seed for reproducibility.

    :param seed: The random seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_data(policy, batch_size, baseline, inference_fn, n_iters_fn):
    """
    Collects data for a single batch.

    :param policy: Policy network (AutoRegressiveTransformerPolicy).
    :param batch_size: Number of steps to collect per batch.
    :param baseline: Current baseline for advantage calculation.
    :param inference_fn: Function to measure inference time and convert it to reward.
    :param n_iters_fn: Function to determine the number of iterations for inference.
    :return: A tuple of (log_probs, actions, rewards, advantages, updated_baseline).
    """
    log_probs_buffer = []
    actions_buffer = []
    rewards_buffer = []

    for i in range(batch_size):
        action, log_prob, _ = policy.sample_action_and_logprob()
        action = action[0]
        log_prob = log_prob[0]

        n_iters = n_iters_fn()
        t = torch.tensor(inference_fn(action, n_iters=n_iters))
        reward = -torch.sqrt(t)  # Example reward function

        logging.info(f"Iteration: {i}, Action: {action}, Reward: {reward.item()}, Time: {t.item()}")

        log_probs_buffer.append(log_prob)
        actions_buffer.append(action)
        rewards_buffer.append(reward)

    rewards_tensor = torch.stack(rewards_buffer)
    advantages = rewards_tensor - baseline
    batch_mean_reward = rewards_tensor.mean().item()
    updated_baseline = baseline * 0.9 + batch_mean_reward * 0.1

    return (
        torch.stack(log_probs_buffer),
        torch.stack(actions_buffer),
        rewards_tensor,
        advantages,
        updated_baseline,
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

    for iteration in range(num_baches):
        log_probs, actions, rewards, advantages, baseline = collect_data(
            policy,
            batch_size,
            baseline,
            measure_inference_time_3devices,
            lambda: 10 if iteration < 200 else 100,
        )

        update_fn(policy, optimizer, log_probs, actions, advantages, **update_kwargs)

        logging.info(
            f"Batch: {iteration + 1}/{num_baches}, "
            f"Mean Reward: {rewards.mean().item()}, "
            f"Baseline: {baseline}"
        )

        logits = policy()
        logit_display = logits.detach().cpu().numpy()
        prob_display = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        logging.info(f"Logits: {logit_display}, \n"
                     f"Probs: {prob_display}")


if __name__ == "__main__":
    ALGO = "PPO"  # "REINFORCE"
    # Configure logging to write to a specific file with a desired format
    logging.basicConfig(
        filename=F'experiment_logs/{ALGO}.log',  # Replace with your log file path
        filemode='a',  # Append mode; change to 'w' for overwrite mode
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    set_seed(42)
    policy = AutoRegressiveTransformerPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    if ALGO == "PPO":
        train(
            policy=policy,
            optimizer=optimizer,
            num_baches=250,
            batch_size=10,
            baseline_decay=0.9,
            update_fn=ppo_loss_update,
            ppo_clip=0.2,
            ppo_epochs=5
        )
    elif ALGO == "REINFORCE":
        train(
            policy=policy,
            optimizer=optimizer,
            num_baches=2000,
            batch_size=10,
            baseline_decay=0.9,
            update_fn=reinforce_loss_update,
        )
    else:
        raise ValueError("Unknown algorithm")
