import torch
from torch import optim

from tf_device_network import AutoRegressiveTransformerPolicy
from tf_device_measure import measure_inference_time_3devices


def train_ppo(
        policy,
        optimizer,
        num_iterations=2000,
        batch_size=10,
        baseline_decay=0.9,
        ppo_clip=0.2,
        ppo_epochs=4
):
    """
    Trains the given policy using single-step PPO with a moving average baseline.
    Gathers 'batch_size' single-step episodes before each PPO update.

    :param policy: Policy network (AutoRegressiveTransformerPolicy).
    :param optimizer: Optimizer (e.g., torch.optim.Adam).
    :param num_iterations: Total number of single-step episodes to run.
    :param batch_size: Number of single-step episodes to collect before each update.
    :param baseline_decay: Factor for moving average baseline.
    :param ppo_clip: The epsilon value for PPO clipping.
    :param ppo_epochs: How many epochs of updates to run per batch.
    """
    # Moving average baseline
    baseline = 0.0

    # Buffers
    old_log_probs_buffer = []  # Log-prob under old parameters
    actions_buffer = []
    rewards_buffer = []

    # Because PPO needs "old" log probabilities, we will temporarily store
    # the policy parameters before collecting the batch.
    old_policy_state_dict = None

    for iteration in range(num_iterations):
        # 1) Sample action and log-prob from *current* policy
        action, log_prob, logits = policy.sample_action_and_logprob()
        action = action[0]
        current_lprob = log_prob[0].detach().cpu()

        # 2) Measure inference time => reward = - sqrt(time)
        n_iters = 10 if iteration < 200 else 100
        t = torch.tensor(measure_inference_time_3devices(action, n_iters=n_iters))
        reward = -torch.sqrt(t)

        print(f"Iteration: {iteration}, Action: {action}, Reward: {reward.item():.5f}, Time: {t.item():.5f}")

        # 3) Store data in buffers
        old_log_probs_buffer.append(current_lprob)
        actions_buffer.append(action)
        rewards_buffer.append(reward)

        # 4) Once we reach batch_size single-step episodes, do a PPO update
        if (iteration + 1) % batch_size == 0:
            # Convert buffers to tensors
            old_log_probs_tensor = torch.stack(old_log_probs_buffer)  # shape [batch_size]
            actions_tensor = torch.stack(actions_buffer)  # shape [batch_size]
            rewards_tensor = torch.stack(rewards_buffer)  # shape [batch_size]

            # Compute advantage = R - baseline
            advantages = rewards_tensor - baseline

            # Update baseline (moving average)
            batch_mean_reward = rewards_tensor.mean().item()
            baseline = baseline_decay * baseline + (1.0 - baseline_decay) * batch_mean_reward

            # PPO typically does multiple epochs over the same batch
            for _ in range(ppo_epochs):
                new_log_probs_tensor, _ = policy.get_log_prob(actions_tensor)

                # Calculate ratio
                ratio = torch.exp(new_log_probs_tensor - old_log_probs_tensor)

                # Clipped objective: L_clip = - mean( min(ratio * A, clip(ratio)* A ) )
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Gradient step
                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

            logit_display = logits.detach().cpu().numpy()
            prob_display = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            print(f"Logits: {logit_display}, \n"
                  f"Probs: {prob_display}")

            # Print debug info (we show info from the *last* step in this batch)
            print(
                f"Step {iteration + 1} / {num_iterations}, "
                f"Policy Loss: {policy_loss.item():.4f}, "
                f"Mean Reward: {batch_mean_reward:.4f}, "
                f"Baseline: {baseline:.4f}, "
                f"Last Action: {action}, "
                f"Last Time: {t.item():.4f}"
            )

            # Clear buffers
            old_log_probs_buffer = []
            actions_buffer = []
            rewards_buffer = []


if __name__ == "__main__":
    policy = AutoRegressiveTransformerPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    train_ppo(
        policy=policy,
        optimizer=optimizer,
    )


