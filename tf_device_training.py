from tf_device_network import AutoRegressiveTransformerPolicy
import torch
from torch import optim
from tf_device_measure import measure_inference_time_3devices


def train_reinforce(
        policy,
        optimizer,
        num_iterations=2000,
        batch_size=10,
        baseline_decay=0.9
):
    """
    Trains the given policy using REINFORCE with a moving average baseline.
    Accumulates trajectories (1-step episodes in this case) for `batch_size` times
    before updating the parameters.

    :param policy: Policy network (AutoRegressiveTransformerPolicy)
    :param optimizer: Optimizer (e.g., torch.optim.Adam)
    :param num_iterations: Total number of episodes to run (1 step per episode)
    :param batch_size: Number of episodes to collect before each update
    :param baseline_decay: Factor for moving average baseline, new baseline is
                          old_baseline * baseline_decay + (1 - baseline_decay) * mean_reward
    """
    # Simple moving average baseline initialized to 0.
    baseline = 0.0

    # Buffers for collected data
    log_probs_buffer = []
    rewards_buffer = []

    for iteration in range(num_iterations):
        # 1) Sample action and log-prob from the policy
        action, log_prob, logits = policy.sample_action_and_logprob()
        action = action[0]
        log_prob = log_prob[0]

        # 2) Measure inference time and convert to reward
        #    We'll treat negative sqrt(time) as the reward
        #    (since you want to minimize time, i.e. negative of cost).
        n_iters = 5 if iteration < 100 else 20
        t = torch.tensor(measure_inference_time_3devices(action, n_iters=n_iters))
        reward = -torch.log(t)

        print(f"Iteration: {iteration}, Action: {action}, Reward: {reward.item()}, Time: {t.item()}")

        # 3) Store samples in buffer
        log_probs_buffer.append(log_prob)
        rewards_buffer.append(reward)

        # 4) Only update when we've collected `batch_size` steps/episodes
        if (iteration + 1) % batch_size == 0:
            # Compute advantage = (reward - baseline) for each step in the batch
            rewards_tensor = torch.stack(rewards_buffer)
            advantages = rewards_tensor - baseline

            # Update baseline via a moving average
            batch_mean_reward = rewards_tensor.mean()
            baseline = baseline_decay * baseline + (1.0 - baseline_decay) * batch_mean_reward.item()

            # Compute loss = - (1/batch_size) * sum(log_prob * advantage)
            loss = 0.0
            for lp, adv in zip(log_probs_buffer, advantages):
                loss += -lp * adv
            loss = loss / batch_size

            # 5) Gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logit_display = [logits[i].detach().cpu().numpy().tolist() for i in range(3)]
            prob_display = [torch.softmax(logits[i], dim=-1).detach().cpu().numpy().tolist() for i in range(3)]

            # Print debug info
            print(
                f"Episode: {iteration + 1}, "
                f"Loss: {loss.item():.4f}, "
                f"Mean Reward: {batch_mean_reward.item():.4f}, "
                f"Baseline: {baseline:.4f}, "
                f"Last Action: {action}, "
                f"Last Time: {t.item():.4f}, \n"
                f"Logits: {logit_display}\n"
                f"Probs: {prob_display}\n"
            )

            # 6) Clear buffers
            log_probs_buffer = []
            rewards_buffer = []


policy = AutoRegressiveTransformerPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

train_reinforce(policy, optimizer, num_iterations=2000)
