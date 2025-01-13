from tf_device_network import AutoRegressiveTransformerPolicy
import torch
from torch import optim
from tf_device_measure import measure_inference_time_3devices


def train_reinforce(policy, optimizer, num_iterations=1000):
    for iteration in range(num_iterations):
        action, log_prob, logits = policy.sample_action_and_logprob()
        action = action[0]
        log_prob = log_prob[0]

        if iteration < 100:
            inference_iter = 10
        else:
            inference_iter = 100
        t = torch.tensor(measure_inference_time_3devices(action, n_iters=inference_iter), dtype=torch.float32)
        reward = -torch.sqrt(t)

        loss = -log_prob * reward

        # 5. Gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 1 == 0:
            print(
                f"Iteration {iteration}, Loss: {loss.item():.4f}, Reward: {reward.item():.4f}, Action: {action}, Time: {t}, logit: {logits}")


policy = AutoRegressiveTransformerPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

train_reinforce(policy, optimizer, num_iterations=2000)
