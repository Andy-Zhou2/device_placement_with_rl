import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import time
import numpy as np



######################################################
# 4) SINGLE-STEP RL: COLLECTOR + PPO-LIKE TRAINING
######################################################

def collect_single_step_episodes_autoreg(policy, batch_size=8, n_steps=16, device="cpu"):
    """
    We do a single-step MDP:
     - For each "step" in [n_steps], we sample a batch of 3-token device configs
       from the policy (auto-regressively).
     - We measure time => reward = -time
     - We store in a small buffer for PPO.

    We'll produce a dictionary with
      {
        "actions": (n_steps, batch_size, 3),   # device indices chosen
        "log_prob": (n_steps, batch_size, 3),  # log probs
        "reward": (n_steps, batch_size),
        "done": (n_steps, batch_size),
        "state_value": (n_steps, batch_size, 3)  # or a single value if we prefer
      }

    Because we must do real auto-regression with sampling, we do the forward pass
    step by step for each sample in the batch. (In practice, you'd want
    a more vectorized approach or a more clever mask.)
    """
    # We'll store arrays then stack => Tensors
    action_buf = []
    logp_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []  # if you want a value per operation or a single overall value

    for _ in range(n_steps):
        # We'll sample for each item in the batch
        batch_actions = []
        batch_logp = []

        for b in range(batch_size):
            # 1) Forward pass to get 3 sets of logits auto-regressively
            logits_list = policy()  # list of 3, each shape [1, NUM_DEVICE_CHOICES]

            chosen_devices = []
            chosen_logps = []
            for step_i, logits in enumerate(logits_list):
                dist = D.Categorical(logits=logits.squeeze(0))  # shape [NUM_DEVICE_CHOICES]
                action_i = dist.sample()  # scalar
                logp_i = dist.log_prob(action_i)
                chosen_devices.append(action_i.item())
                chosen_logps.append(logp_i.item())

            batch_actions.append(chosen_devices)
            batch_logp.append(chosen_logps)

        # 2) measure inference time for each config in the batch
        times_list = []
        for devices_idx in batch_actions:
            # devices_idx = [device_for_X, device_for_dense1, device_for_dense2]
            triple = [DEVICE_OPTIONS[d] for d in devices_idx]
            t = measure_inference_time_3devices(triple, n_warmup=1, n_iters=2, batch_size=1000)
            times_list.append(t)

        # 3) compute reward
        rewards = [-t for t in times_list]

        # For single-step, done=True
        done_flags = [True] * batch_size

        # We'll do a dummy "value" vector => e.g. zero or random
        # A real approach would have a value head that returns a single scalar
        # for the entire 3-step device selection. We'll just do zeros here.
        values = [[0.0, 0.0, 0.0] for _ in range(batch_size)]

        # Accumulate into buffer
        action_buf.append(torch.tensor(batch_actions, dtype=torch.long))
        logp_buf.append(torch.tensor(batch_logp, dtype=torch.float32))
        reward_buf.append(torch.tensor(rewards, dtype=torch.float32))
        done_buf.append(torch.tensor(done_flags, dtype=torch.bool))
        value_buf.append(torch.tensor(values, dtype=torch.float32))

    # Now stack => shape [n_steps, batch_size, *]
    actions = torch.stack(action_buf, dim=0)  # [n_steps, batch_size, 3]
    logps = torch.stack(logp_buf, dim=0)  # [n_steps, batch_size, 3]
    rews = torch.stack(reward_buf, dim=0)  # [n_steps, batch_size]
    dones = torch.stack(done_buf, dim=0)  # [n_steps, batch_size]
    vals = torch.stack(value_buf, dim=0)  # [n_steps, batch_size, 3]

    batch_data = {
        "action": actions,
        "log_prob": logps,
        "reward": rews,
        "done": dones,
        "state_value": vals,  # shape [n_steps, batch_size, 3]
    }
    return batch_data


def ppo_loss(batch_data):
    """
    A placeholder for PPO’s loss with single-step transitions.
    Normally you'd compare old_log_probs to new_log_probs,
    do advantage computation, etc.
    Here we do a *highly simplified* approach to demonstrate the concept.
    """
    # We'll just do a negative log_prob * reward type of "policy gradient" for demonstration.
    # shape: [n_steps, batch_size, 3]
    logp = batch_data["log_prob"]
    rew = batch_data["reward"].unsqueeze(-1)  # broadcast to match [n_steps, batch_size, 1]
    # We'll sum over the 3 tokens
    pg_loss = -(logp.sum(dim=-1) * rew.squeeze(-1)).mean()

    # Similarly, we ignore the value loss in this snippet or do a small penalty
    value_loss = torch.tensor(0.0, requires_grad=True)
    entropy_loss = torch.tensor(0.0, requires_grad=True)

    total_loss = pg_loss + value_loss + entropy_loss
    return {
        "loss_policy": pg_loss,
        "loss_value": value_loss,
        "loss_entropy": entropy_loss,
        "loss_total": total_loss
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the policy
    policy = AutoRegressiveTransformerPolicy(embed_dim=32, n_heads=2, n_layers=2).to(device)

    # Create an optimizer
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # We'll do a small number of epochs
    epochs = 5
    steps_per_epoch = 4
    batch_size = 2

    for epoch in range(epochs):
        # 1) Collect single-step data
        batch_data = collect_single_step_episodes_autoreg(
            policy,
            batch_size=batch_size,
            n_steps=steps_per_epoch,
            device=device
        )

        # 2) PPO-like loss
        loss_dict = ppo_loss(batch_data)
        loss = loss_dict["loss_total"]

        # 3) Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch[{epoch + 1}/{epochs}] -> "
            f"PolicyLoss={loss_dict['loss_policy'].item():.4f}, "
            f"ValueLoss={loss_dict['loss_value'].item():.4f}, "
            f"EntropyLoss={loss_dict['loss_entropy'].item():.4f}"
        )


if __name__ == "__main__":
    main()
