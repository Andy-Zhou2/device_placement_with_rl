import matplotlib.pyplot as plt
import numpy as np
import re

r = re.compile(r'Batch: (\d+)/\d+, Mean Reward: ([\d.-]+), Baseline: ([\d.-]+).*')

def get_batch_data(batch_num, lines):
    batch_line = list(filter(lambda x: f'Batch: {batch_num}' in lines[x], range(len(lines))))[0]
    m = r.search(lines[batch_line])
    if m:
        return float(m.group(2))
    return None



def plot_reward(exp_name, title_name, data):
    # Create the plot
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(len(data)), data, label='Mean Reward')

    plt.xticks(ticks=range(len(data)), labels=range(1, len(data) + 1))

    plt.title(f'Mean Reward vs Batch for {title_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(rf'../plot_figs/opt_125m_{exp_name}_mean_reward.png', dpi=150)


with open(f'../experiment_logs/experiment 13/PPO.log', 'r') as f:
    lines = f.readlines()
data = [get_batch_data(b, lines) for b in range(1, 21)]
plot_reward('one_big_gpu', 'One Big GPU', data)

with open(f'../experiment_logs/experiment 14/PPO.log', 'r') as f:
    lines = f.readlines()
data = [get_batch_data(b, lines) for b in range(1, 21)]
plot_reward('one_constrained_gpu', 'One Constrained GPU', data)


with open(f'../experiment_logs/experiment 15/PPO.log', 'r') as f:
    lines = f.readlines()
data = [get_batch_data(b, lines) for b in range(1, 21)]
plot_reward('two_constrained_gpus', 'Two Constrained GPUs', data)

with open(f'../experiment_logs/experiment 23/PPO.log', 'r') as f:
    lines = f.readlines()
data = [get_batch_data(b, lines) for b in range(1, 21)]
plot_reward('four_constrained_gpus', 'Four Constrained GPUs', data)

with open(f'../experiment_logs/experiment 16/PPO.log', 'r') as f:
    lines = f.readlines()
data = [get_batch_data(b, lines) for b in range(1, 21)]
plot_reward('two_constrained_gpus_slow_update', 'Two Constrained GPUs Slow Update', data)

# plot_prob('one_big_gpu', 10, get_batch_data(2, 10, lines))
# plot_prob('one_big_gpu', 20, get_batch_data(2, 20, lines))
#
#
# with open(f'../experiment_logs/experiment 14/PPO.log', 'r') as f:
#     lines = f.readlines()
# plot_prob('one_constrained_gpu', 10, get_batch_data(2, 10, lines))
# plot_prob('one_constrained_gpu', 20, get_batch_data(2, 20, lines))
#
#
# with open(f'../experiment_logs/experiment 15/PPO.log', 'r') as f:
#     lines = f.readlines()
# plot_prob('two_constrained_gpus', 10, get_batch_data(3, 10, lines))
# plot_prob('two_constrained_gpus', 20, get_batch_data(3, 20, lines))
#
# with open(f'../experiment_logs/experiment 20/PPO.log', 'r') as f:
#     lines = f.readlines()
# plot_prob('four_constrained_gpus', 10, get_batch_data(5, 10, lines))
# plot_prob('four_constrained_gpus', 20, get_batch_data(5, 20, lines))
#
#
# with open(f'../experiment_logs/experiment 16/PPO.log', 'r') as f:
#     lines = f.readlines()
# plot_prob('two_constrained_gpus_slow_update', 10, get_batch_data(3, 10, lines))
# plot_prob('two_constrained_gpus_slow_update', 20, get_batch_data(3, 20, lines))