import matplotlib.pyplot as plt
import numpy as np




def get_batch_data(num_devices, batch_num, lines):
    batch_line = list(filter(lambda x: f'Batch: {batch_num}' in lines[x], range(len(lines))))[0]
    start_line = list(filter(lambda x: 'Probs:' in lines[x], range(batch_line, len(lines))))[0]
    end_line = list(filter(lambda x: 'INFO' in lines[x], range(start_line + 1, len(lines))))[0]

    # # find batch batch_num
    # end_line = 28 + \
    #            (41 + 28) * batch_num
    batch_10_probs = lines[start_line:end_line]
    batch_10_probs[0] = batch_10_probs[0][7:]  # remove 'probs:'

    data = np.zeros((num_devices, 14))

    for col, content in enumerate(batch_10_probs):
        for row, n in enumerate(content[content.rfind('[') + 1:content.find(']')].strip().split()):
            print(row, col, n)
            data[row, col] = float(n)

    print(data)
    return data




def plot_prob(exp_name, batch_num, data):
    # Create the plot
    plt.figure(figsize=(10, 6))
    x_labels = ['pre'] + [str(i) for i in range(1, 13)] + ['post']

    # Plot each row of the matrix as a separate line
    d, n = data.shape
    for i in range(d):
        if i == 0:
            name = 'CPU'
        else:
            name = f'GPU {i}'
        plt.plot(range(n), data[i], marker='s', label=name)

    # Add labels, title, and legend
    plt.xticks(ticks=range(n), labels=x_labels)
    plt.xlabel('Operation Group')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.title(f'Probability of Operations in Batch {batch_num}')

    # legend to the left
    plt.legend(loc='center left')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../plot_figs/opt_125m_{exp_name}_probability_batch_{batch_num}.png', dpi=150)


with open(f'../experiment_logs/experiment 13/PPO.log', 'r') as f:
    lines = f.readlines()
plot_prob('one_big_gpu', 10, get_batch_data(2, 10, lines))
plot_prob('one_big_gpu', 20, get_batch_data(2, 20, lines))


with open(f'../experiment_logs/experiment 14/PPO.log', 'r') as f:
    lines = f.readlines()
plot_prob('one_constrained_gpu', 10, get_batch_data(2, 10, lines))
plot_prob('one_constrained_gpu', 20, get_batch_data(2, 20, lines))


with open(f'../experiment_logs/experiment 15/PPO.log', 'r') as f:
    lines = f.readlines()
plot_prob('two_constrained_gpus', 10, get_batch_data(3, 10, lines))
plot_prob('two_constrained_gpus', 20, get_batch_data(3, 20, lines))

with open(f'../experiment_logs/experiment 23/PPO.log', 'r') as f:
    lines = f.readlines()
plot_prob('four_constrained_gpus', 10, get_batch_data(5, 10, lines))
plot_prob('four_constrained_gpus', 20, get_batch_data(5, 20, lines))


with open(f'../experiment_logs/experiment 16/PPO.log', 'r') as f:
    lines = f.readlines()
plot_prob('two_constrained_gpus_slow_update', 10, get_batch_data(3, 10, lines))
plot_prob('two_constrained_gpus_slow_update', 20, get_batch_data(3, 20, lines))