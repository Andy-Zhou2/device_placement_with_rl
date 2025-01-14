import re
import matplotlib.pyplot as plt
import numpy as np

# read everyline and detect patterns like 2025-01-14 17:30:17,992 - INFO - Batch: 249/250, Mean Reward: -0.0978, Baseline: -0.0977
# extract batch, mean reward and baseline
r = re.compile(r'Batch: (\d+)/\d+, Mean Reward: ([\d.-]+), Baseline: ([\d.-]+)')

data = []

# algo = 'reinforce'
algo = 'ppo'
# read the file
with open(f'../{algo}.log', 'r') as f:
    for line in f:
        m = r.search(line)
        if m:
            data.append([int(m.group(1)), float(m.group(2)), float(m.group(3))])



# plot the data
data = list(zip(*data))

# smooth the data
data[1] = np.convolve(data[1], np.ones(10) / 10, mode='valid')
plt.plot(np.arange(len(data[1])), data[1], label='Mean Reward')



# plt.plot(data[0], data[1], label='Mean Reward')
# plt.plot(data[0], data[2], label='Baseline')
plt.title(f'Mean Reward vs Batch for {algo}')
plt.legend()
plt.show()
