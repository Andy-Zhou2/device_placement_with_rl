import numpy as np
import matplotlib.pyplot as plt

# plot [[[1.6416027e-04 3.6123922e-04 9.9947459e-01]
#   [7.8132878e-05 8.8527758e-04 9.9903667e-01]
#   [1.3316664e-04 4.2754415e-04 9.9943930e-01]]]

ppo_data = np.array([[[1.6416027e-04, 3.6123922e-04, 9.9947459e-01],
                  [7.8132878e-05, 8.8527758e-04, 9.9903667e-01],
                  [1.3316664e-04, 4.2754415e-04, 9.9943930e-01]]])

def plot_data(data, name):
    # plot the data as heatmap, with each cell has a .2f text
    # label each row as layers
    # label each column as devices
    # also plot the color scale on the right side
    fig, ax = plt.subplots()
    cax = ax.matshow(data[0], cmap='viridis')

    for i in range(data[0].shape[0]):
        for j in range(data[0].shape[1]):
            ax.text(j, i, f'{data[0][i, j]:.2f}', ha='center', va='center', color='black')

    plt.xticks(range(data[0].shape[1]), ['CPU', 'GPU 1', 'GPU 2', ])
    plt.yticks(range(data[0].shape[0]), ['Layer 1', 'Layer 2', 'Layer 3'])

    fig.colorbar(cax)
    plt.title(f'{name} Device Probability After Training')
    plt.tight_layout()
    plt.savefig(f'../plot_figs/toy_example_{name}_device_probability.png', dpi=150)

plot_data(ppo_data, 'PPO')

# reinforce data: [[[0.00228287 0.01197117 0.9857459 ]
#   [0.00208597 0.0134245  0.98448956]
#   [0.00229944 0.01275432 0.98494625]]]

reinforce_data = np.array([[[0.00228287, 0.01197117, 0.9857459],
                            [0.00208597, 0.0134245, 0.98448956],
                            [0.00229944, 0.01275432, 0.98494625]]])

plot_data(reinforce_data, 'REINFORCE')


