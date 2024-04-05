import matplotlib.pyplot as plt
import numpy as np

datasets = ("Adults", "Smart factory", "Soccer")
loss_data = {
    'Baseline': ((0.387, 0.161, 0.578), "firebrick"),
    'VAE': ((0.375, 0.158, 0.57), "orange"),
    'Mixup': ((0.362, 0.352, 0), "gray"),
    'S-VAE': ((0.355, 0.154, 0.563), "darkgray"),
    'Grid AE': ((0.35, 0.146, 0.553), "forestgreen"),
}

x = np.arange(len(datasets))
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for name, item in loss_data.items():
    data, color = item
    offset = width * multiplier
    rects = ax.bar(x + offset, data, width, label=name, color=color)
    #ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('RSME (lower is better)')
ax.set_xlabel("Datasets")
ax.set_title('Comparison of sampling methods')
ax.set_xticks(x + width, datasets)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 0.6)

plt.show()