import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Grounding Error', 'OBG Error']
single_means = [1.39, 0.39]
multi_means = [0.94, 0.37]
single_stds = [0.07, 0.18]
multi_stds = [0.31, 0.19]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.82, 5))
colors2 = plt.get_cmap('RdYlBu')(
        np.linspace(0.2, 0.8, 5))

rects1 = ax.bar(x - width/2, single_means, width, yerr=single_stds, label='w/o Multi-step', color = colors[0], hatch="//")
rects2 = ax.bar(x + width/2, multi_means, width, yerr=multi_stds,  label='INVIGORATE', color = colors2[4])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean L1 Error')
ax.set_title('Prediction Error Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#
# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()