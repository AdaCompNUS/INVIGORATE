import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from collections import OrderedDict

category_names = ["Object Detection Failure", "Grounding Failure", "OBG Detection Failure"]
results = []
results.append(['MAttNet+VMRN', 'w/o Uncertainty', 'w/o Multi-step', 'INVIGORATE'])
results.append([[8, 48, 4], [8, 51, 5], [10, 1, 11], [13, 6, 11]])

for j in range(4):
    sum_j = float(sum(results[1][j]))
    for i in range(3):
        results[1][j][i] = float(results[1][j][i]) / sum_j

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results[0])
    data = np.array(list(results[1]))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.4, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 3))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str("{:.1f}%".format(c * 100)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(-0.1, 1),
              loc='lower left', fontsize=12)

    return fig, ax


params = {'axes.labelsize': 20}
pylab.rcParams.update(params)

survey(results, category_names)
plt.show()