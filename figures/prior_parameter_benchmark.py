import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
from numpy import genfromtxt

def main(args):

    x = genfromtxt('./figures/prior_metric.csv', delimiter=',')
    y = genfromtxt('./figures/prior_parameter_loc.csv', delimiter=',')

    colors = genfromtxt('./figures/prior_parameter_a.csv', delimiter=',')
    area = (genfromtxt('./figures/prior_parameter_scale.csv', delimiter=','))**2  # 0 to 15 point radii

    pu.figure_setup()

    fig_size = pu.get_fig_size(8, 8)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    scatter = ax.scatter(x, y, s=area, c=colors, label=area, edgecolors='black')

    pu.figure_setup()

    ax.set_xlabel('Metric [-]')
    ax.set_ylabel('Loc [-]')
    ax.set_axisbelow(True)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper center", title="Classes")
    ax.add_artist(legend1)

    # produce a legend with a cross-section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend2 = ax.legend(handles, ['$\\mathdefault{10}$', '$\\mathdefault{15}$'], loc="center", title="Sizes")

    plt.grid()
    plt.tight_layout()

    if args.save:
        pu.save_fig(fig, args.save)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)