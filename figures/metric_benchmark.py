import argparse
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as pu
from numpy import genfromtxt

def main(args):

    x = genfromtxt('./figures/full_combi_test_cases.csv', delimiter=',')
    y = genfromtxt('./figures/full_combi_metric.csv', delimiter=',')

    x1 = genfromtxt('./figures/t_way_test_cases.csv', delimiter=',')
    y1 = genfromtxt('./figures/t_way_metric.csv', delimiter=',')

    x2 = genfromtxt('./figures/variance_bounded_test_cases.csv', delimiter=',')
    y2 = genfromtxt('./figures/variance_bounded_metric.csv', delimiter=',')

    pu.figure_setup()

    fig_size = pu.get_fig_size(15, 10)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    ax.plot(x, y, c='b', lw=pu.plot_lw(), label="Full combinatorial testing", marker='o')
    ax.plot(x1, y1, c='orange', lw=pu.plot_lw(), label="T-way testing", marker='o')
    ax.plot(x2, y2, c='green', lw=pu.plot_lw(), label="Variance bounded testing", marker='o')

    pu.figure_setup()

    ax.set_xlim(left=0, right=75)
    ax.set_xlabel('Test case amount [-]')
    ax.set_ylabel('Metric [-]')
    ax.legend()
    ax.set_axisbelow(True)

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