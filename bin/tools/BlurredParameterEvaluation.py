#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy
import sys
import csv
import glob


def read_file(filename):
    results = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            results.append(row)
    results = results[1:]
    return numpy.asarray(results, dtype=numpy.float32)


def plot_bars(two_dimension_array, title="", thinout=10):
    two_dimension_array = two_dimension_array.copy()[::, ::thinout]
    plt.boxplot(two_dimension_array)
    plt.xticks(numpy.arange(0, two_dimension_array.shape[1], thinout), numpy.arange(0, two_dimension_array.shape[1] * thinout, thinout * thinout))
    plt.title(title)


def plot_stats(two_dimension_array, title="", thinout=1):
    two_dimension_array = two_dimension_array.copy()[::, ::thinout]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(numpy.arange(0, two_dimension_array.shape[1] * thinout, thinout),
             numpy.std(two_dimension_array, axis=0), label="Std", color=color)
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Std value", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(numpy.arange(0, two_dimension_array.shape[1] * thinout, thinout),
             numpy.mean(two_dimension_array, axis=0), label="Mean", color=color)
    ax2.set_ylabel("Mean value", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()


def main():
    if len(sys.argv) < 2:
        return
    folder = sys.argv[1]

    above_tRet = None
    below_tRet = None
    above_tTra = None
    below_tTra = None

    for file in glob.glob(folder+"/*.csv"):
        iteration_file = read_file(file)

        above_tRet_it = iteration_file[:, 1]
        if above_tRet is None:
            above_tRet = above_tRet_it
        else:
            above_tRet = numpy.column_stack((above_tRet, above_tRet_it))

        below_tRet_it = iteration_file[:, 3]
        if below_tRet is None:
            below_tRet = below_tRet_it
        else:
            below_tRet = numpy.column_stack((below_tRet, below_tRet_it))

        above_tTra_it = iteration_file[:, 4]
        if above_tTra is None:
            above_tTra = above_tTra_it
        else:
            above_tTra = numpy.column_stack((above_tTra, above_tTra_it))

        below_tTra_it = iteration_file[:, 6]
        if below_tTra is None:
            below_tTra = below_tTra_it
        else:
            below_tTra = numpy.column_stack((below_tTra, below_tTra_it))

    above_tRet = numpy.swapaxes(above_tRet, -1, 0)
    plot_bars(above_tRet, title='Above tRet')
    plt.savefig("/tmp/above_tRet_bars.png", dpi=300)
    plt.close()
    plot_stats(above_tRet, title='Above tRet')
    plt.savefig("/tmp/above_tRet_stats.png", dpi=300)
    plt.close()

    below_tRet = numpy.swapaxes(below_tRet, -1, 0)
    plot_bars(below_tRet, title='Below tRet')
    plt.savefig("/tmp/below_tRet_bars.png", dpi=300)
    plt.close()
    plot_stats(below_tRet, title='Below tRet')
    plt.savefig("/tmp/below_tRet_stats.png", dpi=300)
    plt.close()

    above_tTra = numpy.swapaxes(above_tTra, -1, 0)
    plot_bars(above_tTra, title='Above tTra')
    plt.savefig("/tmp/above_tTra_bars.png", dpi=300)
    plt.close()
    plot_stats(above_tTra, title='Above tTra')
    plt.savefig("/tmp/above_tTra_stats.png", dpi=300)
    plt.close()

    below_tTra = numpy.swapaxes(below_tTra, -1, 0)
    plot_bars(below_tTra, title='Below tTra')
    plt.savefig("/tmp/below_tTra_bars.png", dpi=300)
    plt.close()
    plot_stats(below_tTra, title='Below tTra')
    plt.savefig("/tmp/below_tTra_stats.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
