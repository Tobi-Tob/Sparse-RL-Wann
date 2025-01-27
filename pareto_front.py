from wann_src.nsga_sort import nsga_sort
import numpy as np
import matplotlib.pyplot as plt
import argparse


def show_pareto_front(args):
    objVals = []
    if isinstance(args.input, str):
        objVals = np.loadtxt(args.input, delimiter=',')

    # print("Rows: ", objVals.shape[0])
    # print("Cols: ", objVals.shape[1])
    col1 = args.gen * 3
    col2 = col1 + 2
    obj1 = objVals[:, col1]
    obj2 = objVals[:, col2]
    objVals = np.c_[obj1, 1 / obj2]
    # print("MeanFit: ", obj1)
    # print("#conns: ", obj2)

    # Perform NSGA-II sorting to get Pareto fronts
    rank, fronts = nsga_sort(objVals, returnFronts=True)
    # print("Pareto Fronts: ", fronts)

    # Visualization of all points and Pareto fronts
    plt.figure(figsize=(10, 8))

    # Plot all points
    plt.scatter(objVals[:, 0], objVals[:, 1], label="All Points", alpha=0.5, color="gray")

    # Highlight the first Pareto fronts with a unique color
    colors = ["red", "orange", "green"]
    for i, front in enumerate(fronts[:args.nFronts]):  # Only process the first x fronts
        front_points = objVals[front]
        print(f"Pareto Front {i + 1}: {front_points}")
        plt.scatter(
            front_points[:, 0],
            front_points[:, 1],
            color=colors[i % len(colors)],
            alpha=0.5,
            label=f"Pareto Front {i + 1}",
            # edgecolor="black",
            # s=100
        )

    # Add titles, labels, and legend
    plt.title(f"Pareto Front Visualization (Gen={args.gen})", fontsize=16)
    plt.xlabel(f"MeanFit", fontsize=12)
    plt.ylabel(f"1/#conns", fontsize=12)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # python pareto_front.py -i log/lula/lula_256_480_objVals.out -g 255 -n 3
    parser = argparse.ArgumentParser(description='Visualize MOO Pareto Fronts')

    parser.add_argument('-i', '--input', type=str,
                        help='input objective values', default='log/lula/lula_256_480_objVals.out')

    parser.add_argument('-g', '--gen', type=int,
                        help='generation of the population', default=0)

    parser.add_argument('-n', '--nFronts', type=int,
                        help='number of pareto fronts to highlight', default=3)

    show_pareto_front(parser.parse_args())
