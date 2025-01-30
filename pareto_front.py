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
    objVals_inv = np.c_[obj1, 1 / obj2]
    objVals = np.c_[obj1, obj2]
    # print("MeanFit: ", obj1)
    # print("#conns: ", obj2)

    # Perform NSGA-II sorting to get Pareto fronts
    rank, fronts = nsga_sort(objVals_inv, returnFronts=True)
    # print("Pareto Fronts: ", fronts)

    # Visualization of all points and Pareto fronts
    plt.figure(figsize=(10, 8))

    # Plot all points
    plt.scatter(objVals[:, 0], objVals[:, 1], label="All Individuals", alpha=0.7, color="blue", s=70)

    # Highlight the first Pareto fronts with a unique color
    colors = ["red", "orange", "green"]
    for i, front in enumerate(fronts[:args.nFronts]):  # Only process the first x fronts
        front_points = objVals[front]
        print(f"Pareto Front {i + 1}:\n{front_points}")
        plt.scatter(
            front_points[:, 0],
            front_points[:, 1],
            color=colors[i % len(colors)],
            alpha=0.9,
            label=f"Pareto Front {i + 1}",
            edgecolor="black",
            s=130
        )

    # Add titles, labels, and legend
    # plt.title(f"Pareto Front Visualization (Gen={args.gen})", fontsize=22)
    plt.title(f"Pareto Front Visualization", fontsize=22)
    plt.xlabel(f"MeanFit", fontsize=20)
    plt.ylabel(f"#conns", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(args.savePath, bbox_inches='tight') if args.save else None

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # python pareto_front.py -i log/lula/lula_256_480_objVals.out -g 255 -n 3
    # Gen 40 für smc_conti_1024 -> sinnvoller Plot  log/smc_conti/smc_conti_1024_objVals.out
    # Gen 80 für smc_1024 -> sinnvoller Plot  log/smc/smc_1024_objVals.out
    parser = argparse.ArgumentParser(description='Visualize MOO Pareto Fronts')

    parser.add_argument('-i', '--input', type=str,
                        help='input objective values', default='log/smc/smc_1024_objVals.out')

    parser.add_argument('-g', '--gen', type=int,
                        help='generation of the population', default=80)

    parser.add_argument('-n', '--nFronts', type=int,
                        help='number of pareto fronts to highlight', default=1)

    parser.add_argument('-s', '--save', type=bool,
                        help='save the fig?', default=True)

    parser.add_argument('-p', '--savePath', type=str,
                        help='path to save the fig to. Only active if -s is True', default='log/smc/smc_1024_pareto_gen80_v2.pdf')

    show_pareto_front(parser.parse_args())
