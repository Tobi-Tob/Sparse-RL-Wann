import argparse
import numpy as np

from wann_src.viewInd import viewInd
from wann_src import importNet
from domain import SparseMountainCarEnv, SparseMountainCarContiEnv


def visualize(args):
    # Visualize the network graph
    fig, ax = viewInd(ind=args.input, taskName=args.task)
    fig.savefig(f'{args.input}.png')

    # If the task is mountain_car, also generate the policy visualization
    if args.task == 'sparse_mountain_car' or 'sparse_mountain_car_conti':
        # Define parameter
        shared_weight = 1.0  # The one shared weight value that is used for all connections in the network
        granularity = 100  # granularity of the resolution of the state space
        action_bins = 20  # used for the discretion of continuous action spaces

        # Import the model weights and architecture
        wVec, aVec, _ = importNet(args.input)
        wVec[np.isnan(wVec)] = 0
        dim = int(np.sqrt(np.shape(wVec)[0]))
        cMat = np.reshape(wVec, (dim, dim))
        cMat[cMat != 0] = 1.0
        wMat = np.copy(cMat) * shared_weight

        # Create the task environment
        if args.task == 'sparse_mountain_car':
            env = SparseMountainCarEnv()
        elif args.task == 'sparse_mountain_car_conti':
            env = SparseMountainCarContiEnv()

        # Generate the policy visualization
        print(f"Generating policy visualization for {args.task}...")
        color_mesh_fig, color_mesh_ax = env.visualize_model_policy(model=(wMat, aVec), granularity=granularity, action_bins=action_bins)
        color_mesh_fig.savefig(f'{args.input}Policy.png')


if __name__ == "__main__":
    # python visualizer.py -i log/test_best.out -t sparse_mountain_car
    parser = argparse.ArgumentParser(description='Visualize evolved network graphs')

    parser.add_argument('-i', '--input', type=str,
                        help='Input model architecture', default='log/smcc_best/0028.out')

    parser.add_argument('-t', '--task', type=str,
                        help='Task the model was trained for', default='sparse_mountain_car_conti')

    visualize(parser.parse_args())
