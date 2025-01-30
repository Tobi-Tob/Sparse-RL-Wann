import argparse
import numpy as np

from wann_src.viewInd import viewInd
from wann_src import importNet
from domain import SparseMountainCarEnv, SparseMountainCarContiEnv
from domain.make_env import make_env


def visualize(args):
    # Task name mapping
    if args.task == 'SparseMountainCar':
        task_name = 'sparse_mountain_car'
    elif args.task == 'SparseMountainCarConti':
        task_name = 'sparse_mountain_car_conti'
    else:
        task_name = 'sparse_mountain_car'

    # Visualize the network graph
    fig, ax = viewInd(ind=args.input, taskName=task_name)
    fig.show()
    fig.savefig(f'{args.input.split(".")[0]}.pdf', bbox_inches='tight')

    # If the task is mountain_car, also generate the policy visualization
    if args.task == 'SparseMountainCar' or 'SparseMountainCarConti':
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

        # Initialize the environment
        env = make_env(args.task)

        # Generate the policy visualization
        print(f"Generating policy visualization for {args.task}...")
        color_mesh_fig, color_mesh_ax = env.visualize_model_policy(model=(wMat, aVec), granularity=granularity, action_bins=action_bins)
        color_mesh_fig.show()
        color_mesh_fig.savefig(f'{args.input.split(".")[0]}_Policy.pdf', bbox_inches='tight')


if __name__ == "__main__":
    # python visualizer.py -i log/test_best.out -t sparse_mountain_car
    # take log/smc_conti/smc_conti_1024_best/0032.out (has 14 connections)
    parser = argparse.ArgumentParser(description='Visualize evolved network graphs')

    parser.add_argument('-i', '--input', type=str,
                        help='Input model architecture', default='log/smc_conti/smc_conti_1024_best/0032.out')

    parser.add_argument("-t", "--task", type=str,
                        help="Task to use (SparseMountainCar, SparseMountainCarConti)", default="SparseMountainCarConti")

    visualize(parser.parse_args())
