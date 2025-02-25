"""View and record the performance of a WANN with various weight values.

TODO: Parallelize evaluation

"""

import numpy as np
import argparse
import sys

np.set_printoptions(precision=2)
np.set_printoptions(linewidth=160)

from wann_src import *
from domain import *


def main(argv):
    infile = args.infile
    outPref = args.outPref
    hyp_default = args.default
    hyp_adjust = args.hyperparam
    nMean = args.nVals
    nRep = args.nReps
    view = args.view
    seed = args.seed

    # Load task and parameters
    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)
    task = Task(games[hyp['task']], nReps=hyp['alg_nReps'])

    # Bullet needs some extra help getting started
    if hyp['task'].startswith("bullet"):
        task.env.render("human")

    # Import individual for testing
    wVec, aVec, wKey = importNet(infile)

    # Show result
    fitness, wVals = task.getDistFitness(wVec, aVec, hyp,
                                         nVals=nMean, nRep=nRep,
                                         view=view, returnVals=True, seed=seed)

    print("[***]\tFitness:", fitness, '\n' + "[***]\tWeight Values:\t", wVals)
    lsave(outPref + 'reward.out', fitness)
    lsave(outPref + 'wVals.out', wVals)


# -- --------------------------------------------------------------------- -- #
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    ''' Parse input and launch '''
    # python wann_test.py -i log/test_best.out -p p/sparse_mountain_car.json -n 6 -r 1 -v True
    parser = argparse.ArgumentParser(description='Test WANNs on Task')

    parser.add_argument('-i', '--infile', type=str,
                        help='file name for genome input', default='champions/smc_conti_best.out')

    parser.add_argument('-o', '--outPref', type=str,
                        help='file name prefix for result input', default='champions/result_smc_conti_best_')

    parser.add_argument('-d', '--default', type=str,
                        help='default hyperparameter file', default='p/default_wan.json')

    parser.add_argument('-p', '--hyperparam', type=str,
                        help='hyperparameter file', default='p/sparse_mountain_car_conti.json')

    parser.add_argument('-n', '--nVals', type=int,
                        help='Number of weight values to test', default=6)

    parser.add_argument('-r', '--nReps', type=int,
                        help='Number of repetitions to calculate average over', default=100)

    parser.add_argument('-v', '--view', type=str2bool,
                        help='Visualize trial?', default=False)

    parser.add_argument('-s', '--seed', type=int,
                        help='random seed', default=-1)

    args = parser.parse_args()
    main(args)
