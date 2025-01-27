import argparse
from wann_src.viewInd import viewInd


def show_network(args):
    fig, ax = viewInd(ind=args.input, taskName=args.task)
    fig.show()
    fig.savefig(f'{args.input}.png')


if __name__ == "__main__":
    # python visualizer.py -i log/test_best.out -t sparse_mountain_car
    parser = argparse.ArgumentParser(description='Visualize evolved network graphs')

    parser.add_argument('-i', '--input', type=str,
                        help='input model architecture', default='log/smcc_best.out')

    parser.add_argument('-t', '--task', type=str,
                        help='task the model was trained for', default='sparse_mountain_car_conti')

    show_network(parser.parse_args())
