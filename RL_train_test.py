import os
import argparse
from RL.train_rl_agents import train_agent
from RL.eval_rl_agents import evaluate_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")

    parser.add_argument("-a", "--algo", type=str, help="Algorithm to use (PPO, A2C, DQN)", default="PPO")
    parser.add_argument("-t", "--task", type=str,
                        help="Task to use (SparseMountainCar, SparseMountainCarConti)", default="LunarLander")
    parser.add_argument("-o", "--outdir", type=str, help="Directory to save the model and logs", default="RL/log_2")
    parser.add_argument("-m", "--mode", type=str, help="Mode to run (train, test)", default="train")
    parser.add_argument("-s", "--saved", type=str,
                        help="Input dir to the model for testing", default="RL/log_2")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "train":
        train_agent(args)
    elif args.mode == "test":
        evaluate_agent(args)
    else:
        raise ValueError(f"Mode {args.mode} not valide")
