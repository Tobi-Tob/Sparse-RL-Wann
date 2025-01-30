import os

import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from domain.make_env import make_env
from gym.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor


def videos_to_record(episode_id):
    return episode_id in [1, 100, 500, 1000, 2000, 3500, 4900]


def train_agent(args):
    # Initialize the environment
    env = make_env(args.task)
    filename = os.path.join(args.outdir, f"monitor_{args.task}_{args.algo}.csv")
    # env = Monitor(env)

    time_steps = 100000

    # Initialize the model
    if args.algo == 'PPO':
        hyperparams = {
            "learning_rate": 0.01,
            "n_steps": 200,  # Number of steps until the policy is updated
            "batch_size": 50,
            "n_epochs": 20,
            "gae_lambda": 0.98,
            "ent_coef": 6.5e-06,
            "vf_coef": 0.26,
            "max_grad_norm": 1,
            "policy_kwargs": dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[4]
            ),
        }
        log_interval = 10
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=None, **hyperparams)
    elif args.algo == 'A2C':
        hyperparams = {
            "learning_rate": 0.001,
            "n_steps": 5,
            "policy_kwargs": dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[4]
            ),
        }
        log_interval = 1000
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=None, **hyperparams)
    elif args.algo == 'DQN':
        hyperparams = {
            "learning_rate": 1e-2,
            "buffer_size": 10000,
            "learning_starts": 100,
            "batch_size": 32,
            "target_update_interval": 500,
            "exploration_fraction": 0.1,  # 10% of the time a random action is taken
            "exploration_initial_eps": 1.0,  # Initial value of epsilon in the epsilon-greedy exploration
            "exploration_final_eps": 0.05,  # Final value of epsilon in the epsilon-greedy exploration
            "policy_kwargs": dict(  # define the network architecture of the model
                net_arch=[4]
            ),
        }
        log_interval = 40
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=None, **hyperparams)
    else:
        raise ValueError(f"Algorithm {args.algo} is not supported")

    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=args.outdir,
        log_path=args.outdir,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    # Train the agent
    print(f"Training {args.algo} on {args.task}...")
    model.learn(total_timesteps=time_steps, callback=eval_callback, log_interval=log_interval)

    # If env wrpped in Monitor class of
    # print(f"Episode Rewards: {env.get_episode_rewards()}")
    # print(f"Episode Lengths: {env.get_episode_lengths()}")

    # Save the model
    model_save_path = os.path.join(args.outdir, f"{args.task}_{args.algo}")
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")
