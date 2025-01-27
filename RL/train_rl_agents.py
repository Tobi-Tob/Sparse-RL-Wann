import os

import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def train_agent(args):
    # Initialize the environment
    if args.task == 'sparse_mountain_car':
        from domain import SparseMountainCarEnv
        env = make_vec_env(SparseMountainCarEnv, n_envs=1)
    elif args.task == 'sparse_mountain_car_conti':
        from domain import SparseMountainCarContiEnv
        env = make_vec_env(SparseMountainCarContiEnv, n_envs=1)
    elif args.task == 'mountain_car':  # Standard MountainCar-v0
        import gym
        env = make_vec_env(lambda: gym.make('MountainCar-v0'), n_envs=1)
    elif args.task == 'lunar_lander':
        raise ValueError(f"Task {args.task} is not supported yet")
    else:
        raise ValueError(f"Task {args.task} is not supported")

    time_steps = 10000

    # Initialize the model
    if args.algo == 'PPO':
        hyperparams = {
            "learning_rate": 1e-2,
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
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=None, **hyperparams)
    elif args.algo == 'A2C':
        hyperparams = {
            "learning_rate": 1e-3,
            "n_steps": 5,
            "policy_kwargs": dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[8, 8]
            ),
        }
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=None, **hyperparams)
    elif args.algo == 'DQN':
        hyperparams = {
            "learning_rate": 1e-4,
            "buffer_size": 10000,
            "learning_starts": 1000,
            "batch_size": 32,
            "target_update_interval": 500,
            "exploration_fraction": 0.1,  # 10% of the time a random action is taken
            "exploration_initial_eps": 1.0,  # Initial value of epsilon in the epsilon-greedy exploration
            "exploration_final_eps": 0.05,  # Final value of epsilon in the epsilon-greedy exploration
            "policy_kwargs": dict(  # define the network architecture of the model
                net_arch=[8, 8]
            ),
        }
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
    model.learn(total_timesteps=time_steps, callback=eval_callback)

    # Save the model
    model_save_path = os.path.join(args.outdir, f"{args.task}_{args.algo}")
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")
