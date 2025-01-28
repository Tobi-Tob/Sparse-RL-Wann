import numpy as np
import gym


def make_env(env_name, seed=-1, render_mode=False):
    if "Bullet" in env_name:
        import pybullet as p
        import pybullet_envs
        import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

    # -- Bipedal Walker ------------------------------------------------ -- #
    if env_name.startswith("BipedalWalker"):
        if env_name.startswith("BipedalWalkerHardcore"):
            import Box2D
            from domain.bipedal_walker import BipedalWalkerHardcore
            env = BipedalWalkerHardcore()
        elif (env_name.startswith("BipedalWalkerMedium")):
            from domain.bipedal_walker import BipedalWalker
            env = BipedalWalker()
            env.accel = 3
        else:
            from domain.bipedal_walker import BipedalWalker
            env = BipedalWalker()

    # -- VAE Racing ---------------------------------------------------- -- #
    elif env_name.startswith("VAERacing"):
        from domain.vae_racing import VAERacing
        env = VAERacing()

    # -- Classification ------------------------------------------------ -- #
    elif env_name.startswith("Classify"):
        from domain.classify_gym import ClassifyEnv
        if env_name.endswith("digits"):
            from domain.classify_gym import digit_raw
            trainSet, target = digit_raw()

        if env_name.endswith("mnist256"):
            from domain.classify_gym import mnist_256
            trainSet, target = mnist_256()

        env = ClassifyEnv(trainSet, target)

    # -- Cart Pole Swing up -------------------------------------------- -- #
    elif env_name.startswith("CartPoleSwingUp"):
        from domain.cartpole_swingup import CartPoleSwingUpEnv
        env = CartPoleSwingUpEnv()

    # -- Sparse Mountain Car ------------------------------------------- -- #
    elif env_name.startswith("SparseMountainCar"):
        render_mode = input("Choose Render Mode? (human, rgb_array, None): ")
        use_sparse_reward = input("Use Sparse Reward? (0/1): ")
        if render_mode not in ["human", "rgb_array", "None"]:
            print(f"Render Mode {render_mode} not recognized. Using None.")
            render_mode = None
        if use_sparse_reward not in ["0", "1"]:
            print(f"Use Sparse Reward {use_sparse_reward} not recognized. Using True.")
            use_sparse_reward = True
        elif use_sparse_reward == "0":
            use_sparse_reward = False
        else:
            use_sparse_reward = True
        if env_name.startswith("SparseMountainCarConti"):
            from domain.sparse_mountain_car_conti import SparseMountainCarContiEnv
            env = SparseMountainCarContiEnv(render_mode=render_mode, use_sparse_reward=use_sparse_reward)
        else:
            from domain.sparse_mountain_car import SparseMountainCarEnv
            env = SparseMountainCarEnv(render_mode=render_mode, use_sparse_reward=use_sparse_reward)

    # -- Lunar Lander ------------------------------------------- -- #
    elif env_name.startswith("LunarLander"):
        if env_name.startswith("LunarLanderConti"):
            from domain.lunar_lander import LunarLanderEnv
            env = LunarLanderEnv(render_mode="human", continuous=True, enable_wind=False)
        else:
            from domain.lunar_lander import LunarLanderEnv
            env = LunarLanderEnv(render_mode=None, continuous=False, enable_wind=False)

    # -- Other  -------------------------------------------------------- -- #
    else:
        env = gym.make(env_name)

    if (seed >= 0):
        domain.seed(seed)

    return env
