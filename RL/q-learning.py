"""
Repository: Q-Learning Mountain Car
GitHub Link: https://github.com/guillaumefrd/q-learning-mountain-car/tree/master
Permalink: https://github.com/guillaumefrd/q-learning-mountain-car/tree/fd2c88dcbe0f810cfa8824edc2318f94e40f8430
"""

import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from domain.make_env import make_env
import argparse

from gym.wrappers import RecordVideo


class Agent:
    def __init__(self,
                 lr_init,
                 lr_min,
                 lr_decay_rate,
                 gamma,
                 epsilon,
                 epsilon_decay_rate,
                 num_bins,
                 num_actions=3):

        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_decay_rate = lr_decay_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.num_bins = num_bins
        self.state = None
        self.action = None

        # turn the continuous state space into a discrete space (with bins)
        # for the two observations: car position and car velocity
        self.discrete_states = [
            np.linspace(-1.2, 0.6, num=(num_bins + 1))[1:-1],
            np.linspace(-0.07, 0.07, num=(num_bins + 1))[1:-1],
        ]
        # MR: this creates the bins for the state space -> each value is the border between two bins

        # MR: Q-Table can be visualized well
        # initialize the Q-table with zeros
        self.num_actions = num_actions
        num_states = self.num_bins ** len(self.discrete_states)
        self.q = np.zeros(shape=(num_states, self.num_actions))
        print("Q-table shape (number_states, number_actions):", self.q.shape)
        print("Discretize bins of state space:", self.discrete_states)

    def to_state(self, observation):
        # turn the observation features into a space represented by an integer
        state = sum(np.digitize(feature, self.discrete_states[i]) * (self.num_bins ** i)
                    for i, feature in enumerate(observation))
        return state

    def start_episode(self, observation):
        # apply decay on exploration
        self.epsilon *= (1 - self.epsilon_decay_rate)

        # apply decay on learning rate
        self.lr = max(self.lr_min, self.lr * (1 - self.lr_decay_rate))

        # return the first action of the episode
        self.state = self.to_state(observation)
        return np.argmax(self.q[self.state]) if self.num_actions > 1 else np.float32(self.q[self.state])

    def make_action(self, observation, reward):
        next_state = self.to_state(observation)

        if (1 - self.epsilon) <= np.random.uniform():
            # make a random action to explore
            if self.num_actions > 1:  # discrete actions
                next_action = np.random.randint(0, self.num_actions)
            else:  # continuous actions
                next_action = np.float32(np.random.uniform(-1.0, 1.0))
        else:
            # take the best action
            if self.num_actions > 1:  # discrete actions
                next_action = np.argmax(self.q[next_state])
            else:  # continuous actions
                next_action = np.float32(self.q[next_state])

        # update the Q-table
        if self.num_actions > 1:  # discrete actions
            self.q[self.state, self.action] += self.lr * \
                  (reward + self.gamma * np.max(self.q[next_state, :]) -
                   self.q[self.state, self.action])
        else:  # continuous actions
            self.q[self.state] += self.lr * \
                   (reward + self.gamma * np.max(self.q[next_state]) -
                    self.q[self.state])

        self.state = next_state
        self.action = next_action  # Only necessary for discrete action space
        return next_action


class Monitor:
    def __init__(self,
                 num_episodes):

        self.num_episodes = num_episodes
        self.rewards = np.zeros(num_episodes, dtype=float)
        self.sparse_rewards = np.zeros(num_episodes, dtype=float)
        self.episode_plot = None
        self.avg_plot = None
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.rewards[episode_index]

    def __setitem__(self, episode_index, value_tuple):
        episode_reward, sparse_reward = value_tuple  # Unpack the tuple
        self.rewards[episode_index] = episode_reward
        self.sparse_rewards[episode_index] = sparse_reward

    def create_plot(self):
        plt.style.use("ggplot")
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        #self.fig.canvas.manager.set_title("Episode Reward History")
        self.fig.suptitle("Episode Reward History")  # Updated line
        self.ax.set_xlim(0, self.num_episodes + 5)
        self.ax.set_ylim(-210, -110)
        self.ax.set_title("Episode Reward History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Total Reward")
        self.episode_plot, = plt.plot([], [], linewidth=0.5, alpha=0.5,
                                      c="#1d619b", label="reward per episode")
        self.avg_plot, = plt.plot([], [], linewidth=3.0, alpha=0.8, c="#df3930",
                                  label="average reward over the 200 last episodes")
        self.ax.legend(loc="upper left")

    def update_plot(self, episode_index):
        # update the episode plot
        x = range(episode_index)
        y = self.rewards[:episode_index]
        self.episode_plot.set_xdata(x)
        self.episode_plot.set_ydata(y)

        # update the average plot
        mean_kernel_size = 201
        rolling_mean_data = np.concatenate((np.full(mean_kernel_size, fill_value=-200),
                                            self.rewards[:episode_index]))
        rolling_mean_data = pd.Series(rolling_mean_data)

        rolling_means = rolling_mean_data.rolling(window=mean_kernel_size,
                                                  min_periods=0).mean()[mean_kernel_size:]
        self.avg_plot.set_xdata(range(len(rolling_means)))
        self.avg_plot.set_ydata(rolling_means)

        plt.draw()
        plt.pause(0.0001)


def videos_to_record(episode_id):
    return episode_id in [1, 100, 500, 1000, 2000, 3500, 4900]


def create_fig(x, y1, y2, filename=None):
    plt.figure(figsize=(10, 8))

    # Plot all points
    plt.plot(x, y1, color="orange", label="Total Reward")
    plt.plot(x, y2, color="blue", label="Goal Reward", alpha=0.5)

    # Add titles, labels, and legend
    plt.title(f"Episode Reward History", fontsize=22)
    plt.xlabel(f"Episode", fontsize=20)
    plt.ylabel(f"Total Reward", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)

    plt.savefig(filename, bbox_inches='tight') if filename else None
    plt.show()


def main(args):
    # parameters
    verbose = False
    seed = 42
    num_episodes = args.episodes
    plot_redraw_frequency = 10

    # create the environment
    env = make_env(args.task)

    # set seed to reproduce the same results
    env.seed(seed)
    np.random.seed(seed)

    # monitor the training
    env = RecordVideo(env, video_folder=args.outdir, episode_trigger=videos_to_record)

    if "Conti" in args.task:
        num_actions = 1
    elif "LunarLander" in args.task:
        num_actions = 4
    else:  # SparseMountainCar
        num_actions = 3
    agent = Agent(
        lr_init=0.8,
        lr_min=1e-5,
        lr_decay_rate=5e-4,
        gamma=0.99,  # 0.98,
        epsilon=1.0,  # 0.9,
        epsilon_decay_rate=3e-3,  # 5e-3,
        num_bins=4,  # 15
        num_actions=num_actions,
    )

    '''
    # MR: Modifications to the original settings because more exploration is needed in sparse reward environments
    agent = Agent(
        lr_init=0.1,  # 0.3,
        lr_min=1e-3,  # 1e-5,
        lr_decay_rate=5e-4,
        gamma=0.99,  # 0.98,
        epsilon=1.0,  # 0.9,
        epsilon_decay_rate=1e-3,  # 5e-3,
        num_bins=30,  # 15
    )
    '''

    monitor = Monitor(num_episodes=num_episodes)
    monitor.create_plot()

    for episode_index in range(num_episodes):
        observation = env.reset()
        action = agent.start_episode(observation)
        total_reward = 0
        sparse_reward = 0  # stores reward if goal was reached, no intermediate reward
        timestep = 0
        done = False
        stop = False

        while not done and not stop:
            # make an action and get the new observations
            observation, reward, done, stop, info = env.step(action)
            total_reward += reward
            if reward > 5:
                sparse_reward =+ reward
            timestep += 1

            if verbose:
                env.render(mode="rgb_array")
                print("Timestep: {0:3d}, Action: {1:2d}, Reward: {2:5.1f}, Car \
                      position: {3:6.3f}, Car velocity: {4:6.3f}"
                      .format(timestep, action, reward, *observation))

            # compute the next action
            action = agent.make_action(observation, reward)

        print("Episode {} finished after {} timesteps (total reward: {})"
              .format(episode_index + 1, timestep, total_reward))

        # update the plot
        monitor[episode_index] = total_reward, sparse_reward
        if verbose or episode_index % plot_redraw_frequency == 0:
            monitor.update_plot(episode_index)

    # save the history in a csv file
    df = pd.DataFrame({
        "reward": list(monitor.rewards),
        "sparse_reward": list(monitor.sparse_rewards)
    })
    df.to_csv(os.path.join(args.outdir, "history.csv"), index_label="episode")

    # Create and save the plot
    create_fig(df.index.to_list(), df['reward'].to_list(), df['sparse_reward'].to_list(), os.path.join(args.outdir, "history.pdf"))

    env.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")

    parser.add_argument("-t", "--task", type=str,
                        help="Task to use (SparseMountainCar, SparseMountainCarConti, LunarLander)", default="LunarLander")
    parser.add_argument("-o", "--outdir", type=str, help="Directory to save the logs", default="q-learning_logs/lula")
    parser.add_argument("-e", "--episodes", type=int, help="Number of Episodes", default=5001)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    main(args)
