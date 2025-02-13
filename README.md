# Exploring Sparse Reward Environments with Weight Agnostic Neural Networks

![Python](https://img.shields.io/badge/python-3.8-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
Sparse reward environments present a significant challenge in reinforcement learning (RL), as agents receive little to no feedback for extended periods, making effective learning difficult. Traditional RL algorithms struggle in these settings without human-engineered feedback to guide training.

In this work, we explore a novel direction using Weight Agnostic Neural Networks (WANNs), which leverage evolutionary search to discover proper network architectures that, even without trained weights, are capable of performing such tasks. We evaluate WANNs on modified versions of the **MountainCar** and **LunarLander** environments, where rewards are only given upon successful task completion. Our results demonstrate that WANNs can successfully learn compact, interpretable policies in these sparse settings, whereas conventional RL methods fail without auxiliary rewards.

This project builds on the original WANN framework from [Google Brain Tokyo Workshop](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease).

## Directory Structure
```
.
├── wann_train.py        # Evolutionary search for networks solving the task
├── wann_test.py         # Evaluation and visualization of trained WANNs
├── visualizer.py        # Network structure and policy visualization tool
├── pareto_front.py      # Displays the Pareto front (fitness vs. complexity)
├── Sparse-Reward-WANN.pdf # Project report summary
│
├── wann_src/            # Helper functions for WANN EA process
├── p/                   # JSON config files for experiment parameters
├── domain/              # Task environments
├── champions/           # Best evolved models stored here
├── RL/                  # PPO, DQN, and Q-Learning implementations
│
└── requirements.txt      # Required dependencies
```

## Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/Tobi-Tob/WANN-Sparse-RL.git
cd WANN-Sparse-RL
pip install -r requirements.txt
```

## Running Experiments

### 1. Train a WANN
Run evolutionary search to discover a weight-agnostic network for a given environment:
```bash
python wann_train.py
```

### 2. Test and Visualize WANN Policies
Evaluate a trained WANN model on an environment:
```bash
python wann_test.py
```
Visualize the discovered network structure and policy decisions:
```bash
python visualizer.py
```

## Results & Findings
Our experiments show that **WANNs successfully solve sparse reward RL tasks**, while traditional RL methods (PPO, DQN, Q-Learning) fail without additional rewards. Key findings include:
- **WANNs evolve compact and interpretable policies** that generalize well.
- **Traditional RL agents fail to learn** in purely sparse environments.
- **Pareto analysis** reveals a tradeoff between model simplicity and task performance.


