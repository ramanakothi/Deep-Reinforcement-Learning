## Continues Control using Deep Deterministic Policy Gradient (DDPG) agent


## Introduction

The goal of this project is to train an agent deep reinforcement learning algorithms so that it can control the robot arm to touch the ball which is moving around it as long as it can. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

## Environment details

The environment is based on Unity ML-agents. The project environment provided by Udacity is similar to the Reacher environment on the Unity ML-Agents GitHub page.

Set-up: Double-jointed arm which can move to target locations.
Goal: The agents must move it's hand to the goal location, and keep it there.
Agents: The Unity environment contains 10 agent linked to a single Brain.

The provided Udacity agent versions are Single Agent or 20-Agents

Agent Reward Function (independent): +0.1 Each step agent's hand is in goal location.
Brains: One Brain with the following observation/action space.
Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
Visual Observations: None.
Reset Parameters: Two, corresponding to goal size, and goal movement speed.
Benchmark Mean Reward: 30

This implementation will solve the version of the environment with Single Agent using the off-policy DDPG algorithm. This activity is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.


## Hyperparamters

After exploring several combinations, values below for hyperparameters allow the agent to solve the problem in stable manner.

Hyperparameter	Value
Batch size	64
Gamma	0.99
τ	1e-3
LR_ACTOR	1e-4
LR_CRITIC	1e-4
WEIGHT_DECAY	0
N_step	10
UPDATE_EVERY	5
Vmax	5
Vmin	0
N_ATOMS	51
Network Structures

Actor

Layer	Dimension
Input	N x 33
Linear Layer, Leaky Relu	N x 256
Linear Layer, Leaky Relu	N x 4
Batchnormalization1D	N x 4
Tanh Output	N x 4
Critic

Layer	Dimension
Input	N x 33
Linear Layer, Leaky Relu	N x 128
Linear Layer + Actor Output, Leaky Relu	N x (128 + 4)
Linear Layer, Leaky Relu	N x 128
Linear Layer	N x 51

## Training Results

Below are the number of episodes needed to solve the environment and the evolution of rewards per episode during training.

![Random Agent]("images/Screen Shot 2019-12-19 at 1.53.57 PM.png") 
