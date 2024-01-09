import gymnasium as gym
from relay_buffer import ReplayBuffer
from agent import Agent
import torch
import numpy as np
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import itertools
import random

if __name__ == "__main__":
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

    obs, info = env.reset()
    relay_buffer = ReplayBuffer(500_000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        relay_buffer=relay_buffer,
        device=device
    )
    agent.policy_net.load_state_dict(torch.load("model.pth"))

    
    epsilon_max = 1.0
    epsilon_min = 0.1
    decay_rate = 0.0001
    rewards = []
    reward_per_run = 0

    obs, info = env.reset()
    i = 0
    while True:
        i += 1
        action = agent.act(obs)
        
        next_obs, reward, done, truncated, info = env.step(action)
        reward_per_run += reward

        obs = next_obs
        if done:
            obs, info = env.reset()
            rewards.append(reward_per_run)
            reward_per_run = 0
            continue
        
        
        if i % 1_000 == 0:
            print(f"Episode: {i}, Reward: {np.mean(rewards[-101:-1])}")