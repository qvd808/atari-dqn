import gymnasium as gym
from relay_buffer import ReplayBuffer
from agent import Agent
import torch
import numpy as np
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import itertools
import random

if __name__ == "__main__":
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
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
    agent.policy_net.load_state_dict(torch.load("./model/model.pth"))

    episode = 100_000
    for i in range(10_000):
        if random.random() < 0.4:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        relay_buffer.add(
            state = obs,
            reward = reward,
            action = action,
            next_state = next_obs,
            done = done
        )
        obs = next_obs
        if done:
            obs, info = env.reset()
            continue
        
    epsilon_max = 0.3
    epsilon_min = 0.1
    decay_rate = 0.001
    rewards = []
    reward_per_run = 0

    obs, info = env.reset()
    for i in range(episode):
        epsilon = np.interp(i, [0, 20_000], [epsilon_max, epsilon_min])
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)
        
        next_obs, reward, done, truncated, info = env.step(action)
        relay_buffer.add(
            state = obs,
            reward = reward,
            action = action,
            next_state = next_obs,
            done = done
        )
        reward_per_run += reward
        obs = next_obs
        if done:
            obs, info = env.reset()
            rewards.append(reward_per_run)
            reward_per_run = 0
            continue
        
        if len(relay_buffer) > 10_000 and i % 4 == 0:
            agent.compute_loss()
        
        if len(relay_buffer) > 10_000 and i % 2_000 == 0:
            agent.update_target_net()
            if i % 10_000 == 0:
                torch.save(agent.policy_net.state_dict(), f"model_{i}.pth")
                torch.save(agent.policy_net.state_dict(), f"model.pth")
        
        if i % 1_000 == 0:
            print(f"Episode: {i}, Reward: {np.mean(rewards[-101:-1])}, Epsilon: {epsilon}")