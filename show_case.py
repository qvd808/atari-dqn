import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import numpy as np
from relay_buffer import RelayBuffer, Buffer
from model import Model
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":

    n_training_episode = 10_000
    max_step = 1

    ## Environment and things related
    env = gym.make("ALE/Pong-v5", render_mode = "human")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, (80, 80))
    env =FrameStack(env, 4)
    obs, info = env.reset()

    ## Relay buffer
    relay_buffer = RelayBuffer(capacity=100_000)
    model = Model(
        input_size=env.observation_space.shape[0], 
        output_size=env.action_space.n
    )
    model.to(model.device) #Why on earth we need to do this?
    model.load_state_dict(torch.load("model_pong.pt"))

    #Parameters
    gamma = 0.99
    epsilon = 1
    min_epsilon = 0.05
    max_expsilon = 1
    epsilon_decay = 0.005

    # MIN_MEMORY = 10_000
    MIN_MEMORY = 1000
    
    #Plot
    reward_list = []
    episode_list = []
    epsilons_list = []
    reward_per_run = 0
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plt.ion()

    for episode in range(n_training_episode):
        
        epsilon = 0
        
        for step in range(max_step):

            action = model.choose_action(obs, epsilon)
            new_obs, reward, terminate, done, info = env.step(action)
            reward_per_run += reward

            ## Reset obs
            obs = new_obs
            if terminate or done:
                obs, info = env.reset()
                break


        reward_list.append(reward_per_run)
        reward_per_run = 0
        episode_list.append(episode)
        epsilons_list.append(epsilon)

        if episode % 1 == 0:

            axs[0].clear()
            axs[0].plot(episode_list, reward_list)
            axs[1].clear()
            axs[1].plot(episode_list, epsilons_list)
            plt.draw()
            plt.pause(0.001)

            #Detect if you are currently running in IPython
            if "get_ipython" in globals():
                from IPython import display
                display.clear_output(wait=True)
                display.display(fig)


    plt.tight_layout()
    plt.show()
