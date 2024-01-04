import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import numpy as np
from relay_buffer import RelayBuffer, Buffer
from model import Model
import matplotlib.pyplot as plt
import torch

def plot_reward(reward_list, episode_list):
    plt.plot(episode_list, reward_list)
    plt.show()

if __name__ == "__main__":

    n_training_episode = 100_000
    max_step = 1

    ## Environment and things related
    env = gym.make("ALE/Pong-v5", render_mode = "rgb_array")
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

    #Parameters
    gamma = 0.99
    epsilon = 1
    min_epsilon = 0.05
    max_expsilon = 1
    epsilon_decay = 0.005

    MIN_MEMORY = 100
    # MIN_MEMORY = 10_000
    
    #Plot
    reward_list = []
    episode_list = []
    epsilons_list = []
    reward_per_run = 0
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plt.ion()

    for episode in range(n_training_episode):
        
        epsilon = min_epsilon + (max_expsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        
        for step in range(max_step):

            action = model.choose_action(obs, epsilon)
            new_obs, reward, terminate, done, info = env.step(action)
            reward_per_run += reward

            item = Buffer(
                state=obs, 
                next_state=new_obs,
                action=action,
                reward = reward,
                done=done
            )

            relay_buffer.push(item)

            if len(relay_buffer) > MIN_MEMORY:
                if episode % 3000 == 0:

                    with torch.no_grad():
                        buffer = relay_buffer.sample(min(50_000, len(relay_buffer)))
                        state_batch, next_state_batch, reward_batch, action_batch, done_batch = model.process_buffer(buffer)
                        # import ipdb; ipdb.set_trace()
                        q_value = model(state_batch).to(model.device)
                        q_value_next_state = model(next_state_batch)
                        expected_q_value = reward + gamma * q_value_next_state * (1 - done)

                        loss = (q_value - expected_q_value).pow(2).mean()
                        loss.requires_grad = True
                        # Optimize the model
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()

                    # buffer = relay_buffer.sample(MIN_MEMORY)
                    # state_batch, next_state_batch, reward_batch, action_batch, done_batch = model.process_buffer(buffer)
                    # # import ipdb; ipdb.set_trace()
                    # q_value = model(state_batch).to(model.device)
                    # q_value_next_state = model(next_state_batch)
                    # expected_q_value = reward + gamma * q_value_next_state * (1 - done)

                    # loss = (q_value - expected_q_value).pow(2).mean()
                    # # Optimize the model
                    # model.optimizer.zero_grad()
                    # loss.backward()
                    # model.optimizer.step()

            ## Reset obs
            obs = new_obs
            if terminate or done:
                obs, info = env.reset()
                break


        reward_list.append(reward_per_run)
        reward_per_run = 0
        episode_list.append(episode)
        epsilons_list.append(epsilon)

        if episode % 500 == 0:

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
            if episode % 10_000 == 0:
                model.save_model(f"model_step_{episode}.pt")

    plt.tight_layout()
    plt.show()
    model.save_model("model_pong.pt")
