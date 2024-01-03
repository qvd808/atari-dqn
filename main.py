import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import numpy as np
from relay_buffer import RelayBuffer, Buffer
from model import Model

if __name__ == "__main__":

    n_training_episode = 10_000
    max_step = 600

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

    MIN_MEMORY = 100
    

    for episode in range(n_training_episode):
        for step in range(max_step):
            action = env.action_space.sample()

            new_obs, reward, terminate, done, info = env.step(action)
            
            item = Buffer(
                state=obs, 
                next_state=new_obs,
                action=action,
                reward = reward,
                done=done
            )

            relay_buffer.push(item)
            if len(relay_buffer) > MIN_MEMORY:
                buffer = relay_buffer.sample(MIN_MEMORY)
                state_batch, next_state_batch, reward_batch, action_batch, done_batch = model.process_buffer(buffer)

            ## Reset obs
            obs = new_obs
            if terminate or done:
                obs, info = env.reset()
                break