import gymnasium as gym

if __name__ == "__main__":

    n_training_episode = 10_000
    max_step = 600

    env = gym.make("ALE/Pong-v5", render_mode = "human")
    obs, info = env.reset()

    for episode in range(n_training_episode):
        for step in range(max_step):
            action = env.action_space.sample()

            obs, reward, terminate, done, info = env.step(action)
            
            if terminate or done:
                obs, info = env.reset()
                break