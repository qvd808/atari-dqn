from dqn import DQN
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class Agent():
    def __init__(
            self,
            observation_space,
            action_space,
            relay_buffer,
            device,
            learning_rate=5e-5,
            gamma=0.99,
            batch_size=32,
        ) -> None:
        
        self.device = device
        self.relay_buffer = relay_buffer
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size

        self.policy_net = DQN(
            observation_space,
            action_space
        ).to(self.device)
        
        self.target_net = DQN(
            observation_space,
            action_space
        ).to(self.device)

        self.update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def compute_loss(self):
        obs, act, rew, next_obs, done = self.relay_buffer.sample(self.batch_size)
        obs = np.asarray(obs) / 255.0
        next_obs = np.asarray(next_obs) / 255.0
        
        obs_t = torch.from_numpy(obs).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        act_t = torch.from_numpy(act).float().to(self.device)
        rew_t = torch.from_numpy(rew).float().to(self.device)
        done_t = torch.from_numpy(done).float().to(self.device)

        with torch.no_grad():
            max_action = self.policy_net(next_obs_t).argmax(dim=1)
            q_next_max = self.target_net(next_obs_t).gather(1, max_action.unsqueeze(1)).squeeze(1)

            target_q_max = rew_t + (1 - done_t) * self.gamma * q_next_max
        
        current_max_action = self.policy_net(obs_t).argmax(dim = 1)
        current_max_q_val = self.policy_net(obs_t).gather(1, current_max_action.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(current_max_q_val, target_q_max)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def act(self, obs):
        obs = np.asarray(obs) / 255.0
        obs = torch.from_numpy(obs).float().to(self.device)
        q_val = self.policy_net(obs.unsqueeze(dim = 0))
        action = q_val.argmax()
        return action.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
