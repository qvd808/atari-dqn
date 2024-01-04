import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Model, self).__init__()

        self.middle = 128

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )
     
        self.fc = nn.Sequential(
            nn.Linear(6 * 6 * 32, self.middle),
            nn.ReLU(),
            nn.Linear(self.middle, output_size)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def choose_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, self.output_size)
        else:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(dim=0).to(self.device)
            q_value = self.forward(state)
            action = torch.argmax(q_value).item()
            return action
    
    def process_buffer(self, buffer):
        state = torch.Tensor( np.array([item.state for item in buffer]) ).to(self.device)
        next_state = torch.Tensor( np.array([item.next_state for item in buffer]) ).to(self.device)
        action = torch.Tensor( np.array([item.action for item in buffer]) ).to(self.device)
        reward = torch.Tensor( np.array([item.reward for item in buffer]) ).to(self.device)
        done = torch.Tensor( np.array([(1 if item.done else 0) for item in buffer]) ).to(self.device)

        return state, next_state, action, reward, done

    def save_model(self, path):
        torch.save(self.state_dict(), path)