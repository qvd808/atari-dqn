import torch
from torch.nn import nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, input_size, output_size) -> None:

        self.middle = 128

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, self.middle),
            nn.ReLU(),
            nn.Linear(self.middle, output_size)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def choose_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.output_size + 1)
        else:
            state = torch.from_numpy(state).float().unsqueeze().to(self.device)
            q_value = self.forward(state)
            action = torch.argmax(q_value)
            return action