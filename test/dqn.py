import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self, observation_space, action_space) -> None:
        super(DQN, self).__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 81 , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x