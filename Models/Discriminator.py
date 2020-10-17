import torch
import torch.functional as F

class Discriminator(torch.nn.module):
    def __init__(self, side_length):
        super(Discriminator, self).__init__()
        f1, w1, f2 = [64, 5, 32]
        self.conv_1 = torch.nn.Conv2d(1, f1, w1)
        self.conv_2 = torch.nn.Conv2d(f1, f2, w1)

        size1 = (side_length - w1) + 1
        self.fc_size = f2 * size1 ** 2

        self.disc_fc_1 = torch.nn.Linear(self.fc_size, 64)
        self.disc_fc_2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.actor_layer1.weight)
        torch.nn.init.xavier_uniform_(self.actor_layer2.weight)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = x.view(-1, self.fc_size)
        x = self.disc_fc_1(x)
        x = F.relu(x)
        x = self.disc_fc_2(x)
        return x
