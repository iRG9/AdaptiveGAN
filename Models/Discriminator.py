import torch.nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, side_length):
        super(Discriminator, self).__init__()
        f1, w1, f2 = [64, 5, 32]
        self.conv_1 = torch.nn.Conv2d(3, f1, w1)
        self.conv_2 = torch.nn.Conv2d(f1, f2, w1)

        size1 = (side_length - w1) + 1
        size2 = (size1-w1)+1
        self.fc_size = f2 * size2 ** 2

        self.disc_fc_1 = torch.nn.Linear(self.fc_size, 64)
        self.disc_fc_2 = torch.nn.Linear(64, 1)
        torch.nn.init.xavier_uniform_(self.disc_fc_1.weight)
        torch.nn.init.xavier_uniform_(self.disc_fc_2.weight)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = x.view(-1, self.fc_size)

        x = self.disc_fc_1(x)
        x = F.relu(x)

        x = self.disc_fc_2(x)
        x = self.sigmoid(x)
        return x
