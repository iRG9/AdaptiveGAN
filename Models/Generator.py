import torch.nn.functional as F
import torch


class Generator(torch.nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        f1, w1, stride = [4, 4, 2]

        self.conv_t1 = torch.nn.ConvTranspose2d(input_dim, f1, w1, stride, 1)
        self.conv_t2 = torch.nn.ConvTranspose2d(f1, f1 * 4, w1, stride, 1)
        self.conv_t3 = torch.nn.ConvTranspose2d(f1 * 4, f1 * 8, w1, stride, 1)
        self.conv_t4 = torch.nn.ConvTranspose2d(f1 * 8, f1 * 2, w1, stride, 1)
        self.conv_t5 = torch.nn.ConvTranspose2d(f1 * 2, 3, w1, 2, 1)

    def forward(self, x):
        # Initial size (1) *100
        x = self.conv_t1(x)
        x = F.relu(x)


        # Size  = (f1) * 2 * 2
        x = self.conv_t2(x)
        x = F.relu(x)


        # Size  = (f1*4) * 4 * 4
        x = self.conv_t3(x)
        x = F.relu(x)

        # Size  = (f1*8) * 8 * 8
        x = self.conv_t4(x)
        x = F.relu(x)


        # Size  = (f1*2) * 16 * 16
        x = self.conv_t5(x)
        x = torch.tanh(x)

        # Size  = (3) * 32 * 32
        return x
