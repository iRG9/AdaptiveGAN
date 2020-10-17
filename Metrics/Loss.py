import torch
import torch.nn


class Loss:
    def __init__(self):
        self.criterion = torch.nn.CrossEntropy()

    def disc_loss(self, real, gen):
        return self.criterion(real, torch.ones_like(real)) + self.criterion(gen, torch.zeros_like(gen))

    def gen_loss(self, disc_out):
        return self.criterion(disc_out, torch.ones_like(disc_out))

