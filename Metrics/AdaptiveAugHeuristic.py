import torch


class Heuristic:
    def __init__(self):
        self.ls = []

    def get_heur_inf_dec(self, disc_out):
        h = torch.sign(disc_out).sum() / len(disc_out)
        self.ls.append(h)
        if h > 0:
            return True
        return False
