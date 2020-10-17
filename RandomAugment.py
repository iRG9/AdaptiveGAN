import torchvision.transforms as transforms
import numpy as np
import random


class RandomAugment:
    def __init__(self):
        self.p = 0
    # Creating Set of Transforms
    def apply_transforms(self, inp):
        inp = transforms.ToPILImage()(inp)
        transformz = [transforms.Grayscale(num_output_channels=3), transforms.ColorJitter(),
                      transforms.RandomAffine(360), transforms.RandomPerspective(0.5)]
        picked = []
        for e in transformz:
            if random.random() <= self.p:
                picked.append(e)
        transfrm = transforms.Compose(picked)
        return transforms.ToTensor()(transfrm(inp))