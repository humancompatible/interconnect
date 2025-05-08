from abc import ABC, abstractmethod
import torch


class InputSpace(ABC):
    @abstractmethod
    def get_random_point(self):
        pass


class Hypercube(InputSpace):
    def __init__(self, dimension, center, side):
        self.dimension = dimension
        self.lb = torch.ones(dimension) * center - side / 2
        self.ub = torch.ones(dimension) * center + side / 2
        self.side = side

    def get_random_point(self):
        return (torch.rand(self.dimension) * self.side + self.lb).requires_grad_()


if __name__ == '__main__':
    cube = Hypercube(2, 1, 1)
    print(cube.get_random_point())
