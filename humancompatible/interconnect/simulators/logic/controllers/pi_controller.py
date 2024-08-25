import torch


class PiControllerLogic:
    def __init__(self, kappa=0.5, alpha=0.1, sp=0.0):
        self.tensors = {"S": torch.tensor([0.0], requires_grad=True),
                        "kappa": torch.tensor([kappa], requires_grad=True, dtype=torch.float),
                        "alpha": torch.tensor([alpha], requires_grad=True, dtype=torch.float),
                        "sp": torch.tensor([sp], requires_grad=True, dtype=torch.float),
                        "e": torch.tensor([0.0], requires_grad=True),
                        "e_prev": torch.tensor([0.0], requires_grad=True),
                        "pi_prev": torch.tensor([0.0], requires_grad=True)
                        }
        self.variables = ["S"]

    def forward(self, values):
        self.tensors["S"] = torch.tensor([values["S"]], requires_grad=True, dtype=torch.float)
        self.tensors["e"] = self.tensors["sp"] - self.tensors["S"]
        result = (self.tensors["pi_prev"] + self.tensors["kappa"] * (
                    self.tensors["e"] - self.tensors["alpha"] * self.tensors["e_prev"]))
        self.tensors["e_prev"] = self.tensors["e"]
        self.tensors["pi_prev"] = result
        return result
