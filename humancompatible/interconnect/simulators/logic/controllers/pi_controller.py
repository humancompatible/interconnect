import torch


class PiControllerLogic:
    def __init__(self, kappa=0.000001, alpha=0.000001):
        self.tensors = {"kappa": torch.tensor([kappa], requires_grad=True, dtype=torch.float),
                        "alpha": torch.tensor([alpha], requires_grad=True, dtype=torch.float),
                        "e": torch.tensor([0.0], requires_grad=True),
                        "e_prev": torch.tensor([0.0], requires_grad=True),
                        "pi_prev": torch.tensor([0.0], requires_grad=True)
                        }
        self.variables = ["e"]

    def forward(self, values):
        # controller accepts error (agg1_output = refsig + (-filter))
        self.tensors["e"] = values["e"]
        # Compute the output based on input values
        result = (self.tensors["pi_prev"] + self.tensors["kappa"] * (
                    self.tensors["e"] - self.tensors["alpha"] * self.tensors["e_prev"]))
        # Update internal state
        self.tensors["e_prev"] = self.tensors["e"]
        self.tensors["pi_prev"] = result
        return result