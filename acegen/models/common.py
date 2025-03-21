import torch

class Temperature(torch.nn.Module):
    """Implements a temperature layer.

    Simple Module that applies a temperature value to the logits for RL inference.

    Args:
        temperature (float): The temperature value.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperature: torch.tensor) -> torch.Tensor:
        return logits / temperature