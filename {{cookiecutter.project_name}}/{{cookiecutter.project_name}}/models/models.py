""" This module contains the network architecture(s) of the project."""

import torch.nn as nn

from torchtyping import TensorType

from .custom_modules import CustomModuleExample


class {{cookiecutter.model_name}}(nn.Module):
    @classmethod
    def from_state_dict_path(
        cls, model_path: Path = None, device: str = None
    ) -> "{{cookiecutter.mode_name}}":
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if model_path is None:
            model_path = (
                Path(__file__).parent.parent
                / "checkpoints"
                / "default_weights.pt"
            )

        model = cls()
        model.load_state_dict(torch.jit.load(model_path).state_dict())
        model.to(device)

        return model

    def __init__(self):
        super().__init__()

        # TODO: configure the modules of the network here

        raise NotImplementedError()

    def forward(self, x: TensorType) -> TensorType:
        # TODO: implement the forward pass of the network here
        
        raise NotImplementedError()
    