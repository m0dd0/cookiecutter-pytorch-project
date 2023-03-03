"""The class in this modules acts as a base class for all models in the project.
It contains useful generic functions for consructing models from config files or
from state dict paths. Other models should inherit from this class."""

from abc import ABC
from pathlib import Path

import torch
import torch.nn as nn

from grconvnet.utils.misc import get_root_dir, exists_in_subfolder


class BaseModel(nn.Module, ABC):
    @classmethod
    def from_jit(cls, jit_path: Path = None, device: str = None) -> "BasepModel":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        jit_path = (
            jit_path
            or "cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_15_iou_97.pt"
        )

        jit_path = exists_in_subfolder(jit_path, get_root_dir() / "checkpoints")

        # we changed the jitted model class so we only take the state_dict
        # the jitted model expects the default parameters
        # normally we would simply return model=torch.jit.load(jit_path)
        model = cls()
        model.load_state_dict(torch.jit.load(jit_path).state_dict())

        model.to(device)

        return model

    @classmethod
    def from_state_dict_path(
        cls, state_dict_path: Path = None, device: str = None, **kwargs
    ) -> "BaseModel":
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        state_dict_path = exists_in_subfolder(
            state_dict_path, get_root_dir() / "checkpoints"
        )

        model = cls(**kwargs)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))

        model.to(device)

        return model

    # TODO define other methods here which might be shared by all models
    # e.g. save, load, error, ...
