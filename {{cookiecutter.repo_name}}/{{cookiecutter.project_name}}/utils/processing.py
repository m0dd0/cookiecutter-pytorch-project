"""This module mainly contains the End2EndProcessor class. This class is intended
to unify the full processing pipeline from input to output. IT however is not intended
to be used for exporting or visualization. Depending on the use case it might be
useful to add more converters to the End2EndProcessor class.
"""

from typing import Dict, Any, Callable
import importlib
from copy import deepcopy

import torch

from {{cookiecutter.project_name}}.datatypes import DatasetSample
from {{cookiecutter.project_name}}.preprocessing import PreprocessorBase, {{cookiecutter.dataset_name}}Preprocessor
from {{cookiecutter.project_name}}.postprocessing import PostprocessorBase, Postprocessor
from {{cookiecutter.project_name}}.models import {{cookiecutter.model_name}}


class End2EndProcessor:
    @staticmethod
    def check_model_path(model_path: Path) -> Path:
        model_path = Path(model_path)
        
        if not model_path.exists():
            model_path = Path(__file__).parent.parent / "checkpoints" / model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        
        return model_path

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "End2EndProcessor":
        config = deepcopy(config)

        # load the model
        if "jit" in config["model"]:
            jit_path = cls.check_model_path(config["model"].pop("jit"))
            device = config["model"].pop("device", None)
            
            if len(config["model"]) > 0:
                raise ValueError("If a jit model is used, only jit path and device can be specified. No other model parameters are allowed.")
            
            model = torch.jit.load(jit_path)
            
            if device is not None:
                model = model.to(device)

        else:
            state_dict_path = cls.check_model_path(config["model"].pop("state_dict_path"))
            model_cls = importlib.import_module(config["model"].pop("class"))
            device = config["model"].pop("device", None)
            model_args = config["model"]
            model = model_cls.from_state_dict_path(state_dict_path, device, **model_args)

        # load the preprocessor
        preprocessor_cls = importlib.import_module(config["preprocessor"].pop("class"))
        preprocessor = preprocessor_cls.from_config(config["preprocessor"])

        # load the postprocessor
        postprocessor_cls = importlib.import_module(config["postprocessor"].pop("class"))
        postprocessor = postprocessor_cls.from_config(config["postprocessor"])

        # more converters here
        # ...

        return cls(model=model, preprocessor=preprocessor, postprocessor=postprocessor)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "End2EndProcessor":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    def __init__(
        self,
        model: {{cookiecutter.model_name}} = None,
        preprocessor: PreprocessorBase = None,
        postprocessor: PostprocessorBase = None,
        # more converters here
    ):
        self.model = model or {{cookiecutter.model_name}}.from_state_dict_path()

        self.preprocessor = preprocessor or {{cookiecutter.dataset_name}}Preprocessor
        self.postprocessor = postprocessor or Postprocessor()

        # more converters here

    def __call__(self, sample: DatasetSample) -> Dict[str, Any]:

        input_tensor = self.preprocessor(sample)
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            model_output = self.model(input_tensor)

        result = self.postprocessor(model_output)

        # further processing here with more converters here

        process_data = {
            "preprocessor": self.preprocessor.intermediate_results,
            "postprocessor": self.postprocessor.intermediate_results,
            "model_input": input_tensor,
            "sample": sample,
            "result": result,
        }

        return process_data