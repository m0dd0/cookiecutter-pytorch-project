"""This module mainly contains the End2EndProcessor class. This class is intended
to unify the full processing pipeline from input to output. IT however is not intended
to be used for exporting or visualization. Depending on the use case it might be
useful to add more converters to the End2EndProcessor class.
"""

from typing import Dict, Any, Callable

import torch

from {{cookiecutter.project_name}}.datatypes import DatasetSample
from {{cookiecutter.project_name}}.preprocessing import PreprocessorBase, {{cookiecutter.dataset_name}}Preprocessor
from {{cookiecutter.project_name}}.postprocessing import PostprocessorBase, Postprocessor
from {{cookiecutter.project_name}}.models import {{cookiecutter.model_name}}


class End2EndProcessor:
    def __init__(
        self,
        model: {{cookiecutter.model_name}} = None,
        preprocessor: PreprocessorBase = None,
        postprocessor: PostprocessorBase = None,
        # more converters here
    ):
        if model is None:
            model = GenerativeResnet.from_state_dict_path()
        self.model = model

        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor

        if postprocessor is None:
            postprocessor = Postprocessor()
        self.postprocessor = postprocessor

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