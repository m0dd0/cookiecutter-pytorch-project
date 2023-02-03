""" This modules contains the postprocessing pipelines. A postprocessing pipeline should
accept the output of the model (i.e. a tensor) and return a usable, interpretable result.
The result type should be a subclass of ResultBase.
The postprocessing pipeline should also store intermediate results in the
intermediate_results dictionary. These results can be used in the end for closer evauation
and debugging.
Since a pipeline is not used in backpropagation it is not necessary to implement it as
a torch.nn.Module.
A pipeline should consist out of multiple submodules that are called in a specific order.
For clarity it should be avoided to have logic inthe pipeline itself. Instead the logic
should be implemented in the submodules. The pipeline itself should only manage the
flow of information through the submodules. If a pipeline has no or only a single
submodule it might be more suitable to implement it as a submodule instead of a pipeline
for improved reusability.
Somteimes it might be useful to have a subpipeline that is used in multiple pipelines.
This subpipeline will output an intermediate result that is used in multiple pipelines
but not the final result. Therefore this pipeline should not inherit from PreprocessorBase.
Keep in mind that you need to manage the intermediate result of the subpipeline yourself.
Also it might be usefule to have further pipelines which process the result further.
These pipelines might take an ResultBase as input and process it further with more
information etc. These pipelines should NOT inherit from PreprocessorBase.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

# from torchvision import transforms as T
# import torch
# import numpy as np
from torchtyping import TensorType

from {{cookiecutter.project_name}}.datatypes import ResultBase, SomeResult
from . import custom_transforms as CT


class PostprocessorBase(ABC):
    def __init__(self):
        self.intermediate_results: Dict[str, Any] = {}

    @abstractmethod
    def __call__(self, network_output: TensorType) -> ResultBase:
        pass


class Postprocessor(PostprocessorBase):
    def __init__(self):
        super().__init__()

        # configration
        # TODO add configuration options here
        # self.do_something = True

        # submodules
        # TODO add submodules here
        # self.example_submodule = CT.ExampleSubmodule()
        # self.example_submodule_2 = CT.ExampleSubmodule()

    def __call__(self, network_output: TensorType) -> SomeResult:
        # TODO implement preprocessing pipeline
        # result = self.example_submodule(sample)
        # if self.do_something:
        #     result = self.example_submodule_2(network_input)
        raise NotImplementedError()
        return result


# ... other postprocessors here which might use SomeResult as input and process it
# further with more information etc.