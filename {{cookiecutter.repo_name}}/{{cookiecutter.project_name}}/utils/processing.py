"""This module mainly contains the End2EndProcessor class. This class is intended
to unify the full processing pipeline from input to output. IT however is not intended
to be used for exporting or visualization. Depending on the use case it might be
useful to add more converters to the End2EndProcessor class.
"""
from typing import Dict, Any, Callable, List, Tuple
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import yaml


from {{cookiecutter.project_name}}.dataloading.datasets import {{cookiecutter.dataset_name}}
from {{cookiecutter.project_name}}.datatypes import {{cookiecutter.dataset_name}}Sample
from {{cookiecutter.project_name}}.postprocessing import Postprocessor, PostprocessorBase
from {{cookiecutter.project_name}}.models import {{cookiecutter.model_name}}
from {{cookiecutter.project_name}}.utils import visualization as vis
from {{cookiecutter.project_name}}.utils.export import Exporter
from {{cookiecutter.project_name}}.utils.config import module_from_config


def process_dataset(
    dataloader: DataLoader,
    model: {{cookiecutter.model_name}},
    postprocessor: PostprocessorBase,
    img2world_converter: Img2WorldConverter,
    exporter: Exporter,
    device: str,
):
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        preprocessor_results = list(dataloader.dataset.transform.intermediate_results)[
            -len(batch) :
        ]
        samples = [res["initial_sample"] for res in preprocessor_results]

        with torch.no_grad():
            prediction_batch = model(batch)

        postprocessed_batch = [postprocessor(pred) for pred in prediction_batch]
        postprocessor_results = list(postprocessor.intermediate_results)[-len(batch) :]

        for sample, pre_result, post_result in zip(
            samples,
            preprocessor_results,
            postprocessor_results,
            postprocessed_batch
        ):
            fig = vis.overview_fig(
                fig=plt.figure(figsize=(20, 20)),
                # .... TODO
            )
            plt.close(fig)

            export_data = {
                # TODO
            }

            _ = exporter(export_data, f"{sample.name}")


def process_dataset_from_config(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exporter = module_from_config(config["exporter"])
    dataloader = module_from_config(config["dataloader"])
    model = module_from_config(config["model"])
    postprocessor = module_from_config(config["postprocessor"])
    img2world_converter = module_from_config(config["img2world_converter"])

    shutil.copy(config_path, Path(config["exporter"]["export_dir"]) / "config.yaml")

    process_dataset(
        dataloader,
        model,
        postprocessor,
        exporter,
        config["model"]["device"],
    )


if __name__ == "__main__":
    config_path = get_root_dir() / "configs" / "ycb_inference.yaml"

    process_dataset_from_config(config_path)
