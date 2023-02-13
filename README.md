# cookiecutter-data-science-project
This cookiecutter provides a template for data intensive projects. It is focussed on projects using pytorch DNNs. 

## Generated Project Structure
    {{cookiecutter.project_name}}
    ├── {{cookiecutter.project_name}}/
    │   ├── checkoints/
    │   ├── data/
    │   |   └── examples/
    │   ├── dataloading/
    │   |   └── datasets.py/
    │   ├── datatypes/
    │   |   └── datatypes.py/
    │   ├── models/
    │   |   ├── custom_modules.py/
    │   |   └── models.py/
    │   ├── postprocessing/
    │   |   ├── custom_modules.py/
    │   |   └── pipelines.py/
    │   ├── preprocessing/
    │   |   ├── custom_modules.py/
    │   |   └── pipelines.py/
    │   ├── results/
    │   ├── training/
    │   |   └── dataloaders.py/
    │   └── utils/
    │       ├── export.py/
    │       ├── misc.py/
    │       ├── processing.py/
    │       └── visualization.py/
    ├── notebooks/
    │   └── 00_experiments.ipynb/
    ├── .gitignore/
    ├── setup.py/
    ├── LICENSE/
    └── README.md/

## Design Decisions
TODO
