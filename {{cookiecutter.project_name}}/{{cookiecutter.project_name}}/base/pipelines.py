from abc import ABC

class Pipeline(ABC):
    def __init__(self):
        self.intermediate_results: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any], submodule_source: str) -> "Pipeline":
        config = deepcopy(config)

        cls_kwargs = {}
        for key, value in config.items():
            if isinstance(value, dict) and "class" in value:
                import_path = f"{{cookiecutter.project_name}}.{submodule_source}.{value.pop('class')}"
                submodule_cls = importlib.import_module(import_path)
                submodule = submodule_cls.from_config(value)
                cls_kwargs[key] = submodule

            else:
                cls_kwargs[key] = value
                
        return cls(**cls_kwargs)

    @abstractmethod
    def __call__(self, sample: DatasetSample) -> TensorType["batch", ...]:
        raise NotImplementedError()