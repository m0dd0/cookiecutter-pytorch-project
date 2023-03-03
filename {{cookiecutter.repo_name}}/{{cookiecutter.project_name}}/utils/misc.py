from pathlib import Path


def get_root_dir() -> Path:
    return Path(__file__).parent.parent


def exists_in_subfolder(path: Path, subfolder: Path) -> Path:
    path = Path(path)

    if not path.exists():
        path = subfolder / path

    if not path.exists():
        raise FileNotFoundError(f"Model path {path} does not exist.")

    return path