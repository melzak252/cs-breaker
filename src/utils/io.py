"""IO and serialization utilities for Kedro 8-layer simulation and benchmark pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def compute_sha256(path: Path | str) -> str:
    """Compute the SHA-256 digest of a file in streaming chunks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder handling NumPy scalar and array types, NaN, and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj) if math.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Save serializable data to JSON using custom encoder and atomic file write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=indent, cls=NumpyJSONEncoder)
    tmp_path.replace(path)
