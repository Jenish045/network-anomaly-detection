"""
Common utilities and helper functions for the Network Anomaly Detection System.

This module provides generic system utilities (e.g. file path assurance and JSON loading/saving)
to keep our data and model code focused on machine learning logic instead of file system operations.
"""

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: Path | str) -> Path:
    """
    Checks if the parent directory of the given file path exists, and creates it if not.

    This is extremely helpful for avoiding FileNotFoundError errors when saving models,
    plots, or metrics, especially during automated runs or fresh setups.
    """
    p = Path(path)
    # We make sure the folder structure exists. parents=True allows creating nested
    # directories (like outputs/plots/reports) in a single call.
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Path | str) -> None:
    """
    Serializes a Python object directly to JSON format.

    This wrapper automatically takes care of creating the parent directories before saving,
    ensuring we do not have duplicate directory-creation code scattered across our model modules.
    """
    p = ensure_directory(path)
    try:
        with open(p, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    except (TypeError, ValueError) as err:
        # If serialization fails (e.g. non-serializable objects in the dict), raise a clear error.
        raise TypeError(f"Failed to serialize object to JSON at {p}. Error: {err}")
    except OSError as err:
        raise OSError(f"Failed to write JSON file at {p}. Error: {err}")


def load_json(path: Path | str) -> Any:
    """
    Reads and deserializes a JSON file from disk.

    Raises FileNotFoundError with a clear explanation if the target file is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON configuration/metrics file not found at: {p}")

    try:
        with open(p, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON file at {p}. Failed to decode: {err}")
    except OSError as err:
        raise OSError(f"Failed to read JSON file at {p}. Error: {err}")
