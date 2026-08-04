"""Dataset loading and source assignment."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset WITH measured SO3, required by Eq. (1). Supersedes new_data_points.txt,
# which omitted SO3; the strength column is identical between the two files.
#
# Searched in order, so the same code runs on Windows and Linux without editing:
#   1. the OPC_DATA environment variable, if set
#   2. data/ew-points_finalN.txt next to the package
#   3. ew-points_finalN.txt in the current directory or any parent
_DATA_FILENAME = "ew-points_finalN.txt"


def _find_default_data() -> str:
    import os
    if os.environ.get("OPC_DATA"):
        return os.environ["OPC_DATA"]
    here = Path(__file__).resolve().parent
    candidates = []
    for base in (here, *here.parents):
        candidates += [base / "data" / _DATA_FILENAME, base / _DATA_FILENAME]
    candidates.append(Path.cwd() / _DATA_FILENAME)
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        f"Could not find {_DATA_FILENAME}.\n"
        "Put it in a data/ folder next to the package, or set the OPC_DATA "
        "environment variable, or pass an explicit path:\n"
        "    load_dataset(r'D:\\KFUPM\\Adil\\Self-Verifying\\ew-points_finalN.txt')"
    )

OXIDES = ["SiO2", "Al2O3", "Fe2O3", "CaO", "SO3"]


def load_dataset(path: str | None = None) -> pd.DataFrame:
    path = path or _find_default_data()
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Curing time": "age", "Compressive Strength": "fc",
                            "w/c": "wc"})
    # One distinct oxide composition per literature source (Section III-A).
    df["source_id"] = df.groupby(OXIDES, sort=False).ngroup()
    return df.reset_index(drop=True)
