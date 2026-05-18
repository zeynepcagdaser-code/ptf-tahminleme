from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def chronological_train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(data) < 3:
        raise ValueError("Kronolojik split icin yeterli veri yok.")

    test_start = int(len(data) * train_ratio)
    if test_start <= 0 or test_start >= len(data):
        raise ValueError("Kronolojik split icin yeterli veri yok.")

    train_val = data.iloc[:test_start]
    test = data.iloc[test_start:]

    val_start = int(len(train_val) * (1 - validation_ratio))
    if val_start <= 0 or val_start >= len(train_val):
        raise ValueError("Validation split icin yeterli train verisi yok.")

    train = train_val.iloc[:val_start].copy()
    validation = train_val.iloc[val_start:].copy()
    return train, validation, test.copy()


def project_relative_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
