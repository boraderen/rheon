from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EncodingSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categories: dict[str, list[str]]


def fit_encoding_schema(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    min_category_count: int = 1,
) -> EncodingSchema:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    categories: dict[str, list[str]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        means[column] = float(values.mean())
        std = float(values.std(ddof=0))
        stds[column] = std if std > 0 else 1.0
    for column in categorical_columns:
        counts = df[column].fillna("UNK").astype(str).value_counts()
        cats = sorted([str(value) for value, count in counts.items() if count >= min_category_count])
        if "UNK" not in cats:
            cats.append("UNK")
        categories[column] = cats
    return EncodingSchema(numeric_columns, categorical_columns, means, stds, categories)


def transform_with_encoding_schema(df: pd.DataFrame, schema: EncodingSchema) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for column in schema.numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").fillna(schema.numeric_mean[column])
        standardized = (values - schema.numeric_mean[column]) / schema.numeric_std[column]
        blocks.append(pd.DataFrame({column: standardized.astype(float)}))
    for column in schema.categorical_columns:
        known = set(schema.categories[column])
        values = df[column].fillna("UNK").astype(str).map(lambda value: value if value in known else "UNK")
        one_hot = pd.DataFrame(
            {
                f"{column}={category}": (values == category).astype(float)
                for category in schema.categories[column]
            }
        )
        blocks.append(one_hot)
    if not blocks:
        return pd.DataFrame(index=df.index)
    return pd.concat(blocks, axis=1)


def stable_dirichlet_probs(size: int, rng: np.random.Generator) -> np.ndarray:
    probs = rng.dirichlet(np.ones(size))
    return probs / probs.sum()
