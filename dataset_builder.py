# dataset_builder.py

import pandas as pd
import numpy as np


def append_anchors(target_df, anchor_df):
    """
    Append anchor dataset to targets.
    """
    return pd.concat([target_df, anchor_df], ignore_index=True)


def enforce_main_type(df, value=30):
    """
    Required classifier column.
    """
    df = df.copy()
    df["Main_type"] = value
    return df


def ensure_unique_ids(df, id_col="TargetID", prefix="T"):
    """
    Prevent ID collisions safely (no manual edits needed).
    """
    df = df.copy()

    seen = set()
    new_ids = []

    for i, x in enumerate(df[id_col]):
        if x in seen:
            x = f"{prefix}_{x}_{i}"
        seen.add(x)
        new_ids.append(x)

    df[id_col] = new_ids
    return df


def build_training_dataset(
    target_df,
    anchor_df,
    output_path
):
    """
    FULL PIPELINE STEP:
    target + anchors → clean classifier dataset
    """

    df = append_anchors(target_df, anchor_df)

    df = enforce_main_type(df, 30)

    df = ensure_unique_ids(df)

    # clean formatting for export
    if "Path" in df.columns:
        df["Path"] = df["Path"].astype(str)

    df.to_csv(output_path, index=False)

    return df