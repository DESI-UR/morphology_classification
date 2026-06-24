# cutout_manager.py

import os
import numpy as np
from tqdm import tqdm


def get_raslice(ra):
    """
    RA-based directory binning (scales to large datasets)
    """
    return f"{int(ra):03d}"


def build_cutout_path(cutout_dir, target_id, ra, dec):
    """
    Standardized cutout filename.
    """
    return os.path.join(
        cutout_dir,
        get_raslice(ra),
        f"{target_id}_grz_152_{ra:.4f}_{dec:.4f}.fits"
    )


def attach_cutout_paths(df, cutout_dir):
    """
    Adds a Path column without opening FITS files.
    """
    paths = []

    missing = []

    for row in tqdm(df.itertuples(), total=len(df)):
        path = build_cutout_path(
            cutout_dir,
            row.TargetID,
            row.Target_RA,
            row.Target_DEC
        )

        if os.path.exists(path):
            paths.append(path)
        else:
            paths.append("")
            missing.append(path)

    df = df.copy()
    df["Path"] = paths

    return df, missing


def verify_cutouts(df):
    """
    Quick sanity check.
    """
    missing = df[df["Path"] == ""]
    return len(missing), missing