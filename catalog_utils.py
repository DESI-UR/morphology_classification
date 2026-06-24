# catalog_utils.py

import pandas as pd
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# Load a CSV or FITS catalog
def load_catalog(path, hdu=0): 
    
    if path.endswith(".csv"):
        return pd.read_csv(path)
        
    elif path.endswith(".fits"):        
        return Table.read(path, hdu=hdu).to_pandas()
        
    else:
        raise ValueError("Currently only CSV and FITS supported.")

# Rename useful columns
def standardize_columns(df, ra, dec, id, *, g_mag=None, r_mag=None, z_mag=None, diam=None, morph=None, t_type=None, m_type=None, path=None):

    standards = {
        # Input name: "Standard name",
        ra: "Target_RA",
        dec: "Target_DEC",
        id: "TargetID",
        g_mag: "mag_g",
        r_mag: "mag_r",
        z_mag: "mag_z",
        diam: "diam",
        morph: "Morphology",
        t_type: "T_type",
        m_type: "Main_Type",
        path: "Path"
    }
    cleaned_standards = {input_name: std_name for input_name, std_name in standards.items() if input_name is not None}
    return df.rename(columns=cleaned_standards)

# Remove galaxies via coordinate matching. Useful for excluding binaries
def remove_objects(df, sep_arcsec, save_removed=False):
    coords = SkyCoord(ra=df["Target_RA"].values * u.deg,
                      dec=df["Target_DEC"].values * u.deg)

    idx, sep2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)

    mask = sep2d.arcsec > sep_arcsec
    if save_removed == True:
        return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
    return df[mask].reset_index(drop=True)

# Remove galaxies via existing catalog. Useful for excluding already classified
def remove_reference_catalog(target_df, ref_df, sep_arcsec=4.0, save_removed=False):
    target_coords = SkyCoord(
        ra=target_df["Target_RA"].values * u.deg,
        dec=target_df["Target_DEC"].values * u.deg
    )

    ref_coords = SkyCoord(
        ra=ref_df["Target_RA"].values * u.deg,
        dec=ref_df["Target_DEC"].values * u.deg
    )

    idx, sep2d, _ = target_coords.match_to_catalog_sky(ref_coords)

    mask = sep2d.arcsec > sep_arcsec
    if save_removed == True:
        return target_df[mask].reset_index(drop=True), target_df[~mask].reset_index(drop=True)
    return target_df[mask].reset_index(drop=True)