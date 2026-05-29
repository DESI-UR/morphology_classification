#!/usr/bin/env python3
"""
Generate 152x152 Legacy Surveys cutouts for the DESI Y3 Fundamental Plane (FP) sample,
and save matching PNG previews.

Dependencies: pandas, numpy, requests, astropy (for FITS), matplotlib
"""

import os
import sys
import time
import pathlib
import logging
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

# ======================
# --- User defaults (CLI can override)
# ======================
PIX_SCALE       = 0.262       # arcsec/pixel for Legacy Surveys
CUTOUT_SIZE_PIX = 152         # fixed 152x152 cutouts
BANDS           = "grz"       # bands to request
LAYER           = "ls-dr10"   # LS data release
VIEWER_URL      = "https://www.legacysurvey.org/viewer/cutout.fits"

TIMEOUT         = 15          # seconds per HTTP request
MAX_RETRIES     = 3           # total tries per target (including the first)
SUCCESS_DELAY   = 0.05        # polite pause after each successful download (seconds)

ASINH_NONLIN    = 2.5         # asinh stretch nonlinearity
P_LOW           = 0.5         # low percentile for scaling
P_HIGH          = 99.5        # high percentile for scaling

# ======================
# --- HTTP session (connection pooling)
# ======================
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=32,
    pool_maxsize=32,
    max_retries=0  # we'll do our own retries/backoff
)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)

# ======================
# --- Helpers
# ======================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def viewer_cutout_url(ra, dec, size_pix, bands=BANDS, layer=LAYER, pixscale=PIX_SCALE):
    """Build LS viewer cutout URL."""
    params = dict(ra=ra, dec=dec, layer=layer, size=size_pix, pixscale=pixscale, bands=bands)
    return f"{VIEWER_URL}?{urlencode(params)}"

def fetch_cutout(ra, dec, size_pix, outpath, session=SESSION, timeout=TIMEOUT,
                 max_retries=MAX_RETRIES, success_delay=SUCCESS_DELAY):
    """Download a single LS cutout FITS file via the viewer API with retries/backoff."""
    url = viewer_cutout_url(ra, dec, size_pix)
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            with open(outpath, "wb") as f:
                f.write(r.content)
            if success_delay > 0:
                time.sleep(success_delay)
            return True
        except Exception as e:
            wait = 2 + attempt  # 2s, 3s, 4s...
            logging.warning("Attempt %d failed for RA=%.6f DEC=%.6f: %s; sleeping %.1fs",
                            attempt+1, ra, dec, str(e), wait)
            time.sleep(wait)
    return False

def safe_stem(targetid, ra, dec, bands=BANDS, size_pix=CUTOUT_SIZE_PIX):
    """Generate a consistent filename stem."""
    return f"{int(targetid)}_{bands}_{size_pix}_{ra:.4f}_{dec:.4f}".replace(" ", "_")

def _percentile_clip(img, plo=P_LOW, phi=P_HIGH):
    finite = np.isfinite(img)
    if not np.any(finite):
        return 0.0, 1.0
    lo = np.nanpercentile(img[finite], plo)
    hi = np.nanpercentile(img[finite], phi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi

def _asinh_stretch(img, nonlin=ASINH_NONLIN, plo=P_LOW, phi=P_HIGH):
    """Asinh stretch into 0..1 using percentile scaling."""
    lo, hi = _percentile_clip(img, plo, phi)
    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0, None)
    return np.arcsinh(nonlin * x) / np.arcsinh(nonlin)

def save_png_preview_cube_only(fits_path, png_path):
    """
    Read a FITS cutout assumed to be a 3-D cube in PRIMARY and save a color PNG.
    Expected shapes:
      (3, N, N)  -> bands-first (g, r, z)
      (N, N, 3)  -> bands-last  (g, r, z)
    No grayscale fallback; anything else raises.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None or data.ndim != 3:
            raise RuntimeError(f"Expected 3-D cube in PRIMARY; found ndim="
                               f"{None if data is None else data.ndim}")

        # Map cube to g,r,z
        if data.shape[0] == 3:          # (3, N, N)
            g, r, z = data[0], data[1], data[2]
        elif data.shape[-1] == 3:       # (N, N, 3)
            g, r, z = data[..., 0], data[..., 1], data[..., 2]
        else:
            raise RuntimeError(f"Unexpected cube shape {data.shape} (need 3 planes)")

        # Compose RGB = (z, r, g) with asinh stretch
        R = _asinh_stretch(z)
        G = _asinh_stretch(r)
        B = _asinh_stretch(g)

        rgb = np.dstack([R, G, B])
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.clip(rgb, 0, 1)
        plt.imsave(png_path, rgb)

# ======================
# --- Main
# ======================

def main(fp_csv, outdir, bands=BANDS, size=CUTOUT_SIZE_PIX, layer=LAYER,
         png_only=False, timeout=TIMEOUT, retries=MAX_RETRIES, sleep_ms=int(SUCCESS_DELAY*1000)):
    setup_logger()
    ensure_dir(outdir)

    # Read CSV and normalize column names to lowercase
    df = pd.read_csv(fp_csv)
    df.columns = [c.lower() for c in df.columns]

    required = ["targetid", "target_ra", "target_dec"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"CSV must contain column '{col}'")

    logging.info("Loaded %d FP targets from %s", len(df), fp_csv)

    # Per-run overrides
    success_delay = max(0.0, sleep_ms / 1000.0)

    for _, row in df.iterrows():
        tid = int(row["targetid"])
        ra  = float(row["target_ra"])
        dec = float(row["target_dec"])
        size_pix = int(size)

        stem = safe_stem(tid, ra, dec, bands=bands, size_pix=size_pix)
        fits_out = os.path.join(outdir, f"{stem}.fits")
        png_out  = os.path.join(outdir, f"{stem}.png")

        # PNG-only mode (don’t hit the network)
        if png_only:
            if not os.path.exists(fits_out):
                logging.warning("Missing FITS for targetid=%d; skipping PNG-only.", tid)
                continue
            try:
                save_png_preview_cube_only(fits_out, png_out)
                logging.info("PNG regenerated for targetid=%d", tid)
            except Exception as e:
                logging.warning("PNG regen failed for targetid=%d: %s", tid, str(e))
            continue

        # Normal mode: fetch FITS if needed, then write PNG
        if not os.path.exists(fits_out):
            t0 = time.time()
            ok = fetch_cutout(ra, dec, size_pix, fits_out,
                              session=SESSION, timeout=timeout,
                              max_retries=retries, success_delay=success_delay)
            if not ok:
                logging.warning("Failed all retries for targetid=%d (%.1fs)", tid, time.time()-t0)
                continue

        # Save PNG (cube-only)
        try:
            save_png_preview_cube_only(fits_out, png_out)
            logging.info("Saved FITS+PNG for targetid=%d", tid)
        except Exception as e:
            logging.warning("PNG preview failed for targetid=%d: %s", tid, str(e))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate 152x152 LS cutouts + PNG previews for DESI FP sample")
    ap.add_argument("--csv", required=True, help="Path to FP_TARGETS_SSL.csv (lowercase columns)")
    ap.add_argument("--outdir", required=True, help="Output directory for FITS/PNGs")
    ap.add_argument("--bands", default=BANDS, help="Bands to request (e.g., grz)")
    ap.add_argument("--size", type=int, default=CUTOUT_SIZE_PIX, help="Cutout size (pixels per side)")
    ap.add_argument("--layer", default=LAYER, help="Legacy Surveys layer (ls-dr9 or ls-dr10) [kept for completeness]")
    ap.add_argument("--png-only", action="store_true", help="Do not download; build PNGs for existing FITS only")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="Per-request timeout (s)")
    ap.add_argument("--retries", type=int, default=MAX_RETRIES, help="Max attempts per target")
    ap.add_argument("--sleep-ms", type=int, default=int(SUCCESS_DELAY*1000),
                    help="Delay in milliseconds after each successful request (polite pacing)")
    args = ap.parse_args()

    main(args.csv, args.outdir, bands=args.bands, size=args.size, layer=args.layer,
         png_only=args.png_only, timeout=args.timeout, retries=args.retries, sleep_ms=args.sleep_ms)