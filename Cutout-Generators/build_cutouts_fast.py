#!/usr/bin/env python3
"""
build_cutouts_fast.py

Generate fixed-angular-size (width * pixscale arcsec) galaxy cutouts for
the full SGA-2025 sample, packed into chunked HDF5 files for SSL/ML use.

Each cutout is resampled from its SGA GROUP's pre-built mosaic
({SGAGROUP}-image-{band}.fits.fz), centered on the INDIVIDUAL galaxy's own
RA/DEC via WCS -- not the group mosaic's center -- so non-primary group
members get a correctly-centered view of themselves rather than the
group-wide shot every member would otherwise share.

Zero-padding is used when a galaxy's crop window extends past the edge of
its group's mosaic file (NOT necessarily a true DR11 data gap -- see
edge_frac in the output). Every cutout's edge_frac is recorded so any
padding can be audited after the run rather than discovered by surprise.

Usage:
    python build_cutouts_fast.py \
        --catalog /path/to/file.fits \
        --region dr11-south \
        --outdir /path/to/output \
        --width 152 --pixscale 0.262 \
        --chunksize 50000 --mp 32
"""
import os
import sys
import logging
import argparse
import multiprocessing

import numpy as np
import h5py
import fitsio
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from tqdm import tqdm

# These already exist in your SGA package -- import, don't redefine, so this
# script always matches the canonical directory-resolution logic.
from SGA.SGA import get_galaxy_galaxydir, sga_data_dir, sga_html_dir
from SGA.io import get_raslice

log = logging.getLogger("build_ssl_cutouts")


# ---------------------------------------------------------------------
# Coadd reading
# ---------------------------------------------------------------------

def _read_coadd_bands(galaxydir, group_prefix, bands_str):
    """Return ({band: 2D image}, {band: FITS header}) for each coadd found.

    Coadds may be stored either fpack-compressed (.fits.fz) or uncompressed
    (.fits), depending on the data release -- check both, preferring .fits.fz
    if present (smaller, and the historically-expected format) but falling
    back to plain .fits without complaint.
    """
    images, headers = {}, {}
    for band in bands_str:
        base = os.path.join(galaxydir, f'{group_prefix}-image-{band}.fits')
        candidates = [base + '.fz', base]
        fitsfile = next((f for f in candidates if os.path.isfile(f)), None)

        if fitsfile is not None:
            data, hdr = fitsio.read(fitsfile, header=True)
            images[band] = data
            headers[band] = hdr
        else:
            log.warning(f'Missing {band}-band coadd for {group_prefix} '
                       f'(checked {candidates})')
    return images, headers


# ---------------------------------------------------------------------
# Per-band resampling, centered on the individual galaxy's own position
# ---------------------------------------------------------------------

def _rescale_one_band(img, center_x, center_y, crop_pix, width=152):
    """Lanczos-3 resample a crop_pix x crop_pix window centered at
    (center_x, center_y) -- this galaxy's own pixel position, from WCS --
    to (width, width). Out-of-bounds samples are zero-padded.

    Returns (resampled_array, edge_frac): edge_frac is the fraction of
    output pixels that fell outside the source mosaic (0.0 = fully
    covered, 1.0 = entirely outside). This reflects the mosaic's own
    extent, not necessarily a true gap in DR11 imaging -- see module
    docstring.
    """
    from astrometry.util.util import lanczos3_interpolate

    sh, sw = img.shape
    pscale = width / crop_pix

    cox = np.arange(width, dtype=np.float64) / pscale
    cox += center_x - cox[width // 2]
    coy = np.arange(width, dtype=np.float64) / pscale
    coy += center_y - coy[width // 2]

    fx, fy = np.meshgrid(cox, coy)
    fx = fx.ravel().astype(np.float32)
    fy = fy.ravel().astype(np.float32)
    ix = (fx + 0.5).astype(np.int32)
    iy = (fy + 0.5).astype(np.int32)
    dx = (fx - ix).astype(np.float32)
    dy = (fy - iy).astype(np.float32)

    valid = (ix >= 0) & (ix < sw) & (iy >= 0) & (iy < sh)
    edge_frac = 1.0 - (valid.sum() / valid.size)

    out = np.zeros(width * width, np.float32)
    if np.any(valid):
        ix_c = np.clip(ix, 0, sw - 1)
        iy_c = np.clip(iy, 0, sh - 1)
        tmp = np.zeros(width * width, np.float32)
        lanczos3_interpolate(ix_c, iy_c, dx, dy, [tmp], [np.ascontiguousarray(img, np.float32)])
        out[valid] = tmp[valid]

    return out.reshape((width, width)), edge_frac


def rescale_galaxy_cutout(row, galaxydir, width=152, pixscale=0.262):
    group_prefix = row['SGAGROUP']
    raw, headers = _read_coadd_bands(galaxydir, group_prefix, row['BANDS'])

    if not raw:
        raise FileNotFoundError(f'No .fits.fz coadds found for {group_prefix} in {galaxydir}')

    if 'r' not in raw and 'i' in raw:
        raw['r'] = raw['i']
        headers.setdefault('r', headers.get('i'))

    ref_band = next(iter(headers))
    wcs = WCS(headers[ref_band])
    x0, y0 = wcs.all_world2pix(row['RA'], row['DEC'], 0)

    native_scale = np.mean(proj_plane_pixel_scales(wcs)) * 3600.0
    crop_pix = (width * pixscale) / native_scale

    planes = np.zeros((3, width, width), np.float32)
    edge_fracs = []
    for iband, b in enumerate(('g', 'r', 'z')):
        if b in raw:
            planes[iband], frac = _rescale_one_band(raw[b], x0, y0, crop_pix, width)
            edge_fracs.append(frac)

    edge_frac = max(edge_fracs) if edge_fracs else 1.0
    return planes, edge_frac


def _rescale_worker(args):
    sgagroup, bands_str, ra, dec, galaxydir, width, pixscale = args
    row = {'SGAGROUP': sgagroup, 'BANDS': bands_str, 'RA': ra, 'DEC': dec}
    return rescale_galaxy_cutout(row, galaxydir, width=width, pixscale=pixscale)


# ---------------------------------------------------------------------
# Chunked HDF5 packing
# ---------------------------------------------------------------------

def build_ssl_hdf5(sample, region, datadir, outdir,
                   chunksize=50000, width=152, pixscale=0.262,
                   overwrite=False, mp=1):
    """Pack per-galaxy cutouts (centered on each galaxy's own RA/DEC) into
    chunked HDF5 files.

    Parameters
    ----------
    sample : astropy.table.Table
        Full SGA-2025 sample -- ALL members, primaries and non-primaries
        alike. Must have SGAID, RA, DEC, SGAGROUP, GROUP_RA, GROUP_NAME,
        BANDS columns.
    region : str
        Survey region, e.g. 'dr11-south'.
    datadir : str
        Root of the SGA data tree ($SGA_DATA_DIR).
    outdir : str
        Output directory; files written as ssl-cutouts-{region}-chunk{N:04d}.hdf5.
    chunksize : int
        Galaxies per HDF5 file.
    width : int
        Output image size in pixels (default 152).
    pixscale : float
        Output pixel scale in arcsec/pixel (default 0.262) -- fixes the
        angular footprint of every cutout to width*pixscale arcsec,
        matching the single-galaxy cutout convention used elsewhere in
        this pipeline.
    overwrite : bool
        Overwrite existing output files.
    mp : int
        Worker processes for image rescaling.
    """
    os.makedirs(outdir, exist_ok=True)

    # Decode byte-string columns once
    sample = sample.copy()
    for col in ["GROUP_NAME", "SGAGROUP"]:
        if col in sample.colnames:
            sample[col] = [
                x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in sample[col]
            ]
        
    # get_galaxy_galaxydir resolves each galaxy's GROUP directory (mosaics
    # are shared per SGAGROUP) -- group=True uses GROUP_RA/SGAGROUP/GROUP_NAME.
    _, galaxydirs = get_galaxy_galaxydir(sample, region=region, group=True, datadir=datadir)

    nobj   = len(sample)
    chunks = np.array_split(np.arange(nobj), max(1, int(np.ceil(nobj / chunksize))))
    log.info(f'build_ssl_hdf5: {nobj:,d} objects -> {len(chunks)} chunk(s) in {outdir}')

    for ichunk, idx in enumerate(chunks):
        outfile = os.path.join(outdir, f'ssl-cutouts-{region}-chunk{ichunk:04d}.hdf5')
        if os.path.isfile(outfile) and not overwrite:
            log.info(f'Skipping existing {outfile}')
            continue

        chunk_sample = sample[idx]
        chunk_dirs   = galaxydirs[idx]

        args = [
            (str(row['SGAGROUP']), str(row['BANDS']), float(row['RA']), float(row['DEC']),
             str(gdir), width, pixscale)
            for row, gdir in zip(chunk_sample, chunk_dirs)
        ]

        images, edge_fracs = [], []

        with tqdm(
            total=len(args),
            desc=f'chunk {ichunk:04d}',
            unit='galaxy',
            dynamic_ncols=True,
            file=sys.stdout,
        ) as progress:

            if mp > 1:
                with multiprocessing.Pool(mp) as P:
                    for img, frac in P.imap(_rescale_worker, args):
                        images.append(img)
                        edge_fracs.append(frac)
                        progress.update(1)
            else:
                for a in args:
                    img, frac = _rescale_worker(a)
                    images.append(img)
                    edge_fracs.append(frac)
                    progress.update(1)

        edge_fracs = np.array(edge_fracs, dtype=np.float32)
        n_affected = int(np.count_nonzero(edge_fracs > 0))
        if n_affected:
            log.info(f'  chunk {ichunk:04d}: {n_affected:,d} / {len(idx):,d} galaxies had '
                      f'some out-of-bounds padding (max edge_frac={edge_fracs.max():.3f})')

        tmpfile = outfile + '.tmp'
        with h5py.File(tmpfile, 'w') as F:
            F.attrs['region'] = region
            F.attrs['width'] = width
            F.attrs['pixscale'] = pixscale
            F.create_dataset('images', data=np.stack(images).astype(np.float32),
                             chunks=(1, 3, width, width))
            F.create_dataset('sgaid', data=chunk_sample['SGAID'].value)
            F.create_dataset('ra',    data=chunk_sample['RA'].value)
            F.create_dataset('dec',   data=chunk_sample['DEC'].value)
            F.create_dataset('edge_frac', data=edge_fracs)
        os.rename(tmpfile, outfile)
        log.info(f'Wrote {outfile}: {len(idx):,d} objects')

def filter_to_region(sample, region, datadir):
    """
    Filter SGA catalog to one physical SGA region.

    REGION 1 = south
    REGION 2 = north
    REGION 3 = ambiguous -- resolved per-call by checking whether this
    run's target region actually has the group's mosaic directory on disk.
    Since this function is called once per --region invocation (matching
    every other stage of this pipeline), an ambiguous group whose mosaic
    genuinely exists in both regions will correctly be picked up by both
    the dr11-south and dr11-north runs independently.
    """
    if region == "dr11-south":
        target_region_num = 1
    elif region == "dr11-north":
        target_region_num = 2
    else:
        raise ValueError(f"Unknown region: {region}")

    region_col = np.asarray(sample["REGION"])

    # Unambiguous rows need no filesystem lookup.
    mask = region_col == target_region_num

    # Only REGION=3 needs filesystem resolution.
    ambiguous = sample[region_col == 3]
    if len(ambiguous) == 0:
        return sample[mask]

    # Resolve each unique group's directory only once, using the same
    # {datadir}/{region}/{raslice}/{group} convention as
    # get_galaxy_galaxydir(group=True) -- GROUP_RA determines the raslice,
    # and every row sharing an SGAGROUP should share the same GROUP_RA, so
    # one lookup per group is enough.
    groups = np.asarray(ambiguous["SGAGROUP"]).astype(str)
    group_ra = np.asarray(ambiguous["GROUP_RA"]).astype(float)

    group_ra_lookup = {}
    for grp, ra in zip(groups, group_ra):
        group_ra_lookup.setdefault(grp, ra)

    unique_groups = np.unique(groups)
    valid_groups = set()

    for group in unique_groups:
        ra = group_ra_lookup[group]
        raslice = get_raslice(ra)
        group_dir = os.path.join(datadir, region, raslice, group.removeprefix("SGA2025_"))
        if os.path.isdir(group_dir):
            valid_groups.add(group)

    n_checked = len(unique_groups)
    n_found = len(valid_groups)
    log.info(f'  Ambiguous (REGION=3) groups: {n_found:,d} / {n_checked:,d} '
             f'found in {region} mosaic tree')

    ambiguous_mask = np.isin(groups, list(valid_groups))
    ambiguous_indices = np.flatnonzero(region_col == 3)
    mask[ambiguous_indices[ambiguous_mask]] = True

    return sample[mask]

# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True,
                        help='FITS/CSV table of the full sample; needs SGAID, RA, DEC, '
                             'SGAGROUP, GROUP_RA, GROUP_NAME, BANDS columns.')
    parser.add_argument('--region', required=True, help="e.g. 'dr11-south' or 'dr11-north'")
    parser.add_argument('--datadir', default=None, help='Defaults to $SGA_DATA_DIR')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--chunksize', type=int, default=50000)
    parser.add_argument('--width', type=int, default=152)
    parser.add_argument('--pixscale', type=float, default=0.262)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stdout)

    datadir = args.datadir or sga_data_dir()
    
    if args.catalog.endswith('.fits'):
        sample = Table.read(args.catalog)
    else:
        sample = Table.from_pandas(__import__('pandas').read_csv(args.catalog))
    
    before = len(sample)
    
    sample = filter_to_region(
        sample,
        region=args.region,
        datadir=datadir,
    )
    
    print(
        f"Region filter: {args.region} -> "
        f"{len(sample):,} / {before:,} galaxies"
    )
    
    build_ssl_hdf5(
        sample=sample,
        region=args.region,
        datadir=datadir,
        outdir=args.outdir,
        chunksize=args.chunksize,
        width=args.width,
        pixscale=args.pixscale,
        overwrite=args.overwrite,
        mp=args.mp,
    )


if __name__ == '__main__':
    main()