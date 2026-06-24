def main(targets_csv, outdir, bands=BANDS, size=CUTOUT_SIZE_PIX,
         layer=LAYER, png_only=False,
         timeout=TIMEOUT, retries=MAX_RETRIES,
         sleep_ms=int(SUCCESS_DELAY*1000)):

    setup_logger()
    ensure_dir(outdir)

    df = pd.read_csv(targets_csv)

    required = ["targetid", "target_ra", "target_dec"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column {col}")

    success_delay = max(0.0, sleep_ms / 1000.0)

    for row in df.itertuples(index=False):

        tid = int(row.targetid)
        ra  = float(row.target_ra)
        dec = float(row.target_dec)

        # RA-binned directory (IMPORTANT FOR SCALE)
        out_subdir = os.path.join(outdir, f"{int(ra):03d}")
        ensure_dir(out_subdir)

        fits_out = os.path.join(
            out_subdir,
            f"{tid}_grz_{size}.fits"
        )

        png_out = fits_out.replace(".fits", ".png")

        # PNG-only mode
        if png_only:
            if not os.path.exists(fits_out):
                continue
            save_png_preview_cube_only(fits_out, png_out)
            continue

        # Download if missing
        if not os.path.exists(fits_out):
            ok = fetch_cutout(
                ra, dec, size, PIX_SCALE,
                fits_out,
                session=SESSION,
                timeout=timeout,
                max_retries=retries,
                success_delay=success_delay
            )
            if not ok:
                continue

        # PNG preview
        try:
            save_png_preview_cube_only(fits_out, png_out)
        except Exception as e:
            logging.warning("PNG failed for %d: %s", tid, str(e))