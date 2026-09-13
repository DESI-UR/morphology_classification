#!/usr/bin/env python3
"""
Legacy Surveys DR11 JPG downloader — Image / Model / Residual,
keyed by SGAID (master_anchor_catalog.csv), threaded with adaptive throttling.
"""

import os
import time
import logging
import pathlib
import threading
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# ======================
# CONFIG
# ======================

VIEWER_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"

LAYERS = {
    "Image": "ls-dr11",
    "Model": "ls-dr11-model",
    "Residual": "ls-dr11-resid",
}

CUTOUT_SIZE_PIX = 152
PIX_SCALE = 0.262
TIMEOUT = 15
WORKERS = 4
GLOBAL_COOLDOWN = 10
REQUEST_SEMAPHORE = threading.Semaphore(2)
SUCCESS_DELAY = 0.0

failed = []
failed_lock = threading.Lock()   # see note below

SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AdaptiveController:
    def __init__(self):
        self.lock = threading.Lock()
        self.cooldown_until = 0

    def wait_for_turn(self):
        while time.time() < self.cooldown_until:
            time.sleep(0.5)

    def report_failure(self, status):
        if status != 429:
            return
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.time() + GLOBAL_COOLDOWN)
            logging.warning("429 received -> cooldown %.0fs", GLOBAL_COOLDOWN)


class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.images = 0
        self.failed = 0
        self.skipped = 0
        self.status429 = 0
        self.status503 = 0
        self.completed_galaxies = 0
        self.start = time.time()

    def record_image(self):
        with self.lock:
            self.images += 1

    def record_failed(self, status):
        with self.lock:
            self.failed += 1
            if status == 429:
                self.status429 += 1
            elif status == 503:
                self.status503 += 1

    def record_skipped(self):
        with self.lock:
            self.skipped += 1

    def record_galaxy(self):
        with self.lock:
            self.completed_galaxies += 1

    def elapsed(self):
        return time.time() - self.start

    def img_per_sec(self):
        e = self.elapsed()
        return self.images / e if e else 0

    def galaxies_per_sec(self):
        e = self.elapsed()
        return self.completed_galaxies / e if e else 0

    def eta(self, total_galaxies):
        if self.completed_galaxies == 0:
            return "--:--:--"
        remaining = total_galaxies - self.completed_galaxies
        seconds = remaining / self.galaxies_per_sec()
        hrs, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def progress_printer(stats, controller, total_galaxies):
    while stats.completed_galaxies < total_galaxies:
        elapsed = stats.elapsed()
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        logging.info(
            "Galaxies %5d/%5d | Images %6d | Skipped %5d | Failed %4d | "
            "%.2f img/s | 429=%d | Elapsed %02d:%02d:%02d | ETA %s",
            stats.completed_galaxies, total_galaxies, stats.images, stats.skipped,
            stats.failed, stats.img_per_sec(), stats.status429, hrs, mins, secs,
            stats.eta(total_galaxies),
        )
        time.sleep(5)


def viewer_url(ra, dec, size, layer):
    params = {"ra": ra, "dec": dec, "size": size, "pixscale": PIX_SCALE, "layer": layer}
    return f"{VIEWER_URL}?{urlencode(params)}"


def safe_name(sgaid, ra, dec):
    return f"{int(sgaid)}_{ra:.4f}_{dec:.4f}"


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def fetch_jpg(ra, dec, size, outpath, layer, controller, stats):
    url = viewer_url(ra, dec, size, layer)
    controller.wait_for_turn()

    with REQUEST_SEMAPHORE:
        for attempt in range(3):
            try:
                r = SESSION.get(url, timeout=TIMEOUT)

                if r.status_code == 429:
                    controller.report_failure(429)
                    stats.record_failed(429)
                    return False, 429

                if r.status_code == 503:
                    stats.record_failed(503)
                    time.sleep(2 + attempt * 3)
                    continue

                r.raise_for_status()
                with open(outpath, "wb") as f:
                    f.write(r.content)
                return True, r.status_code

            except requests.exceptions.RequestException as e:
                logging.warning("Request failed (%s): %s", url, e)
                time.sleep(2 + attempt * 2)

        stats.record_failed(None)
        return False, None


def process_galaxy(row, outdir, controller, stats):
    sgaid = row.sgaid
    ra = row.ra
    dec = row.dec
    name = safe_name(sgaid, ra, dec)
    ok_all = True

    for folder, layer in LAYERS.items():
        outpath = os.path.join(outdir, folder, f"{name}.jpg")

        if os.path.exists(outpath):
            stats.record_skipped()
            continue

        ok, status = fetch_jpg(ra, dec, CUTOUT_SIZE_PIX, outpath, layer, controller, stats)

        if ok:
            stats.record_image()
            logging.debug("%s | %.4f %.4f | %s", layer, ra, dec, os.path.basename(outpath))
        else:
            with failed_lock:
                failed.append({"sgaid": sgaid, "ra": ra, "dec": dec, "layer": folder})
            ok_all = False

        time.sleep(SUCCESS_DELAY)

    stats.record_galaxy()
    return ok_all


def main(csv_path, outdir):
    setup_logger()
    ensure_dir(outdir)
    for folder in LAYERS:
        ensure_dir(os.path.join(outdir, folder))

    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    required = ["sgaid", "ra", "dec"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"CSV must contain column '{col}'")

    rows = list(df.itertuples(index=False))

    controller = AdaptiveController()
    stats = DownloadStats()

    threading.Thread(target=progress_printer, args=(stats, controller, len(rows)), daemon=True).start()

    idx = 0
    while idx < len(rows):
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            batch = rows[idx: idx + 50]
            future_to_row = {
                executor.submit(process_galaxy, row, outdir, controller, stats): row
                for row in batch
            }
            for future in as_completed(future_to_row):
                future.result()
            idx += len(batch)

    if failed:
        pd.DataFrame(failed).to_csv("failed_downloads.csv", index=False)
        logging.info("Saved %d failed downloads.", len(failed))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to master_anchor_catalog.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    main(args.csv, args.outdir)