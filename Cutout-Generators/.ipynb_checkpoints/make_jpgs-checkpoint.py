#!/usr/bin/env python3
"""
Adaptive Legacy Surveys JPG downloader
- Image / Model / Residual
- Threaded per-galaxy processing
- Adaptive throttling based on server response
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
    "Image": "ls-dr10",
    "Model": "ls-dr10-model",
    "Residual": "ls-dr10-resid",
}

CUTOUT_SIZE_PIX = 152
PIX_SCALE = 0.262

TIMEOUT = 15

WORKERS = 4
GLOBAL_COOLDOWN = 10      # seconds
REQUEST_SEMAPHORE = threading.Semaphore(2)
SUCCESS_DELAY = 0.0

failed = []

# ======================
# SESSION
# ======================

SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=2,
    pool_maxsize=2,
    max_retries=0
)
SESSION.mount("http://", ADAPTER)
SESSION.mount("https://", ADAPTER)

# ======================
# LOGGING
# ======================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# ======================
# ADAPTIVE CONTROLLER
# ======================

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

            self.cooldown_until = max(
                self.cooldown_until,
                time.time() + GLOBAL_COOLDOWN
            )

            logging.warning(
                "429 received → cooldown %.0fs",
                GLOBAL_COOLDOWN,
            )

# ======================
# DOWNLOAD STATISTICS
# ======================

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

        if e == 0:
            return 0

        return self.images / e

    def galaxies_per_sec(self):
    
        e = self.elapsed()
    
        if e == 0:
            return 0
    
        return self.completed_galaxies / e
    
    
    def eta(self, total_galaxies):
    
        if self.completed_galaxies == 0:
            return "--:--:--"
    
        remaining = total_galaxies - self.completed_galaxies
    
        seconds = remaining / self.galaxies_per_sec()
    
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
    
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# ======================
# PROGRESS BAR
# ======================

def progress_printer(stats, controller, total_galaxies):

    while stats.completed_galaxies < total_galaxies:

        elapsed = stats.elapsed()

        hrs = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        logging.info(
            "Galaxies %5d/%5d | "
            "Images %6d | "
            "Skipped %5d | "
            "Failed %4d | "
            "%.2f img/s | "
            "429=%d | "
            "Elapsed %02d:%02d:%02d | "
            "ETA %s",
            stats.completed_galaxies,
            total_galaxies,
            stats.images,
            stats.skipped,
            stats.failed,
            stats.img_per_sec(),
            stats.status429,
            hrs,
            mins,
            secs,
            stats.eta(total_galaxies),
        )

        time.sleep(5)

# ======================
# URL + FILE HELPERS
# ======================

def viewer_url(ra, dec, size, layer):
    params = {
        "ra": ra,
        "dec": dec,
        "size": size,
        "pixscale": PIX_SCALE,
        "layer": layer,
    }
    return f"{VIEWER_URL}?{urlencode(params)}"


def safe_name(tid, ra, dec):
    return f"{int(tid)}_{ra:.4f}_{dec:.4f}"

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# ======================
# DOWNLOAD LOGIC
# ======================

def fetch_jpg(
    ra,
    dec,
    size,
    outpath,
    layer,
    controller,
    stats
):
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
    
                logging.warning(
                    "Request failed (%s): %s",
                    url,
                    e,
                )
    
                time.sleep(2 + attempt * 2)
                
        stats.record_failed(None)
        return False, None

# ======================
# WORKER
# ======================

def process_galaxy(row, outdir, controller, stats):
    tid = row.targetid
    ra = row.target_ra
    dec = row.target_dec

    name = safe_name(tid, ra, dec)

    ok_all = True

    for folder, layer in LAYERS.items():
        folder_path = os.path.join(outdir, folder)

        outpath = os.path.join(folder_path, f"{name}.jpg")

        if os.path.exists(outpath):

            stats.record_skipped()
        
            continue
            
        ok, status = fetch_jpg(
            ra,
            dec,
            CUTOUT_SIZE_PIX,
            outpath,
            layer,
            controller,
            stats,
        )

        if ok:
            stats.record_image()

            logging.debug(
                "%s | %.4f %.4f | %s",
                layer,
                ra,
                dec,
                os.path.basename(outpath),
            )
            
        else:
            failed.append({
                "targetid": tid,
                "ra": ra,
                "dec": dec,
                "layer": folder,
            })
            ok_all = False

        time.sleep(SUCCESS_DELAY)

    stats.record_galaxy()
    
    return ok_all
    
# ======================
# MAIN LOOP
# ======================

def main(csv_path, outdir):
    setup_logger()
    ensure_dir(outdir)
    for folder in LAYERS:
        ensure_dir(os.path.join(outdir, folder))
        
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    rows = list(df.itertuples(index=False))

    controller = AdaptiveController()
    stats = DownloadStats()

    threading.Thread(
        target=progress_printer,
        args=(stats, controller, len(rows)),
        daemon=True,
    ).start()

    idx = 0

    while idx < len(rows):

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:

            batch = rows[idx: idx + 50]

            future_to_row = {
                executor.submit(
                    process_galaxy,
                    row,
                    outdir,
                    controller,
                    stats,
                ): row
                for row in batch
            }
            
            for future in as_completed(future_to_row):
            
                row = future_to_row[future]
            
                success = future.result()
            

            idx += len(batch)

    if failed:
        pd.DataFrame(failed).to_csv("failed_downloads.csv", index=False)
        logging.info("Saved %d failed downloads.", len(failed))

# ======================
# ENTRY
# ======================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)

    args = ap.parse_args()

    main(args.csv, args.outdir)