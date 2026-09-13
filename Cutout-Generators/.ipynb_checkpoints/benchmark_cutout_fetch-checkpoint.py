#!/usr/bin/env python3
"""
Benchmark concurrency levels for fetching DR11 Image-only cutouts.
Run once on a small sample before committing to a setting for the full batch.
"""
import time
import logging
import pathlib
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

VIEWER_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"
LAYER = "ls-dr11"
SIZE = 152
PIXSCALE = 0.262
TIMEOUT = 15
N_PER_TRIAL = 30
WORKER_LEVELS = [2, 4, 8, 16]

SESSION = requests.Session()


def viewer_url(ra, dec):
    params = {"ra": ra, "dec": dec, "size": SIZE, "pixscale": PIXSCALE, "layer": LAYER}
    return f"{VIEWER_URL}?{urlencode(params)}"


def fetch_one(ra, dec, outpath):
    try:
        r = SESSION.get(viewer_url(ra, dec), timeout=TIMEOUT)
        if r.status_code == 200:
            with open(outpath, "wb") as f:
                f.write(r.content)
            return "ok"
        return f"http_{r.status_code}"
    except requests.exceptions.RequestException:
        return "error"


def run_trial(rows, n_workers, outdir):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(fetch_one, r.ra, r.dec, f"{outdir}/{int(r.sgaid)}.jpg") for r in rows]
        for f in as_completed(futures):
            results.append(f.result())
    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r == "ok")
    n_429 = sum(1 for r in results if r == "http_429")
    return {
        "workers": n_workers, "n": len(rows), "elapsed_s": round(elapsed, 2),
        "img_per_sec": round(n_ok / elapsed, 2) if elapsed else 0,
        "ok": n_ok, "429": n_429, "other_fail": len(results) - n_ok - n_429,
    }


def main(csv_path, outdir="benchmark_tmp"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    rows = list(df.itertuples(index=False))

    report, idx = [], 0
    for n_workers in WORKER_LEVELS:
        trial_rows = rows[idx: idx + N_PER_TRIAL] or rows[:N_PER_TRIAL]
        idx += N_PER_TRIAL
        logging.info(f"Testing {n_workers} workers on {len(trial_rows)} galaxies...")
        result = run_trial(trial_rows, n_workers, f"{outdir}/w{n_workers}")
        report.append(result)
        logging.info(result)
        time.sleep(3)

    report_df = pd.DataFrame(report)
    print("\n" + report_df.to_string(index=False))
    report_df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="benchmark_tmp")
    args = ap.parse_args()
    main(args.csv, args.outdir)