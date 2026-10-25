"""
jpg_service.py -- background Legacy Surveys JPG downloader for the live vetting session.

Same recipe as make_jpgs.py (same viewer URL, layers, pixscale, folder layout and file names
"<SGAID>_<ra:.4f>_<dec:.4f>.jpg"), but as an importable, non-blocking service: the vetter calls
enqueue() whenever votes are saved and the Image / Model / Residual JPGs for those galaxies are
fetched in the background while vetting continues.

* Polite by default: 2 concurrent requests per kernel, global cooldown on HTTP 429.
* Safe with several vetters at once: files are written atomically and existing files are skipped.
* Updates the live JPG index (the dict your load_jpg_band reads), so a new JPG shows up in the
  vetter without restarting anything.
"""
import logging
import os
import queue
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

VIEWER_URL = "https://www.legacysurvey.org/viewer/cutout.jpg"
LAYERS = {"Image": "ls-dr11", "Model": "ls-dr11-model", "Residual": "ls-dr11-resid"}
PIX_SCALE = 0.262
TIMEOUT = 15
GLOBAL_COOLDOWN = 10


class JpgService:
    def __init__(self, jpg_root, jpg_index, workers=2, default_size_pix=152, http_get=None):
        """jpg_root : folder containing Image/ Model/ Residual/ (created if missing).
        jpg_index : the SAME dict your load_jpg_band reads ({sgaid: {"Image": path, ...}});
                    new files are added to it as they land.
        http_get  : optional callable(url, timeout) -> response (for testing)."""
        self.root = Path(jpg_root)
        self.index = jpg_index
        self.workers = workers
        self.default_size_pix = int(default_size_pix)
        self._http_get = http_get
        self._session = None
        for folder in LAYERS:
            (self.root / folder).mkdir(parents=True, exist_ok=True)

        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._threads = []
        self._pending = set()           # sgaids queued or in progress
        self._cooldown_until = 0.0
        self.n_queued = self.n_done = self.n_images = self.n_skipped = 0
        self.failed = []                # dicts: sgaid, ra, dec, layer, size_pix, reason

    # ---------------------------------------------------------------- public
    def enqueue(self, items, size_pix=None):
        """items: iterable of (sgaid, ra, dec). Returns how many were actually queued
        (galaxies that already have all three JPGs, or are already queued, are skipped)."""
        size = int(size_pix or self.default_size_pix)
        n = 0
        for sgaid, ra, dec in items:
            sgaid = int(sgaid)
            if ra is None or dec is None or ra != ra or dec != dec:
                self._fail(sgaid, ra, dec, "all", size, "no RA/DEC")
                continue
            with self._lock:
                if sgaid in self._pending:
                    continue
                have = self.index.get(sgaid, {})
                if all(k in have for k in LAYERS):
                    self.n_skipped += 1
                    continue
                self._pending.add(sgaid)
                self.n_queued += 1
            self._q.put((sgaid, float(ra), float(dec), size))
            n += 1
        if n:
            self._ensure_threads()
        return n

    def backfill(self, store, size_bins=None, dry_run=True, limit=None):
        """Queue JPGs for every galaxy ANY vetter has already voted on that is missing one or more of
        Image/Model/Residual. Sizes follow each vote's size bin (SizeBinSource.jpg_pix); votes with no
        bin (legacy imports) use the default size. Run with dry_run=True first to see the counts."""
        votes = store.all_votes()
        if votes.empty:
            print("No votes yet -- nothing to backfill.")
            return 0
        pix = {b.label: int(getattr(b, "jpg_pix", self.default_size_pix)) for b in (size_bins or [])}
        votes = votes.drop_duplicates("SGAID", keep="last")
        votes["_pix"] = votes["size_bin"].map(pix).fillna(self.default_size_pix).astype(int)
        with self._lock:
            missing = [int(s) for s in votes["SGAID"]
                       if not all(k in self.index.get(int(s), {}) for k in LAYERS)]
        todo = votes[votes["SGAID"].isin(missing)].merge(
            store.catalog[["SGAID", "RA", "DEC"]], on="SGAID", how="left")
        if limit:
            todo = todo.head(int(limit))
        print(f"{'[DRY RUN] ' if dry_run else ''}{len(todo):,d} of {len(votes):,d} vetted galaxies are missing JPGs "
              f"(by size: {todo['_pix'].value_counts().to_dict()}); "
              f"{int(todo['RA'].isna().sum())} have no RA/DEC in the catalog.")
        if dry_run:
            return len(todo)
        n = 0
        for pix_size, grp in todo.groupby("_pix"):
            n += self.enqueue(zip(grp["SGAID"], grp["RA"], grp["DEC"]), size_pix=int(pix_size))
        print(f"Queued {n:,d} galaxies. {self.status()}  (check progress with jpg_service.status())")
        return n

    def retry_failed(self):
        items = {(f["sgaid"], f["ra"], f["dec"], f["size_pix"]) for f in self.failed
                 if f["ra"] == f["ra"] and f["ra"] is not None}
        with self._lock:
            self.failed = []
        return sum(self.enqueue([(s, r, d)], size_pix=z) for s, r, d, z in items)

    def pending(self):
        with self._lock:
            return len(self._pending)

    def status(self):
        with self._lock:
            return (f"JPGs: {len(self._pending)} pending, {self.n_done} done "
                    f"({self.n_images} images), {len(self.failed)} failed")

    def wait(self, timeout=None):
        """Block until the queue is empty (optional; vetting never needs this)."""
        t0 = time.time()
        while self.pending() and (timeout is None or time.time() - t0 < timeout):
            time.sleep(0.2)
        return self.pending() == 0

    # ---------------------------------------------------------------- workers
    def _ensure_threads(self):
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            while len(self._threads) < self.workers:
                t = threading.Thread(target=self._worker, daemon=True, name="jpg-worker")
                t.start()
                self._threads.append(t)

    def _worker(self):
        while True:
            try:
                sgaid, ra, dec, size = self._q.get(timeout=30)
            except queue.Empty:
                return                                   # idle -> thread exits; restarted on next enqueue
            try:
                self._process(sgaid, ra, dec, size)
            except Exception as e:                       # never let a worker die silently
                self._fail(sgaid, ra, dec, "all", size, f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._pending.discard(sgaid)
                    self.n_done += 1
                self._q.task_done()

    def _process(self, sgaid, ra, dec, size):
        name = f"{sgaid}_{ra:.4f}_{dec:.4f}"
        for folder, layer in LAYERS.items():
            path = self.root / folder / f"{name}.jpg"
            if path.exists():
                self._register(sgaid, folder, path)
                continue
            ok, reason = self._fetch(ra, dec, size, layer, path)
            if ok:
                with self._lock:
                    self.n_images += 1
                self._register(sgaid, folder, path)
            else:
                self._fail(sgaid, ra, dec, folder, size, reason)

    def _register(self, sgaid, folder, path):
        with self._lock:
            self.index.setdefault(sgaid, {})[folder] = str(path)

    def _fail(self, sgaid, ra, dec, layer, size, reason):
        with self._lock:
            self.failed.append({"sgaid": sgaid, "ra": ra, "dec": dec, "layer": layer,
                                "size_pix": size, "reason": reason})

    # ---------------------------------------------------------------- http
    def _get(self, url):
        if self._http_get is not None:
            return self._http_get(url, TIMEOUT)
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session.get(url, timeout=TIMEOUT)

    def _fetch(self, ra, dec, size, layer, outpath):
        url = f"{VIEWER_URL}?{urlencode({'ra': ra, 'dec': dec, 'size': size, 'pixscale': PIX_SCALE, 'layer': layer})}"
        last = "unknown"
        for attempt in range(4):
            while time.time() < self._cooldown_until:    # honour a global 429 cooldown
                time.sleep(0.5)
            try:
                r = self._get(url)
                if r.status_code == 429:
                    with self._lock:
                        self._cooldown_until = max(self._cooldown_until, time.time() + GLOBAL_COOLDOWN)
                    logging.warning("jpg_service: 429 -> cooling down %ss", GLOBAL_COOLDOWN)
                    last = "HTTP 429"
                    continue
                if r.status_code == 503:
                    last = "HTTP 503"
                    time.sleep(2 + attempt * 3)
                    continue
                if r.status_code != 200:
                    return False, f"HTTP {r.status_code}"
                if r.content[:2] != b"\xff\xd8":
                    return False, "response was not a JPEG"
                tmp = Path(f"{outpath}.part.{os.getpid()}.{threading.get_ident()}")
                tmp.write_bytes(r.content)
                os.replace(tmp, outpath)                 # atomic: readers never see a half-written file
                return True, ""
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 + attempt * 2)
        return False, last