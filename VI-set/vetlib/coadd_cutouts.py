"""
coadd_cutouts.py -- coadd-derived HDF5 cutouts for the vetter's "Coadd" view.

ON-DISK LAYOUT (one folder per size bin, files keep their existing names):

    /global/cfs/cdirs/desicollab/users/qshimp/h5py_files/
        bin0/ssl-cutouts-dr11-north-chunk0000.hdf5
        bin0/ssl-cutouts-dr11-south-chunk0000.hdf5      <- any number of chunks, either region
        bin1/ssl-cutouts-dr11-north-chunk0000.hdf5      <- partial sets are fine
        ...

  * `bin<N>` is the size-bin NUMBER, which never changes even if the display label does.
  * Region and chunk live in the file name (as for the SSL cutouts), not in extra folders, so a
    regeneration job can just drop new chunk files into the right bin folder.
  * Every file holds datasets  "sgaid" (N,)  and  "images" (N, 3, H, W)  with bands g, r, z
    (the same schema as the SSL cutout files, so the same colour stretch applies).
  * SGAIDs are unique (v1.0), so lookup is by SGAID alone. If a galaxy appears in two chunks the
    later file (sorted by name) wins and a warning is printed.

CoaddIndex re-scans its folder cheaply (only new / changed files are opened), so chunks that are
generated later show up in a running session without restarting anything.
"""
import glob
import os
import re
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _region_of(filename):
    m = re.search(r"dr11-(north|south)", os.path.basename(filename))
    return m.group(1) if m else "unknown"


def _validate(path):
    """Open one file read-only and return (sgaid array, images shape) or raise ValueError."""
    with h5py.File(path, "r") as H:
        for name in ("sgaid", "images"):
            if name not in H:
                raise ValueError(f"missing dataset {name!r} (has {list(H.keys())})")
        ids = H["sgaid"][:]
        shape = tuple(H["images"].shape)
    if len(shape) != 4 or shape[1] < 3:
        raise ValueError(f"images has shape {shape}; expected (N, >=3 bands, H, W)")
    if shape[0] != len(ids):
        raise ValueError(f"images has {shape[0]} rows but sgaid has {len(ids)}")
    return ids, shape


class CoaddIndex:
    """{SGAID: (file, row)} for all chunks in one bin folder, refreshed incrementally."""

    def __init__(self, directory, pattern="*.hdf5"):
        self.directory = Path(directory)
        self.pattern = pattern
        self.map = {}
        self.skipped = {}               # path -> reason (unreadable, wrong schema, still being written...)
        self._sig = {}                  # path -> (mtime_ns, size)
        self._ids = {}                  # path -> sgaid array
        self._shape = {}
        self._last = 0.0

    def refresh(self, force=False, min_interval=30.0, verbose=True):
        """Rescan the folder. Returns the list of files that were added or changed (callers should drop
        any open handles on them). Rate-limited unless force=True."""
        if not force and time.time() - self._last < min_interval:
            return []
        self._last = time.time()
        if not self.directory.is_dir():
            if verbose:
                print(f"[coadd] folder not found: {self.directory}")
            return []
        changed = []
        for f in sorted(glob.glob(str(self.directory / self.pattern))):
            try:
                st = os.stat(f)
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            if self._sig.get(f) == sig:
                continue
            try:
                ids, shape = _validate(f)
            except Exception as e:      # half-written chunk, wrong schema, ...: skip now, retry next refresh
                self.skipped[f] = f"{type(e).__name__}: {e}"
                continue
            self.skipped.pop(f, None)
            for sg in self._ids.get(f, ()):                 # forget the old contents of a replaced file
                if self.map.get(int(sg), (None,))[0] == f:
                    del self.map[int(sg)]
            n_dup = 0
            for i, sg in enumerate(ids.tolist()):
                prev = self.map.get(sg)
                if prev is not None and prev[0] != f:
                    n_dup += 1
                self.map[sg] = (f, i)
            self._ids[f], self._shape[f], self._sig[f] = ids, shape, sig
            changed.append(f)
            if verbose and n_dup:
                print(f"[coadd] WARNING: {n_dup} SGAID(s) in {os.path.basename(f)} were already in an earlier chunk; "
                      f"keeping the later file.")
        if verbose and (changed or self.skipped):
            print(self.summary())
        return changed

    def lookup(self, sgaid):
        return self.map.get(int(sgaid))

    def coverage(self, sgaids):
        """(n_have, n_total) for a collection of SGAIDs -- how complete is this bin's coadd set?"""
        ids = [int(s) for s in sgaids]
        return sum(1 for s in ids if s in self.map), len(ids)

    def summary(self):
        by_region = {}
        for f, ids in self._ids.items():
            r = _region_of(f)
            by_region[r] = by_region.get(r, 0) + len(ids)
        lines = [f"[coadd] {self.directory.name}: {len(self.map):,d} galaxies in {len(self._ids)} file(s) "
                 f"{by_region}"]
        for f, why in self.skipped.items():
            lines.append(f"[coadd]   skipped {os.path.basename(f)}: {why}")
        return "\n".join(lines)


def migrate_h5(src_dir, root, bin_id, dry_run=True, remove_source=False, pattern="*.hdf5"):
    """Copy a folder of cutout chunks into <root>/bin<bin_id>/, verifying each copy.

    Every source file is opened and checked first (datasets, shapes). Each copy is written to a
    temporary name, compared with the source (size + the full SGAID list), and only then renamed
    into place. The SOURCE is left untouched unless remove_source=True, and even then it is
    deleted only after its copy verified. Files already present at the destination with the same
    size are skipped, so an interrupted run can simply be repeated. Run with dry_run=True first."""
    dest = Path(root) / f"bin{int(bin_id)}"
    files = sorted(glob.glob(str(Path(src_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {src_dir}")
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        name, size = os.path.basename(f), os.path.getsize(f)
        out = dest / name
        row = {"file": name, "region": _region_of(f), "galaxies": None, "GB": round(size / 1e9, 2), "result": ""}
        try:
            ids, shape = _validate(f)
            row["galaxies"] = len(ids)
        except Exception as e:
            row["result"] = f"SKIPPED (invalid source: {e})"
            rows.append(row)
            print(f"  {name}: {row['result']}")
            continue
        if out.exists() and out.stat().st_size == size:
            row["result"] = "already at destination"
        elif dry_run:
            row["result"] = "would copy"
        else:
            t0 = time.time()
            tmp = Path(f"{out}.part")
            shutil.copy2(f, tmp)
            try:
                if tmp.stat().st_size != size:
                    raise ValueError("size mismatch after copy")
                new_ids, _ = _validate(tmp)
                if not np.array_equal(ids, new_ids):
                    raise ValueError("SGAID list differs after copy")
            except Exception as e:
                tmp.unlink(missing_ok=True)
                row["result"] = f"FAILED verification ({e}); source untouched"
                rows.append(row)
                print(f"  {name}: {row['result']}")
                continue
            os.replace(tmp, out)
            row["result"] = f"copied + verified ({time.time() - t0:.0f}s)"
        if remove_source and not dry_run and out.exists() and out.stat().st_size == size:
            os.remove(f)
            row["result"] += "; source removed"
        rows.append(row)
        print(f"  {name}: {row['galaxies']:,d} galaxies, {row['GB']} GB -> {row['result']}", flush=True)
    report = pd.DataFrame(rows)
    print(f"{'[DRY RUN] ' if dry_run else ''}{len(report)} file(s), "
          f"{int(report['galaxies'].fillna(0).sum()):,d} galaxies -> {dest}")
    return report