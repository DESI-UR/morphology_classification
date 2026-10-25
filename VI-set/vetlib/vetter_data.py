"""
vetter_data.py -- the data layer behind the anchor vetter (no widgets in here).

Layout on disk (everything lives under one MASTER_DIR):

    MASTER_DIR/
        vetters/<name>.csv      one file per vetter -- the SOURCE OF TRUTH for that vetter's votes.
                                Only that vetter ever writes it, so two people saving at the same
                                moment can never overwrite each other.
        vetting_master.csv      DERIVED, rebuilt atomically after every save:
                                SGAID, RA, DEC, D26, <one column per vetter>
                                cell = Elliptical / Lenticular / Spiral / Irregular / Bad anchor,
                                blank = that vetter has not vetted the galaxy.

Rows are added to the master dynamically the first time any vetter labels a galaxy (the master
is therefore small and never has to be pre-built). "Unvetted" galaxies are defined against the
per-size-bin ML catalogs, which hold the full galaxy list and the ML-proposed morphology.
SGAID is assumed unique (true for the v1.0 SGA-2025 catalog).
"""
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LABELS = ["Elliptical", "Lenticular", "Spiral", "Irregular"]
BAD = "Bad anchor"
UNKNOWN = "Unknown"                       # ML could not classify it; there is nothing to "affirm"
VALID_LABELS = set(LABELS + [BAD])

MODE_UNVETTED = "Unvetted galaxies (ML-proposed morphology)"
MODE_AGREED = "Galaxies other vetters already agree on"

FIXED_COLS = ["SGAID", "RA", "DEC", "D26"]
VOTE_COLS = ["SGAID", "label", "proposed", "source", "size_bin", "notes", "timestamp"]


@dataclass
class SizeBinSource:
    """One selectable size bin.
    label      : what the vetter sees, e.g. "0-39arcsec".
    ml_catalog : CSV with the FULL ML morphology catalog for this bin (SGAID + morphology column).
    ssl_dir    : optional bin-specific SSL cutout directory (defaults to the shared one).
    jpg_pix    : size of the downloaded Image/Model/Residual JPGs; use the same size as this bin's
                 cutouts (e.g. 304 for the larger bin) so big galaxies aren't cropped."""
    label: str
    ml_catalog: str
    ssl_dir: Optional[str] = None
    jpg_pix: int = 152                    # Legacy Surveys JPG cutout size (pixels @ 0.262"/px) for this bin
    coadd_dir: Optional[str] = None       # folder of coadd-derived HDF5 chunks for this bin (enables the "Coadd" view)


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def safe_username(raw):
    """Normalise to lowercase (so 'QShimp' and 'qshimp' are one column) and reject anything that
    could break a file name or collide with the master's fixed columns."""
    name = (raw or "").strip().lower()
    if not name:
        raise ValueError("Please enter a username.")
    if not _NAME_RE.match(name):
        raise ValueError("Use only letters, numbers, '_', '-' or '.' (no spaces or other symbols).")
    if name.upper() in FIXED_COLS:
        raise ValueError(f"'{name}' is reserved -- please pick another username.")
    return name


def _atomic_csv(df, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)                  # readers see the old file or the new one, never half of one


class VettingStore:
    def __init__(self, master_dir, catalog, catalog_cols=None, ml_id_col="SGAID",
                 ml_morph_col="Galaxy_Type", ml_proba_col="Top1_Proba"):
        """catalog : DataFrame or CSV path of the SGA-2025 catalog (supplies RA, DEC, D26).
        catalog_cols : override column names, e.g. {"D26": "D26_ARCMIN"}.
        """
        self.master_dir = Path(master_dir)
        self.vetters_dir = self.master_dir / "vetters"
        self.vetters_dir.mkdir(parents=True, exist_ok=True)
        self.master_path = self.master_dir / "vetting_master.csv"
        self.ml_id_col, self.ml_morph_col = ml_id_col, ml_morph_col
        self.ml_proba_col = ml_proba_col

        cols = dict(zip(FIXED_COLS, FIXED_COLS))
        cols.update(catalog_cols or {})
        if isinstance(catalog, (str, Path)):                 # read only the 4 columns we need (much faster)
            cat = pd.read_csv(catalog, usecols=[c for c in cols.values()], low_memory=False)
        else:
            cat = catalog
        missing = [v for v in cols.values() if v not in cat.columns]
        if missing:
            raise KeyError(f"Catalog is missing {missing}. Columns available: {list(cat.columns)}. "
                           f"Pass catalog_cols={{'RA': ..., 'DEC': ..., 'D26': ...}} to map them.")
        cat = cat[[cols[c] for c in FIXED_COLS]].copy()
        cat.columns = FIXED_COLS
        cat["SGAID"] = cat["SGAID"].astype("int64")
        self.catalog = cat.drop_duplicates("SGAID").reset_index(drop=True)

    # ---------------------------------------------------------------- locking
    @contextmanager
    def _lock(self, timeout=15.0):
        """Best-effort cross-process lock (flock) with a timeout. Votes live in per-vetter files and
        the master is derived, so if the lock can't be had (another kernel holds it, or the
        filesystem doesn't support flock) we warn and carry on instead of hanging."""
        import time
        fh = None
        try:
            import fcntl
            fh = open(self.master_dir / ".lock", "w")
            t0, warned = time.time(), False
            while True:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - t0 > timeout:
                        print(f"WARNING: could not get the vetting lock in {timeout:.0f}s "
                              f"(another session holding it?). Continuing without it.")
                        fh.close()
                        fh = None
                        break
                    if not warned and time.time() - t0 > 3:
                        print("Waiting for another vetter's save to finish...", flush=True)
                        warned = True
                    time.sleep(0.2)
        except Exception:
            fh = None
        try:
            yield
        finally:
            if fh is not None:
                try:
                    import fcntl
                    fcntl.flock(fh, fcntl.LOCK_UN)
                finally:
                    fh.close()

    # ---------------------------------------------------------------- reading
    def _vetter_path(self, vetter):
        return self.vetters_dir / f"{vetter}.csv"

    def _read_vetter(self, vetter):
        p = self._vetter_path(vetter)
        if not p.exists():
            return pd.DataFrame({c: pd.Series(dtype="object") for c in VOTE_COLS}).astype({"SGAID": "int64"})
        d = pd.read_csv(p, dtype=str, keep_default_na=False).reindex(columns=VOTE_COLS, fill_value="")
        d["SGAID"] = d["SGAID"].astype("int64")
        return d

    def list_vetters(self):
        return sorted(p.stem for p in self.vetters_dir.glob("*.csv"))

    def all_votes(self):
        """Long table: vetter, SGAID, label, proposed, source, size_bin, notes, timestamp."""
        frames = []
        for v in self.list_vetters():
            d = self._read_vetter(v)
            d.insert(0, "vetter", v)
            frames.append(d)
        if not frames:
            return pd.DataFrame(columns=["vetter"] + VOTE_COLS).astype({"SGAID": "int64"})
        return pd.concat(frames, ignore_index=True)

    def my_votes(self, vetter):
        """{SGAID: {label, proposed, notes, ...}} for one vetter."""
        d = self._read_vetter(vetter)
        return {int(r["SGAID"]): r for r in d.to_dict("records")}

    # ---------------------------------------------------------------- writing
    def save_votes(self, vetter, rows):
        """rows: list of dicts with SGAID, label, proposed, source, size_bin, notes (timestamp optional).
        Merges into this vetter's file (a re-vote replaces the old one), then refreshes the master."""
        if not rows:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        for r in rows:
            if r.get("label") not in VALID_LABELS:
                raise ValueError(f"Invalid label {r.get('label')!r} for SGAID {r.get('SGAID')}")
            r.setdefault("timestamp", now)
        new = pd.DataFrame(rows).reindex(columns=VOTE_COLS, fill_value="")
        new["SGAID"] = new["SGAID"].astype("int64")
        with self._lock():
            cur = self._read_vetter(vetter)
            merged = pd.concat([cur[~cur["SGAID"].isin(new["SGAID"])], new], ignore_index=True)
            merged = merged.sort_values("SGAID")
            _atomic_csv(merged, self._vetter_path(vetter))
            self._write_master()
        return len(rows)

    def build_wide(self):
        votes = self.all_votes()
        if votes.empty:
            return pd.DataFrame(columns=FIXED_COLS)
        wide = votes.pivot(index="SGAID", columns="vetter", values="label")
        wide = wide.reindex(columns=sorted(wide.columns))
        out = self.catalog.set_index("SGAID")[["RA", "DEC", "D26"]].reindex(wide.index).join(wide)
        n_missing = int(out["RA"].isna().sum())
        if n_missing:
            print(f"WARNING: {n_missing} vetted SGAID(s) are not in the catalog (RA/DEC/D26 left blank).")
        return out.reset_index()

    def _write_master(self):
        wide = self.build_wide()
        _atomic_csv(wide, self.master_path)
        return wide

    def rebuild_master(self):
        with self._lock():
            return self._write_master()

    # ---------------------------------------------------------------- pools for the vetter UI
    def build_pool(self, vetter, source, mode, seed_extra=""):
        """The galaxies this vetter will see for (size bin, mode).
        Unvetted mode is ordered by the ML catalog's Top1_Proba, highest first (so page 1 is the
        most confident proposals). Agreed mode is ordered by how many other vetters agree
        (most first), shuffled reproducibly within each tier.
        Returns DataFrame: SGAID, RA, DEC, D26, Morphology (the label shown/proposed), source, n_agree."""
        use = [self.ml_id_col, self.ml_morph_col]
        if mode == MODE_UNVETTED:
            use.append(self.ml_proba_col)
        try:
            ml = pd.read_csv(source.ml_catalog, usecols=use)
        except ValueError as e:
            raise ValueError(f"{source.ml_catalog} must contain columns {use}: {e}")
        ml = ml.rename(columns={self.ml_id_col: "SGAID", self.ml_morph_col: "Morphology",
                                self.ml_proba_col: "Top1_Proba"})
        ml["SGAID"] = ml["SGAID"].astype("int64")
        ml = ml.drop_duplicates("SGAID")
        morph = ml["Morphology"].astype(str).str.strip()
        n_odd = int((~morph.isin(LABELS)).sum())
        ml["Morphology"] = morph.where(morph.isin(LABELS), UNKNOWN)

        votes = self.all_votes()
        mine = votes[votes["vetter"] == vetter]

        if mode == MODE_UNVETTED:
            tag = f"ml:{source.label}"
            keep = ~ml["SGAID"].isin(votes["SGAID"]) | ml["SGAID"].isin(
                mine.loc[mine["source"] == tag, "SGAID"])
            pool = ml[keep].copy()
        elif mode == MODE_AGREED:
            tag = f"consensus:{source.label}"
            others = votes[votes["vetter"] != vetter]
            g = others.groupby("SGAID")["label"].agg(n="size", u="nunique", label="first")
            agreed = g[g["u"] == 1]            # every other vote is the same label (1 vote qualifies)
            mine_any = set(mine["SGAID"])
            mine_here = set(mine.loc[mine["source"] == tag, "SGAID"])
            ok = agreed.index[agreed.index.isin(mine_here) | ~agreed.index.isin(mine_any)]
            pool = ml[ml["SGAID"].isin(ok)].copy()
            pool["Morphology"] = agreed["label"].reindex(pool["SGAID"]).to_numpy()
            pool["n_agree"] = agreed["n"].reindex(pool["SGAID"]).to_numpy()
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        pool["source"] = tag
        if "n_agree" not in pool:
            pool["n_agree"] = 0
        pool = pool.merge(self.catalog, on="SGAID", how="left")
        if mode == MODE_UNVETTED:
            # highest ML confidence first; ties and NaNs broken by SGAID so the order is stable
            pool = pool.sort_values(["Top1_Proba", "SGAID"], ascending=[False, True],
                                    na_position="last").reset_index(drop=True)
        else:
            seed = int(hashlib.sha256(f"{vetter}|{source.label}|{mode}|{seed_extra}".encode()).hexdigest()[:8], 16)
            pool = pool.iloc[np.random.default_rng(seed).permutation(len(pool))]
            # most-agreed galaxies first (4 others, then 3, 2, 1); random but reproducible within a tier
            pool = pool.sort_values("n_agree", ascending=False, kind="stable").reset_index(drop=True)
        pool.attrs["n_non_standard_ml_labels"] = n_odd
        pool.attrs["n_vetters_total"] = len(votes["vetter"].unique())
        return pool[["SGAID", "RA", "DEC", "D26", "Morphology", "source", "n_agree"]]

    # ---------------------------------------------------------------- reports
    def merge_vetter(self, source, target, overwrite=False, dry_run=True):
        """Fold one vetter's votes into another (e.g. a mistyped username into the real one).
        Where BOTH voted on a galaxy the target's vote is kept, unless overwrite=True.
        The source file is MOVED to MASTER_DIR/vetters_archive/ (not deleted), so it is recoverable,
        and the master is rebuilt without the source's column. Run with dry_run=True first."""
        if source == target:
            raise ValueError("source and target are the same vetter.")
        for v in (source, target):
            if not self._vetter_path(v).exists() and v == source:
                raise FileNotFoundError(f"No vetter file for {source!r}. Existing: {self.list_vetters()}")
        src, tgt = self._read_vetter(source), self._read_vetter(target)
        both = src["SGAID"].isin(tgt["SGAID"])
        same = src[both].merge(tgt[["SGAID", "label"]], on="SGAID", suffixes=("", "_t"))
        n_conflict = int((same["label"] != same["label_t"]).sum())
        take = src if overwrite else src[~both]
        print(f"{'[DRY RUN] ' if dry_run else ''}merge {source!r} ({len(src)} votes) -> {target!r} ({len(tgt)} votes): "
              f"{len(src) - int(both.sum())} new, {int(both.sum())} already voted by {target!r} "
              f"({n_conflict} with a DIFFERENT label; {'overwritten' if overwrite else 'target kept'}).")
        if dry_run:
            return
        with self._lock():
            merged = pd.concat([tgt[~tgt["SGAID"].isin(take["SGAID"])], take], ignore_index=True).sort_values("SGAID")
            _atomic_csv(merged, self._vetter_path(target))
            arch = self.master_dir / "vetters_archive"
            arch.mkdir(exist_ok=True)
            os.replace(self._vetter_path(source), arch / f"{source}.csv")
            self._write_master()
        print(f"Done. {target!r} now has {len(merged)} votes; {source!r} archived to {arch}.")

    def summary(self):
        """Votes per vetter per label."""
        v = self.all_votes()
        if v.empty:
            return pd.DataFrame()
        return v.pivot_table(index="vetter", columns="label", values="SGAID", aggfunc="count", fill_value=0)

    # ---------------------------------------------------------------- one-time migration
    def import_legacy(self, save_dir, skip=("debug",), dry_run=True):
        """Import old per-user review_log.csv files (<save_dir>/anchor_review/<user>/review_log.csv)
        into the new per-vetter files. Existing new-style votes are never overwritten.
        confirmed_correct -> the morphology reviewed; reclassified -> alt_morphology; bad_anchor -> Bad anchor."""
        report = []
        for p in sorted(Path(save_dir).glob("anchor_review/*/review_log.csv")):
            raw_name = p.parent.name
            if raw_name in skip:
                continue
            try:
                vetter = safe_username(raw_name)
            except ValueError as e:
                report.append({"vetter": raw_name, "imported": 0, "note": f"skipped: {e}"})
                continue
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
            label = np.where(d["status"] == "confirmed_correct", d["morph_reviewed"],
                    np.where(d["status"] == "reclassified", d["alt_morphology"],
                    np.where(d["status"] == "bad_anchor", BAD, "")))
            rows = pd.DataFrame({
                "SGAID": d["SGAID"].astype("int64"), "label": label, "proposed": d["morph_reviewed"],
                "source": "legacy", "size_bin": "", "notes": d.get("notes", ""),
                "timestamp": d.get("timestamp", ""),
            })
            rows = rows[rows["label"].isin(VALID_LABELS)]
            n_multi_region = int(d.groupby("SGAID")["VI_REGION"].nunique().gt(1).sum()) if "VI_REGION" in d else 0
            rows = rows.drop_duplicates("SGAID", keep="last")
            have = set(self._read_vetter(vetter)["SGAID"])
            rows = rows[~rows["SGAID"].isin(have)]
            note = f"{n_multi_region} SGAID(s) appear in more than one VI_REGION (kept last)" if n_multi_region else ""
            if not dry_run and len(rows):
                self.save_votes(vetter, rows.to_dict("records"))
            report.append({"vetter": vetter, "imported": len(rows), "note": note})
        out = pd.DataFrame(report)
        print(("[DRY RUN] would import:\n" if dry_run else "Imported:\n") + (out.to_string(index=False) if len(out) else "(nothing found)"))
        return out


def confusion_matrix_report(store, vetter=None, size_bin=None):
    """Rows = the morphology that was PROPOSED (ML or consensus), columns = what the vetter decided
    (four morphologies + Bad anchor). Diagonal = vetter agreed with the proposal.
    Filter by vetter and/or size_bin; legacy imports count under their original reviewed label."""
    v = store.all_votes()
    if vetter is not None:
        v = v[v["vetter"] == vetter]
    if size_bin is not None:
        v = v[v["source"].str.endswith(f":{size_bin}")]
    cols = LABELS + [BAD]
    rows = [m for m in LABELS + [UNKNOWN, BAD] if (v["proposed"] == m).any()]
    matrix = pd.DataFrame(0, index=rows, columns=cols)
    for (prop, lab), n in v.groupby(["proposed", "label"]).size().items():
        if prop in matrix.index and lab in matrix.columns:
            matrix.loc[prop, lab] += n
    return matrix