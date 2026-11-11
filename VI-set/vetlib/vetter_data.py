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
    def _lock(self, timeout=15.0, stale=60.0):
        """Cross-process lock that works on NERSC CFS, where flock() is unsupported (errno 524).

        Uses an atomically-created lock FILE (open with O_CREAT|O_EXCL, which every shared
        filesystem supports). The file records host:pid and is deleted on release. A lock older
        than `stale` seconds (a crashed kernel) is broken. Votes live in per-vetter files and the
        master is derived, so if the lock still can't be had within `timeout` we warn and carry on
        rather than hang."""
        import socket
        import time
        path = self.master_dir / ".vetting.lock"
        token = f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}"
        held = False
        t0, warned = time.time(), False
        while True:
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as fh:
                    fh.write(token)
                held = True
                break
            except FileExistsError:
                try:
                    if time.time() - path.stat().st_mtime > stale:       # crashed holder -> break it
                        path.unlink()
                        continue
                except OSError:
                    continue                                              # vanished/raced: retry now
            except OSError as e:                                          # unwritable dir etc.
                print(f"WARNING: cannot create the vetting lock ({e}); saving without it.")
                break
            if time.time() - t0 > timeout:
                print(f"WARNING: could not get the vetting lock in {timeout:.0f}s "
                      f"(held by {self._lock_holder(path)}). Continuing without it.")
                break
            if not warned and time.time() - t0 > 3:
                print(f"Waiting for another vetter's save to finish ({self._lock_holder(path)})...", flush=True)
                warned = True
            time.sleep(0.2)
        try:
            yield
        finally:
            if held:
                try:
                    if path.read_text() == token:                         # only remove our own lock
                        path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _lock_holder(path):
        try:
            return Path(path).read_text().strip() or "unknown"
        except OSError:
            return "unknown"

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
            # unvetted galaxies, plus everything THIS vetter has voted on (any source/mode), so their
            # earlier work stays visible under "All"; the vetter's default "Unmarked only" view hides it
            keep = ~ml["SGAID"].isin(votes["SGAID"]) | ml["SGAID"].isin(mine["SGAID"])
            pool = ml[keep].copy()
        elif mode == MODE_AGREED:
            tag = f"consensus:{source.label}"
            others = votes[votes["vetter"] != vetter]
            g = others.groupby("SGAID")["label"].agg(n="size", u="nunique", label="first")
            agreed = g[g["u"] == 1]            # every other vote is the same label (1 vote qualifies)
            # every galaxy the others agree on stays in the pool, including ones this vetter already
            # voted on (any source): they show as marked under "All" and are hidden by "Unmarked only"
            ok = agreed.index
            # the pool comes from the master catalog (anchors need not be in the ML catalog), restricted
            # to THIS size bin by D26 so a bin never shows galaxies from another bin
            pool = self.catalog[self.catalog["SGAID"].isin(ok)].copy()
            want = self.bin_index_of_label(source.label)
            if want is None:
                print(f"WARNING: can't read a size range from the bin label {source.label!r}; not filtering by size.")
                in_bin = pd.Series(True, index=pool.index)
            else:
                in_bin = pool["SGAID"].map(self.size_bin_index()) == want
            n_other = int((~in_bin).sum())
            pool = pool[in_bin].copy()
            sub = agreed.loc[pool["SGAID"]]
            pool["Morphology"] = sub["label"].to_numpy()
            pool["n_agree"] = sub["n"].to_numpy()
            pool.attrs["n_agreed_other_bins"] = n_other
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        pool["source"] = tag
        if "n_agree" not in pool:
            pool["n_agree"] = 0
        if mode == MODE_UNVETTED:                                # ML rows carry no RA/DEC/D26 yet
            pool = pool.merge(self.catalog, on="SGAID", how="left")
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

    def backfill_declared(self, vetter, from_vetters=None, sources=("legacy",), dry_run=True):
        """Give `vetter` a vote on every galaxy they never voted on, using the morphology that was DECLARED
        when other vetters reviewed it (their `proposed` field -- in imported legacy votes this is the
        anchor-catalog label the galaxy was shown under, i.e. what the owner of the set had assigned).

        Guards: only votes whose source is in `sources` count (default "legacy": for votes made in the new
        tool, `proposed` is an ML or consensus label, NOT the owner's judgement); the declared label must be
        one of the four morphologies; galaxies where the other vetters saw DIFFERENT declared labels are
        skipped and reported; existing votes by `vetter` are never touched. New votes are tagged
        source="backfill:declared" so they stay identifiable. from_vetters restricts which vetters'
        rows are used. Run with dry_run=True first."""
        votes = self.all_votes()
        mine = set(votes.loc[votes["vetter"] == vetter, "SGAID"])
        cand = votes[(votes["vetter"] != vetter) & votes["source"].isin(sources) & votes["proposed"].isin(LABELS)]
        if from_vetters is not None:
            cand = cand[cand["vetter"].isin(from_vetters)]
        g = cand.groupby("SGAID").agg(n_declared=("proposed", "nunique"), declared=("proposed", "first"),
                                      seen_by=("vetter", lambda x: ",".join(sorted(set(x)))))
        covered = g.index.isin(mine)
        conflict = (g["n_declared"] > 1) & ~covered
        new = g[(g["n_declared"] == 1) & ~covered]
        print(f"{'[DRY RUN] ' if dry_run else ''}backfill for {vetter!r}: {len(g):,d} galaxies have a declared label "
              f"in {sorted(sources)} votes by others; {int(covered.sum()):,d} already voted by {vetter!r}; "
              f"{int(conflict.sum()):,d} skipped (others saw different declared labels); "
              f"{len(new):,d} would be added.")
        if len(new):
            print("by declared label:", new["declared"].value_counts().to_dict())
            print("by who showed it :", new["seen_by"].value_counts().head(5).to_dict())
        if dry_run or not len(new):
            return new.reset_index()
        rows = [{"SGAID": int(i), "label": r["declared"], "proposed": r["declared"], "source": "backfill:declared",
                 "size_bin": "", "notes": "backfilled from the morphology declared at vetting time"}
                for i, r in new.iterrows()]
        self.save_votes(vetter, rows)
        print(f"Added {len(rows):,d} votes for {vetter!r}. Run store.import_size_bins(dry_run=False) to fill their bins.")
        return new.reset_index()

    @staticmethod
    def _bin_label(k, bin_width=39.824):
        """Classifier-style label for bin k: 0-39arcsec, 40-79arcsec, 80-119arcsec, ..."""
        lo = 0 if k == 0 else int(np.floor(k * bin_width)) + 1
        return f"{lo}-{int(np.floor((k + 1) * bin_width))}arcsec"

    @staticmethod
    def bin_index_of_label(label, bin_width=39.824):
        """Bin number a label like '40-79arcsec' refers to (also accepts the older '39-79arcsec' spelling);
        None if the label can't be parsed."""
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", str(label))
        if not m:
            return None
        lo = int(m.group(1))
        return 0 if lo == 0 else int(round((lo - 1) / bin_width))

    def size_bin_index(self, bin_width=39.824, d26_units="arcmin"):
        """Series SGAID -> integer size-bin number from D26 (NaN where D26 is missing)."""
        to_arcsec = {"arcsec": 1.0, "arcmin": 60.0, "deg": 3600.0}[d26_units]
        d26 = self.catalog.set_index("SGAID")["D26"].astype(float) * to_arcsec
        return np.ceil(d26 / bin_width - 1e-12).clip(lower=1) - 1

    def size_bin_of(self, bin_width=39.824, d26_units="arcmin"):
        """Series SGAID -> size-bin label from D26, in the classifier's spelling. The one definition used
        by import_size_bins and by the agreed-mode pool."""
        idx = self.size_bin_index(bin_width, d26_units)
        return idx.map(lambda k: None if k != k else self._bin_label(int(k), bin_width))

    def import_size_bins(self, bin_width=39.824, d26_units="arcmin", overwrite=False, dry_run=True):
        """Backfill the size_bin column of every vote from the catalog's D26.

        Edges are multiples of bin_width arcsec starting at 0 (the classifier's 152 px x 0.262" = 39.824"),
        labelled like the classifier ("0-39arcsec", "39-79arcsec", ...). Only votes with an EMPTY size_bin are
        filled (legacy imports and anything saved before bins were tracked) unless overwrite=True. Votes
        whose existing size_bin disagrees with D26 are counted and reported either way. The vote's `source`
        tag is never touched, so resuming sessions keeps working. Run with dry_run=True first."""
        bin_of = self.size_bin_of(bin_width, d26_units)
        labels = sorted(bin_of.dropna().unique(), key=lambda x: int(x.split("-")[0]))
        report = []
        with self._lock():
            for v in self.list_vetters():
                d = self._read_vetter(v)
                new = d["SGAID"].map(bin_of)
                have = d["size_bin"].fillna("") != ""
                no_d26 = new.isna()
                mismatch = have & ~no_d26 & (d["size_bin"] != new)
                fill = (~have | overwrite) & ~no_d26
                report.append({"vetter": v, "votes": len(d), "filled": int(fill.sum()),
                               "already_set_ok": int((have & ~mismatch & ~no_d26).sum()),
                               "mismatch_with_D26": int(mismatch.sum()), "no_D26": int(no_d26.sum())})
                if not dry_run and fill.any():
                    d.loc[fill, "size_bin"] = new[fill]
                    _atomic_csv(d, self._vetter_path(v))
            if not dry_run:
                self._write_master()
        out = pd.DataFrame(report)
        print(f"{'[DRY RUN] ' if dry_run else ''}size bins ({bin_width}\" wide): {labels[:6]}{'...' if len(labels) > 6 else ''}")
        print(out.to_string(index=False) if len(out) else "(no vetter files)")
        if len(out) and out["mismatch_with_D26"].sum():
            print("NOTE: some votes have a size_bin that disagrees with D26 -- they were "
                  f"{'overwritten' if overwrite else 'left as they are (pass overwrite=True to recompute them)'}.")
        return out

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
        v = v[v["size_bin"] == size_bin]              # run store.import_size_bins() to fill this for older votes
    cols = LABELS + [BAD]
    rows = [m for m in LABELS + [UNKNOWN, BAD] if (v["proposed"] == m).any()]
    matrix = pd.DataFrame(0, index=rows, columns=cols)
    for (prop, lab), n in v.groupby(["proposed", "label"]).size().items():
        if prop in matrix.index and lab in matrix.columns:
            matrix.loc[prop, lab] += n
    return matrix


def anchor_progress(store, target=1000, min_votes=1):
    """How many anchors do we have, and how many more do we need, per size bin per class?
    A galaxy counts for a class when every vote on it is that class and it has at least min_votes votes
    (Bad-anchor galaxies are counted in their own column). Returns (counts, still_needed) DataFrames
    indexed by size bin; votes without a size_bin are listed under '(unbinned)' -- run
    store.import_size_bins() to fill them. `target` may be an int or {class: n}."""
    v = store.all_votes()
    if v.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = v.groupby("SGAID").agg(n=("label", "size"), u=("label", "nunique"), label=("label", "first"),
                               size_bin=("size_bin", lambda x: next((b for b in x if b), "")))
    g = g[(g["u"] == 1) & (g["n"] >= min_votes)]
    g["size_bin"] = g["size_bin"].replace("", "(unbinned)")
    counts = g.pivot_table(index="size_bin", columns="label", values="n", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=LABELS + [BAD], fill_value=0)
    tgt = target if isinstance(target, dict) else {c: target for c in LABELS}
    needed = pd.DataFrame({c: (tgt.get(c, 0) - counts[c]).clip(lower=0) for c in LABELS})
    needed = needed.drop("(unbinned)", errors="ignore")
    return counts, needed


def vetting_depth(store, cumulative=False, show=True):
    """One table per number of vetters: how many galaxies have been vetted by exactly n people
    (or by at least n with cumulative=True), by size bin x class. A galaxy whose vetters don't all
    agree is counted under 'Disputed'; Bad-anchor consensus has its own column.
    Returns {n: DataFrame}; with show=True each table is also printed."""
    v = store.all_votes()
    if v.empty:
        return {}
    g = v.groupby("SGAID").agg(n=("vetter", "nunique"), u=("label", "nunique"), label=("label", "first"),
                               size_bin=("size_bin", lambda x: next((b for b in x if b), "")))
    g["cls"] = np.where(g["u"] == 1, g["label"], "Disputed")
    g["size_bin"] = g["size_bin"].replace("", "(unbinned)")
    cols = LABELS + [BAD, "Disputed"]
    out = {}
    for n in sorted(int(x) for x in g["n"].unique()):
        sel = g[g["n"] >= n] if cumulative else g[g["n"] == n]
        t = sel.pivot_table(index="size_bin", columns="cls", values="n", aggfunc="count", fill_value=0)
        t = t.reindex(columns=cols, fill_value=0)
        t["Total"] = t.sum(axis=1)
        t.loc["All bins"] = t.sum()
        out[n] = t.astype(int)
        if show:
            print(f"\n=== galaxies vetted by {'at least' if cumulative else 'exactly'} {n} vetter(s) "
                  f"({int(t.loc['All bins', 'Total']):,d}) ===")
            print(out[n].to_string())
    return out