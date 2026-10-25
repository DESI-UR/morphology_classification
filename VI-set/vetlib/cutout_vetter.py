"""
cutout_vetter.py -- interactive click-to-vet grid of galaxy cutouts, backed by a shared master.

Flow:  username  ->  size bin + mode  ->  vet.   All data handling lives in vetting_store.py.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from vetlib.coadd_cutouts import CoaddIndex
from vetlib.vetter_data import (BAD, LABELS, MODE_AGREED, MODE_UNVETTED, UNKNOWN, SizeBinSource,
                           VettingStore, confusion_matrix_report, safe_username)


class ReviewerLogin:
    def __init__(self, store, sdss_rgb_fn, size_bins, cutout_index=None, build_cutout_index_fn=None,
                 load_jpg_fn=None, load_jpg_band_fn=None, ncols=10, n_per_page=50,
                 figsize_per=2, title_fontsize=None, n_workers=8, jpg_service=None):
        """store : VettingStore.   size_bins : list of SizeBinSource.
        cutout_index : shared {(region, sgaid): (h5_path, row)} index (used when a bin has no ssl_dir).
        build_cutout_index_fn : callable(ssl_dir) -> index, for bins that carry their own ssl_dir."""
        self.store = store
        self.size_bins = {b.label: b for b in size_bins}
        self.sdss_rgb_fn = sdss_rgb_fn
        self.default_index = cutout_index
        self.build_cutout_index_fn = build_cutout_index_fn
        self._index_cache = {}
        self._coadd_cache = {}
        self.vetter_kwargs = dict(load_jpg_fn=load_jpg_fn, load_jpg_band_fn=load_jpg_band_fn,
                                  jpg_service=jpg_service,
                                  ncols=ncols, n_per_page=n_per_page, figsize_per=figsize_per,
                                  title_fontsize=title_fontsize, n_workers=n_workers)
        self.vetter = None
        self.username = None
        self.vetter_output = widgets.Output()

        # step 1: username
        self.username_box = widgets.Text(description="Username:", placeholder="Enter your username",
                                         layout=widgets.Layout(width="350px"))
        self.enter_btn = widgets.Button(description="Enter", button_style="success",
                                        layout=widgets.Layout(width="100px"))
        self.status = widgets.HTML("<i>Enter your username to begin.</i>")
        self.enter_btn.on_click(lambda _: self._on_enter())
        self.username_box.on_submit(lambda _: self._on_enter())
        self.step1 = widgets.VBox([self.username_box, self.enter_btn])

        # step 2: size bin + mode
        self.bin_dd = widgets.Dropdown(options=list(self.size_bins), description="Size bin:",
                                       layout=widgets.Layout(width="350px"))
        self.mode_rb = widgets.RadioButtons(
            options=[MODE_UNVETTED, MODE_AGREED], description="Vet:",
            layout=widgets.Layout(width="600px"))
        self.mode_help = widgets.HTML(
            f"<small><b>Unvetted</b>: galaxies nobody has vetted yet, starting from the ML-proposed morphology.<br>"
            f"<b>Agreed</b>: galaxies that have been vetted by others, where <i>every</i> vote so far is the same "
            f"label (one vote counts; any disagreement hides it), and you haven't vetted yet. "
            f"Ordered by how many others agree (most first), so you start with the clearest examples.</small>")
        self.start_btn = widgets.Button(description="Start", button_style="success",
                                        layout=widgets.Layout(width="100px"))
        self.back_btn = widgets.Button(description="Back", layout=widgets.Layout(width="100px"))
        self.start_btn.on_click(lambda _: self._on_start())
        self.back_btn.on_click(lambda _: self._show_step(1))
        self.step2 = widgets.VBox([self.bin_dd, self.mode_rb, self.mode_help,
                                   widgets.HBox([self.start_btn, self.back_btn])])
        self.step2.layout.display = "none"

        self.login_box = widgets.VBox([widgets.HTML("<h2>Galaxy Anchor Review</h2>"),
                                       self.step1, self.step2, self.status])
        display(widgets.VBox([self.login_box, self.vetter_output]))

    # ---------------------------------------------------------------- steps
    def _show_step(self, n):
        self.step1.layout.display = "" if n == 1 else "none"
        self.step2.layout.display = "" if n == 2 else "none"
        self.username_box.disabled = (n == 2)
        self.status.value = ("<i>Enter your username to begin.</i>" if n == 1 else
                             f"<i>Hi <b>{self.username}</b> -- choose a size bin and what you want to vet.</i>")

    def _on_enter(self):
        try:
            self.username = safe_username(self.username_box.value)
        except ValueError as e:
            self.status.value = f"<b>{e}</b>"
            return
        self.username_box.value = self.username
        self._show_step(2)

    def _cutout_index_for(self, source):
        if source.ssl_dir is None:
            if self.default_index is None:
                raise ValueError("No shared cutout_index given and this size bin has no ssl_dir.")
            return self.default_index
        if source.ssl_dir not in self._index_cache:
            if self.build_cutout_index_fn is None:
                raise ValueError("size bin has an ssl_dir but build_cutout_index_fn was not given.")
            self._index_cache[source.ssl_dir] = self.build_cutout_index_fn(source.ssl_dir)
        return self._index_cache[source.ssl_dir]

    def _coadd_index_for(self, source):
        """One CoaddIndex per bin folder, kept for the whole notebook session and refreshed on Start."""
        if not getattr(source, "coadd_dir", None):
            return None
        if source.coadd_dir not in self._coadd_cache:
            self._coadd_cache[source.coadd_dir] = CoaddIndex(source.coadd_dir)
        idx = self._coadd_cache[source.coadd_dir]
        idx.refresh(force=True)
        return idx

    def _on_start(self):
        source, mode = self.size_bins[self.bin_dd.value], self.mode_rb.value
        for w in (self.start_btn, self.back_btn, self.bin_dd, self.mode_rb):
            w.disabled = True
        self.status.value = (f"<i>⏳ Loading <b>{source.label}</b> for <b>{self.username}</b> -- "
                             f"please don't click again.</i>")
        try:
            self.store.rebuild_master()                       # make sure the master reflects everyone's latest
            pool = self.store.build_pool(self.username, source, mode)
            index = self._cutout_index_for(source)
            coadd = self._coadd_index_for(source)
        except Exception as e:
            self.status.value = f"<b>Could not start:</b> {type(e).__name__}: {e}"
            self._enable_step2()
            return
        if len(pool) == 0:
            hint = (f" Only {pool.attrs['n_vetters_total']} vetter(s) have votes so far; "
                    f"no galaxy yet has votes from others that all agree." if mode == MODE_AGREED else "")
            self.status.value = f"<b>Nothing to vet in {source.label} for this mode.</b>{hint}"
            self._enable_step2()
            return
        if pool.attrs.get("n_non_standard_ml_labels"):
            print(f"Note: {pool.attrs['n_non_standard_ml_labels']:,d} ML labels were not one of {LABELS}; "
                  f"they are shown as '{UNKNOWN}'.")
        if coadd is not None:
            have, tot = coadd.coverage(pool["SGAID"])
            print(f"Coadd cutouts cover {have:,d} of {tot:,d} galaxies in this pool ({100 * have / max(tot, 1):.1f}%); "
                  f"the rest show 'not found' in the Coadd view.")
        with self.vetter_output:
            self.vetter = CutoutVetter(
                store=self.store, vetter=self.username, source=source, mode=mode, pool=pool,
                cutout_index=index, coadd_index=coadd, sdss_rgb_fn=self.sdss_rgb_fn,
                on_exit=self._on_vetter_exit,
                **self.vetter_kwargs)
        self.login_box.layout.display = "none"

    def _enable_step2(self):
        for w in (self.start_btn, self.back_btn, self.bin_dd, self.mode_rb):
            w.disabled = False

    def _on_vetter_exit(self):
        if self.vetter is not None:
            self.vetter.close()
        self.vetter_output.clear_output()
        self.username_box.value = ""
        self.username_box.disabled = False
        self._enable_step2()
        self._show_step(1)
        self.login_box.layout.display = ""


class CutoutVetter:
    ALT_MORPHOLOGIES = LABELS

    def __init__(self, store, vetter, source, mode, pool, cutout_index, sdss_rgb_fn,
                 load_jpg_fn=None, load_jpg_band_fn=None, ncols=10, n_per_page=50, figsize_per=2,
                 title_fontsize=None, n_workers=8, on_exit=None, jpg_service=None, coadd_index=None):
        self.store, self.vetter, self.source, self.mode = store, vetter, source, mode
        self.pool = pool
        self.sdss_rgb_fn = sdss_rgb_fn
        self.load_jpg_fn, self.load_jpg_band_fn = load_jpg_fn, load_jpg_band_fn
        self.jpg_service = jpg_service
        self.coadd_index = coadd_index
        self.ncols, self.n_per_page, self.figsize_per = ncols, n_per_page, figsize_per
        self.title_fontsize = title_fontsize or max(6, int(figsize_per * 4))
        self.n_workers = n_workers
        self.on_exit = on_exit
        self.view_mode = "Image" if load_jpg_fn is not None else "SSL"
        self.filter_mode = "All"

        # SGAID -> (region, sgaid) key into the cutout index (SGAIDs are unique in v1.0)
        self.sgaid_to_cutout, n_dup = {}, 0
        for (region, sgaid) in cutout_index:
            if sgaid in self.sgaid_to_cutout:
                n_dup += 1
            else:
                self.sgaid_to_cutout[sgaid] = (region, sgaid)
        if n_dup:
            print(f"WARNING: {n_dup} SGAID(s) appear in both regions of the cutout index; using the first.")
        self.cutout_index = cutout_index

        sg = pool["SGAID"].to_numpy()
        self.radec = dict(zip(sg.tolist(), zip(pool["RA"].tolist(), pool["DEC"].tolist())))
        self.pool_pos = {s: i for i, s in enumerate(sg.tolist())}
        self.proposed = dict(zip(sg.tolist(), pool["Morphology"].tolist()))
        self.n_agree = dict(zip(sg.tolist(), pool["n_agree"].tolist()))
        self.groups = {}
        for s, m in zip(sg.tolist(), pool["Morphology"].tolist()):
            self.groups.setdefault(m, []).append(s)
        self.source_tag = pool["source"].iloc[0]

        # restore this vetter's earlier votes on galaxies in this pool
        mine = store.my_votes(vetter)
        self.votes = {s: {"label": r["label"], "notes": r["notes"]} for s, r in mine.items()
                      if s in self.pool_pos}
        self.dirty = set()
        self.selected = set()

        self._image_cache = {}
        self._h5_handles = {}
        self._h5_lock = threading.Lock()
        self.fig, self.axes, self.ax_to_key = None, None, {}
        self.page = 0

        order = [m for m in LABELS + [UNKNOWN, BAD] if m in self.groups]
        order += [m for m in self.groups if m not in order]
        self.morph_selector = widgets.Dropdown(
            options=[(f"{m} ({len(self.groups[m]):,d})", m) for m in order], value=order[0],
            description="Morphology:", layout=widgets.Layout(width="350px"))
        self.group = self.morph_selector.value
        self.keys = self.groups[self.group]
        self.morph_selector.observe(self._on_morph_change, names="value")

        header = widgets.HTML(
            f"<b>{vetter}</b> &middot; size bin <b>{source.label}</b> &middot; {mode} "
            f"&middot; {len(pool):,d} galaxies")
        self._build_controls(header)
        self._recompute_pagination()
        self._render_page()

    def close(self):
        for H in self._h5_handles.values():
            try:
                H.close()
            except Exception:
                pass
        self._h5_handles.clear()

    # ---------------------------------------------------------------- controls
    def _build_controls(self, header):
        self.view_selector = widgets.ToggleButtons(
            options=((["Image", "Model", "Residual", "SSL"] if self.load_jpg_fn is not None else ["SSL"])
                     + (["Coadd"] if self.coadd_index is not None else [])),
            value=self.view_mode, description="View:")
        self.view_selector.observe(self._on_view_mode_change, names="value")
        self.filter_selector = widgets.ToggleButtons(options=["All", "Unmarked only"],
                                                     value=self.filter_mode, description="Show:")
        self.filter_selector.observe(self._on_filter_change, names="value")

        self.morph_btns, morph_row = {}, []
        for name in self.ALT_MORPHOLOGIES:
            b = widgets.Button(description=name)
            b.on_click(lambda _, n=name: self._apply_to_selection(n))
            self.morph_btns[name] = b
            morph_row.append(b)
        self.bad_anchor_btn = widgets.Button(description="Bad anchor", button_style="danger")
        self.bad_anchor_btn.on_click(lambda _: self._apply_to_selection(BAD))
        self.clear_btn = widgets.Button(description="Clear selection", button_style="warning")
        self.clear_btn.on_click(self._on_clear_selection)
        self.mark_remaining_btn = widgets.Button(description="Affirm All", button_style="info")
        self.mark_remaining_btn.on_click(self._on_mark_remaining_correct)
        self.notes_box = widgets.Textarea(placeholder="Notes for the selected galaxies...",
                                          layout=widgets.Layout(width="400px", height="60px"))
        self.save_btn = widgets.Button(description="Save", button_style="success")
        self.save_btn.on_click(lambda _: self.save())
        self.exit_btn = widgets.Button(description="Exit", button_style="danger")
        self.exit_btn.on_click(self._on_exit_click)

        self.prev_btn = widgets.Button(description="← Previous")
        self.prev_btn.on_click(lambda _: self._go_to_page(self.page - 1))
        self.next_btn = widgets.Button(description="Next →")
        self.next_btn.on_click(lambda _: self._go_to_page(self.page + 1))
        self.page_label = widgets.HTML()
        self.page_jump = widgets.BoundedIntText(value=1, min=1, max=1, description="Go to page:",
                                                layout=widgets.Layout(width="150px"))
        self.page_jump.observe(self._on_page_jump, names="value")
        self.status = widgets.HTML("<i>Click a galaxy to select it.</i>")
        self.plot_output = widgets.Output()

        display(widgets.VBox([
            header, self.morph_selector,
            widgets.HBox([self.view_selector, self.filter_selector]),
            widgets.HBox([self.prev_btn, self.page_label, self.next_btn, self.page_jump]),
            self.plot_output,
            widgets.HBox(morph_row + [self.bad_anchor_btn, self.clear_btn]),
            widgets.HBox([self.mark_remaining_btn]),
            self.notes_box,
            widgets.HBox([self.save_btn, self.exit_btn]),
            self.status,
        ]))

    # ---------------------------------------------------------------- morphology group switching
    def _on_morph_change(self, change):
        if getattr(self, "_reverting_morph_change", False):
            return
        if self.selected:
            self._reverting_morph_change = True
            self.morph_selector.value = change["old"]
            self._reverting_morph_change = False
            self.status.value = "<b>Commit or clear your selection before switching morphology.</b>"
            return
        self.save()
        self.group = change["new"]
        self.keys = self.groups[self.group]
        self.page = 0
        self._recompute_pagination()
        self._render_page()

    def _on_filter_change(self, change):
        if getattr(self, "_reverting_filter_change", False):
            return
        if self.selected:
            self._reverting_filter_change = True
            self.filter_selector.value = change["old"]
            self._reverting_filter_change = False
            self.status.value = "<b>Commit or clear your selection before changing the filter.</b>"
            return
        self.filter_mode = change["new"]
        self.page = 0
        self._recompute_pagination()
        self._render_page()

    def _on_view_mode_change(self, change):
        self.view_mode = change["new"]
        if self.view_mode == "Coadd" and self.coadd_index is not None:
            for f in self.coadd_index.refresh(verbose=False):         # pick up chunks generated since last look
                with self._h5_lock:
                    H = self._h5_handles.pop(f, None)
                if H is not None:
                    try:
                        H.close()
                    except Exception:
                        pass
        self._render_page()

    # ---------------------------------------------------------------- pagination
    def _filtered_keys(self):
        if self.filter_mode == "Unmarked only":
            return [k for k in self.keys if k not in self.votes]
        return self.keys

    def _page_keys(self):
        keys = self._filtered_keys()
        return keys[self.page * self.n_per_page:(self.page + 1) * self.n_per_page]

    def _recompute_pagination(self):
        total = len(self._filtered_keys())
        self.n_pages = max(1, int(np.ceil(total / self.n_per_page)))
        self.page = min(self.page, self.n_pages - 1)
        self._sync_page_jump(update_max=True)

    def _sync_page_jump(self, update_max=False):
        self._updating_page_jump = True
        if update_max:
            self.page_jump.max = self.n_pages
        self.page_jump.value = self.page + 1
        self._updating_page_jump = False

    def _on_page_jump(self, change):
        if not getattr(self, "_updating_page_jump", False):
            self._go_to_page(change["new"] - 1)

    def _go_to_page(self, new_page):
        new_page = max(0, min(new_page, self.n_pages - 1))
        if new_page == self.page:
            self._sync_page_jump()
            return
        if self.selected:
            self.status.value = "<b>Commit or clear your selection before changing pages.</b>"
            self._sync_page_jump()
            return
        self.page = new_page
        self._render_page()

    def _update_page_label(self):
        total = len(self._filtered_keys())
        if total == 0:
            shown = "0-0 of 0"
        else:
            start = self.page * self.n_per_page + 1
            shown = f"{start}-{min(start + self.n_per_page - 1, total)} of {total}"
        unmarked = sum(1 for k in self.keys if k not in self.votes)
        tier = ""
        if self.mode == MODE_AGREED:
            ns = [self.n_agree[k] for k in self._page_keys()]
            if ns:
                lo, hi = int(min(ns)), int(max(ns))
                tier = (f"&nbsp;|&nbsp;agreed by <b>{hi if lo == hi else f'{lo}-{hi}'}</b> other vetter(s)")
        self.page_label.value = (f"&nbsp;&nbsp;Page {self.page + 1}/{self.n_pages}&nbsp;(showing {shown}){tier}"
                                 f"&nbsp;|&nbsp;<b>{unmarked} unmarked</b> in this group"
                                 f"&nbsp;|&nbsp;{len(self.dirty)} unsaved&nbsp;&nbsp;")
        self.prev_btn.disabled = (self.page == 0)
        self.next_btn.disabled = (self.page == self.n_pages - 1)

    # ---------------------------------------------------------------- images
    def _row(self, key):
        d = self.pool.iloc[self.pool_pos[key]].to_dict()
        return d

    def _read_h5_rgb(self, fname, idx):
        with self._h5_lock:
            H = self._h5_handles.get(fname)
            if H is None:
                H = h5py.File(fname, "r")
                self._h5_handles[fname] = H
            img = H["images"][idx]
        return self.sdss_rgb_fn([img[0], img[1], img[2]], ["g", "r", "z"])

    def _fetch_image_raw(self, key, view_mode):
        if view_mode == "SSL":
            ckey = self.sgaid_to_cutout.get(key)
            if ckey is None:
                return None
            fname, idx = self.cutout_index[ckey]
            return self._read_h5_rgb(fname, idx)
        if view_mode == "Coadd":
            loc = self.coadd_index.lookup(key) if self.coadd_index is not None else None
            return None if loc is None else self._read_h5_rgb(*loc)
        row = self._row(key)
        if self.load_jpg_band_fn is not None:
            try:
                return self.load_jpg_band_fn(row, view_mode)
            except (KeyError, IndexError, FileNotFoundError):
                return None
        try:
            model, residual, image = self.load_jpg_fn(row)
        except (KeyError, IndexError, FileNotFoundError):
            return None
        return {"Model": model, "Residual": residual, "Image": image}

    def _prefetch_page_images(self, page_keys):
        # re-check anything that came back empty last time (e.g. a JPG the background service has since fetched)
        to_fetch = [k for k in page_keys if self._image_cache.get((k, self.view_mode)) is None]
        if not to_fetch:
            return
        progress = widgets.IntProgress(value=0, min=0, max=len(to_fetch), bar_style="info",
                                       layout=widgets.Layout(width="400px"))
        label = widgets.HTML(f"Loading images: 0 / {len(to_fetch)} (0%)")
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            display(widgets.VBox([label, progress]))
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {ex.submit(self._fetch_image_raw, k, self.view_mode): k for k in to_fetch}
            for i, fut in enumerate(as_completed(futures), start=1):
                key = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if isinstance(result, dict):
                    for band, arr in result.items():
                        self._image_cache[(key, band)] = arr
                else:
                    self._image_cache[(key, self.view_mode)] = result
                progress.value = i
                label.value = f"Loading images: {i} / {len(to_fetch)} ({int(100 * i / len(to_fetch))}%)"

    # ---------------------------------------------------------------- rendering
    def _color(self, key):
        if key in self.selected:
            return "yellow" if key in self.votes else "blue"
        return "red" if key in self.votes else None

    def _set_border(self, ax, color):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color if color else "none")
            spine.set_linewidth(3 if color else 1)

    def _ax_for_key(self, key):
        for ax, k in self.ax_to_key.items():
            if k == key:
                return ax
        return None

    def _render_page(self):
        page_keys = self._page_keys()
        if not page_keys:
            with self.plot_output:
                self.plot_output.clear_output(wait=True)
                msg = ("🎉 No unmarked galaxies in this group -- nice work!"
                       if self.filter_mode == "Unmarked only" else "No galaxies to show.")
                display(widgets.HTML(f"<i>{msg}</i>"))
            self.fig, self.ax_to_key = None, {}
            self._update_page_label()
            self._sync_page_jump()
            return

        self._prefetch_page_images(page_keys)
        ncols = self.ncols or 1
        nrows = int(np.ceil(len(page_keys) / ncols))
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)
            self.fig, axes = plt.subplots(nrows, ncols,
                                          figsize=(ncols * self.figsize_per, nrows * self.figsize_per))
            self.axes = np.array(axes).ravel()
            self.ax_to_key = {}
            for ax, key in zip(self.axes, page_keys):
                rgb = self._image_cache.get((key, self.view_mode))
                if rgb is not None:
                    ax.imshow(rgb, origin="lower")
                else:
                    ax.text(0.5, 0.5, f"SGAID\n{key}\nnot found", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7)
                ra, dec = self.radec[key]
                ax.set_title(f"{key} ({ra:.4f}, {dec:.4f})" if ra == ra else f"{key}",
                             fontsize=self.title_fontsize)
                ax.set_xticks([]); ax.set_yticks([])
                self._set_border(ax, self._color(key))
                self.ax_to_key[ax] = key
            for ax in self.axes[len(page_keys):]:
                ax.axis("off")
            self.fig.tight_layout()
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            plt.show()
            self.fig.canvas.draw()
        self._update_page_label()
        self._sync_page_jump()

    # ---------------------------------------------------------------- interaction
    def _on_click(self, event):
        ax = event.inaxes
        if ax is None or ax not in self.ax_to_key:
            return
        key = self.ax_to_key[ax]
        self.selected.symmetric_difference_update({key})
        self._set_border(ax, self._color(key))
        n_reedit = sum(1 for k in self.selected if k in self.votes)
        msg = f"{len(self.selected)} galaxy(ies) selected."
        if n_reedit:
            msg = (f"{len(self.selected)} galaxy(ies) selected "
                   f"(<b>{n_reedit} re-editing an existing vet</b> -- this will overwrite it).")
        self.status.value = msg
        self.fig.canvas.draw_idle()

    def _on_clear_selection(self, _):
        if not self.selected:
            return
        n = len(self.selected)
        cleared, self.selected = list(self.selected), set()
        for key in cleared:
            ax = self._ax_for_key(key)
            if ax is not None:
                self._set_border(ax, self._color(key))
        self.status.value = f"Cleared {n} pending selection(s)."
        self.fig.canvas.draw_idle()

    def _vote(self, key, label, notes=""):
        self.votes[key] = {"label": label, "notes": notes}
        self.dirty.add(key)

    def _after_marking(self):
        self._recompute_pagination()
        if self.filter_mode == "Unmarked only":
            self._render_page()
        else:
            self._update_page_label()
            if self.fig is not None:
                self.fig.canvas.draw_idle()

    def _apply_to_selection(self, label):
        if not self.selected:
            self.status.value = "<b>Select at least one galaxy first.</b>"
            return
        note = self.notes_box.value
        n_reedit = sum(1 for k in self.selected if k in self.votes)
        done, self.selected = list(self.selected), set()
        for key in done:
            self._vote(key, label, note)
            ax = self._ax_for_key(key)
            if ax is not None:
                self._set_border(ax, self._color(key))
        self.notes_box.value = ""
        suffix = f" ({n_reedit} were re-edits)" if n_reedit else ""
        self.status.value = f"Tagged {len(done)} galaxy(ies) as {label}{suffix}."
        self._after_marking()

    def _on_mark_remaining_correct(self, _):
        if self.group == UNKNOWN:
            self.status.value = ("<b>Nothing to affirm:</b> these galaxies have no proposed morphology. "
                                 "Select them and choose a label.")
            return
        marked = 0
        for key in self._page_keys():
            if key in self.votes or key in self.selected:
                continue
            self._vote(key, self.group)                         # affirming == agreeing with the proposal
            ax = self._ax_for_key(key)
            if ax is not None:
                self._set_border(ax, self._color(key))
            marked += 1
        self.status.value = f"Affirmed {marked} remaining galaxy(ies) on this page as {self.group}."
        self._after_marking()

    # ---------------------------------------------------------------- saving
    def save(self):
        if not self.dirty:
            self.status.value = "Nothing new to save."
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [{"SGAID": k, "label": self.votes[k]["label"], "proposed": self.proposed[k],
                 "source": self.source_tag, "size_bin": self.source.label,
                 "notes": self.votes[k]["notes"], "timestamp": now} for k in sorted(self.dirty)]
        try:
            n = self.store.save_votes(self.vetter, rows)
        except Exception as e:                                  # keep votes in memory, tell the user
            self.status.value = f"<b>SAVE FAILED</b> ({type(e).__name__}: {e}). Your votes are still in memory -- try Save again."
            return
        saved = sorted(self.dirty)
        self.dirty.clear()
        jpg_note = ""
        if self.jpg_service is not None:                        # fetch Image/Model/Residual JPGs in the background
            ra_dec = [(k, *self.radec[k]) for k in saved]
            self.jpg_service.enqueue(ra_dec, size_pix=getattr(self.source, "jpg_pix", None))
            jpg_note = f" | {self.jpg_service.status()}"
        self.status.value = f"Saved {n} vote(s). Master updated: {self.store.master_path}{jpg_note}"
        self._update_page_label()

    def _on_exit_click(self, _):
        self.save()
        if self.dirty:                                           # save failed -- don't throw the votes away
            return
        if callable(self.on_exit):
            self.on_exit()