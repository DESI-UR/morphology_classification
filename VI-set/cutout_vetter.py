"""
CutoutVetter: interactive click-to-vet grid of galaxy cutouts.
"""
import os
import csv
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


MASTER_LOG_FIELDNAMES = ["SGAID", "VI_REGION", "RA", "DEC", "morph_reviewed", "status",
                          "alt_morphology", "notes", "timestamp"]


class ReviewerLogin:
    def __init__(self, morph_options, data_lookup_fn, df, cutout_index, sdss_rgb_fn,
                 load_jpg_fn=None, load_jpg_band_fn=None, initial_morph=None, ncols=4, n_per_page=25,
                 figsize_per=2, title_fontsize=None, n_workers=8,
                 save_dir="/global/cfs/cdirs/desicollab/users/qshimp/anchors"):
        self.morph_options = morph_options
        self.data_lookup_fn = data_lookup_fn
        self.df = df
        self.cutout_index = cutout_index
        self.sdss_rgb_fn = sdss_rgb_fn
        self.load_jpg_fn = load_jpg_fn
        self.load_jpg_band_fn = load_jpg_band_fn
        self.initial_morph = initial_morph
        self.ncols = ncols
        self.n_per_page = n_per_page
        self.figsize_per = figsize_per
        self.title_fontsize = title_fontsize
        self.n_workers = n_workers
        self.save_dir = save_dir

        self.vetter = None
        self.vetter_output = widgets.Output()

        self.title = widgets.HTML("<h2>Galaxy Anchor Review</h2>")
        self.username_box = widgets.Text(
            description="Username:", placeholder="Enter your username",
            layout=widgets.Layout(width="350px"),
        )
        self.enter_btn = widgets.Button(description="Enter", button_style="success",
                                         layout=widgets.Layout(width="100px"))
        self.status = widgets.HTML(value="<i>Enter your username to begin.</i>")

        self.enter_btn.on_click(self._on_enter)
        self.username_box.on_submit(self._on_username_submit)

        self.login_box = widgets.VBox([self.title, self.username_box, self.enter_btn, self.status])
        display(widgets.VBox([self.login_box, self.vetter_output]))

    def _on_username_submit(self, _):
        self._start_review()

    def _on_enter(self, _):
        self._start_review()

    def _start_review(self):
        username = self.username_box.value.strip()

        if not username:
            self.status.value = "<b>Please enter a username.</b>"
            return

        invalid_chars = set('/\\:*?"<>|')
        if any(char in invalid_chars for char in username):
            self.status.value = "<b>Invalid username.</b> Please avoid / \\ : * ? \" &lt; &gt; |"
            return

        # Disable inputs immediately so repeated Enter presses can't queue
        # up duplicate session builds while loading.
        self.username_box.disabled = True
        self.enter_btn.disabled = True
        self.status.value = (
            f"<i>⏳ Loading your review session for <b>{username}</b> — this can "
            f"take a little while on a slow kernel. Please don't click again.</i>"
        )

        review_dir = os.path.join(self.save_dir, "anchor_review", username)
        os.makedirs(review_dir, exist_ok=True)

        self.username = username
        self.review_dir = review_dir

        with self.vetter_output:
            self.vetter = CutoutVetter(
                morph_options=self.morph_options,
                data_lookup_fn=self.data_lookup_fn,
                df=self.df,
                cutout_index=self.cutout_index,
                sdss_rgb_fn=self.sdss_rgb_fn,
                load_jpg_fn=self.load_jpg_fn,
                load_jpg_band_fn=self.load_jpg_band_fn,
                initial_morph=self.initial_morph,
                ncols=self.ncols,
                n_per_page=self.n_per_page,
                figsize_per=self.figsize_per,
                title_fontsize=self.title_fontsize,
                n_workers=self.n_workers,
                save_dir=self.review_dir,
                on_exit=self._on_vetter_exit,
            )
        self.vetter.username = self.username

        # Only hide the login screen once the vetter is actually built and ready.
        self.login_box.layout.display = "none"

    def _on_vetter_exit(self):
        if self.vetter is not None:
            self.vetter.close()
        self.vetter_output.clear_output()
        self.username_box.value = ""
        self.username_box.disabled = False
        self.enter_btn.disabled = False
        self.status.value = "<i>Enter your username to begin.</i>"
        self.login_box.layout.display = ""


class CutoutVetter:
    ALT_MORPHOLOGIES = ["Elliptical", "Lenticular", "Spiral", "Irregular"]

    def __init__(self, morph_options, data_lookup_fn, df, cutout_index, sdss_rgb_fn,
                 load_jpg_fn=None, load_jpg_band_fn=None, initial_morph=None, ncols=4, n_per_page=25, figsize_per=2,
                 title_fontsize=None, n_workers=8,
                 save_dir="/pscratch/sd/q/qshimp/Sorter", on_exit=None):
        self.data_lookup_fn = data_lookup_fn
        self.df = df
        self.cutout_index = cutout_index
        self.load_jpg_fn = load_jpg_fn
        self.load_jpg_band_fn = load_jpg_band_fn
        self.sdss_rgb_fn = sdss_rgb_fn
        self.ncols = ncols
        self.n_per_page = n_per_page
        self.figsize_per = figsize_per
        self.title_fontsize = title_fontsize or max(6, int(figsize_per * 4))
        self.n_workers = n_workers
        self.save_dir = save_dir
        self.on_exit = on_exit
        self.view_mode = "Image" if load_jpg_fn is not None else "SSL"

        # image loading: cache + kept-open HDF5 handles + a lock around HDF5 reads
        # (HDF5 isn't reliably safe for concurrent reads from multiple threads)
        self._image_cache = {}          # (key, view_mode) -> rgb array or None
        self._h5_handles = {}           # hdf5 filename -> open h5py.File
        self._h5_lock = threading.Lock()

        self.morph_selector = widgets.Dropdown(
            options=morph_options,
            value=initial_morph or morph_options[0],
            description="Morphology:",
        )
        self.morph_selector.observe(self._on_morph_change, names="value")

        self.body_output = widgets.Output()
        self.top_box = widgets.VBox([self.morph_selector, self.body_output])
        display(self.top_box)

        self._load_morphology(self.morph_selector.value, save_old=False)

    def close(self):
        """Close any open HDF5 handles. Call when the session is torn down."""
        for H in self._h5_handles.values():
            try:
                H.close()
            except Exception:
                pass
        self._h5_handles.clear()

    # ---------- morphology switching ----------

    def _on_morph_change(self, change):
        if getattr(self, "_reverting_morph_change", False):
            return
        if self.selected_keys:
            self._reverting_morph_change = True
            self.morph_selector.value = change["old"]
            self._reverting_morph_change = False
            self.status.value = "<b>Commit or clear your selection before switching morphology.</b>"
            return
        self._load_morphology(change["new"], save_old=True, old_morph=change["old"])

    def _load_morphology(self, morph, save_old, old_morph=None):
        if save_old:
            self.save(morph_override=old_morph)

        data = self.data_lookup_fn(morph, self.df)
        self.gal_keys = [(int(r["SGAID"]), str(r["VI_REGION"])) for _, r in data.iterrows()]
        self.radec = {(int(r["SGAID"]), str(r["VI_REGION"])): (float(r["RA"]), float(r["DEC"]))
                      for _, r in data.iterrows()}
        self.rows_by_key = {(int(r["SGAID"]), str(r["VI_REGION"])): r for _, r in data.iterrows()}

        self.state = {k: None for k in self.gal_keys}
        self.selected_keys = set()
        self.selected_morph = {}
        self.notes = {}
        self.border_color = {}
        self.records = {}
        self.bad_anchors = {}
        self.confirmed = {}

        self._restore_from_disk(morph)

        self.n_pages = max(1, int(np.ceil(len(self.gal_keys) / self.n_per_page)))
        self.page = 0
        self.fig = None
        self.axes = None
        self.ax_to_key = {}

        with self.body_output:
            self.body_output.clear_output(wait=True)
            self.plot_output = widgets.Output()
            self._build_body_controls()
        self._render_page()

    def _logfile_path(self):
        return os.path.join(self.save_dir, "review_log.csv")

    def _restore_from_disk(self, morph):
        valid_keys = set(self.gal_keys)
        path = self._logfile_path()
        if not os.path.exists(path):
            return

        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("morph_reviewed") != morph:
                    continue
                key = (int(row["SGAID"]), row.get("VI_REGION", ""))
                if key not in valid_keys:
                    continue

                self.state[key] = "reviewed"
                self.border_color[key] = "red"
                if row.get("notes"):
                    self.notes[key] = row["notes"]

                status = row.get("status")
                ra, dec = row.get("RA", ""), row.get("DEC", "")
                sgaid, region = key
                if status == "reclassified":
                    if row.get("alt_morphology"):
                        self.selected_morph[key] = row["alt_morphology"]
                    self.records[key] = {
                        "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                        "alt_morphology": row.get("alt_morphology", ""),
                        "notes": row.get("notes", ""), "timestamp": row.get("timestamp", ""),
                    }
                elif status == "bad_anchor":
                    self.bad_anchors[key] = {
                        "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                        "notes": row.get("notes", ""), "timestamp": row.get("timestamp", ""),
                    }
                elif status == "confirmed_correct":
                    self.confirmed[key] = {
                        "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                        "notes": row.get("notes", ""), "timestamp": row.get("timestamp", ""),
                    }

    # ---------- controls ----------

    def _build_body_controls(self):
        self.view_selector = widgets.ToggleButtons(
            options=["Image", "Model", "Residual", "SSL"] if self.load_jpg_fn is not None else ["SSL"],
            value=self.view_mode, description="View:",
        )
        self.view_selector.observe(self._on_view_mode_change, names="value")

        self.morph_btns = {}
        morph_row = []
        for name in self.ALT_MORPHOLOGIES:
            b = widgets.Button(description=name)
            b.on_click(lambda _, n=name: self._on_morph_click(n))
            self.morph_btns[name] = b
            morph_row.append(b)

        self.bad_anchor_btn = widgets.Button(description="Bad anchor", button_style="danger")
        self.bad_anchor_btn.on_click(self._on_bad_anchor)

        self.clear_btn = widgets.Button(description="Clear selection", button_style="warning")
        self.clear_btn.on_click(self._on_clear_selection)

        self.mark_remaining_btn = widgets.Button(description="Affirm All", button_style="info")
        self.mark_remaining_btn.on_click(self._on_mark_remaining_correct)

        self.notes_box = widgets.Textarea(
            placeholder="Notes for the selected galaxies...",
            layout=widgets.Layout(width="400px", height="60px"),
        )
        self.save_btn = widgets.Button(description="Save", button_style="success")
        self.save_btn.on_click(self._on_save)

        self.exit_btn = widgets.Button(description="Exit", button_style="danger")
        self.exit_btn.on_click(self._on_exit_click)

        self.prev_btn = widgets.Button(description="← Previous")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn = widgets.Button(description="Next →")
        self.next_btn.on_click(self._on_next)
        self.page_label = widgets.HTML()

        self.page_jump = widgets.BoundedIntText(
            value=1, min=1, max=self.n_pages, description="Go to page:",
            layout=widgets.Layout(width="150px"),
        )
        self.page_jump.observe(self._on_page_jump, names="value")

        self.status = widgets.HTML(value="<i>Click a galaxy to select it.</i>")

        nav_row = widgets.HBox([self.prev_btn, self.page_label, self.next_btn, self.page_jump])
        display(widgets.VBox([
            self.view_selector,
            nav_row,
            self.plot_output,
            widgets.HBox(morph_row + [self.bad_anchor_btn, self.clear_btn]),
            widgets.HBox([self.mark_remaining_btn]),
            self.notes_box,
            widgets.HBox([self.save_btn, self.exit_btn]),
            self.status,
        ]))

    def _on_clear_selection(self, _):
        if not self.selected_keys:
            return
        for key in list(self.selected_keys):
            self.border_color[key] = None
            ax = self._ax_for_key(key)
            if ax is not None:
                self._set_border(ax, None)
        n = len(self.selected_keys)
        self.selected_keys.clear()
        self.status.value = f"Cleared {n} pending selection(s)."
        self.fig.canvas.draw_idle()

    def _on_mark_remaining_correct(self, _):
        page_keys = self._page_keys()
        marked = 0
        for key in page_keys:
            if self.border_color.get(key) is not None:
                continue
            sgaid, region = key
            ra, dec = self.radec.get(key, (None, None))
            self.confirmed[key] = {
                "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                "notes": "", "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._mark_evaluated(key)
            marked += 1
        self.status.value = f"Marked {marked} remaining galaxy(ies) on this page as correct."
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def _on_exit_click(self, _):
        self.save()
        if callable(self.on_exit):
            self.on_exit()

    # ---------- pagination ----------

    def _page_keys(self):
        start = self.page * self.n_per_page
        end = start + self.n_per_page
        return self.gal_keys[start:end]

    def _on_prev(self, _):
        self._go_to_page(self.page - 1)

    def _on_next(self, _):
        self._go_to_page(self.page + 1)

    def _on_page_jump(self, change):
        if getattr(self, "_updating_page_jump", False):
            return
        self._go_to_page(change["new"] - 1)

    def _go_to_page(self, new_page):
        new_page = max(0, min(new_page, self.n_pages - 1))
        if new_page == self.page:
            self._sync_page_jump()
            return
        if self.selected_keys:
            self.status.value = "<b>Commit or clear your selection before changing pages.</b>"
            self._sync_page_jump()
            return
        self.page = new_page
        self._render_page()

    def _sync_page_jump(self):
        if not hasattr(self, "page_jump"):
            return
        self._updating_page_jump = True
        self.page_jump.value = self.page + 1
        self._updating_page_jump = False

    # ---------- image loading (threaded, cached, with progress) ----------

    def _fetch_image_raw(self, key, view_mode):
        """Pure fetch, no widget/matplotlib calls — safe to run in a worker thread."""
        sgaid, region = key
        if view_mode == "SSL":
            cutout_key = (region, sgaid)
            if cutout_key not in self.cutout_index:
                return None
            fname, idx = self.cutout_index[cutout_key]
            with self._h5_lock:   # HDF5 reads are serialized even though fetch scheduling is parallel
                H = self._h5_handles.get(fname)
                if H is None:
                    H = h5py.File(fname, "r")
                    self._h5_handles[fname] = H
                img = H["images"][idx]
                return self.sdss_rgb_fn([img[0], img[1], img[2]], ["g", "r", "z"])

        row = self.rows_by_key.get(key)
        if row is None:
            return None
    
        if self.load_jpg_band_fn is not None:
            try:
                return self.load_jpg_band_fn(row, view_mode)   # fetch only the active band
            except (KeyError, IndexError, FileNotFoundError):
                return None
    
        # fallback: old 3-tuple API
        try:
            model, residual, image = self.load_jpg_fn(row)
        except (KeyError, IndexError, FileNotFoundError):
            return None
        return {"Model": model, "Residual": residual, "Image": image}

    def _prefetch_page_images(self, page_keys):
        """Concurrently fetch whatever isn't already cached for the current view_mode,
        showing a live progress bar in plot_output while it works."""
        to_fetch = [k for k in page_keys if (k, self.view_mode) not in self._image_cache]
        if not to_fetch:
            return

        progress = widgets.IntProgress(
            value=0, min=0, max=len(to_fetch), bar_style="info",
            layout=widgets.Layout(width="400px"),
        )
        progress_label = widgets.HTML(value=f"Loading images: 0 / {len(to_fetch)} (0%)")
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            display(widgets.VBox([progress_label, progress]))

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._fetch_image_raw, key, self.view_mode): key
                      for key in to_fetch}
            for i, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                try:
                    result = future.result()
                except Exception:
                    result = None

                if isinstance(result, dict):
                    for band, arr in result.items():
                        self._image_cache[(key, band)] = arr
                else:
                    self._image_cache[(key, self.view_mode)] = result

                progress.value = i
                pct = int(100 * i / len(to_fetch))
                progress_label.value = f"Loading images: {i} / {len(to_fetch)} ({pct}%)"

    def _get_cached_image(self, key):
        return self._image_cache.get((key, self.view_mode))

    # ---------- figure rendering ----------

    def _render_page(self):
        page_keys = self._page_keys()
        self._prefetch_page_images(page_keys)

        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            display(widgets.HTML("<i>Rendering...</i>"))   # visible the instant fetching finishes

        ncols = self.ncols if self.ncols else 1
        nrows = int(np.ceil(len(page_keys) / ncols)) if page_keys else 1
        self.ncols = ncols

        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)

            self.fig, axes = plt.subplots(
                nrows, ncols, figsize=(ncols * self.figsize_per, nrows * self.figsize_per)
            )
            self.axes = np.array(axes).ravel()
            self.ax_to_key = {}

            for ax, key in zip(self.axes, page_keys):
                sgaid, region = key
                rgb = self._get_cached_image(key)   # instant — already prefetched
                if rgb is not None:
                    ax.imshow(rgb, origin="lower")
                else:
                    ax.text(0.5, 0.5, f"SGAID\n{sgaid}\n[{region}]\nnot found",
                            ha="center", va="center", transform=ax.transAxes, fontsize=7)

                title = (f"{sgaid} ({self.radec[key][0]:.4f}, {self.radec[key][1]:.4f})"
                         if key in self.radec else f"{sgaid} [{region}]")
                ax.set_title(title, fontsize=self.title_fontsize)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3)
                self._set_border(ax, self.border_color.get(key))
                self.ax_to_key[ax] = key

            for ax in self.axes[len(page_keys):]:
                ax.axis("off")

            self.fig.tight_layout()
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            plt.show()
            self.fig.canvas.draw()
            plt.show()

        self._update_page_label()
        self._sync_page_jump()

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

    def _update_page_label(self):
        start = self.page * self.n_per_page + 1
        end = min(start + self.n_per_page - 1, len(self.gal_keys))
        self.page_label.value = (
            f"&nbsp;&nbsp;Page {self.page + 1}/{self.n_pages}"
            f"&nbsp;(showing {start}-{end} of {len(self.gal_keys)})&nbsp;&nbsp;"
        )
        self.prev_btn.disabled = (self.page == 0)
        self.next_btn.disabled = (self.page == self.n_pages - 1)

    def _on_view_mode_change(self, change):
        self.view_mode = change["new"]
        self._render_page()   # cheap/no-op prefetch if this mode's images are already cached

    def _mark_evaluated(self, key):
        self.state[key] = "reviewed"
        self.border_color[key] = "red"
        ax = self._ax_for_key(key)
        if ax is not None:
            self._set_border(ax, "red")

    def _on_click(self, event):
        ax = event.inaxes
        if ax is None or ax not in self.ax_to_key:
            return
        key = self.ax_to_key[ax]

        if self.border_color.get(key) == "red":
            return

        if key in self.selected_keys:
            self.selected_keys.discard(key)
            self.border_color[key] = None
        else:
            self.selected_keys.add(key)
            self.border_color[key] = "blue"

        self._set_border(ax, self.border_color[key])
        self.status.value = f"{len(self.selected_keys)} galaxy(ies) selected."
        self.fig.canvas.draw_idle()

    def _apply_to_selection(self, alt_morph=None, bad_anchor=False):
        if not self.selected_keys:
            self.status.value = "<b>Select at least one galaxy first.</b>"
            return

        note_text = self.notes_box.value
        for key in list(self.selected_keys):
            sgaid, region = key
            if note_text:
                self.notes[key] = note_text
            if bad_anchor:
                ra, dec = self.radec.get(key, (None, None))
                self.bad_anchors[key] = {
                    "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                    "notes": self.notes.get(key, ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.records.pop(key, None)
                self.confirmed.pop(key, None)
            else:
                self.selected_morph[key] = alt_morph
                self.confirmed.pop(key, None)
            self._mark_evaluated(key)

        n = len(self.selected_keys)
        self.selected_keys.clear()
        self.notes_box.value = ""
        label = "bad anchor" if bad_anchor else alt_morph
        self.status.value = f"Tagged {n} galaxy(ies) as {label}."
        self.fig.canvas.draw_idle()

    def _on_morph_click(self, name):
        self._apply_to_selection(alt_morph=name)

    def _on_bad_anchor(self, _):
        self._apply_to_selection(bad_anchor=True)

    def save(self, morph_override=None):
        morph = morph_override if morph_override is not None else self.morph_selector.value
        now = datetime.now(timezone.utc).isoformat()
        new_rows = {}

        for key, color in self.border_color.items():
            if color != "red":
                continue
            sgaid, region = key
            ra, dec = self.radec.get(key, (None, None))
            if key in self.bad_anchors:
                status, alt = "bad_anchor", ""
            elif key in self.confirmed:
                status, alt = "confirmed_correct", ""
            elif self.selected_morph.get(key) is not None:
                status, alt = "reclassified", self.selected_morph[key]
            else:
                continue

            new_rows[key] = {
                "SGAID": sgaid, "VI_REGION": region, "RA": ra, "DEC": dec,
                "morph_reviewed": morph, "status": status,
                "alt_morphology": alt, "notes": self.notes.get(key, ""),
                "timestamp": now,
            }

        path = self._logfile_path()
        self._write_master_log(path, new_rows)
        self.status.value = f"Saved {len(new_rows)} record(s) to {path}."

    def _on_save(self, _):
        self.save()

    @staticmethod
    def _write_master_log(path, new_rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    existing[(row["SGAID"], row["VI_REGION"])] = row

        for key, v in new_rows.items():
            sgaid, region = key
            existing[(str(sgaid), region)] = {fn: v.get(fn, "") for fn in MASTER_LOG_FIELDNAMES}

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MASTER_LOG_FIELDNAMES)
            writer.writeheader()
            for row in existing.values():
                writer.writerow(row)


def confusion_matrix_report(save_dir, morph_options, username=None):
    review_root = os.path.join(save_dir, "anchor_review")
    if username is not None:
        paths = [os.path.join(review_root, username, "review_log.csv")]
    else:
        paths = glob.glob(os.path.join(review_root, "*", "review_log.csv"))

    columns = list(morph_options) + ["Bad anchor"]
    matrix = pd.DataFrame(0, index=morph_options, columns=columns)

    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                orig = row.get("morph_reviewed")
                if orig not in matrix.index:
                    continue
                status = row.get("status")
                if status == "confirmed_correct":
                    col = orig
                elif status == "reclassified":
                    col = row.get("alt_morphology")
                elif status == "bad_anchor":
                    col = "Bad anchor"
                else:
                    continue
                if col not in matrix.columns:
                    continue
                matrix.loc[orig, col] += 1

    return matrix