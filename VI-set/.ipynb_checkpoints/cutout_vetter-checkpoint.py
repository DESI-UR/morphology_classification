"""
CutoutVetter: interactive click-to-vet grid of galaxy cutouts.
"""
import os
import csv
from datetime import datetime, timezone

import numpy as np
import h5py
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


class CutoutVetter:
    ALT_MORPHOLOGIES = ["Elliptical", "Lenticular", "Spiral", "Irregular"]

    def __init__(self, morph_options, data_lookup_fn, df, cutout_index, sdss_rgb_fn, load_jpg_fn = None, 
                 initial_morph=None, ncols=4, n_per_page=25, figsize_per=2,
                 title_fontsize=None, save_dir="/pscratch/sd/q/qshimp/Sorter"):
        self.data_lookup_fn = data_lookup_fn   # callable: morph_label -> DataFrame(SGAID, RA, DEC, VI_REGION)
        self.df = df
        self.cutout_index = cutout_index
        self.load_jpg_fn = load_jpg_fn         # callable: row -> (model, residual, image)
        self.sdss_rgb_fn = sdss_rgb_fn         # callable: ([g,r,z] bands) -> RGB array
        self.ncols = ncols
        self.n_per_page = n_per_page
        self.figsize_per = figsize_per
        self.title_fontsize = title_fontsize or max(6, int(figsize_per * 4))
        self.save_dir = save_dir
        self.view_mode = "Image" if load_jpg_fn is not None else "DR-11"   # persists across morphology switches

        self.morph_selector = widgets.Dropdown(
            options=morph_options,
            value=initial_morph or morph_options[0],
            description="Morphology:",
        )
        self.morph_selector.observe(self._on_morph_change, names="value")

        self.body_output = widgets.Output()   # everything below the dropdown; rebuilt per morphology
        display(widgets.VBox([self.morph_selector, self.body_output]))

        self._load_morphology(self.morph_selector.value, save_old=False)

    # ---------- morphology switching ----------

    def _on_morph_change(self, change):
        if getattr(self, "_reverting_morph_change", False):
            return
        if self.selected_sgaids:
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
        self.sgaids = [int(s) for s in data["SGAID"]]
        self.radec = {int(r["SGAID"]): (float(r["RA"]), float(r["DEC"])) for _, r in data.iterrows()}
        self.region_lookup = {
            int(r["SGAID"]): str(r["VI_REGION"]) for _, r in data.iterrows() if "VI_REGION" in r
        }
        self.rows_by_sgaid = {int(r["SGAID"]): r for _, r in data.iterrows()}

        self.state = {s: None for s in self.sgaids}
        self.selected_sgaids = set()
        self.selected_morph = {}
        self.notes = {}
        self.border_color = {}
        self.records = {}
        self.bad_anchors = {}

        self._restore_from_disk(morph)

        self.n_pages = max(1, int(np.ceil(len(self.sgaids) / self.n_per_page)))
        self.page = 0
        self.fig = None
        self.axes = None
        self.ax_to_sgaid = {}

        with self.body_output:
            self.body_output.clear_output(wait=True)
            self.plot_output = widgets.Output()
            self._build_body_controls()
        self._render_page()

    def _restore_from_disk(self, morph):
        """Repopulate in-memory state from previously-saved CSVs for this morphology."""
        valid_sgaids = set(self.sgaids)

        misclass_path = os.path.join(self.save_dir, f"misclassification_{morph}.csv")
        if os.path.exists(misclass_path):
            with open(misclass_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    sgaid = int(row["SGAID"])
                    if sgaid not in valid_sgaids:
                        continue
                    self.state[sgaid] = "reviewed"
                    self.border_color[sgaid] = "red"
                    if row.get("alt_morphology"):
                        self.selected_morph[sgaid] = row["alt_morphology"]
                    if row.get("notes"):
                        self.notes[sgaid] = row["notes"]
                    self.records[sgaid] = {
                        "SGAID": sgaid,
                        "RA": row.get("RA", ""),
                        "DEC": row.get("DEC", ""),
                        "alt_morphology": row.get("alt_morphology", ""),
                        "notes": row.get("notes", ""),
                        "timestamp": row.get("timestamp", ""),
                    }

        bad_path = os.path.join(self.save_dir, f"bad_anchors_{morph}.csv")
        if os.path.exists(bad_path):
            with open(bad_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    sgaid = int(row["SGAID"])
                    if sgaid not in valid_sgaids:
                        continue
                    self.state[sgaid] = "reviewed"
                    self.border_color[sgaid] = "red"
                    if row.get("notes"):
                        self.notes[sgaid] = row["notes"]
                    self.bad_anchors[sgaid] = {
                        "SGAID": sgaid,
                        "RA": row.get("RA", ""),
                        "DEC": row.get("DEC", ""),
                        "notes": row.get("notes", ""),
                        "timestamp": row.get("timestamp", ""),
                    }

    # ---------- controls (per-morphology body, not the dropdown) ----------

    def _build_body_controls(self):
        self.view_selector = widgets.ToggleButtons(
            options=["Image", "Model", "Residual", "DR-11"] if self.load_jpg_fn is not None else ["DR-11"],
            value=self.view_mode,
            description="View:",
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
        
        self.notes_box = widgets.Textarea(
            placeholder="Notes for the selected galaxies...",
            layout=widgets.Layout(width="400px", height="60px"),
        )
        self.save_btn = widgets.Button(description="Save", button_style="success")
        self.save_btn.on_click(self._on_save)

        self.prev_btn = widgets.Button(description="← Previous")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn = widgets.Button(description="Next →")
        self.next_btn.on_click(self._on_next)
        self.page_label = widgets.HTML()

        self.status = widgets.HTML(value="<i>Click a galaxy to select it.</i>")

        nav_row = widgets.HBox([self.prev_btn, self.page_label, self.next_btn])
        display(widgets.VBox([
            self.view_selector,
            nav_row,
            self.plot_output,
            widgets.HBox(morph_row + [self.bad_anchor_btn, self.clear_btn]),
            self.notes_box,
            widgets.HBox([self.save_btn]),
            self.status,
        ]))

    def _on_clear_selection(self, _):
        if not self.selected_sgaids:
            return
        for sgaid in list(self.selected_sgaids):
            self.border_color[sgaid] = None
            ax = self._ax_for_sgaid(sgaid)
            if ax is not None:
                self._set_border(ax, None)
        n = len(self.selected_sgaids)
        self.selected_sgaids.clear()
        self.status.value = f"Cleared {n} pending selection(s)."
        self.fig.canvas.draw_idle()

    # ---------- pagination ----------

    def _page_sgaids(self):
        start = self.page * self.n_per_page
        end = start + self.n_per_page
        return self.sgaids[start:end]

    def _on_prev(self, _):
        self._go_to_page(self.page - 1)

    def _on_next(self, _):
        self._go_to_page(self.page + 1)

    # ---------- figure / click handling ----------

    def _get_display_image(self, sgaid):
        """Return an RGB array to display for this SGAID, honoring self.view_mode."""
        if self.view_mode == "DR-11":
            region = self.region_lookup.get(sgaid)
            key = (region, sgaid) if region is not None else None
            if key is None or key not in self.cutout_index:
                return None
            fname, idx = self.cutout_index[key]
            with h5py.File(fname, "r") as H:
                img = H["images"][idx]
            return self.sdss_rgb_fn([img[0], img[1], img[2]], ["g", "r", "z"])

        row = self.rows_by_sgaid.get(sgaid)
        if row is None:
            return None
        try:
            model, residual, image = self.load_jpg_fn(row)
        except (IndexError, FileNotFoundError):
            return None

        if self.view_mode == "Model":
            return model
        elif self.view_mode == "Residual":
            return residual
        else:
            return image

    def _render_page(self):
        page_sgaids = self._page_sgaids()
        ncols = self.ncols if self.ncols else 1
        nrows = int(np.ceil(len(page_sgaids) / ncols)) if page_sgaids else 1
        self.ncols = ncols  # keep in sync, in case anything else reads it

        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)

            self.fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(ncols * self.figsize_per, nrows * self.figsize_per)
            )
            self.axes = np.array(axes).ravel()
            self.ax_to_sgaid = {}

            for ax, sgaid in zip(self.axes, page_sgaids):
                rgb = self._get_display_image(sgaid)
                if rgb is not None:
                    ax.imshow(rgb, origin="lower")
                else:
                    ax.text(0.5, 0.5, f"SGAID\n{sgaid}\nnot found",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=7)

                title = (f"{sgaid} ({self.radec[sgaid][0]:.4f}, {self.radec[sgaid][1]:.4f})"
                         if sgaid in self.radec else str(sgaid))
                ax.set_title(title, fontsize=self.title_fontsize)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(3)
                self._set_border(ax, self.border_color.get(sgaid))
                self.ax_to_sgaid[ax] = sgaid

            for ax in self.axes[len(page_sgaids):]:
                ax.axis("off")

            self.fig.tight_layout()
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            plt.show()

        self._update_page_label()

    def _set_border(self, ax, color):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color if color else "none")
            spine.set_linewidth(3 if color else 1)

    def _ax_for_sgaid(self, sgaid):
        for ax, s in self.ax_to_sgaid.items():
            if s == sgaid:
                return ax
        return None

    # ---------- controls ----------

    def _update_page_label(self):
        start = self.page * self.n_per_page + 1
        end = min(start + self.n_per_page - 1, len(self.sgaids))
        self.page_label.value = (
            f"&nbsp;&nbsp;Page {self.page + 1}/{self.n_pages}"
            f"&nbsp;(showing {start}-{end} of {len(self.sgaids)})&nbsp;&nbsp;"
        )
        self.prev_btn.disabled = (self.page == 0)
        self.next_btn.disabled = (self.page == self.n_pages - 1)

    def _on_view_mode_change(self, change):
        self.view_mode = change["new"]
        self._render_page()

    def _mark_evaluated(self, sgaid):
        """Lock in a galaxy as reviewed: border goes red and stays red."""
        self.state[sgaid] = "reviewed"
        self.border_color[sgaid] = "red"
        ax = self._ax_for_sgaid(sgaid)
        if ax is not None:
            self._set_border(ax, "red")

    def _on_click(self, event):
        ax = event.inaxes
        if ax is None or ax not in self.ax_to_sgaid:
            return
        sgaid = self.ax_to_sgaid[ax]

        if self.border_color.get(sgaid) == "red":
            return  # already reviewed; re-editing isn't supported yet

        if sgaid in self.selected_sgaids:
            self.selected_sgaids.discard(sgaid)
            self.border_color[sgaid] = None
        else:
            self.selected_sgaids.add(sgaid)
            self.border_color[sgaid] = "blue"

        self._set_border(ax, self.border_color[sgaid])
        self.status.value = f"{len(self.selected_sgaids)} galaxy(ies) selected."
        self.fig.canvas.draw_idle()

    def _apply_to_selection(self, alt_morph=None, bad_anchor=False):
        if not self.selected_sgaids:
            self.status.value = "<b>Select at least one galaxy first.</b>"
            return

        note_text = self.notes_box.value
        for sgaid in list(self.selected_sgaids):
            if note_text:
                self.notes[sgaid] = note_text
            if bad_anchor:
                ra, dec = self.radec.get(sgaid, (None, None))
                self.bad_anchors[sgaid] = {
                    "SGAID": sgaid, "RA": ra, "DEC": dec,
                    "notes": self.notes.get(sgaid, ""),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.records.pop(sgaid, None)
            else:
                self.selected_morph[sgaid] = alt_morph
            self._mark_evaluated(sgaid)

        n = len(self.selected_sgaids)
        self.selected_sgaids.clear()
        self.notes_box.value = ""
        label = "bad anchor" if bad_anchor else alt_morph
        self.status.value = f"Tagged {n} galaxy(ies) as {label}."
        self.fig.canvas.draw_idle()

    def _on_morph_click(self, name):
        self._apply_to_selection(alt_morph=name)

    def _on_bad_anchor(self, _):
        self._apply_to_selection(bad_anchor=True)

    def _go_to_page(self, new_page):
        new_page = max(0, min(new_page, self.n_pages - 1))
        if new_page == self.page:
            return
        if self.selected_sgaids:
            self.status.value = "<b>Commit or clear your selection before changing pages.</b>"
            return
        self.page = new_page
        self._render_page()

    def _current_savefile(self, morph=None):
        morph = morph if morph is not None else self.morph_selector.value
        return os.path.join(self.save_dir, f"misclassification_{morph}.csv")

    def _current_bad_anchor_file(self, morph=None):
        morph = morph if morph is not None else self.morph_selector.value
        return os.path.join(self.save_dir, f"bad_anchors_{morph}.csv")

    def save(self, morph_override=None):
        for sgaid, color in self.border_color.items():
            if color != "red":
                continue
            if sgaid in self.bad_anchors:
                continue
            alt = self.selected_morph.get(sgaid)
            if alt is None:
                continue
            ra, dec = self.radec.get(sgaid, (None, None))
            self.records[sgaid] = {
                "SGAID": sgaid, "RA": ra, "DEC": dec,
                "alt_morphology": alt,
                "notes": self.notes.get(sgaid, ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        save_path = self._current_savefile(morph_override)
        bad_path = self._current_bad_anchor_file(morph_override)

        self._write_csv(save_path, self.records,
                         fieldnames=["SGAID", "RA", "DEC", "alt_morphology", "notes", "timestamp"])
        self._write_csv(bad_path, self.bad_anchors,
                         fieldnames=["SGAID", "RA", "DEC", "notes", "timestamp"])

        self.status.value = (
            f"Saved {len(self.records)} record(s) to {save_path} and "
            f"{len(self.bad_anchors)} bad anchor(s) to {bad_path}."
        )

    def _on_save(self, _):
        self.save()

    @staticmethod
    def _write_csv(path, records_dict, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    existing[row["SGAID"]] = row

        for k, v in records_dict.items():
            existing[str(k)] = {fn: v.get(fn, "") for fn in fieldnames}

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing.values():
                writer.writerow(row)

def export_implicit_correct(df, morph_options, data_lookup_fn, save_dir):
    """
    For each morphology in morph_options, write correct_{morph}.csv containing
    every galaxy in df's morph subset that does NOT appear in that morph's
    misclassification_{morph}.csv or bad_anchors_{morph}.csv — i.e. galaxies
    never flagged, meaning they're implicitly confirmed as correctly classified.
    """
    for morph in morph_options:
        data = data_lookup_fn(morph, df)
        all_sgaids = {int(s) for s in data["SGAID"]}
        radec = {int(r["SGAID"]): (float(r["RA"]), float(r["DEC"])) for _, r in data.iterrows()}

        flagged = set()
        for fname_pattern in (f"misclassification_{morph}.csv", f"bad_anchors_{morph}.csv"):
            path = os.path.join(save_dir, fname_pattern)
            if os.path.exists(path):
                with open(path, newline="") as fh:
                    flagged.update(int(row["SGAID"]) for row in csv.DictReader(fh))

        correct_sgaids = all_sgaids - flagged
        records = {
            sgaid: {
                "SGAID": sgaid,
                "RA": radec[sgaid][0],
                "DEC": radec[sgaid][1],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for sgaid in correct_sgaids
        }

        out_path = os.path.join(save_dir, f"correct_{morph}.csv")
        os.makedirs(save_dir, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["SGAID", "RA", "DEC", "timestamp"])
            writer.writeheader()
            for row in records.values():
                writer.writerow(row)

        print(f"{morph}: {len(correct_sgaids)} implicitly correct / {len(all_sgaids)} total "
              f"-> {out_path}")