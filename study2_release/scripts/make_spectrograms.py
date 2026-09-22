#!/usr/bin/env python3
"""
make_spectrograms.py -- regenerates the (omega/omega0, t) spectrogram for
every finished/checkpointed grid point at w_max=30 (the range requested for
this study; run_coupled.py's own automatic per-run figures use the
project-wide default w_max=8). Safe to re-run: skips a point whose w30
figure already exists and is newer than its data file.

    python make_spectrograms.py [--force]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from run_coupled import setup_style, log, fig_spectrogram
from run_study2 import J_I_VALUES, OMEGA0_VALUES, tag_for

DATA_DIR = HERE / "data"
IMG_DIR = HERE / "img"
MODELS = ["AFM", "FM"]
W_MAX = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    setup_style()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = n_missing = 0
    for model in MODELS:
        for ji in J_I_VALUES:
            for w0 in OMEGA0_VALUES:
                tag = tag_for(model, ji, w0)
                final = DATA_DIR / f"{tag}_coupled.npz"
                ckpt = DATA_DIR / f"{tag}_ckpt.npz"
                src = final if final.exists() else (ckpt if ckpt.exists() else None)
                if src is None:
                    n_missing += 1
                    continue
                out_png = IMG_DIR / f"{tag}_spectrogram_w30.png"
                if not args.force and out_png.exists() and out_png.stat().st_mtime > src.stat().st_mtime:
                    n_skipped += 1
                    continue
                with np.load(src) as z:
                    if "t" not in z.files or len(z["t"]) == 0:
                        n_missing += 1
                        continue
                    out = {k: z[k] for k in z.files}
                meta = {"omega0": float(w0)}
                fig_spectrogram(out, meta, IMG_DIR / f"{tag}_spectrogram_w30", w_max=W_MAX)
                n_written += 1
    log(f"wrote {n_written}, skipped {n_skipped} (up to date), "
        f"{n_missing} points have no data yet")


if __name__ == "__main__":
    main()
