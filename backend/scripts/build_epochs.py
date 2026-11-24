"""S2 - Preprocess every subject once and save the epochs to data/processed/.

Pipeline per subject:  load raw -> filter 8-30 Hz + notch 60 Hz -> epochs 0.5-4 s
-> save S###_epo.npz (all 64 channels; configs A/B/C are selected at load time).

Usage (from backend/):
    python -m scripts.build_epochs                 # all valid subjects
    python -m scripts.build_epochs --subjects 1 2 3
    python -m scripts.build_epochs --force         # recompute existing files

Then, anywhere in the project:
    from handmind.preprocessing import load_epochs
    d = load_epochs(1, config_name="B", classes=["main_gauche", "main_droite"])
    d["X"].shape   # (n_epochs, 17, 561)
"""

import argparse
from collections import Counter

from handmind import config
from handmind.data import load_physionet_raw
from handmind.preprocessing import (epochs_path, filter_raw, make_epochs,
                                    save_epochs)


def process_subject(subject, force=False):
    if epochs_path(subject).exists() and not force:
        return "skipped (exists)"
    raw = load_physionet_raw(subject)
    if raw.info["sfreq"] != config.SFREQ:
        return f"skipped (sfreq={raw.info['sfreq']})"
    epochs = make_epochs(filter_raw(raw))
    save_epochs(epochs, subject)
    counts = Counter(config.CLASSES[c] for c in epochs.events[:, 2])
    return f"{len(epochs)} epochs {dict(counts)}"


def main():
    parser = argparse.ArgumentParser()
    valid = [s for s in range(1, config.N_SUBJECTS + 1)
             if s not in config.EXCLUDED_SUBJECTS]
    parser.add_argument("--subjects", type=int, nargs="+", default=valid)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for s in args.subjects:
        try:
            print(f"S{s:03d}: {process_subject(s, args.force)}")
        except Exception as e:
            print(f"S{s:03d}: FAILED ({e})")


if __name__ == "__main__":
    main()
