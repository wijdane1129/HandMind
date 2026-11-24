"""S1 - Exploration of PhysioNet: raw signals, spectrum, T0/T1/T2 annotations.

Usage (from backend/):
    python -m scripts.explore                      # subjects 1, 2, 3
    python -m scripts.explore --subjects 1 7 42
    python -m scripts.explore --check-all          # sanity check of all 109

Figures are saved to docs/figures/ (reuse them in the report).
Tip: you can also paste these functions into a Jupyter notebook
(docs/exploration.ipynb) to look at the plots interactively.
"""

import argparse #permet de donner des options au script depuis le terminal.
from collections import Counter#counter permet de calculer le nbr d'occurence

import matplotlib# matplotlib est une bibliothèque de visualisation de données en Python.
matplotlib.use("Agg")  # Agg pour generer des figures sans interface graphique 
import matplotlib.pyplot as plt  # noqa: E402 #pyplot est une module de traçage pour créer des graphiques en 2D et 3D.
import mne  # noqa: E402

from handmind import config  # noqa: E402
from handmind.data import get_events, load_physionet_raw  # noqa: E402


def describe(raw, subject):
    """Print the basic facts to compare with the dataset documentation (T1)."""
    events, event_id = get_events(raw)
    inv = {v: k for k, v in event_id.items()}
    counts = Counter(inv[c] for c in events[:, 2])
    print(f"\n=== Subject {subject} ===")
    print(f"  channels: {len(raw.ch_names)}  |  sfreq: {raw.info['sfreq']} Hz"
          f"  |  duration: {raw.times[-1]:.0f} s")
    print(f"  trials per class: {dict(counts)}")
    return counts


def plot_subject(raw, subject):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"S{subject:03d}"

    # 1. 10 s of raw signal on the 3 key electrodes, with the cue annotations
    fig = raw.copy().pick(config.CHANNEL_CONFIGS["C"]).plot(
        duration=10, scalings=dict(eeg=100e-6), show=False)
    fig.savefig(config.FIGURES_DIR / f"{tag}_raw.png", dpi=120)
    plt.close(fig)

    # 2. Power spectrum: look for the alpha/mu peak (~10 Hz) and the 60 Hz line
    fig = raw.compute_psd(fmax=80, picks=config.CHANNEL_CONFIGS["B"]).plot(
        show=False)
    fig.savefig(config.FIGURES_DIR / f"{tag}_psd.png", dpi=120)
    plt.close(fig)

    # 3. Timeline of events: shows the T0/T1/T2 alternation
    events, event_id = get_events(raw)
    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], event_id=event_id, show=False)
    fig.savefig(config.FIGURES_DIR / f"{tag}_events.png", dpi=120)
    plt.close(fig)
    print(f"  figures -> {config.FIGURES_DIR}/{tag}_*.png")


def check_all():
    """Flag subjects whose sfreq or trial count differs from the majority."""
    for s in range(1, config.N_SUBJECTS + 1):
        try:
            raw = load_physionet_raw(s)
        except Exception as e:
            print(f"S{s:03d}: cannot load ({e})")
            continue
        n = len(get_events(raw)[0])
        flag = "" if raw.info["sfreq"] == config.SFREQ else "  <-- sfreq!"
        print(f"S{s:03d}: sfreq={raw.info['sfreq']:.0f}  trials={n}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()

    if args.check_all:
        check_all()
        return
    for s in args.subjects:
        raw = load_physionet_raw(s)
        describe(raw, s)
        plot_subject(raw, s)


if __name__ == "__main__":
    main()
