"""S4 - Test T8: benchmark on BCI Competition IV 2a (4 classes).

Why: PhysioNet scores are hard to compare with the literature. BCI IV 2a is
THE reference dataset: if our pipeline gets scores close to published ones
here, the pipeline itself is correct, and low PhysioNet scores come from the
data, not from a bug.

Protocol = the official competition one:
    train on session 1 ("T"), test on session 2 ("E"), per subject,
    4 classes (left hand, right hand, feet, tongue), 22 EEG channels, 250 Hz,
    band-pass 8-30 Hz + notch 50 Hz, epochs 0.5-2.5 s after the cue.

    Careful: MOABB's events mark the trial start; the cue is 2 s later
    (cue_offset()). A window taken after the event instead of after the cue
    lands on the fixation cross and gives chance-level scores.

Reference: the competition winner (FBCSP, Ang et al. 2012) reached a mean
kappa of about 0.57 over the 9 subjects. Riemannian methods are usually in
the same range.

Usage (from backend/):
    python -m scripts.benchmark_bci2a                   # 9 subjects, 3 models
    python -m scripts.benchmark_bci2a --subjects 1 3 --models riemann

Outputs: docs/results/s4_bci2a.csv / .md
Data: downloaded by MOABB to ~/mne_data the first time (~1.5 GB).
"""

import argparse

import mne
import numpy as np
import pandas as pd

from handmind import config
from handmind.evaluation import session_transfer
from handmind.models import MODELS, make_model
from handmind.preprocessing import filter_signal


def bci2a_events(raw):
    """Cue events with codes 1-4, whatever way MOABB stored them."""
    codes = set(config.BCI2A_EVENT_ID.values())
    if "stim" in raw.get_channel_types():
        events = mne.find_events(raw, shortest_event=0, verbose="error")
    else:  # annotation-based (some MOABB versions)
        events, _ = mne.events_from_annotations(
            raw, event_id=config.BCI2A_EVENT_ID, verbose="error")
    return events[np.isin(events[:, 2], list(codes))]


def cue_offset():
    """Seconds between a MOABB event (trial start) and the cue.

    Read from MOABB itself (dataset.interval[0]) when available, so the
    window can never silently fall before the cue again.
    """
    try:
        from moabb.datasets import BNCI2014_001
        return float(BNCI2014_001().interval[0])
    except Exception:
        return config.BCI2A_CUE_OFFSET


def epochs_from_raw(raw, offset=config.BCI2A_CUE_OFFSET):
    """Filter the continuous signal, then cut epochs 0.5-2.5 s after the CUE.

    `offset` = time from the event to the cue (2 s in MOABB's BCI IV 2a).
    Returns (X, y, sfreq).
    """
    events = bci2a_events(raw)
    raw = raw.copy().pick("eeg")
    raw.apply_function(filter_signal, sfreq=raw.info["sfreq"],
                       notch=config.BCI2A_NOTCH, channel_wise=False)
    present = {k: v for k, v in config.BCI2A_EVENT_ID.items()
               if v in set(events[:, 2])}
    epochs = mne.Epochs(raw, events, event_id=present,
                        tmin=offset + config.BCI2A_TMIN,
                        tmax=offset + config.BCI2A_TMAX,
                        baseline=None, preload=True, verbose="error")
    return epochs.get_data(copy=True), epochs.events[:, 2].copy(), raw.info["sfreq"]


def benchmark_subject(subject, model_names):
    from handmind.data import load_bci_iv_2a  # heavy import (MOABB)

    sessions = load_bci_iv_2a(subject)
    names = sorted(sessions)  # "0train" < "1test" (or "session_E" > "session_T")
    if len(names) != 2:
        raise ValueError(f"expected 2 sessions, got {names}")
    train_name, test_name = (names if "E" not in names[0] else names[::-1])
    offset = cue_offset()
    X_tr, y_tr, sfreq = epochs_from_raw(sessions[train_name], offset)
    X_te, y_te, _ = epochs_from_raw(sessions[test_name], offset)

    rows = []
    for m in model_names:
        s = session_transfer(make_model(m, X_tr.shape[1], sfreq),
                             X_tr, y_tr, X_te, y_te)
        rows.append({"subject": subject, "model": m,
                     "n_train": len(y_tr), "n_test": len(y_te),
                     "acc": s["acc"], "kappa": s["kappa"]})
    return rows


def to_markdown(df):
    lines = ["# Benchmark BCI Competition IV 2a (test T8)", "",
             "4 classes, train session T -> test session E, 22 EEG channels, "
             f"{config.BCI2A_TMIN}-{config.BCI2A_TMAX} s, 8-30 Hz.", "",
             "| Model | Accuracy | Kappa |", "|---|---|---|"]
    for m, g in df.groupby("model"):
        lines.append(f"| {m} | {g['acc'].mean():.3f} ± {g['acc'].std():.3f} | "
                     f"{g['kappa'].mean():.3f} ± {g['kappa'].std():.3f} |")
    lines += ["", "Reference: competition winner (FBCSP) mean kappa ≈ 0.57.", "",
              "## Kappa per subject", ""]
    piv = df.pivot_table(index="subject", columns="model", values="kappa")
    lines.append("| Subject | " + " | ".join(piv.columns) + " |")
    lines.append("|---" * (len(piv.columns) + 1) + "|")
    for s, row in piv.iterrows():
        lines.append(f"| A{s:02d} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=config.BCI2A_SUBJECTS)
    parser.add_argument("--models", nargs="+", choices=list(MODELS),
                        default=list(MODELS))
    args = parser.parse_args()

    rows = []
    for s in args.subjects:
        try:
            for r in benchmark_subject(s, args.models):
                rows.append(r)
                print(f"A{s:02d} {r['model']:9s} acc={r['acc']:.3f} "
                      f"kappa={r['kappa']:.3f}")
        except Exception as e:
            print(f"A{s:02d}: FAILED ({e})")
    if not rows:
        raise SystemExit("No result produced.")

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = config.RESULTS_DIR / "s4_bci2a.csv"
    md = config.RESULTS_DIR / "s4_bci2a.md"
    df.to_csv(csv, index=False, float_format="%.4f")
    md.write_text(to_markdown(df), encoding="utf-8")
    print("\n" + df.groupby("model")[["acc", "kappa"]].mean()
          .to_string(float_format="%.3f"))
    print(f"\n-> {csv}\n-> {md}")


if __name__ == "__main__":
    main()
