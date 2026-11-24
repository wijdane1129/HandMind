"""S3/S5 - Train a model and save it (joblib).

Two modes:

1) S3 - offline model, one subject, full 3.5 s epochs (from data/processed):
    python -m scripts.train --subject 1 --model riemann --config B
    -> models/S001_riemann_B.joblib

2) S5 - ONLINE model, the one used by the real-time classifier:
    python -m scripts.train --online                       # S4 choice, 5 classes, 20 train subjects
    python -m scripts.train --online --classes 2 --subjects 1 2 3
    -> models/online_<model>_<config>_<k>c.joblib

Why a separate online model? Online, the classifier sees 1 s windows that
went through a CAUSAL filter. A model trained on 3.5 s zero-phase epochs
would see different signals. So the online model is trained on exactly
what it will receive: 1 s windows, cut every 0.5 s inside each trial
(0.5-1.5 s, 1.0-2.0 s, ... after the cue), from the raw recording filtered
with the same StreamFilter as online.

It is trained on several TRAIN subjects pooled (never the demo subjects),
because the demo subjects are new people for the model (cf. LOSO, S4).

The saved file is a dict {"model", "classes", "ch_names", "sfreq", ...} so
the online side knows exactly which electrodes, order and labels to use.
"""

import argparse

import joblib
import numpy as np

from handmind import config
from handmind.models import MODELS, make_model
from handmind.preprocessing import get_channels, load_epochs


# ---------------------------------------------------------------------------
# S3 - offline, one subject
# ---------------------------------------------------------------------------
def train(subject, model_name, config_name, classes=config.BASELINE_CLASSES):
    d = load_epochs(subject, config_name, classes=classes)
    model = make_model(model_name, n_channels=d["X"].shape[1])
    model.fit(np.asarray(d["X"], dtype=np.float64), d["y"])
    return {
        "model": model,
        "model_name": model_name,
        "classes": list(classes),
        "label_codes": [config.EVENT_ID[c] for c in classes],
        "ch_names": d["ch_names"],
        "sfreq": d["sfreq"],
        "tmin": config.TMIN,
        "tmax": config.TMAX,
        "subject": subject,
        "config": config_name,
        "online": False,
    }


# ---------------------------------------------------------------------------
# S5 - online windows
# ---------------------------------------------------------------------------
def causal_windows(data, sfreq, events, classes,
                   starts=config.ONLINE_TRAIN_STARTS,
                   window_s=config.ONLINE_WINDOW_S, balance=True,
                   seed=config.RANDOM_SEED):
    """Cut 1 s training windows from a continuous, causally filtered signal.

    Parameters
    ----------
    data : (n_channels, n_times), ALREADY filtered with filter_causal.
    events : (n_events, 3) mne events, sample index relative to data[:, 0].
    classes : class names to keep.
    starts : window starts after each cue (s).
    balance : keep as many "repos" trials as the most frequent movement.

    Returns X (n_windows, n_channels, n_win) float32, y (n_windows,).
    """
    codes = [config.EVENT_ID[c] for c in classes]
    ev = events[np.isin(events[:, 2], codes)]
    if balance and config.EVENT_ID["repos"] in codes and len(codes) > 1:
        rest = config.EVENT_ID["repos"]
        others = [np.sum(ev[:, 2] == c) for c in codes if c != rest]
        keep_n = max(others) if others else 0
        idx_rest = np.flatnonzero(ev[:, 2] == rest)
        if len(idx_rest) > keep_n:
            rng = np.random.default_rng(seed)
            drop = rng.choice(idx_rest, len(idx_rest) - keep_n, replace=False)
            ev = np.delete(ev, drop, axis=0)

    win = int(round(window_s * sfreq))
    X, y = [], []
    for onset, _, code in ev:
        for s in starts:
            a = onset + int(round(s * sfreq))
            if a >= 0 and a + win <= data.shape[1]:
                X.append(data[:, a:a + win])
                y.append(code)
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def online_dataset(subjects, config_name, classes):
    """Pool the online windows of several subjects (needs data/raw)."""
    from handmind.data import get_events, load_physionet_raw
    from handmind.online.filters import filter_causal

    channels = get_channels(config_name)
    Xs, ys = [], []
    for s in subjects:
        raw = load_physionet_raw(s)
        if raw.info["sfreq"] != config.SFREQ:
            print(f"  S{s:03d}: skipped (sfreq={raw.info['sfreq']})")
            continue
        data = filter_causal(raw.get_data(picks=channels), raw.info["sfreq"])
        events, _ = get_events(raw)
        events[:, 0] -= raw.first_samp       # -> index in `data`
        X, y = causal_windows(data, raw.info["sfreq"], events, classes)
        print(f"  S{s:03d}: {len(y)} windows")
        Xs.append(X)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys), channels


def train_online(subjects, model_name, config_name, n_classes):
    classes = config.CLASS_SETS[n_classes]
    X, y, _ = online_dataset(subjects, config_name, classes)
    print(f"Training {model_name} on {len(y)} windows of "
          f"{config.ONLINE_WINDOW_S} s, {X.shape[1]} channels ...")
    model = make_model(model_name, n_channels=X.shape[1])
    model.fit(X, y)
    return {
        "model": model,
        "model_name": model_name,
        "classes": list(classes),
        "label_codes": [config.EVENT_ID[c] for c in classes],
        "ch_names": get_channels(config_name),
        "sfreq": config.SFREQ,
        "window_s": config.ONLINE_WINDOW_S,
        "train_subjects": list(subjects),
        "config": config_name,
        "online": True,
    }


def online_model_path(model_name=None, config_name=None, n_classes=None):
    model_name = model_name or config.online_model_name()
    config_name = config_name or config.online_config_name()
    n_classes = n_classes or config.ONLINE_CLASSES
    return config.MODELS_DIR / f"online_{model_name}_{config_name}_{n_classes}c.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true",
                        help="S5: 1 s causal windows, pooled train subjects")
    parser.add_argument("--subject", type=int, help="S3 mode: one subject")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="online mode: default = first 20 train subjects")
    parser.add_argument("--model", choices=list(MODELS), default=None)
    parser.add_argument("--config", choices=list(config.CHANNEL_CONFIGS),
                        default=None)
    parser.add_argument("--classes", type=int, choices=sorted(config.CLASS_SETS),
                        default=None)
    args = parser.parse_args()

    model_name = args.model or config.online_model_name()
    config_name = args.config or config.online_config_name()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.online:
        from scripts.evaluate import default_subjects
        subjects = args.subjects or default_subjects()
        n_classes = args.classes or config.ONLINE_CLASSES
        bundle = train_online(subjects, model_name, config_name, n_classes)
        path = online_model_path(model_name, config_name, n_classes)
    else:
        if args.subject is None:
            parser.error("--subject is required without --online")
        classes = config.CLASS_SETS[args.classes or 2]
        bundle = train(args.subject, model_name, config_name, classes)
        path = config.MODELS_DIR / f"S{args.subject:03d}_{model_name}_{config_name}.joblib"

    joblib.dump(bundle, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
