"""Preprocessing: filtering, channel selection, epoching, save/load.

Order of operations (always the same):
    raw  ->  filter_raw  ->  make_epochs  ->  (save)  ->  select_channels
Filtering is done on the *continuous* signal, before cutting epochs, so the
filter does not create edge artefacts at the start/end of each epoch.
"""

import mne
import numpy as np

from . import config
from .data import get_events


# ---------------------------------------------------------------------------
# Filtering  (test T3)
# ---------------------------------------------------------------------------
def filter_signal(data, sfreq, l_freq=config.L_FREQ, h_freq=config.H_FREQ,
                  notch=config.NOTCH_FREQ):
    """Band-pass + notch filter a NumPy array of shape (n_channels, n_times).

    This array version is the single source of truth: it is what the tests
    check and what filter_raw() calls. Zero-phase FIR (offline use only;
    the real-time version in S5 will need a causal filter).
    """
    data = np.asarray(data, dtype=np.float64)
    if notch is not None and notch < sfreq / 2: #ne peut pas détecter correctement une fréquence supérieure à la fréquence de Nyquist.
        data = mne.filter.notch_filter(data, sfreq, freqs=notch, verbose="error")
    return mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose="error")#(mne.filter.filter_data) is zero-phase: it runs forwards AND backwards, so it needs samples from the future


def filter_raw(raw):
    """Apply filter_signal() to a Raw object (returns a filtered copy)."""
    raw = raw.copy()
    raw.apply_function(filter_signal, sfreq=raw.info["sfreq"],
                       channel_wise=False, picks="eeg")
    return raw


# ---------------------------------------------------------------------------
# Channel selection  (test T2)
# ---------------------------------------------------------------------------
def get_channels(config_name):
    """Return the channel list of configuration "A", "B" or "C"."""
    try:
        return list(config.CHANNEL_CONFIGS[config_name])
    except KeyError:
        raise ValueError(
            f"Unknown config {config_name!r}, expected one of "
            f"{list(config.CHANNEL_CONFIGS)}"
        ) from None


def select_channels(inst, config_name):
    """Keep only the channels of a configuration, in the configuration order.

    Works on mne Raw / Epochs (returns a copy) or on a dict produced by
    load_epochs() (returns a new dict with X restricted).
    """
    channels = get_channels(config_name)

    if isinstance(inst, dict):  # {"X", "y", "ch_names", ...} from load_epochs
        index = [inst["ch_names"].index(ch) for ch in channels]
        out = dict(inst)
        out["X"] = inst["X"][:, index, :]
        out["ch_names"] = channels
        return out

    inst = inst.copy().pick(channels)
    inst.reorder_channels(channels)  # guarantee the order, not just the set
    return inst


# ---------------------------------------------------------------------------
# Epoching  (test T4)
# ---------------------------------------------------------------------------
def make_epochs(raw, tmin=config.TMIN, tmax=config.TMAX):
    """Cut epochs from tmin to tmax seconds after each cue, with class labels.

    No baseline correction (the band-pass already removes the DC offset).
    Returns an mne.Epochs; labels are in epochs.events[:, 2].
    """
    events, event_id = get_events(raw)
    return mne.Epochs(
        raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
        baseline=None, picks="eeg", preload=True, verbose="error",
    )#preload=True: charge les données en mémoire pour un accès plus rapide, mais utilise plus de mémoire.


def epochs_to_arrays(epochs):
    """Return X: signaux(n_epochs, n_channels, n_times) in volts and y les etiquettes des classes(n_epochs,)."""
    return epochs.get_data(copy=True), epochs.events[:, 2].copy()


# ---------------------------------------------------------------------------
# Save / load  (so we never recompute the preprocessing)
# ---------------------------------------------------------------------------
def epochs_path(subject, folder=config.PROCESSED_DIR):
    return folder / f"S{subject:03d}_epo.npz"


def save_epochs(epochs, subject, folder=config.PROCESSED_DIR):
    """Save epochs of one subject as a compressed .npz (all 64 channels)."""
    folder.mkdir(parents=True, exist_ok=True)
    X, y = epochs_to_arrays(epochs)
    path = epochs_path(subject, folder)
    np.savez_compressed(
        path, X=X.astype(np.float32), y=y, subject=subject,
        ch_names=np.array(epochs.ch_names), sfreq=epochs.info["sfreq"],
        tmin=epochs.tmin, classes=np.array(config.CLASSES),
    )
    return path


def load_epochs(subject, config_name="A", classes=None,
                folder=config.PROCESSED_DIR):
    """Load the saved epochs of one subject.

    Parameters
    ----------
    config_name : "A", "B" or "C" - electrode configuration to keep.
    classes : list of class names to keep (e.g. ["main_gauche", "main_droite"]
        for the 2-class baseline). None keeps all 5 classes.

    Returns
    -------
    dict with keys X, y, ch_names, sfreq, subject.
    """
    with np.load(epochs_path(subject, folder)) as f:
        d = {
            "X": f["X"], "y": f["y"], "ch_names": list(f["ch_names"]),
            "sfreq": float(f["sfreq"]), "subject": int(f["subject"]),
        }
    if classes is not None:
        keep = np.isin(d["y"], [config.EVENT_ID[c] for c in classes])
        d["X"], d["y"] = d["X"][keep], d["y"][keep]
    return select_channels(d, config_name)
