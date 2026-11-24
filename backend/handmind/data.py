"""Dataset loading: PhysioNet EEGMMIDB (main) and BCI Competition IV 2a (benchmark).

The key job of this file: turn PhysioNet's ambiguous codes (T1 means "left
hand" in runs 4/8/12 but "both hands" in runs 6/10/14) into our 5 explicit
class names, *before* the runs are concatenated.
"""

import mne
from mne.datasets import eegbci

from . import config


def download_physionet(subject, runs=config.ALL_RUNS, path=config.RAW_DIR):
    """Download (if needed) the EDF files of one subject and return their paths."""
    path.mkdir(parents=True, exist_ok=True)
    return eegbci.load_data(subject, runs, path=str(path), update_path=False)


def _annotation_map(run):
    """Return the T0/T1/T2 -> class-name mapping for a given run number."""
    if run in config.RUNS_HANDS:
        return config.ANNOTATION_MAP_HANDS
    if run in config.RUNS_FEET:
        return config.ANNOTATION_MAP_FEET
    raise ValueError(f"Run {run} is not an imagery run used by HandMind")


# MNE >= 1.12 renamed "standard_1005" to "colin27_1005" (FutureWarning)
_MONTAGE = ("colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages()
            else "standard_1005")


def load_physionet_raw(subject, runs=config.ALL_RUNS, path=config.RAW_DIR):
    """Load one subject as a single continuous Raw with explicit class annotations.

    Returns
    -------
    raw : mne.io.Raw
        64 EEG channels, 160 Hz, standard 10-10 names (C3, Cz, ...),
        annotations renamed to "repos", "main_gauche", "main_droite",
        "deux_mains", "deux_pieds".
    """
    files = download_physionet(subject, runs, path)
    raws = []
    for run, fname in zip(runs, files):
        raw = mne.io.read_raw_edf(fname, preload=True, verbose="error")
        raw.annotations.rename(_annotation_map(run))  # T1 -> main_gauche, ...
        raws.append(raw)

    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)                 # normaliser les noms des électrodes selon les conventions EEGBCI
    raw.set_montage(_MONTAGE)        # 3D positions, needed for topographic so MNE knows where C3, Cz and C4 are located.
    return raw


def get_events(raw):
    """Extract (events, event_id) using our fixed class -> integer mapping.

    Only the classes present in the recording are kept in event_id, so that
    loading a single run (e.g. only hands) still works.
    """
    present = set(raw.annotations.description)
    event_id = {k: v for k, v in config.EVENT_ID.items() if k in present}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
    return events, event_id


def load_bci_iv_2a(subject):
    """Load one subject of BCI Competition IV 2a through MOABB.

    Returns
    -------
    dict : {session_name: mne.io.Raw}, each Raw being the concatenation of
    the session's runs (22 EEG + 3 EOG channels, 250 Hz, 4 classes).
    """
    from moabb.datasets import BNCI2014_001  # imported here: heavy import

    data = BNCI2014_001().get_data(subjects=[subject])[subject]
    return {
        session: mne.concatenate_raws(list(runs.values()))
        for session, runs in data.items()
    }
