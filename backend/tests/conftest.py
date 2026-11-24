"""Shared pytest fixtures: a fake PhysioNet-like recording (no download needed)."""

import sys
from pathlib import Path 

import mne
import numpy as np
import pytest 

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from handmind import config  # noqa: E402

# Known sequence of cues in the fake recording: (onset_s, class_name)
FAKE_CUES = [
    (2.0, "repos"), (8.0, "main_gauche"), (14.0, "repos"),
    (20.0, "main_droite"), (26.0, "deux_mains"), (32.0, "pieds"),
    (38.0, "main_gauche"),
]


@pytest.fixture #decorateur definissant une fonction comme fixture (une fct qui prépare des données ou des ressources dont les tests ont besoin)
def fake_raw():
    """45 s of random 'EEG' with the 64 PhysioNet channels and known cues."""
    rng = np.random.default_rng(0)
    sfreq = config.SFREQ
    info = mne.create_info(config.CHANNELS_ALL, sfreq, ch_types="eeg")
    data = rng.normal(0, 20e-6, (64, int(45 * sfreq)))
    raw = mne.io.RawArray(data, info, verbose="error")
    onsets, names = zip(*FAKE_CUES)
    raw.set_annotations(mne.Annotations(onsets, [4.1] * len(onsets), names))
    return raw
