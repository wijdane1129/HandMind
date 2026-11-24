"""Test T1 - dataset loading matches the documentation.

These tests need the real data, so they are skipped automatically until
`python -m scripts.download_data --subjects 1` has been run.
Run them explicitly with:   pytest -v tests/test_data.py
"""

import pytest

from handmind import config
from handmind.data import get_events

SUBJECT = 1
_needs_physionet = pytest.mark.skipif(
    not (config.RAW_DIR / "MNE-eegbci-data").exists(),
    reason="PhysioNet not downloaded (python -m scripts.download_data)",
)#skipif sauter si une condition est vraie 


@_needs_physionet
def test_T1_physionet_subject():
    from handmind.data import load_physionet_raw

    raw = load_physionet_raw(SUBJECT)
    assert raw.info["sfreq"] == config.SFREQ
    assert raw.ch_names == config.CHANNELS_ALL  # 64 channels, standard names

    events, event_id = get_events(raw)
    assert set(event_id) == set(config.CLASSES)  # all 5 classes present
    counts = {name: int((events[:, 2] == code).sum())
              for name, code in event_id.items()}
    # 6 imagery runs x 15 cues = 45 hand cues + 45 feet/both-hands cues
    assert counts["main_gauche"] + counts["main_droite"] == 45
    assert counts["deux_mains"] + counts["pieds"] == 45
    assert counts["repos"] >= 85  # ~15 rest periods per run


@pytest.mark.slow #slow marque que le test est long 
def test_T1_bci_iv_2a_subject():
    pytest.importorskip("moabb")
    from handmind.data import load_bci_iv_2a

    sessions = load_bci_iv_2a(SUBJECT)
    assert len(sessions) == 2  # train + eval sessions
    for raw in sessions.values():
        assert raw.info["sfreq"] == 250
        assert len(raw.copy().pick("eeg").ch_names) == 22
