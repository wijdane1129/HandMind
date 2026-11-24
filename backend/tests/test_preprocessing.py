"""Tests T2, T3, T4 of the cahier de tests (+ save/load and split checks).

Run from backend/:   pytest -v
These tests use synthetic data only: no download, they run in seconds.
"""

import numpy as np
import pytest

from handmind import config
from handmind.preprocessing import (filter_signal, load_epochs, make_epochs,
                                    save_epochs, select_channels)
from handmind.split import draw_demo_subjects, load_split, save_split

from .conftest import FAKE_CUES

SFREQ = config.SFREQ


# ---------------------------------------------------------------------------
# T2 - Electrode configurations: 64, 17 and 3 channels, right names and order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, n", [("A", 64), ("B", 17), ("C", 3)]) #parametrize permet de d'exécuter le même test plusieurs fois avec des valeurs différentes.
def test_T2_channel_configs(fake_raw, name, n):
    out = select_channels(fake_raw, name)
    assert len(out.ch_names) == n #assert il vérifie qu'une condition est vraie.
    assert out.ch_names == config.CHANNEL_CONFIGS[name]  # names AND order


def test_T2_order_is_config_order_not_file_order(fake_raw):
    # In the file, C3 comes before Cz before C4; reversing must be respected
    config.CHANNEL_CONFIGS["_test"] = ["C4", "Cz", "C3"]
    try:
        assert select_channels(fake_raw, "_test").ch_names == ["C4", "Cz", "C3"]
    finally:
        del config.CHANNEL_CONFIGS["_test"]


def test_T2_unknown_config():
    with pytest.raises(ValueError):
        select_channels(None, "Z")


# ---------------------------------------------------------------------------
# T3 - Filtering: >= 20 dB attenuation outside 8-30 Hz and at 60 Hz
# ---------------------------------------------------------------------------
def _gain_db(freq, duration=20.0):
    """Gain (dB) of the filter for a pure sine at `freq` Hz."""
    t = np.arange(int(duration * SFREQ)) / SFREQ
    x = np.sin(2 * np.pi * freq * t)[None, :]
    y = filter_signal(x, SFREQ)
    mid = slice(len(t) // 4, 3 * len(t) // 4)  # ignore filter edges
    return 20 * np.log10(np.std(y[0, mid]) / np.std(x[0, mid]))


@pytest.mark.parametrize("freq", [2.0, 4.0, 40.0, 50.0, 60.0, 70.0])
def test_T3_stopband_attenuated(freq):
    assert _gain_db(freq) <= -20, f"{freq} Hz not attenuated enough"


@pytest.mark.parametrize("freq", [10.0, 12.0, 20.0, 25.0])
def test_T3_passband_kept(freq):
    assert _gain_db(freq) >= -1, f"{freq} Hz should pass almost untouched"


def test_T3_shape_preserved():
    x = np.random.default_rng(0).normal(size=(3, 1600))
    assert filter_signal(x, SFREQ).shape == x.shape


# ---------------------------------------------------------------------------
# T4 - Epoching: 100 % correct labels, right window
# ---------------------------------------------------------------------------
def test_T4_labels_match_annotations(fake_raw):
    epochs = make_epochs(fake_raw)
    expected = [config.EVENT_ID[name] for _, name in FAKE_CUES]
    assert list(epochs.events[:, 2]) == expected


def test_T4_epoch_window(fake_raw):
    epochs = make_epochs(fake_raw)
    assert epochs.tmin == pytest.approx(config.TMIN)
    assert epochs.tmax == pytest.approx(config.TMAX)
    n_times = int(round((config.TMAX - config.TMIN) * SFREQ)) + 1
    assert epochs.get_data().shape == (len(FAKE_CUES), 64, n_times)


def test_T4_no_leak_between_train_and_demo(tmp_path):
    split = draw_demo_subjects()
    assert len(split["demo"]) == config.N_DEMO_SUBJECTS
    assert not set(split["demo"]) & set(split["train"])
    assert not set(split["demo"]) & set(config.EXCLUDED_SUBJECTS)
    path = tmp_path / "splits.json"
    save_split(split, path)
    assert load_split(path) == split
    with pytest.raises(FileExistsError):  # demo subjects can't be redrawn
        save_split(split, path)


def test_split_is_reproducible():
    assert draw_demo_subjects() == draw_demo_subjects()


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(fake_raw, tmp_path):
    epochs = make_epochs(fake_raw)
    save_epochs(epochs, subject=1, folder=tmp_path)

    d = load_epochs(1, "C", classes=["main_gauche", "main_droite"],
                    folder=tmp_path)
    assert d["ch_names"] == ["C3", "Cz", "C4"]
    assert d["X"].shape[:2] == (3, 3)  # 2 x main_gauche + 1 x main_droite
    assert set(d["y"]) == {1, 2}
