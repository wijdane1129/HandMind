"""S4 tests - FBCSP, 3/5 classes, class balancing, LOSO, BCI IV 2a epoching.

Synthetic data only (no download), runs in well under a minute.
Run from backend/:   pytest -v tests/test_s4.py
"""

import mne
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from handmind import config
from handmind.evaluation import (balance_classes, chance_level,
                                 cross_validate_subject, loso, session_transfer)
from handmind.models import FilterBankCSP, MODELS, make_model

SFREQ = config.SFREQ
N_TIMES = int(round((config.TMAX - config.TMIN) * SFREQ)) + 1  # 561
T = np.arange(N_TIMES) / SFREQ


def multiclass_mi(classes, n_per_class=24, n_channels=17, seed=0, boost=1.5):
    """Each class has more power on its own electrode (class k -> channel k).

    boost stays realistic (x1.5 amplitude): multi-class CSP ranks its filters
    with an approximation of mutual information that only holds for moderate
    power changes - like real ERD/ERS, not like a x6 power jump.
    """
    rng = np.random.default_rng(seed)
    codes = [config.EVENT_ID[c] for c in classes]
    y = np.repeat(codes, n_per_class)
    X = rng.normal(0, 10e-6, (len(y), n_channels, N_TIMES))
    for k, code in enumerate(codes):
        X[y == code, k * (n_channels // len(codes))] *= boost
    return X.astype(np.float32), y


def band_only_mi(n_trials=60, n_channels=6, seed=0):
    """Two classes that differ ONLY by frequency, not by broadband power.

    Left: 10 Hz on ch0 and 22 Hz on ch1. Right: the opposite.
    Same total variance on every channel -> broadband CSP / covariance
    cannot separate them, a filter bank can.
    """
    rng = np.random.default_rng(seed)
    left, right = config.EVENT_ID["main_gauche"], config.EVENT_ID["main_droite"]
    y = np.array([left, right] * (n_trials // 2))
    X = rng.normal(0, 5e-6, (n_trials, n_channels, N_TIMES))
    a = 10e-6
    for i, lab in enumerate(y):
        f0, f1 = (10, 22) if lab == left else (22, 10)
        p = rng.uniform(0, 2 * np.pi, 2)
        X[i, 0] += a * np.sin(2 * np.pi * f0 * T + p[0])
        X[i, 1] += a * np.sin(2 * np.pi * f1 * T + p[1])
    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# FBCSP + SVM
# ---------------------------------------------------------------------------
def test_filter_bank_feature_count():
    X, y = multiclass_mi(config.CLASS_SETS[2], n_channels=17)
    fb = FilterBankCSP().fit(X, y)
    n_bands = len(config.FBCSP_BANDS)
    assert fb.transform(X).shape == (len(y), n_bands * config.FBCSP_N_COMPONENTS)


def test_filter_bank_clips_components_for_3_channels():
    X, y = multiclass_mi(config.CLASS_SETS[2], n_channels=3)
    fb = FilterBankCSP(n_components=4).fit(X, y)
    assert fb.transform(X).shape[1] == len(config.FBCSP_BANDS) * 3


def test_fbcsp_finds_frequency_only_difference():
    X, y = band_only_mi()
    fb = cross_validate_subject(make_model("fbcsp_svm", X.shape[1]), X, y)
    csp = cross_validate_subject(make_model("csp_lda", X.shape[1]), X, y)
    assert fb["acc_mean"] >= 0.9
    assert csp["acc_mean"] < fb["acc_mean"]  # the reason FBCSP exists


def test_fbcsp_uses_given_sfreq():
    model = make_model("fbcsp_svm", 22, sfreq=250.0)
    assert model.steps[0][1].sfreq == 250.0


# ---------------------------------------------------------------------------
# 3 and 5 classes: every model works and beats chance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("k", [3, 5])
def test_multiclass_above_chance(name, k):
    X, y = multiclass_mi(config.CLASS_SETS[k])
    r = cross_validate_subject(make_model(name, X.shape[1]), X, y)
    assert r["acc_mean"] > chance_level(len(y), k)
    assert r["confusion"].shape == (k, k)
    assert r["confusion"].sum() == len(y)  # every trial tested exactly once


@pytest.mark.parametrize("name", list(MODELS))
def test_predict_proba_5_classes(name):
    X, y = multiclass_mi(config.CLASSES, n_per_class=12, n_channels=3)
    m = make_model(name, 3).fit(X.astype(np.float64), y)
    proba = m.predict_proba(X.astype(np.float64))
    assert proba.shape == (len(y), 5)
    assert np.allclose(proba.sum(axis=1), 1)


# ---------------------------------------------------------------------------
# Class balancing (repos has ~2x more trials)
# ---------------------------------------------------------------------------
def test_balance_classes():
    y = np.array([0] * 90 + [1] * 22 + [2] * 23)
    X = np.arange(len(y))[:, None, None].astype(float)
    Xb, yb = balance_classes(X, y)
    _, counts = np.unique(yb, return_counts=True)
    assert list(counts) == [22, 22, 22]
    assert np.all(np.diff(Xb[:, 0, 0]) > 0)              # order preserved
    assert np.array_equal(yb, y[Xb[:, 0, 0].astype(int)])  # labels follow X
    Xb2, _ = balance_classes(X, y)
    assert np.array_equal(Xb, Xb2)                         # reproducible


# ---------------------------------------------------------------------------
# LOSO: the left-out subject never leaks into training
# ---------------------------------------------------------------------------
class SpyClassifier(BaseEstimator, ClassifierMixin):
    """Remembers which subject ids it was trained on (stored in X[:, 0, 0])."""

    seen = []

    def fit(self, X, y):
        SpyClassifier.seen.append(set(np.unique(X[:, 0, 0]).astype(int)))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _subject_datasets(n=4, tag=False):
    out = []
    for s in range(1, n + 1):
        X, y = multiclass_mi(config.CLASS_SETS[2], n_per_class=10,
                             n_channels=3, seed=s, boost=2.0)
        X = X.astype(np.float64)
        if tag:
            X[:, 0, 0] = s  # tag every trial with its subject id (spy only)
        out.append({"X": X, "y": y, "subject": s})
    return out


def test_loso_never_trains_on_test_subject():
    SpyClassifier.seen = []
    rows = loso(_subject_datasets(4, tag=True), lambda n_ch: SpyClassifier())
    assert [r["subject"] for r in rows] == [1, 2, 3, 4]
    for r, seen in zip(rows, SpyClassifier.seen):
        assert r["subject"] not in seen
        assert seen == {1, 2, 3, 4} - {r["subject"]}
        assert r["n_train_subjects"] == 3


def test_loso_real_model_learns_shared_pattern():
    # Same spatial pattern for every subject -> transfer must work
    rows = loso(_subject_datasets(4), lambda n: make_model("riemann", n))
    assert np.mean([r["acc"] for r in rows]) > 0.8


# ---------------------------------------------------------------------------
# BCI IV 2a: epoching from a MOABB-like raw + session transfer
# ---------------------------------------------------------------------------
def _fake_bci2a_raw(seed=0, n_trials=40):
    """MOABB-like raw: the event marks the trial START, the class-specific
    activity only appears after the cue, 2 s later (like the real dataset)."""
    sfreq = 250.0
    rng = np.random.default_rng(seed)
    n_eeg = 22
    n_times = int((n_trials * 8 + 5) * sfreq)
    data = rng.normal(0, 10e-6, (n_eeg + 1, n_times))
    data[-1] = 0
    codes = np.tile([1, 2, 3, 4], n_trials // 4)
    onsets = ((np.arange(n_trials) * 8 + 2) * sfreq).astype(int)
    for code, t0 in zip(codes, onsets):
        data[-1, t0:t0 + 10] = code
        # ERD builds up after the cue: activity from cue + 0.5 s to cue + 4 s
        start = t0 + int((config.BCI2A_CUE_OFFSET + 0.5) * sfreq)
        data[code - 1, start:start + int(3.5 * sfreq)] *= 3
    info = mne.create_info([f"EEG{i}" for i in range(n_eeg)] + ["stim"], sfreq,
                           ch_types=["eeg"] * n_eeg + ["stim"])
    return mne.io.RawArray(data, info, verbose="error"), codes


def test_bci2a_epochs_and_transfer():
    from scripts.benchmark_bci2a import epochs_from_raw

    raw_tr, codes = _fake_bci2a_raw(0)
    raw_te, _ = _fake_bci2a_raw(1)
    X_tr, y_tr, sfreq = epochs_from_raw(raw_tr)
    X_te, y_te, _ = epochs_from_raw(raw_te)
    n_times = int(round((config.BCI2A_TMAX - config.BCI2A_TMIN) * 250)) + 1
    assert sfreq == 250.0
    assert X_tr.shape == (len(codes), 22, n_times)  # stim channel dropped
    assert list(y_tr) == list(codes)
    s = session_transfer(make_model("riemann", 22, sfreq), X_tr, y_tr, X_te, y_te)
    assert s["acc"] > 0.9


def test_bci2a_window_before_cue_is_chance():
    # Regression test: the first version cut 0.5-2.5 s after the EVENT
    # (= before the cue) and every model was at chance on the real data.
    from scripts.benchmark_bci2a import epochs_from_raw

    raw_tr, _ = _fake_bci2a_raw(0)
    raw_te, _ = _fake_bci2a_raw(1)
    X_tr, y_tr, sfreq = epochs_from_raw(raw_tr, offset=0.0)
    X_te, y_te, _ = epochs_from_raw(raw_te, offset=0.0)
    s = session_transfer(make_model("riemann", 22, sfreq), X_tr, y_tr, X_te, y_te)
    assert s["acc"] < chance_level(len(y_te), 4)
