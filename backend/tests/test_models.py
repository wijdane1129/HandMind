"""S3 tests - baselines and evaluation, on synthetic data only (runs in seconds).

The synthetic trials imitate motor imagery: class "main_gauche" has more
power on one electrode, class "main_droite" on another - exactly the kind of
variance difference CSP and covariance-based models should pick up.

Run from backend/:   pytest -v tests/test_models.py
"""

import numpy as np
import pandas as pd
import pytest

from handmind import config
from handmind.evaluation import (chance_level, cross_validate_subject,
                                 evaluate_subject, summarize, to_markdown)
from handmind.models import MODELS, make_model

LEFT, RIGHT = config.EVENT_ID["main_gauche"], config.EVENT_ID["main_droite"]
N_TIMES = int(round((config.TMAX - config.TMIN) * config.SFREQ)) + 1  # 561


def synthetic_mi(n_trials=60, n_channels=17, separable=True, seed=0):
    """Fake epochs (volts, float32 like the saved .npz) + labels LEFT/RIGHT."""
    rng = np.random.default_rng(seed)
    y = np.array([LEFT, RIGHT] * (n_trials // 2))
    X = rng.normal(0, 10e-6, (n_trials, n_channels, N_TIMES))
    if separable:
        X[y == LEFT, 0] *= 2.5    # e.g. more power on "C4" for left hand
        X[y == RIGHT, -1] *= 2.5  # and on "C3" for right hand
    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# Models build and fit on the 3 electrode configurations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("cfg", ["A", "B", "C"])
def test_models_fit_all_configs(name, cfg):
    n_ch = len(config.CHANNEL_CONFIGS[cfg])
    X, y = synthetic_mi(n_trials=40, n_channels=n_ch)
    model = make_model(name, n_ch).fit(X.astype(np.float64), y)
    pred = model.predict(X.astype(np.float64))
    assert pred.shape == y.shape
    assert set(pred) <= {LEFT, RIGHT}
    # predict_proba is needed later for the confidence bar of the avatar UI
    proba = model.predict_proba(X.astype(np.float64))
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_csp_components_clipped_for_3_channels():
    model = make_model("csp_lda", n_channels=3)
    assert model.steps[0][1].n_components == 3


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("eegnet", 3)


# ---------------------------------------------------------------------------
# Cross-validation: learns real signal, does NOT learn noise (no leak)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(MODELS))
def test_cv_learns_separable_data(name):
    X, y = synthetic_mi(separable=True)
    r = cross_validate_subject(make_model(name, X.shape[1]), X, y)
    assert r["acc_mean"] >= 0.9
    assert r["kappa_mean"] >= 0.8
    assert len(r["acc_folds"]) == config.N_FOLDS


@pytest.mark.parametrize("name", list(MODELS))
def test_cv_stays_at_chance_on_noise(name):
    # Pure noise: if the model were fitted on test trials (leak), accuracy
    # would climb well above chance. It must stay below the p<0.05 threshold.
    X, y = synthetic_mi(n_trials=90, separable=False, seed=1)
    r = cross_validate_subject(make_model(name, X.shape[1]), X, y)
    assert r["acc_mean"] < chance_level(len(y))


def test_cv_is_reproducible():
    X, y = synthetic_mi()
    a = cross_validate_subject(make_model("riemann", X.shape[1]), X, y)
    b = cross_validate_subject(make_model("riemann", X.shape[1]), X, y)
    assert np.array_equal(a["acc_folds"], b["acc_folds"])


# ---------------------------------------------------------------------------
# Chance level and result tables
# ---------------------------------------------------------------------------
def test_chance_level():
    # 45 trials, 2 classes: the real threshold is well above 50 %
    assert 0.55 < chance_level(45, 2) < 0.70
    # more trials -> threshold gets closer to 50 %
    assert chance_level(1000, 2) < chance_level(45, 2)


def test_summary_and_markdown():
    rows = []
    for s in (1, 2):
        X, y = synthetic_mi(n_trials=40, n_channels=3, seed=s)
        rows_s, _ = evaluate_subject({"X": X, "y": y, "subject": s},
                                     ["csp_lda", "riemann"])
        for r in rows_s:
            r["config"] = "C"
            rows.append(r)
    df = pd.DataFrame(rows)
    assert len(df) == 2 * 2
    assert {"subject", "model", "config", "acc_mean", "kappa_mean",
            "chance_p05"} <= set(df.columns)

    summary = summarize(df)
    assert len(summary) == 2                      # one row per model x config
    assert (summary["n_subjects"] == 2).all()

    md = to_markdown(summary, df)
    assert "| csp_lda | C | 2 |" in md
    assert "S001" in md and "S002" in md
