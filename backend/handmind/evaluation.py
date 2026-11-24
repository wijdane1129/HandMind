"""S3/S4 - Offline evaluation.

    cross_validate_subject : intra-subject stratified k-fold (S3, S4)
    loso                   : leave-one-subject-out, inter-subject (S4, test T9)
    session_transfer       : train on one session, test on another (S4, test T8)

Rule that protects every number in the report: the model is fitted INSIDE
each fold on training trials only (a fresh model each time), so CSP filters
/ covariance means / feature selection never see test trials.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from . import config
from .models import make_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chance_level(n_trials, n_classes=2, alpha=0.05):
    """Accuracy above which a result is better than chance (p < alpha).

    With few trials, 1/n_classes is NOT the real threshold: by luck a random
    classifier often gets 60 % on 45 trials in 2 classes. Binomial threshold,
    see Combrisson & Jerbi (2015), "Exceeding chance level by chance".
    """
    return binom.ppf(1 - alpha, n_trials, 1.0 / n_classes) / n_trials


def balance_classes(X, y, seed=config.RANDOM_SEED):
    """Random under-sampling: every class keeps as many trials as the rarest.

    Needed for 3 / 5 classes because "repos" has ~2x more trials than each
    movement; otherwise a model that always says "repos" looks good.
    Trial order is preserved.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.min()
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == c), size=n, replace=False)
        for c in classes])
    keep.sort()
    return X[keep], y[keep]


def _scores(y_true, y_pred, labels):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


# ---------------------------------------------------------------------------
# Intra-subject k-fold  (S3, S4)
# ---------------------------------------------------------------------------
def cross_validate_subject(model, X, y, n_splits=config.N_FOLDS,
                           seed=config.RANDOM_SEED):
    """Stratified, shuffled k-fold CV of one model on one subject.

    Returns a dict: acc_mean, acc_std, kappa_mean, kappa_std, fit_time
    (s, mean per fold), per-fold arrays acc_folds / kappa_folds, and
    `confusion` (rows = true, cols = predicted, summed over folds,
    labels = sorted(unique(y))).
    """
    X = np.asarray(X, dtype=np.float64)  # pyriemann is happier in float64
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, kappas, times = [], [], []
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for train, test in cv.split(X, y):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X[train], y[train])
        times.append(time.perf_counter() - t0)
        s = _scores(y[test], m.predict(X[test]), labels)
        accs.append(s["acc"])
        kappas.append(s["kappa"])
        confusion += s["confusion"]
    accs, kappas = np.array(accs), np.array(kappas)
    return {
        "acc_mean": accs.mean(), "acc_std": accs.std(),
        "kappa_mean": kappas.mean(), "kappa_std": kappas.std(),
        "fit_time": float(np.mean(times)),
        "acc_folds": accs, "kappa_folds": kappas,
        "confusion": confusion, "labels": labels,
    }


def evaluate_subject(d, model_names, n_splits=config.N_FOLDS, balance=False,
                     sfreq=config.SFREQ):
    """Run every model on one subject's data dict (from load_epochs).

    Returns (rows, confusions): rows is a list of result dicts (one per
    model), confusions maps model name -> confusion matrix of this subject.
    """
    X, y = d["X"], d["y"]
    if balance:
        X, y = balance_classes(X, y)
    n_channels = X.shape[1]
    n_classes = len(np.unique(y))
    rows, confusions = [], {}
    for name in model_names:
        t0 = time.perf_counter()
        r = cross_validate_subject(make_model(name, n_channels, sfreq),
                                   X, y, n_splits)
        confusions[name] = r["confusion"]
        rows.append({
            "subject": d["subject"],
            "model": name,
            "n_classes": n_classes,
            "n_channels": n_channels,
            "n_trials": len(y),
            "acc_mean": r["acc_mean"],
            "acc_std": r["acc_std"],
            "kappa_mean": r["kappa_mean"],
            "kappa_std": r["kappa_std"],
            "chance_p05": chance_level(len(y), n_classes),
            "seconds": time.perf_counter() - t0,
        })
    return rows, confusions


# ---------------------------------------------------------------------------
# Leave-one-subject-out  (S4, test T9 part 1)
# ---------------------------------------------------------------------------
def loso(datasets, model_factory, balance=False):
    """Inter-subject evaluation: train on N-1 subjects, test on the last one.

    Parameters
    ----------
    datasets : list of dicts {"X", "y", "subject"} (from load_epochs),
        all with the same channels and classes.
    model_factory : callable(n_channels) -> unfitted sklearn estimator.
    balance : under-sample each subject's classes before pooling.

    Returns a list of rows, one per left-out subject.
    The left-out subject's trials are NEVER in the training set.
    """
    data = []
    for d in datasets:
        X, y = (balance_classes(d["X"], d["y"]) if balance
                else (d["X"], d["y"]))
        data.append((d["subject"], X, y))
    labels = np.unique(np.concatenate([y for _, _, y in data]))

    rows = []
    for i, (subject, X_test, y_test) in enumerate(data):
        X_train = np.concatenate([X for j, (_, X, _) in enumerate(data) if j != i])
        y_train = np.concatenate([y for j, (_, _, y) in enumerate(data) if j != i])
        model = model_factory(X_test.shape[1])
        t0 = time.perf_counter()
        model.fit(np.asarray(X_train, dtype=np.float64), y_train)
        s = _scores(y_test, model.predict(np.asarray(X_test, dtype=np.float64)),
                    labels)
        rows.append({
            "subject": subject,
            "n_train_subjects": len(data) - 1,
            "n_train_trials": len(y_train),
            "n_trials": len(y_test),
            "n_classes": len(labels),
            "acc": s["acc"],
            "kappa": s["kappa"],
            "chance_p05": chance_level(len(y_test), len(labels)),
            "seconds": time.perf_counter() - t0,
        })
    return rows


# ---------------------------------------------------------------------------
# Session transfer  (S4, test T8 - BCI IV 2a competition protocol)
# ---------------------------------------------------------------------------
def session_transfer(model, X_train, y_train, X_test, y_test):
    """Fit on the training session, score on the evaluation session."""
    m = clone(model).fit(np.asarray(X_train, dtype=np.float64), y_train)
    labels = np.unique(np.concatenate([y_train, y_test]))
    return _scores(y_test, m.predict(np.asarray(X_test, dtype=np.float64)),
                   labels)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _group_keys(df):
    return [k for k in ("n_classes", "model", "config") if k in df.columns]


def summarize(df):
    """Mean +/- std over subjects, one row per (n_classes, model, config).

    Also counts how many subjects beat the chance threshold.
    """
    df = df.assign(above_chance=df["acc_mean"] > df["chance_p05"])
    g = df.groupby(_group_keys(df))
    out = pd.DataFrame({
        "n_subjects": g["subject"].nunique(),
        "acc_mean": g["acc_mean"].mean(),
        "acc_std": g["acc_mean"].std(),
        "kappa_mean": g["kappa_mean"].mean(),
        "kappa_std": g["kappa_mean"].std(),
        "above_chance": g["above_chance"].sum(),
    })
    return out.reset_index()


def to_markdown(summary, per_subject, title=None):
    """Build the Markdown result page (no extra dependency like tabulate)."""
    n_classes = (int(per_subject["n_classes"].iloc[0])
                 if "n_classes" in per_subject else 2)
    title = title or f"Résultats {n_classes} classes (intra-sujet)"
    lines = [f"# {title}", "",
             f"Intra-subject, {config.N_FOLDS}-fold stratified CV, "
             f"seed {config.RANDOM_SEED}, epochs {config.TMIN}-{config.TMAX} s, "
             f"band-pass {config.L_FREQ:g}-{config.H_FREQ:g} Hz.", "",
             "## Summary (mean ± std over subjects)", "",
             "| Model | Config | Subjects | Accuracy | Kappa | > chance |",
             "|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(
            f"| {r.model} | {r.config} | {r.n_subjects} | "
            f"{r.acc_mean:.3f} ± {r.acc_std:.3f} | "
            f"{r.kappa_mean:.3f} ± {r.kappa_std:.3f} | "
            f"{r.above_chance}/{r.n_subjects} |")

    lines += ["", "## Accuracy per subject", ""]
    piv = per_subject.pivot_table(index="subject", columns=["model", "config"],
                                  values="acc_mean")
    cols = list(piv.columns)
    lines.append("| Subject | " + " | ".join(f"{m} {c}" for m, c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for subject, row in piv.iterrows():
        lines.append(f"| S{subject:03d} | "
                     + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    chance = per_subject["chance_p05"].iloc[0]
    lines += ["", f"Chance level: {1 / n_classes:.2f} in theory, significance "
              f"threshold (p < 0.05, {int(per_subject['n_trials'].iloc[0])} "
              f"trials): {chance:.2f}"]
    return "\n".join(lines) + "\n"
