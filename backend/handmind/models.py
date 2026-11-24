"""S3/S4 - Offline classifiers.

Three classic motor-imagery pipelines, all scikit-learn Pipelines that take
X of shape (n_epochs, n_channels, n_times) directly and work for 2, 3 or 5
classes:

    csp_lda    : CSP (log-variance of spatially filtered signals) -> LDA      (S3)
    riemann    : covariance matrices -> tangent space -> logistic regression  (S3)
    fbcsp_svm  : filter bank (5 bands) -> CSP per band -> best features
                 (mutual information) -> SVM                                  (S4)

Because they are plain sklearn objects, cross_validate / joblib.dump work on
them unchanged. A new model = one factory + one entry in MODELS.
"""

from functools import partial

import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


# ---------------------------------------------------------------------------
# S3 - CSP + LDA
# ---------------------------------------------------------------------------
def make_csp_lda(n_channels, n_components=config.CSP_N_COMPONENTS, **_):
    """CSP + LDA.

    CSP finds spatial filters whose output variance differs most between
    the classes (mu/beta desynchronisation over C3 vs C4). We keep the
    log-variance of `n_components` filters as features, then LDA separates
    them with straight lines. With more than 2 classes MNE's CSP uses a
    multi-class (joint diagonalisation) version, nothing to change here.

    n_components is clipped to n_channels: config C has only 3 electrodes.
    """
    n = min(n_components, n_channels)
    return make_pipeline(
        CSP(n_components=n, reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )


# ---------------------------------------------------------------------------
# S3 - Riemann
# ---------------------------------------------------------------------------
def make_riemann(n_channels=None, **_):
    """Covariances + tangent space + logistic regression.

    Each trial becomes its (n_channels x n_channels) covariance matrix. These
    matrices live on a curved space (SPD matrices); TangentSpace "flattens"
    them around their mean so a linear classifier can be used.
    OAS shrinkage keeps the covariance well conditioned with 64 channels and
    few trials.
    """
    return make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(max_iter=1000),
    )


# ---------------------------------------------------------------------------
# S4 - FBCSP + SVM
# ---------------------------------------------------------------------------
class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter Bank CSP (Ang et al., 2008 - winner of BCI Competition IV 2a).

    The signal is split into several frequency bands; one CSP is learned per
    band and their log-variance features are concatenated. Useful because
    the most discriminant rhythm (mu ~10 Hz, beta ~20 Hz) differs from one
    person to another.

    Filtering is stateless (same filter for every trial), so doing it inside
    fit/transform does not leak anything between CV folds.
    """

    def __init__(self, bands=tuple(config.FBCSP_BANDS), sfreq=config.SFREQ,
                 n_components=config.FBCSP_N_COMPONENTS):
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components

    def _band(self, X, band):
        # 4th-order Butterworth, zero-phase (offline). Fast on short epochs.
        #(mne.filter.filter_data) is zero-phase: it runs forwards(from past to future) AND backwards(from future to past), so it needs samples from the future and the past.
        return mne.filter.filter_data(
            np.asarray(X, dtype=np.float64), self.sfreq, band[0], band[1],
            method="iir", verbose="error")

    def fit(self, X, y):
        n = min(self.n_components, X.shape[1])
        self.csps_ = []
        for band in self.bands:
            csp = CSP(n_components=n, reg="ledoit_wolf", log=True,
                      norm_trace=False)
            csp.fit(self._band(X, band), y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        return np.concatenate(
            [csp.transform(self._band(X, band))
             for csp, band in zip(self.csps_, self.bands)], axis=1)


def make_fbcsp_svm(n_channels, sfreq=config.SFREQ,
                   bands=tuple(config.FBCSP_BANDS),
                   n_components=config.FBCSP_N_COMPONENTS,
                   k_best=config.FBCSP_K_BEST, **_):
    """FBCSP features -> standardise -> keep the k best (mutual info) -> SVM.

    The SVM is wrapped in Platt calibration (what SVC(probability=True) did,
    deprecated since scikit-learn 1.9) so predict_proba() gives the
    confidence shown in the avatar UI, like the two other models.
    """
    n_features = len(bands) * min(n_components, n_channels)
    mi = partial(mutual_info_classif, random_state=config.RANDOM_SEED)
    return make_pipeline(
        FilterBankCSP(bands=bands, sfreq=sfreq, n_components=n_components),
        StandardScaler(),
        SelectKBest(mi, k=min(k_best, n_features)),
        CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, gamma="scale"),
                               method="sigmoid", cv=3, ensemble=False),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# name -> factory(n_channels, sfreq=...). The names are the ones used in the
# result tables and on the command line (scripts.evaluate --models ...).
MODELS = {
    "csp_lda": make_csp_lda,
    "riemann": make_riemann,
    "fbcsp_svm": make_fbcsp_svm,
}


def make_model(name, n_channels, sfreq=config.SFREQ):
    """Build a fresh, unfitted model by name.

    sfreq only matters for fbcsp_svm (160 Hz PhysioNet, 250 Hz BCI IV 2a).
    """
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}, expected one of {list(MODELS)}"
        ) from None
    return factory(n_channels=n_channels, sfreq=sfreq)
