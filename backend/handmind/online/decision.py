"""S5 - From window probabilities to a stable command for the avatar.

A raw prediction every 250 ms flickers (one noisy window = one wrong
movement). Two simple rules make it stable:

1. Confidence threshold: if the best class has a probability below
   `threshold`, the window counts as "repos" (unsure -> do nothing).
2. Majority vote over the last `n_vote` windows (3 by default): a new
   command needs at least 2 of the last 3 windows. Without a majority the
   previous decision is kept.

Cost: up to (n_vote - 1) x step = 0.5 s of extra reaction time. This
trade-off (stability vs reactivity) is one of the S6 measurements.
"""

from collections import Counter, deque

import numpy as np

from .. import config

REST = "repos"


class DecisionMaker:
    """reçoit les probabilités produites par le modèle et renvoie(classe, confiance)."""

    def __init__(self, classes, threshold=config.ONLINE_THRESHOLD,
                 n_vote=config.ONLINE_N_VOTE, rest=REST):
        self.classes = list(classes)       # column order of predict_proba
        self.threshold = threshold
        self.n_vote = n_vote
        self.rest = rest
        self.reset()

    def reset(self):
        self.labels = deque(maxlen=self.n_vote)
        self.probas = deque(maxlen=self.n_vote)
        self.decision = self.rest

    def window_label(self, proba):
        """Label of ONE window, after the confidence threshold."""
        best = int(np.argmax(proba))
        return self.classes[best] if proba[best] >= self.threshold else self.rest

    def update(self, proba):
        """Add one window's probabilities, return (classe, confiance).

        confiance = mean probability of the decided class over the vote
        window. If the decided class is "repos" but the model has no "repos"
        class (2-class model), confiance = 1 - mean of the best probability.
        """
        proba = np.asarray(proba, dtype=float)
        self.labels.append(self.window_label(proba))
        self.probas.append(proba)

        label, count = Counter(self.labels).most_common(1)[0]
        if count > len(self.labels) / 2:   # strict majority
            self.decision = label
        return self.decision, self.confidence(self.decision)

    def confidence(self, label):
        P = np.array(self.probas)
        if label in self.classes:
            return float(P[:, self.classes.index(label)].mean())
        return float(1.0 - P.max(axis=1).mean())
