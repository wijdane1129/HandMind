"""S5 - Online classifier: stream in, one decision every 250 ms out.

    EEG chunks ──> pick channels ──> causal filter ──> ring buffer (1 s)
                                                          │ every 250 ms
                                       model.predict_proba(last 1 s)
                                                          │
                                 DecisionMaker (threshold + vote on 3)
                                                          │
           {"classe", "confiance", "vraie_classe", "t"}  (Godot contract)

The classifier does not know where the samples come from: an LSL stream
(sources.LSLSource, the dataset player of Etudiant B) or a direct replay of
a recording (sources.ReplaySource). It only needs, per chunk, the samples
and their timestamps; the true class comes from the marker stream.
"""

import time
from bisect import bisect_right

import numpy as np

from .. import config
from .decision import REST, DecisionMaker
from .filters import StreamFilter


def bundle_class_names(bundle):
    """Class names in the column order of model.predict_proba."""
    return [config.CLASSES[int(c)] for c in bundle["model"].classes_]


def marker_to_class(label):
    """Marker -> class name. Accepts our names, codes 0-4 or "T0/T1/T2"-free
    names already converted by the player. Unknown markers -> None."""
    if isinstance(label, (list, tuple)):
        label = label[0]
    s = str(label).strip()
    if s in config.EVENT_ID:
        return s
    try:
        code = int(float(s))
    except ValueError:
        return None
    return config.CLASSES[code] if 0 <= code < len(config.CLASSES) else None


class OnlineClassifier:
    """Sliding-window classifier fed chunk by chunk.

    Parameters
    ----------
    bundle : dict saved by `scripts.train --online` (model, ch_names, sfreq...).
    stream_ch_names : channel names of the incoming stream, in stream order.
    stream_sfreq : sampling rate of the stream (must match the model's).
    """

    def __init__(self, bundle, stream_ch_names, stream_sfreq,
                 window_s=config.ONLINE_WINDOW_S, step_s=config.ONLINE_STEP_S,
                 threshold=config.ONLINE_THRESHOLD, n_vote=config.ONLINE_N_VOTE):
        if abs(stream_sfreq - bundle["sfreq"]) > 1e-6:
            raise ValueError(f"stream at {stream_sfreq} Hz but model trained "
                             f"at {bundle['sfreq']} Hz")
        missing = [c for c in bundle["ch_names"] if c not in stream_ch_names]
        if missing:
            raise ValueError(f"channels missing from the stream: {missing}")

        self.model = bundle["model"]
        self.sfreq = float(stream_sfreq)
        # model channel order, whatever the order of the stream
        self.picks = [list(stream_ch_names).index(c) for c in bundle["ch_names"]]
        self.win = int(round(window_s * self.sfreq))
        self.step = int(round(step_s * self.sfreq))
        self.filter = StreamFilter(self.sfreq, len(self.picks))
        self.decider = DecisionMaker(bundle_class_names(bundle), threshold, n_vote)
        self.reset()

    def reset(self):
        self.buffer = np.zeros((len(self.picks), self.win))
        self.n_seen = 0               # samples received so far
        self.next_decision = self.win  # sample count of the next decision
        self.t0 = None
        self.last_ts = None
        self.marker_times, self.marker_labels = [], []
        self.filter.reset()
        self.decider.reset()
        self.proc_ms = []             # processing time of each decision

    # -- markers (true class) ------------------------------------------------
    def push_marker(self, timestamp, label):
        name = marker_to_class(label)
        if name is not None:
            self.marker_times.append(float(timestamp))
            self.marker_labels.append(name)

    def true_class(self, timestamp):
        i = bisect_right(self.marker_times, timestamp)
        return self.marker_labels[i - 1] if i else None

    # -- EEG -----------------------------------------------------------------
    def push(self, samples, timestamps):
        """Feed one chunk. samples: (n_samples, n_stream_channels) like LSL.

        Returns the list of messages (dicts) produced by this chunk: 0, 1 or
        several, depending on how many 250 ms steps the chunk covers.
        """
        samples = np.asarray(samples, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if samples.ndim != 2 or len(samples) == 0:
            return []
        if self.t0 is None:
            self.t0 = timestamps[0]
        x = self.filter(samples[:, self.picks].T)   # (n_picks, n)

        messages, i = [], 0
        while i < x.shape[1]:
            # feed up to the next decision point
            n = min(x.shape[1] - i, self.next_decision - self.n_seen)
            self._append(x[:, i:i + n])
            i += n
            self.n_seen += n
            if self.n_seen == self.next_decision:
                messages.append(self._decide(timestamps[i - 1]))
                self.next_decision += self.step
        self.last_ts = timestamps[-1]
        return messages

    def _append(self, x):
        n = x.shape[1]
        if n >= self.win:
            self.buffer = x[:, -self.win:].copy()
        else:
            self.buffer = np.roll(self.buffer, -n, axis=1)
            self.buffer[:, -n:] = x

    def _decide(self, ts):
        start = time.perf_counter()
        proba = self.model.predict_proba(self.buffer[None])[0]
        classe, confiance = self.decider.update(proba)
        self.proc_ms.append(1000 * (time.perf_counter() - start))
        return {
            "classe": classe,
            "confiance": round(confiance, 3),
            "vraie_classe": self.true_class(ts) or REST,
            "t": round(float(ts - self.t0), 3),
        }
