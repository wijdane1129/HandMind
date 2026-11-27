"""S5 - Online classifier: stream in, one decision every 250 ms out.

    EEG chunks ──> pick channels ──> causal filter ──> ring buffer (1 s)
                                                          │ every 250 ms
                                       model.predict_proba(last 1 s)
                                                          │
                                 DecisionMaker (threshold + vote on 3)
                                                          │
  {"numero", "classe", "confiance", "vraie_classe", "t"}  (contract v1.2)

The classifier does not know where the samples come from: an LSL stream
(sources.LSLSource, the dataset player of Etudiant B) or a direct replay of
a recording (sources.ReplaySource). It only needs, per chunk, the samples
and their timestamps; the true class comes from the marker stream.

S6 changes (integration with Etudiant B):
- "numero" field, 0, 1, 2... (B's tests detect lost / duplicated messages)
- "t" = timestamp of the LAST sample of the classified window, converted
  by `to_t` (relative seconds by default, Unix time in the bridge, T13)
- "vraie_classe" = "" before the first marker (contract: unknown)
- new_segment(): reset filter + buffer when the player jumps to a new run
- latency_ms: chunk received -> decision ready, per decision (test T12)
"""

import time
from bisect import bisect_right#Retrouver rapidement la classe réelle à partir des timestamps des marqueurs.

import numpy as np

from .. import config
from .decision import REST, DecisionMaker
from .filters import StreamFilter


def bundle_class_names(bundle):# bundle dictionnaire qui contient les éléments nécessaires pour utiliser le modèle entraîné
    """Class names in the column order of model.predict_proba."""
    return [config.CLASSES[int(c)] for c in bundle["model"].classes_]


def marker_to_class(label):#transforme un marqueur EEG en nom de classe.
    #marqueur est une information qui indique à quel mouvement correspond une partie du signal
    """Marker -> class name. Accepts our names, codes 0-4 or "T0/T1/T2"-free
    names already converted by the player. Unknown markers -> None."""
    if isinstance(label, (list, tuple)):
        label = label[0]
    s = str(label).strip()
    s = config.CLASS_ALIASES.get(s, s)
    if s in config.EVENT_ID:
        return s
    try:
        code = int(float(s))
    except ValueError:
        return None
    return config.CLASSES[code] if 0 <= code < len(config.CLASSES) else None


class OnlineClassifier:
    #elle reçoit continuellement des données EEG, les traite et produit des prédictions sans attendre que l'enregistrement complet soit terminé.
    """Sliding-window classifier fed chunk by chunk.

    Parameters
    ----------
    bundle : dict saved by `scripts.train --online` (model, ch_names, sfreq...).
    stream_ch_names : channel names of the incoming stream, in stream order.
    stream_sfreq : sampling rate of the stream (must match the model's).
    """

    def __init__(self, bundle, stream_ch_names, stream_sfreq,
                 window_s=config.ONLINE_WINDOW_S, step_s=config.ONLINE_STEP_S,
                 threshold=config.ONLINE_THRESHOLD, n_vote=config.ONLINE_N_VOTE,
                 scale=1.0, to_t=None):
        """scale: factor applied to the stream samples to get volts (the
        player sends microvolts: scale=1e-6). to_t: timestamp -> "t" field
        (default: seconds since the first sample)."""
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
        self.scale = float(scale)
        self.to_t = to_t
        self.reset()

    def reset(self):
        self.buffer = np.zeros((len(self.picks), self.win))
        self.n_seen = 0               # samples received so far
        self.next_decision = self.win  # sample count of the next decision
        self.t0 = None
        self.last_ts = None
        self.marker_times, self.marker_labels = [], []
        self.boundaries = []          # timestamps of "NOUVEAU_RUN" markers
        self.end_ts = None            # timestamp of the "FIN" marker
        self.filter.reset()
        self.decider.reset()
        self.proc_ms = []             # model + vote time of each decision
        self.latency_ms = []          # chunk received -> decision (T12)
        self.raw_labels = []          # argmax of each window (no threshold/vote)
        self.decision_ts = []         # stream timestamp of each window's last sample
        self.numero = 0

    def new_segment(self):#On ne doit surtout pas mélanger les échantillons des deux runs.
        """The signal jumps (new run in the player): forget the past samples
        so that no window mixes two runs. Markers and counters are kept."""
        self.buffer[:] = 0.0
        self.filter.reset()
        self.decider.reset()
        self.next_decision = self.n_seen + self.win

    # -- markers (true class) ------------------------------------------------
    def push_marker(self, timestamp, label):#Elle traite trois types d'événement: 1. mémorise le timestamp du début du nouveau run.
        #2. enregistre le moment où le replay est terminé 3. convertit le marqueur en classe et on stocke son timestamp.
        """Class markers -> true class. Technical markers of the player:
        "NOUVEAU_RUN:<n>" -> segment boundary, "FIN:<n>" -> end of replay."""
        if isinstance(label, (list, tuple)):
            label = label[0]
        text = str(label).strip()
        if text.startswith(config.LSL_NEW_RUN_PREFIX):
            self.boundaries.append(float(timestamp))
            return
        if text.startswith(config.LSL_END_PREFIX):
            self.end_ts = float(timestamp)
            return
        name = marker_to_class(text)
        if name is not None:
            self.marker_times.append(float(timestamp))
            self.marker_labels.append(name)

    @property
    def ended(self):#Elle permet de savoir si le replay est réellement terminé.
        """True once "FIN" is received AND the EEG up to it has been processed.
        The marker stream is lighter than the EEG stream, so FIN can arrive
        while EEG samples sent before it are still waiting in the inlet."""
        if self.end_ts is None or self.last_ts is None:
            return False
        return self.last_ts >= self.end_ts - 0.01

    def true_class(self, timestamp):#Cette fonction retrouve la vraie classe à un instant donné.
        i = bisect_right(self.marker_times, timestamp)
        return self.marker_labels[i - 1] if i else None

    # -- EEG -----------------------------------------------------------------
    def push(self, samples, timestamps):#Gérer les données et les frontières entre les runs
        """Feed one chunk, cutting it at the run boundaries if needed."""
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if len(timestamps) and self.boundaries and timestamps[-1] >= self.boundaries[0]:
            k = int(np.searchsorted(timestamps, self.boundaries.pop(0)))
            samples = np.asarray(samples)
            before = self._push(samples[:k], timestamps[:k]) if k else []
            if self.n_seen:                 # nothing to forget at the very start
                self.new_segment()
            return before + self.push(samples[k:], timestamps[k:])
        return self._push(samples, timestamps)

    def _push(self, samples, timestamps):#Traiter les données EEG et déclencher les prédictions
        """Feed one chunk (no run boundary inside). samples: (n_samples, n_stream_channels) like LSL.

        Returns the list of messages (dicts) produced by this chunk: 0, 1 or
        several, depending on how many 250 ms steps the chunk covers.
        """
        t_arrival = time.perf_counter()
        samples = np.asarray(samples, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if samples.ndim != 2 or len(samples) == 0:
            return []
        if self.t0 is None:
            self.t0 = timestamps[0]
        x = self.filter(samples[:, self.picks].T * self.scale)  # (n_picks, n), volts

        messages, i = [], 0
        while i < x.shape[1]:
            # feed up to the next decision point
            n = min(x.shape[1] - i, self.next_decision - self.n_seen)
            self._append(x[:, i:i + n])
            i += n
            self.n_seen += n
            if self.n_seen == self.next_decision:
                messages.append(self._decide(timestamps[i - 1], t_arrival))
                self.next_decision += self.step
        self.last_ts = timestamps[-1]
        return messages

    def _append(self, x):#Mettre à jour le buffer EEG
        n = x.shape[1]
        if n >= self.win:
            self.buffer = x[:, -self.win:].copy()
        else:
            self.buffer = np.roll(self.buffer, -n, axis=1)
            self.buffer[:, -n:] = x

    def _decide(self, ts, t_arrival=None):#Effectuer la classification
        start = time.perf_counter()
        proba = self.model.predict_proba(self.buffer[None])[0]
        classe, confiance = self.decider.update(proba)
        end = time.perf_counter()
        self.proc_ms.append(1000 * (end - start))
        self.latency_ms.append(1000 * (end - (t_arrival or start)))
        self.raw_labels.append(self.decider.classes[int(np.argmax(proba))])
        self.decision_ts.append(float(ts))
        t = self.to_t(ts) if self.to_t else round(float(ts - self.t0), 3)
        msg = {
            "numero": self.numero,
            "classe": classe,
            "confiance": round(confiance, 3),
            "vraie_classe": self.true_class(ts) or "",
            "t": t,
        }
        self.numero += 1
        return msg
