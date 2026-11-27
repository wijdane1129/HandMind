"""S5 - Causal (real-time) filtering.

The S2 filter (mne.filter.filter_data) is zero-phase: it runs forwards AND
backwards, so it needs samples from the future. Live, we only have the past.

StreamFilter is a causal IIR filter (Butterworth band-pass 8-30 Hz + notch
60 Hz) that keeps its internal state between chunks: filtering a signal in
chunks of any size gives exactly the same output as filtering it in one go.
The same class is used to build the training windows (scripts.train
--online), so the model sees online-like signals during training.

Price of causality: a small delay (group delay, ~ a few tens of ms in the
8-30 Hz band), measured by group_delay_ms() and reported in the S5 notes.
"""

import numpy as np
from scipy.signal import butter, group_delay, iirnotch, sosfilt, tf2sos # butter Créer butterworth filter
#group_delay for calculating the delay of a filter, iirnotch for a notch filter
# Appliquer un filtre causal en conservant son état interne
# tf2sos for converting transfer function coefficients to second-order sections

from .. import config


def design_sos(sfreq, l_freq=config.L_FREQ, h_freq=config.H_FREQ,
               notch=config.NOTCH_FREQ, order=config.ONLINE_FILTER_ORDER):
    """Second-order sections of the band-pass (+ notch) filter."""
    sos = butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq,
                 output="sos")
    if notch is not None and notch < sfreq / 2:
        b, a = iirnotch(notch, Q=30.0, fs=sfreq)#Le facteur Q contrôle la largeur de la bande rejetée.
        sos = np.vstack([sos, tf2sos(b, a)])#On convertit les coefficients du filtre Notch en représentation SOS.
    return sos
#SOS signifie Second-Order Sections (sections du second ordre).
#Au lieu de représenter tout le filtre avec un seul ensemble de coefficients
#on le décompose en plusieurs petits filtres du second ordre.
#Car cette représentation est généralement plus stable numériquement
#particulièrement pour les filtres d'ordre élevé.

class StreamFilter:
    """Filtre causal qui filtre dans le sens "passe => present, en gardant l'état interne entre les chunks."""

    def __init__(self, sfreq, n_channels, **design_kwargs):
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.sos = design_sos(sfreq, **design_kwargs)
        self.reset()

    def reset(self):# initialise ou reinitialise l'état interne du filtre à zéro. Cela signifie que le filtre n'a pas de mémoire des échantillons précédents et commencera à filtrer les nouveaux échantillons comme s'il n'avait jamais été utilisé auparavant.
        # zero initial state: the first ~0.5 s are a transient, the online
        # classifier waits for a full window anyway
        self.zi = np.zeros((self.sos.shape[0], self.n_channels, 2))#zi the internal state of the filter, because to filter a cunk we need info from the previous chunk 

    def __call__(self, chunk):# appliquer le filtre causal à un nouveau morceau du signal EEG.
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.shape[0] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, "
                             f"got {chunk.shape[0]}")
        out, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)# on applique le filtre causal sur le chunk, on met à jour l'état interne du filtre
        return out


def filter_causal(data, sfreq, **design_kwargs):#iltrer un enregistrement EEG complet en une seule fois, avec le même filtre causal.
    """One-shot causal filtering of a whole recording (n_channels, n_times)."""
    data = np.asarray(data, dtype=np.float64)
    return StreamFilter(sfreq, data.shape[0], **design_kwargs)(data)


def group_delay_ms(sfreq, freqs=(10.0, 20.0), **design_kwargs):
    """Delay (ms) introduced by the causal filter at a few frequencies."""
    sos = design_sos(sfreq, **design_kwargs)
    total = np.zeros(len(freqs))
    for section in sos:
        _, gd = group_delay((section[:3], section[3:]), w=list(freqs), fs=sfreq)
        total += gd
    return {f: 1000 * d / sfreq for f, d in zip(freqs, total)}
