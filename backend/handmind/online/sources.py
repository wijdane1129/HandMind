"""S5 - Where the online classifier gets its samples from.

Every source gives, in a loop, (samples, timestamps, markers):
    samples    : (n_samples, n_channels) in volts, like pylsl.pull_chunk
    timestamps : (n_samples,) seconds
    markers    : list of (timestamp, label) received since the last call
and exposes .ch_names and .sfreq.

    LSLSource     the dataset player of Etudiant B (S3) - the real chain
    ReplaySource  reads a recording directly, no LSL needed (tests, or
                  while the player is not ready)
"""

import time

import numpy as np

from .. import config


class ReplaySource:
    """Replay an array (n_channels, n_times) + markers, optionally in real time.

    speed=1: real time, speed=5: 5x faster, speed=0: as fast as possible.
    """

    def __init__(self, data, sfreq, ch_names, markers=(), chunk_s=0.0625,
                 speed=1.0, loop=False):
        self.data = np.asarray(data)
        self.sfreq = float(sfreq)
        self.ch_names = list(ch_names)
        self.markers = sorted((float(t), m) for t, m in markers)
        self.chunk = max(1, int(round(chunk_s * self.sfreq)))
        self.speed = speed
        self.loop = loop

    @classmethod
    def from_physionet(cls, subject, **kwargs):
        """Continuous recording of one subject, markers = class names."""
        from ..data import load_physionet_raw
        raw = load_physionet_raw(subject)
        markers = [(a["onset"], a["description"]) for a in raw.annotations
                   if a["description"] in config.EVENT_ID]
        return cls(raw.get_data(picks="eeg"), raw.info["sfreq"],
                   raw.copy().pick("eeg").ch_names, markers, **kwargs)

    def __iter__(self):
        n_times = self.data.shape[1]
        offset = 0.0       # time shift for each new loop
        start = time.perf_counter()
        while True:
            m = 0
            for i in range(0, n_times, self.chunk):
                j = min(i + self.chunk, n_times)
                ts = offset + np.arange(i, j) / self.sfreq
                new = []
                while m < len(self.markers) and self.markers[m][0] <= ts[-1] - offset:
                    new.append((self.markers[m][0] + offset, self.markers[m][1]))
                    m += 1
                if self.speed:
                    wait = start + (ts[-1] + 1 / self.sfreq) / self.speed - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                yield self.data[:, i:j].T, ts, new
            if not self.loop:
                return
            offset += n_times / self.sfreq


class LSLSource:
    """Read the EEG + marker LSL streams of the dataset player.

    S6: streams are found by NAME first (HandMind_EEG / HandMind_Markers,
    see backend/reader/lecteur.py), then by type. The unit written by the
    player in desc/unit gives `.scale` (microvolts -> 1e-6, to get volts).
    `.to_unix(ts)` converts an LSL timestamp to Unix time (same machine).
    """

    def __init__(self, eeg_name=config.LSL_EEG_NAME,
                 marker_name=config.LSL_MARKER_NAME,
                 eeg_type=config.LSL_EEG_TYPE,
                 marker_type=config.LSL_MARKER_TYPE, timeout=10.0):
        import pylsl  # only needed for the real chain
        self._pylsl = pylsl

        eeg = self._resolve(eeg_name, eeg_type, timeout)
        if not eeg:
            raise RuntimeError(
                f"No LSL stream {eeg_name!r} (type {eeg_type!r}) found. Is the "
                "dataset player running? (or use --replay SUBJECT)")
        self.inlet = pylsl.StreamInlet(eeg[0], max_buflen=10)
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.ch_names = self._channel_names(info)
        self.unit = info.desc().child_value("unit") or "volts"
        self.scale = self.unit_scale(self.unit)
        print(f"[LSL] EEG: {info.name()}, {info.channel_count()} ch, "
              f"{self.sfreq:g} Hz, unit {self.unit!r} (x{self.scale:g} -> V)")

        markers = self._resolve(marker_name, marker_type, 2.0)
        self.markers = pylsl.StreamInlet(markers[0]) if markers else None
        if self.markers is None:
            print(f"[LSL] no marker stream: vraie_classe will be empty")
        self._sync_clock()

    def _resolve(self, name, stype, timeout):
        found = self._pylsl.resolve_byprop("name", name, 1, timeout)
        return found or self._pylsl.resolve_byprop("type", stype, 1, 1.0)

    def _sync_clock(self):
        """LSL clock -> Unix clock (+ correction if the player runs elsewhere)."""
        try:
            corr = self.inlet.time_correction(1.0)
        except Exception:
            corr = 0.0
        self.lsl_to_unix = time.time() - self._pylsl.local_clock() + corr

    def to_unix(self, ts):
        return round(float(ts + self.lsl_to_unix), 4)

    @staticmethod
    def unit_scale(unit):
        key = str(unit).strip().lower()
        if key not in config.LSL_UNIT_SCALE:
            raise RuntimeError(f"Unknown EEG unit {unit!r} in the stream")
        return config.LSL_UNIT_SCALE[key]

    @staticmethod
    def _channel_names(info):
        names, ch = [], info.desc().child("channels").child("channel")
        while not ch.empty():
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        if len(names) == info.channel_count():
            return names
        if info.channel_count() == len(config.CHANNELS_ALL):
            print("[LSL] no channel labels in the stream: assuming the 64 "
                  "PhysioNet channels in standard order")
            return list(config.CHANNELS_ALL)
        raise RuntimeError("The EEG stream has no channel labels: ask the "
                           "player to send them (desc/channels/channel/label)")

    def pull(self, timeout=0.0):
        """One non-blocking read: (samples, timestamps, markers)."""
        samples, ts = self.inlet.pull_chunk(timeout=timeout, max_samples=1024)
        new = []
        if self.markers is not None:
            m, mts = self.markers.pull_chunk(timeout=0.0)
            new = list(zip(mts, (x[0] for x in m)))
        if samples:
            return np.asarray(samples), np.asarray(ts), new
        return np.empty((0, len(self.ch_names))), np.empty(0), new

    def __iter__(self):
        while True:
            samples, ts, new = self.pull(timeout=0.05)
            if len(ts) or new:
                yield samples, ts, new
