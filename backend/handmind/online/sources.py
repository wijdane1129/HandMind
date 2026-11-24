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
    """Read the EEG + marker LSL streams of the dataset player."""

    def __init__(self, eeg_type=config.LSL_EEG_TYPE,
                 marker_type=config.LSL_MARKER_TYPE, timeout=10.0):
        import pylsl  # only needed for the real chain

        eeg = pylsl.resolve_byprop("type", eeg_type, timeout=timeout)
        if not eeg:
            raise RuntimeError(
                f"No LSL stream of type {eeg_type!r} found. Is the dataset "
                "player running? (or use --replay SUBJECT)")
        self.inlet = pylsl.StreamInlet(eeg[0], max_buflen=10)
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.ch_names = self._channel_names(info)

        markers = pylsl.resolve_byprop("type", marker_type, timeout=2.0)
        self.markers = pylsl.StreamInlet(markers[0]) if markers else None
        if self.markers is None:
            print(f"[LSL] no {marker_type!r} stream: vraie_classe will be 'repos'")

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

    def __iter__(self):
        while True:
            samples, ts = self.inlet.pull_chunk(timeout=0.05)
            new = []
            if self.markers is not None:
                m, mts = self.markers.pull_chunk(timeout=0.0)
                new = list(zip(mts, (x[0] for x in m)))
            if samples:
                yield np.asarray(samples), np.asarray(ts), new
            elif new:
                yield np.empty((0, len(self.ch_names))), np.empty(0), new
