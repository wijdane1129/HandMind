"""S5 tests - causal filter, decision rules, online classifier, LSL chain.

Synthetic data only. The LSL test needs pylsl (skipped otherwise).
Run from backend/:   pytest -v tests/test_online.py
"""

import json
import threading
import time

import numpy as np
import pytest

from handmind import config
from handmind.models import make_model
from handmind.online.classifier import OnlineClassifier, marker_to_class
from handmind.online.decision import DecisionMaker
from handmind.online.filters import (StreamFilter, filter_causal,
                                     group_delay_ms)
from handmind.online.sources import ReplaySource
from handmind.preprocessing import filter_signal
from scripts.train import causal_windows

SFREQ = config.SFREQ
CH_B = config.CHANNEL_CONFIGS["B"]


# ---------------------------------------------------------------------------
# Causal filter
# ---------------------------------------------------------------------------
def _gain_db(freq, duration=20.0):
    t = np.arange(int(duration * SFREQ)) / SFREQ
    x = np.sin(2 * np.pi * freq * t)[None, :]
    y = filter_causal(x, SFREQ)
    keep = slice(len(t) // 3, None)  # skip the start-up transient
    return 20 * np.log10(np.std(y[0, keep]) / np.std(x[0, keep]))


@pytest.mark.parametrize("freq", [2.0, 4.0, 50.0, 60.0, 70.0])
def test_causal_filter_stopband(freq):
    assert _gain_db(freq) <= -20


@pytest.mark.parametrize("freq", [10.0, 12.0, 20.0, 25.0])
def test_causal_filter_passband(freq):
    assert _gain_db(freq) >= -1


def test_chunked_filter_equals_one_shot():
    x = np.random.default_rng(0).normal(size=(3, 1000))
    f = StreamFilter(SFREQ, 3)
    pieces, i = [], 0
    for n in [1, 7, 160, 33, 299, 500]:   # odd chunk sizes on purpose
        pieces.append(f(x[:, i:i + n]))
        i += n
    assert np.allclose(np.concatenate(pieces, axis=1), filter_causal(x, SFREQ))


def test_causal_filter_does_not_look_ahead():
    # changing the future must not change the past
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 800))
    y1 = filter_causal(x, SFREQ)
    x2 = x.copy()
    x2[:, 400:] = rng.normal(size=(1, 400))
    y2 = filter_causal(x2, SFREQ)
    assert np.allclose(y1[:, :400], y2[:, :400])


def test_causal_close_to_offline_in_power():
    # same band power as the S2 zero-phase filter (the phase differs)
    x = np.random.default_rng(2).normal(size=(2, 3200))
    on = filter_causal(x, SFREQ)[:, 800:]
    off = filter_signal(x, SFREQ)[:, 800:]
    assert np.allclose(on.std(axis=1), off.std(axis=1), rtol=0.1)


def test_group_delay_is_small():
    d = group_delay_ms(SFREQ)
    assert all(0 < v < 150 for v in d.values())


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
CLASSES5 = list(config.CLASSES)


def _p(label, p=0.8):
    v = np.full(5, (1 - p) / 4)
    v[CLASSES5.index(label)] = p
    return v


def test_threshold_low_confidence_is_rest():
    d = DecisionMaker(CLASSES5, threshold=0.4, n_vote=1)
    assert d.update(_p("main_gauche", 0.35))[0] == "repos"
    assert d.update(_p("main_gauche", 0.45))[0] == "main_gauche"


def test_vote_needs_2_of_3():
    d = DecisionMaker(CLASSES5, threshold=0.4, n_vote=3)
    assert d.update(_p("main_droite"))[0] == "main_droite"   # 1/1
    assert d.update(_p("deux_pieds"))[0] == "main_droite"         # 1-1: keep
    assert d.update(_p("deux_pieds"))[0] == "deux_pieds"               # 2/3


def test_single_noisy_window_is_ignored():
    d = DecisionMaker(CLASSES5, threshold=0.4, n_vote=3)
    for _ in range(3):
        d.update(_p("main_gauche"))
    assert d.update(_p("deux_mains"))[0] == "main_gauche"     # 2 vs 1
    assert d.update(_p("main_gauche"))[0] == "main_gauche"


def test_no_majority_keeps_previous():
    d = DecisionMaker(CLASSES5, threshold=0.4, n_vote=3)
    d.update(_p("deux_pieds"))
    d.update(_p("deux_pieds"))                  # decision: pieds
    d.update(_p("main_gauche"))            # p / p / g  -> still pieds
    d.update(_p("main_droite"))            # p / g / d  -> no majority
    assert d.decision == "deux_pieds"


def test_confidence_range_and_two_class_rest():
    d = DecisionMaker(["main_gauche", "main_droite"], threshold=0.6, n_vote=1)
    c, conf = d.update(np.array([0.55, 0.45]))
    assert c == "repos" and 0 <= conf <= 1      # repos not in a 2-class model
    c, conf = d.update(np.array([0.9, 0.1]))
    assert c == "main_gauche" and conf == pytest.approx(0.9)


def test_marker_to_class():
    assert marker_to_class("deux_pieds") == "deux_pieds"
    assert marker_to_class(["main_gauche"]) == "main_gauche"
    assert marker_to_class("2") == "main_droite"
    assert marker_to_class("T1") is None


# ---------------------------------------------------------------------------
# Online classifier on a synthetic stream
# ---------------------------------------------------------------------------
def _synthetic_recording(seconds=120, seed=0):
    """64-ch 'EEG' alternating repos / main_gauche / main_droite every 4 s.
    main_gauche: more 12 Hz power on C4, main_droite: on C3."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SFREQ)
    data = rng.normal(0, 5e-6, (64, n))
    t = np.arange(n) / SFREQ
    seq = ["repos", "main_gauche", "repos", "main_droite"]
    markers = []
    for k, onset in enumerate(np.arange(0, seconds, 4.0)):
        label = seq[k % 4]
        markers.append((onset, label))
        a, b = int(onset * SFREQ), int((onset + 4) * SFREQ)
        ch = {"main_gauche": "C4", "main_droite": "C3"}.get(label)
        if ch:
            i = config.CHANNELS_ALL.index(ch)
            data[i, a:b] += 15e-6 * np.sin(2 * np.pi * 12 * t[a:b])
    return data, markers


def _train_bundle(classes=("repos", "main_gauche", "main_droite")):
    data, markers = _synthetic_recording(seed=1)
    picks = [config.CHANNELS_ALL.index(c) for c in CH_B]
    filt = filter_causal(data[picks], SFREQ)
    events = np.array([[int(o * SFREQ), 0, config.EVENT_ID[m]]
                       for o, m in markers])
    X, y = causal_windows(filt, SFREQ, events, list(classes))
    model = make_model("riemann", len(CH_B)).fit(X, y)
    return {"model": model, "ch_names": CH_B, "sfreq": SFREQ,
            "classes": list(classes), "model_name": "riemann", "config": "B"}


@pytest.fixture(scope="module")
def bundle():
    return _train_bundle()


def test_causal_windows_shapes():
    data = np.zeros((3, int(60 * SFREQ)))
    events = np.array([[160 * 5, 0, 1], [160 * 15, 0, 2], [160 * 25, 0, 0],
                       [160 * 35, 0, 0], [160 * 45, 0, 0]])
    X, y = causal_windows(data, SFREQ, events,
                          ["repos", "main_gauche", "main_droite"])
    n_starts = len(config.ONLINE_TRAIN_STARTS)
    assert X.shape[1:] == (3, int(config.ONLINE_WINDOW_S * SFREQ))
    # repos is balanced down to 1 trial (as many as the biggest movement)
    assert np.sum(y == 0) == n_starts
    assert len(y) == 3 * n_starts


def _run(bundle, data, markers, chunk_s=0.0625, stream_names=None):
    names = stream_names or config.CHANNELS_ALL
    src = ReplaySource(data, SFREQ, names, markers, chunk_s=chunk_s, speed=0)
    clf = OnlineClassifier(bundle, names, SFREQ)
    out = []
    for samples, ts, mk in src:
        for t, lab in mk:
            clf.push_marker(t, lab)
        out += clf.push(samples, ts)
    return out, clf


def test_four_decisions_per_second(bundle):
    data, markers = _synthetic_recording(seconds=30)
    msgs, _ = _run(bundle, data, markers)
    n = data.shape[1]
    win, step = int(config.ONLINE_WINDOW_S * SFREQ), int(config.ONLINE_STEP_S * SFREQ)
    assert len(msgs) == (n - win) // step + 1
    dt = np.diff([m["t"] for m in msgs])
    assert np.allclose(dt, config.ONLINE_STEP_S, atol=1e-6)


def test_message_matches_front_contract(bundle):
    data, markers = _synthetic_recording(seconds=10)
    msgs, _ = _run(bundle, data, markers)
    for m in msgs:
        assert set(m) == {"numero", "classe", "confiance", "vraie_classe", "t"}
        assert m["classe"] in config.CLASSES
        assert m["vraie_classe"] in config.CLASSES
        assert 0.0 <= m["confiance"] <= 1.0
        json.dumps(m)   # serialisable as-is


def test_same_output_whatever_the_chunk_size(bundle):
    data, markers = _synthetic_recording(seconds=20)
    a, _ = _run(bundle, data, markers, chunk_s=1 / SFREQ)   # sample by sample
    b, _ = _run(bundle, data, markers, chunk_s=0.37)         # odd chunks
    assert a == b


def test_channel_order_of_stream_does_not_matter(bundle):
    data, markers = _synthetic_recording(seconds=20)
    perm = np.random.default_rng(3).permutation(64)
    names = [config.CHANNELS_ALL[i] for i in perm]
    a, _ = _run(bundle, data, markers)
    b, _ = _run(bundle, data[perm], markers, stream_names=names)
    assert a == b


def test_online_decisions_follow_true_class(bundle):
    data, markers = _synthetic_recording(seconds=120, seed=5)  # unseen data
    msgs, clf = _run(bundle, data, markers)
    ok = np.mean([m["classe"] == m["vraie_classe"] for m in msgs])
    assert ok > 0.7     # includes the ~0.5-1 s lag after every change
    assert np.mean(clf.proc_ms) < config.ONLINE_STEP_S * 1000


def test_rejects_wrong_sfreq_or_missing_channels(bundle):
    with pytest.raises(ValueError):
        OnlineClassifier(bundle, config.CHANNELS_ALL, 128.0)
    with pytest.raises(ValueError):
        OnlineClassifier(bundle, ["C3", "Cz", "C4"], SFREQ)


# ---------------------------------------------------------------------------
# Real LSL chain: outlet (fake player) -> LSLSource -> classifier
# ---------------------------------------------------------------------------
def test_lsl_chain(bundle):
    pylsl = pytest.importorskip("pylsl")
    from handmind.online.sources import LSLSource

    stype, mtype = "EEG_test_hm", "Markers_test_hm"
    info = pylsl.StreamInfo("hm-test-eeg", stype, 64, SFREQ, "float32", "hmtest")
    chans = info.desc().append_child("channels")
    for c in config.CHANNELS_ALL:
        chans.append_child("channel").append_child_value("label", c)
    eeg_out = pylsl.StreamOutlet(info)
    mk_out = pylsl.StreamOutlet(pylsl.StreamInfo(
        "hm-test-mk", mtype, 1, 0, "string", "hmtestmk"))

    data, markers = _synthetic_recording(seconds=6)

    def player():
        time.sleep(1.0)  # let the inlet connect
        t0 = pylsl.local_clock()
        mk = list(markers)
        for i in range(0, data.shape[1], 10):
            ts = t0 + i / SFREQ
            while mk and mk[0][0] <= i / SFREQ:
                onset, label = mk.pop(0)
                mk_out.push_sample([label], t0 + onset)
            eeg_out.push_chunk(data[:, i:i + 10].T.tolist(), ts)
            time.sleep(10 / SFREQ / 4)  # 4x real time

    src = LSLSource(eeg_type=stype, marker_type=mtype, timeout=5)
    assert src.ch_names == config.CHANNELS_ALL and src.sfreq == SFREQ
    clf = OnlineClassifier(bundle, src.ch_names, src.sfreq)
    th = threading.Thread(target=player, daemon=True)
    th.start()
    msgs, deadline = [], time.time() + 15
    for samples, ts, mk in src:
        for t, lab in mk:
            clf.push_marker(t, lab)
        if len(ts):
            msgs += clf.push(samples, ts)
        if clf.n_seen >= data.shape[1] or time.time() > deadline:
            break
    assert clf.n_seen == data.shape[1]
    assert len(msgs) == (data.shape[1] - 160) // 40 + 1
    assert {m["vraie_classe"] for m in msgs} == {"repos", "main_gauche"}
