"""S6 tests - integration of the real classifier with Etudiant B's chain.

No download: synthetic EEG. The last test runs the REAL chain through LSL:
B's player functions (reader/lecteur.py) -> pont_classifieur -> WebSocket
client standing in for Godot.
"""

import asyncio
import json
import os
import sys
import threading
import time

import joblib
import numpy as np
import pytest

from handmind import config
from handmind.online.classifier import OnlineClassifier, marker_to_class
from handmind.online.sources import LSLSource, ReplaySource
from scripts.eval_online import _scored, evaluate_recording, latency_summary
from tests.test_online import SFREQ, _synthetic_recording, _train_bundle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reader"))
import lecteur  # noqa: E402  (Etudiant B)

CONTRACT = set(config.CLASSES)


@pytest.fixture(scope="module")
def bundle():
    return _train_bundle()


def _run(bundle, data, markers, chunk_s=0.0625, **kw):
    src = ReplaySource(data, SFREQ, config.CHANNELS_ALL, markers,
                       chunk_s=chunk_s, speed=0)
    clf = OnlineClassifier(bundle, config.CHANNELS_ALL, SFREQ, **kw)
    out = []
    for samples, ts, mk in src:
        for t, lab in mk:
            clf.push_marker(t, lab)
        out += clf.push(samples, ts)
    return out, clf


# --- class names ------------------------------------------------------------
def test_backend_classes_match_player_and_contract():
    """Same 5 names in config, B's player and the JSON contract."""
    player = {lecteur.code_vers_classe(c, r) for r in (4, 6) for c in ("T0", "T1", "T2")}
    assert set(config.CLASSES) == player
    assert config.CLASSES[4] == "deux_pieds" and config.EVENT_ID["deux_pieds"] == 4


def test_old_name_pieds_still_accepted():
    assert marker_to_class("pieds") == "deux_pieds"
    assert marker_to_class("deux_pieds") == "deux_pieds"
    assert marker_to_class("NOUVEAU_RUN:4") is None
    assert marker_to_class("FIN:1200") is None


# --- units --------------------------------------------------------------------
def test_unit_scale_from_stream_description():
    assert LSLSource.unit_scale("microvolts") == 1e-6
    assert LSLSource.unit_scale("volts") == 1.0
    with pytest.raises(RuntimeError):
        LSLSource.unit_scale("furlongs")


def test_microvolt_stream_with_scale_gives_same_output(bundle):
    data, markers = _synthetic_recording(seconds=20)
    volts, _ = _run(bundle, data, markers)
    micro, _ = _run(bundle, data * 1e6, markers, scale=1e-6)
    assert [m["classe"] for m in micro] == [m["classe"] for m in volts]
    assert np.allclose([m["confiance"] for m in micro],
                       [m["confiance"] for m in volts], atol=2e-3)


def test_microvolt_stream_without_scale_breaks_the_model(bundle):
    """The bug S6 fixes: B's player sends uV, the model was trained in V."""
    data, markers = _synthetic_recording(seconds=30)
    volts, c1 = _run(bundle, data, markers)
    micro, c2 = _run(bundle, data * 1e6, markers)          # no conversion
    assert c1.raw_labels != c2.raw_labels


# --- message fields -----------------------------------------------------------
def test_numero_and_unknown_true_class(bundle):
    data, markers = _synthetic_recording(seconds=12)
    markers = [(t + 2.0, m) for t, m in markers]            # first cue at 2 s
    msgs, _ = _run(bundle, data, markers)
    assert [m["numero"] for m in msgs] == list(range(len(msgs)))
    assert msgs[0]["vraie_classe"] == ""                    # before any marker
    assert all(m["vraie_classe"] in CONTRACT | {""} for m in msgs)
    assert all(m["classe"] in CONTRACT for m in msgs)


def test_t_is_unix_time_with_to_t(bundle):
    data, markers = _synthetic_recording(seconds=5)
    base = 1_790_000_000.0
    msgs, clf = _run(bundle, data, markers, to_t=lambda ts: base + ts)
    assert all(m["t"] > 1e9 for m in msgs)                  # Godot measures T13
    # t = timestamp of the LAST sample of the window
    win = int(config.ONLINE_WINDOW_S * SFREQ)
    assert msgs[0]["t"] == pytest.approx(base + (win - 1) / SFREQ)


# --- technical markers of the player ------------------------------------------
def test_new_run_marker_restarts_like_a_fresh_classifier(bundle):
    d1, m1 = _synthetic_recording(seconds=12, seed=4)
    d2, m2 = _synthetic_recording(seconds=12, seed=5)
    cut = d1.shape[1] / SFREQ
    data = np.concatenate([d1, d2], axis=1)
    markers = m1 + [(cut, "NOUVEAU_RUN:8")] + [(t + cut, m) for t, m in m2]
    joined, clf = _run(bundle, data, markers, chunk_s=0.37)
    fresh, _ = _run(bundle, d2, m2)
    after = [m for m, ts in zip(joined, clf.decision_ts) if ts >= cut]
    assert [m["classe"] for m in after] == [m["classe"] for m in fresh]
    assert [m["confiance"] for m in after] == [m["confiance"] for m in fresh]


def test_end_marker(bundle):
    data, markers = _synthetic_recording(seconds=6)
    _, clf = _run(bundle, data, markers + [(5.0, "FIN:960")])
    assert clf.ended


# --- T15 / T12 ----------------------------------------------------------------
def test_scored_window_region():
    cues = [(10.0, "main_gauche")]
    win_last = lambda start: start + (SFREQ - 1) / SFREQ
    assert _scored(win_last(10.5), cues, SFREQ) == "main_gauche"
    assert _scored(win_last(10.25), cues, SFREQ) is None    # starts too early
    assert _scored(win_last(13.0), cues, SFREQ) == "main_gauche"
    assert _scored(win_last(13.25), cues, SFREQ) is None    # ends after 4 s


def test_t15_offline_and_online_agree_on_synthetic(bundle):
    data, markers = _synthetic_recording(seconds=120, seed=7)
    r, lat = evaluate_recording(bundle, data, SFREQ, config.CHANNELS_ALL, markers)
    assert r["n_windows_online"] > 50 and r["n_windows_offline"] > 50
    assert r["offline"] > 0.8                    # strong synthetic pattern
    assert r["gap"] <= config.T15_MAX_GAP
    s = latency_summary(lat)
    assert s["t12_ok"], s


# --- full chain through LSL ---------------------------------------------------
def test_real_chain_player_to_bridge_to_websocket(bundle, tmp_path):
    """B's player (uV, NOUVEAU_RUN, deux_pieds, FIN) -> pont_classifieur -> WS."""
    pylsl = pytest.importorskip("pylsl")
    websockets = pytest.importorskip("websockets")
    from handmind.server import pont_classifieur as pont

    # player timeline: 2 runs of synthetic EEG with the REAL channel names, in uV
    d1, m1 = _synthetic_recording(seconds=8, seed=11)
    d2, _ = _synthetic_recording(seconds=8, seed=12)
    signal = (np.concatenate([d1, d2], axis=1).T * 1e6).astype(np.float32)
    marqueurs = {0: ["NOUVEAU_RUN:4"]}
    for t, m in m1:
        marqueurs.setdefault(int(t * SFREQ), []).append(m)
    n1 = d1.shape[1]
    marqueurs.setdefault(n1, []).append("NOUVEAU_RUN:6")
    marqueurs.setdefault(n1 + int(2 * SFREQ), []).append("deux_pieds")

    out_eeg, out_m = lecteur.creer_flux(config.CHANNELS_ALL, 1, [4, 6], False)
    model_file = tmp_path / "m.joblib"
    joblib.dump(bundle, model_file)
    port = 8791
    args = pont.lire_arguments(["--modele", str(model_file), "--port", str(port),
                                "--une-seance", "--silencieux", "--duree", "60"])

    def play():
        time.sleep(1.0)
        lecteur.diffuser(signal, marqueurs, out_eeg, out_m, vitesse=4.0)

    received = []

    async def client():
        for _ in range(50):
            try:
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    async for text in ws:
                        received.append(json.loads(text))
            except (OSError, websockets.ConnectionClosed):
                if received:
                    return
                await asyncio.sleep(0.1)

    async def scenario():
        async with websockets.serve(pont.handler, "localhost", port):
            task = asyncio.create_task(client())
            threading.Thread(target=play, daemon=True).start()
            await pont.boucle(bundle, args)
            await asyncio.sleep(0.2)
            task.cancel()

    asyncio.run(scenario())

    preds = [m for m in received if "type" not in m]
    states = [m for m in received if m.get("type") == "etat"]
    assert states and any(s["flux"] for s in states)
    # 16 s at x4 speed = 4 s real -> ~ (16 - 1) * 4 - 4 decisions (1 reset)
    assert len(preds) > 40
    assert all(set(p) == {"numero", "classe", "confiance", "vraie_classe", "t"}
               for p in preds)
    assert all(p["classe"] in CONTRACT for p in preds)
    assert {p["vraie_classe"] for p in preds} <= CONTRACT | {""}
    assert "deux_pieds" in {p["vraie_classe"] for p in preds}
    assert all(abs(p["t"] - time.time()) < 120 for p in preds)   # Unix time
    nums = [p["numero"] for p in preds]
    assert nums == sorted(nums) and len(set(nums)) == len(nums)


def test_model_saved_before_s6_with_old_class_name(bundle):
    old = dict(bundle, classes=[c if c != "deux_pieds" else "pieds"
                                for c in bundle["classes"]] + ["pieds"])
    data, markers = _synthetic_recording(seconds=30)
    r, _ = evaluate_recording(old, data, SFREQ, config.CHANNELS_ALL, markers)
    assert r["n_windows_online"] > 0


def test_session_without_eeg_times_out(bundle):
    """Bridge reconnected to a closing player: no EEG ever -> session dropped."""
    from types import SimpleNamespace
    from handmind.server.pont_classifieur import Seance
    src = SimpleNamespace(ch_names=config.CHANNELS_ALL, sfreq=SFREQ, scale=1.0,
                          to_unix=lambda t: t)
    args = SimpleNamespace(threshold=0.4, vote=3, log=None, envoi="tout")
    s = Seance(src, bundle, args)
    assert not s.terminee(3.0)
    s.debut -= 10.0
    assert s.terminee(3.0)


def test_fin_waits_for_the_eeg_sent_before_it(bundle):
    """FIN read before the last EEG chunks: the session must not stop early."""
    data, markers = _synthetic_recording(seconds=6)
    clf = OnlineClassifier(bundle, config.CHANNELS_ALL, SFREQ)
    ts = np.arange(data.shape[1]) / SFREQ
    clf.push_marker(ts[-1] + 1 / SFREQ, "FIN:960")      # marker arrives first
    half = data.shape[1] // 2
    clf.push(data[:, :half].T, ts[:half])
    assert not clf.ended                                 # EEG still pending
    clf.push(data[:, half:].T, ts[half:])
    assert clf.ended
