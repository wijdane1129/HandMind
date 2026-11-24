"""S6 - Tests T15 (online accuracy) and T12 (processing latency).

For each demo subject (never used for training), the SAME online model is
scored twice on the SAME windows:

  offline : 1 s windows cut inside each trial (0.5-4 s after the cue), as in
            training -> model.predict
  online  : the whole recording replayed through OnlineClassifier, chunk by
            chunk (causal filter with memory, ring buffer, one window every
            250 ms). Only the windows lying fully inside 0.5-4 s after a cue
            are scored, i.e. the same region as offline.

  T15 passes if |offline - online| <= 5 points (cahier, section 6).
  Two online scores are given:
    online_raw      argmax of each window (comparable to offline -> T15)
    online_decision after threshold + vote on 3 (what the avatar does)

T12: for every decision, time from "chunk received" to "decision ready"
(filter + model + vote). Criterion: median < 200 ms, p95 < 300 ms.

Usage (from backend/):
    python -m scripts.eval_online                      # 3 first demo subjects
    python -m scripts.eval_online --subjects 42 57 88
    python -m scripts.eval_online --all                # the 10 demo subjects (S9)
    python -m scripts.eval_online --threshold 0.35 --vote 5
-> docs/results/s6_t15.csv / .md and docs/results/s6_t12.md
"""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import numpy as np
import pandas as pd

from handmind import config
from handmind.online.classifier import OnlineClassifier
from handmind.online.filters import filter_causal
from handmind.online.sources import ReplaySource
from scripts.train import causal_windows, online_model_path


def _scored(t_end, cues, sfreq, window_s=config.ONLINE_WINDOW_S,
            tmin=config.TMIN, tmax=config.TMAX):
    """True class of the window whose LAST sample is at t_end, if the window
    lies fully inside [cue + tmin, cue + tmax] of a cue, else None."""
    n = int(round(window_s * sfreq))
    t_start = t_end - (n - 1) / sfreq      # time of its first sample
    tol = 0.5 / sfreq
    for onset, name in cues:
        if onset + tmin - tol <= t_start and t_end <= onset + tmax + tol:
            return name
    return None


def evaluate_recording(bundle, data, sfreq, ch_names, cues,
                       threshold=config.ONLINE_THRESHOLD,
                       n_vote=config.ONLINE_N_VOTE):
    """Offline vs online on one continuous recording.

    data : (n_channels, n_times) in volts, channels named `ch_names`.
    cues : [(onset_s, class_name)] - all class markers of the recording.
    Returns a dict of scores + the list of latencies (ms).
    """
    # models trained before S6 call class 4 "pieds"
    classes = [config.CLASS_ALIASES.get(c, c) for c in bundle["classes"]]
    cues = [(float(t), c) for t, c in cues if c in classes]

    # --- offline: same model, windows cut from the causally filtered signal
    picks = [list(ch_names).index(c) for c in bundle["ch_names"]]
    filt = filter_causal(data[picks], sfreq)
    events = np.array([[int(round(t * sfreq)), 0, config.EVENT_ID[c]]
                       for t, c in cues], dtype=int).reshape(-1, 3)
    X, y = causal_windows(filt, sfreq, events, classes, balance=False)
    y_off = bundle["model"].predict(X) if len(y) else np.array([])
    acc_off = float(np.mean(y_off == y)) if len(y) else np.nan
    names = np.array(config.CLASSES)

    # --- online: replay chunk by chunk
    clf = OnlineClassifier(bundle, ch_names, sfreq, threshold=threshold,
                           n_vote=n_vote)
    src = ReplaySource(data, sfreq, ch_names, cues, speed=0)
    msgs = []
    for samples, ts, mk in src:
        for t, lab in mk:
            clf.push_marker(t, lab)
        msgs += clf.push(samples, ts)

    truth_l, raw_l, dec_l = [], [], []
    for msg, raw_label, ts in zip(msgs, clf.raw_labels, clf.decision_ts):
        truth = _scored(ts, cues, sfreq)
        if truth is not None:
            truth_l.append(truth)
            raw_l.append(raw_label)
            dec_l.append(msg["classe"])
    raw_ok = [a == b for a, b in zip(raw_l, truth_l)]
    dec_ok = [a == b for a, b in zip(dec_l, truth_l)]

    def bal(t, p):
        # balanced accuracy = mean recall per class: chance = 1/k whatever
        # the class proportions ("repos" is ~half of the trials)
        return float(balanced_accuracy_score(t, p)) if len(t) else np.nan

    acc_raw = float(np.mean(raw_ok)) if raw_ok else np.nan
    return {
        "rest_share": float(np.mean(np.array(truth_l) == "repos")) if truth_l else np.nan,
        "offline_bal": bal(names[y], names[y_off]) if len(y) else np.nan,
        "online_raw_bal": bal(truth_l, raw_l),
        "online_decision_bal": bal(truth_l, dec_l),
        "online_decision_kappa": float(cohen_kappa_score(truth_l, dec_l)) if truth_l else np.nan,
        "n_windows_offline": len(y),
        "n_windows_online": len(raw_ok),
        "offline": acc_off,
        "online_raw": acc_raw,
        "online_decision": float(np.mean(dec_ok)) if dec_ok else np.nan,
        "gap": abs(acc_off - acc_raw),
        "t15_ok": bool(abs(acc_off - acc_raw) <= config.T15_MAX_GAP),
    }, list(clf.latency_ms)


def evaluate_subject(bundle, subject, **kwargs):
    from handmind.data import load_physionet_raw
    raw = load_physionet_raw(subject)
    eeg = raw.copy().pick("eeg")
    cues = [(a["onset"] - raw.first_time, a["description"])
            for a in raw.annotations if a["description"] in config.EVENT_ID]
    return evaluate_recording(bundle, eeg.get_data(), raw.info["sfreq"],
                              eeg.ch_names, cues, **kwargs)


def latency_summary(lat):
    lat = np.asarray(lat)
    med, p95 = float(np.median(lat)), float(np.percentile(lat, 95))
    return {"n": len(lat), "median_ms": med, "p95_ms": p95,
            "max_ms": float(lat.max()),
            "t12_ok": med < config.T12_MEDIAN_MS and p95 < config.T12_P95_MS}


def write_reports(df, lat, bundle, args):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "s6_t15.csv", index=False)
    k = len(bundle["classes"])
    mean = df[["offline", "online_raw", "online_decision", "gap"]].mean()
    lines = [
        "# S6 - T15 : précision en ligne vs hors ligne",
        "",
        f"Modèle `{bundle['model_name']}`, config {bundle['config']}, {k} classes "
        f"({', '.join(bundle['classes'])}), entraîné sur "
        f"{len(bundle.get('train_subjects', []))} sujets d'entraînement.",
        f"Seuil {args.threshold}, vote sur {args.vote}. Hasard : {1 / k:.2f}.",
        "",
        "Même modèle, fenêtres de 1 s dans la même zone (0,5-4 s après la consigne). "
        "Critère T15 : écart hors ligne / en ligne (brut) ≤ 5 points.",
        "",
        "| Sujet | Fenêtres | Hors ligne | En ligne (brut) | En ligne (décision) | Écart | T15 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| S{int(r.subject):03d} | {int(r.n_windows_online)} | "
                     f"{r.offline:.3f} | {r.online_raw:.3f} | "
                     f"{r.online_decision:.3f} | {100 * r.gap:.1f} pts | "
                     f"{'OK' if r.t15_ok else 'ÉCHEC'} |")
    lines.append(f"| **Moyenne** | | {mean.offline:.3f} | {mean.online_raw:.3f} | "
                 f"{mean.online_decision:.3f} | {100 * mean.gap:.1f} pts | "
                 f"{'OK' if df.t15_ok.all() else 'ÉCHEC'} |")
    bm = df[["offline_bal", "online_raw_bal", "online_decision_bal",
             "online_decision_kappa", "rest_share"]].mean()
    lines += ["", "## Précision équilibrée (moyenne des rappels par classe)", "",
              f"« repos » représente {100 * bm.rest_share:.0f} % des fenêtres notées : "
              "un modèle qui répond toujours « repos » aurait déjà "
              f"{100 * bm.rest_share:.0f} % de précision simple. La précision "
              f"équilibrée corrige cela : hasard = {1 / k:.2f} quelles que soient "
              "les proportions.", "",
              "| Sujet | Hors ligne | En ligne (brut) | En ligne (décision) | Kappa (décision) |",
              "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| S{int(r.subject):03d} | {r.offline_bal:.3f} | "
                     f"{r.online_raw_bal:.3f} | {r.online_decision_bal:.3f} | "
                     f"{r.online_decision_kappa:.3f} |")
    lines.append(f"| **Moyenne** | {bm.offline_bal:.3f} | {bm.online_raw_bal:.3f} | "
                 f"{bm.online_decision_bal:.3f} | {bm.online_decision_kappa:.3f} |")
    lines += ["", "- *brut* : classe la plus probable de chaque fenêtre ; "
              "*décision* : après seuil de confiance et vote majoritaire "
              "(ce que joue l'avatar).",
              "- Les sujets de démo ne sont jamais vus à l'entraînement : "
              "c'est une précision **inter-sujets** (sans calibration, voir S7)."]
    (config.RESULTS_DIR / "s6_t15.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")

    s = latency_summary(lat)
    (config.RESULTS_DIR / "s6_t12.md").write_text("\n".join([
        "# S6 - T12 : temps de traitement (fenêtre reçue -> décision)",
        "",
        f"{s['n']} décisions sur {len(df)} sujets, modèle `{bundle['model_name']}` "
        f"config {bundle['config']} ({len(bundle['ch_names'])} canaux).",
        "Mesure : du moment où le morceau de signal arrive au classifieur "
        "jusqu'à la décision prête (filtre causal + modèle + vote).",
        "",
        "| Médiane | 95e centile | Max | Critère | Résultat |",
        "|---|---|---|---|---|",
        f"| {s['median_ms']:.1f} ms | {s['p95_ms']:.1f} ms | {s['max_ms']:.1f} ms | "
        f"< {config.T12_MEDIAN_MS:.0f} ms / < {config.T12_P95_MS:.0f} ms | "
        f"{'OK' if s['t12_ok'] else 'ÉCHEC'} |",
        "",
        "Hors de cette mesure : la fenêtre elle-même (1 s de signal à attendre), "
        "le retard du filtre causal (~30-80 ms) et le vote (jusqu'à 0,5 s). "
        "Le trajet pont -> Godot est mesuré par l'Étudiant B (T13).",
    ]) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="the 10 demo subjects")
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--threshold", type=float, default=config.ONLINE_THRESHOLD)
    parser.add_argument("--vote", type=int, default=config.ONLINE_N_VOTE)
    args = parser.parse_args()

    if args.subjects is None:
        from handmind.split import demo_subjects
        demo = demo_subjects()
        args.subjects = demo if args.all else demo[:config.N_T15_SUBJECTS]
    path = Path(args.model_file) if args.model_file else online_model_path()
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m scripts.train --online` first.")
    bundle = joblib.load(path)
    train = set(bundle.get("train_subjects", []))
    leak = train.intersection(args.subjects)
    if leak:
        raise SystemExit(f"Subjects {sorted(leak)} were used to train {path}: "
                         "T15 must use unseen (demo) subjects.")
    print(f"Model {path.name}, subjects {args.subjects}")

    rows, lat = [], []
    for s in args.subjects:
        r, l = evaluate_subject(bundle, s, threshold=args.threshold,
                                n_vote=args.vote)
        rows.append({"subject": s, **r})
        lat += l
        print(f"  S{s:03d}: offline {r['offline']:.3f}  online raw "
              f"{r['online_raw']:.3f}  decision {r['online_decision']:.3f}  "
              f"gap {100 * r['gap']:.1f} pts -> {'OK' if r['t15_ok'] else 'FAIL'}"
              f"  | balanced: raw {r['online_raw_bal']:.3f}  decision "
              f"{r['online_decision_bal']:.3f}  (rest {100 * r['rest_share']:.0f} %)")
    df = pd.DataFrame(rows)
    write_reports(df, lat, bundle, args)
    s = latency_summary(lat)
    print(f"T12: median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms "
          f"-> {'OK' if s['t12_ok'] else 'FAIL'}")
    print(f"-> {config.RESULTS_DIR / 's6_t15.md'} and s6_t12.md")


if __name__ == "__main__":
    main()
