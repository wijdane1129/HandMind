"""S5 - Run the online classifier.

Reads the EEG stream, makes one decision every 250 ms and outputs the JSON
message of the front contract:
    {"classe": "main_gauche", "confiance": 0.81, "vraie_classe": "main_gauche", "t": 12.75}

Usage (from backend/):
    # 0. once: train the online model (S4 choice, 5 classes, 20 train subjects)
    python -m scripts.train --online

    # 1. with the dataset player of Etudiant B (LSL) - the real chain
    python -m scripts.run_online

    # 2. without the player: replay a demo subject directly
    python -m scripts.run_online --replay 42 --speed 1
    python -m scripts.run_online --replay 42 --speed 0 --duration 120   # fast, for scores

    # 3. drive the Godot avatar (WebSocket server on ws://127.0.0.1:8765)
    python -m scripts.run_online --replay 42 --ws

Options: --quiet (no JSON lines), --log FILE.csv (all decisions + timing).
At the end: decision accuracy vs the true class, and processing time.

Note: --ws is a minimal server for testing the chain end to end. If
Etudiant B's server (handmind/server) is used instead, it can import
OnlineClassifier and call .push() itself - same messages.
"""

import argparse
import asyncio
import csv
import json
from collections import Counter

import joblib
import numpy as np

from handmind import config
from handmind.online.classifier import OnlineClassifier
from handmind.online.filters import group_delay_ms
from handmind.online.sources import LSLSource, ReplaySource
from scripts.train import online_model_path


def make_source(args):
    if args.replay is not None:
        print(f"Source: replay of S{args.replay:03d} (speed x{args.speed or 'max'})")
        return ReplaySource.from_physionet(args.replay, speed=args.speed,
                                           loop=args.loop)
    print("Source: LSL (waiting for the dataset player...)")
    return LSLSource()


class Run:
    """Glue: source -> classifier -> outputs (stdout, csv, websocket)."""

    def __init__(self, args):
        path = args.model_file or online_model_path()
        if not path.exists():
            raise SystemExit(f"{path} not found: run `python -m scripts.train "
                             "--online` first.")
        self.bundle = joblib.load(path)
        print(f"Model: {path.name}  ({self.bundle['model_name']}, config "
              f"{self.bundle['config']}, classes {self.bundle['classes']})")
        self.source = make_source(args)
        self.clf = OnlineClassifier(self.bundle, self.source.ch_names,
                                    self.source.sfreq,
                                    threshold=args.threshold, n_vote=args.vote)
        self.args = args
        self.messages = []
        self.clients = set()
        self.writer = None
        if args.log:
            f = open(args.log, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(f)
            self.writer.writerow(["t", "classe", "confiance", "vraie_classe",
                                  "proc_ms"])

    def handle(self, msg):
        self.messages.append(msg)
        if not self.args.quiet:
            print(json.dumps(msg, ensure_ascii=False))
        if self.writer:
            self.writer.writerow([msg["t"], msg["classe"], msg["confiance"],
                                  msg["vraie_classe"],
                                  round(self.clf.proc_ms[-1], 2)])

    async def broadcast(self, msg):
        text = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send(text)
            except Exception:
                self.clients.discard(ws)

    async def ws_handler(self, ws, *_):
        self.clients.add(ws)
        print(f"[WS] client connected ({len(self.clients)})")
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    async def loop(self):
        for samples, ts, markers in self.source:
            for t, label in markers:
                self.clf.push_marker(t, label)
            for msg in self.clf.push(samples, ts) if len(ts) else []:
                self.handle(msg)
                if self.clients:
                    await self.broadcast(msg)
            await asyncio.sleep(0)   # let the websocket server breathe
            if self.args.duration and self.clf.last_ts is not None and \
                    self.clf.last_ts - self.clf.t0 >= self.args.duration:
                break

    async def main(self):
        if self.args.ws:
            import websockets
            async with websockets.serve(self.ws_handler, config.WS_HOST,
                                        config.WS_PORT):
                print(f"[WS] server on ws://{config.WS_HOST}:{config.WS_PORT}"
                      " - start the Godot scene (it reconnects every 2 s)")
                await self.loop()
        else:
            await self.loop()

    def summary(self):
        if not self.messages:
            print("No decision made.")
            return
        m = self.messages
        ok = np.mean([x["classe"] == x["vraie_classe"] for x in m])
        print(f"\n{len(m)} decisions over {m[-1]['t']:.1f} s "
              f"({len(m) / max(m[-1]['t'], 1e-9):.2f} per s)")
        print(f"Decision = true class: {ok:.1%} of decisions")
        print("Decisions per class:", dict(Counter(x["classe"] for x in m)))
        p = np.array(self.clf.proc_ms)
        gd = group_delay_ms(self.clf.sfreq)
        print(f"Processing time per decision: mean {p.mean():.1f} ms, "
              f"max {p.max():.1f} ms (budget: {config.ONLINE_STEP_S * 1000:.0f} ms)")
        print("Causal filter delay: "
              + ", ".join(f"{d:.0f} ms at {f:.0f} Hz" for f, d in gd.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=int, default=None, metavar="SUBJECT",
                        help="replay a PhysioNet subject instead of LSL")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay speed: 1 = real time, 5, 0 = max")
    parser.add_argument("--loop", action="store_true", help="replay in a loop")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds of signal")
    parser.add_argument("--model-file", type=lambda p: __import__("pathlib").Path(p),
                        default=None)
    parser.add_argument("--threshold", type=float, default=config.ONLINE_THRESHOLD)
    parser.add_argument("--vote", type=int, default=config.ONLINE_N_VOTE)
    parser.add_argument("--ws", action="store_true",
                        help=f"serve decisions on ws://{config.WS_HOST}:{config.WS_PORT}")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log", default=None, help="CSV file of all decisions")
    args = parser.parse_args()

    run = Run(args)
    try:
        asyncio.run(run.main())
    except KeyboardInterrupt:
        pass
    run.summary()


if __name__ == "__main__":
    main()
