"""
HandMind - Semaine 6 (Étudiant A)
Pont RÉEL : flux LSL du lecteur -> classifieur en ligne -> WebSocket -> Godot.

C'est le pont_lsl.py de l'Étudiant B (S5) avec le vrai classifieur à la place
du classifieur factice. Même port, mêmes deux types de messages
(docs/protocole_json.md v1.2), donc rien à changer côté Godot :

  prédiction : {"numero", "classe", "confiance", "vraie_classe", "t"}
               une toutes les 250 ms (ou seulement aux changements, --envoi changement)
               t = heure Unix du DERNIER échantillon de la fenêtre classifiée
  état       : {"type": "etat", "flux": true/false, "t"} toutes les 0,5 s

Différences avec le classifieur factice :
- la prédiction ne part plus du marqueur : elle vient du signal EEG (fenêtre
  glissante de 1 s), le marqueur ne sert qu'à remplir "vraie_classe" ;
- le lecteur envoie des microvolts, le modèle attend des volts : conversion
  automatique à partir de desc/unit du flux ;
- "NOUVEAU_RUN" remet le filtre à zéro (le signal saute d'un run à l'autre),
  "FIN" termine la séance et affiche le bilan (précision en ligne, T12).

Lancement (3 terminaux, depuis backend/) :
    python reader/lecteur.py --sujet 42 --runs 4 8 12 --attendre
    python -m handmind.server.pont_classifieur
    (puis la scène Godot)
"""
import argparse
import asyncio
import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import websockets

from handmind import config
from handmind.online.classifier import OnlineClassifier
from handmind.online.sources import LSLSource

CLIENTS = set()


async def handler(ws, *_):
    CLIENTS.add(ws)
    print(f"Client connecté ({len(CLIENTS)} au total)")
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)
        print(f"Client déconnecté ({len(CLIENTS)} restant)")


def envoyer(message):
    if CLIENTS:
        websockets.broadcast(CLIENTS, json.dumps(message, ensure_ascii=False))


def chemin_modele(path=None):
    from scripts.train import online_model_path
    p = Path(path) if path else online_model_path()
    if not p.exists():
        raise SystemExit(f"{p} introuvable : lancer d'abord "
                         "`python -m scripts.train --online`")
    return p


class Seance:
    """Une connexion au lecteur : source LSL + classifieur + bilan."""

    def __init__(self, source, bundle, args):
        self.source = source
        self.clf = OnlineClassifier(bundle, source.ch_names, source.sfreq,
                                    threshold=args.threshold, n_vote=args.vote,
                                    scale=source.scale, to_t=source.to_unix)
        self.args = args
        self.messages = []
        self.derniere_classe = None
        self.dernier_eeg = 0.0
        self.debut = time.time()
        self.fichier = None
        self.writer = None
        if args.log:
            self.fichier = open(args.log, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.fichier)
            self.writer.writerow(["numero", "t", "classe", "confiance",
                                  "vraie_classe", "latence_ms"])

    def tour(self):
        """Une lecture non bloquante du flux. Renvoie les messages produits."""
        samples, ts, marqueurs = self.source.pull(timeout=0.0)
        for t, label in marqueurs:
            if str(label) not in config.EVENT_ID:
                print("Marqueur technique :", label)
            self.clf.push_marker(t, label)
        if not len(ts):
            return []
        self.dernier_eeg = time.time()
        sortie = []
        for msg in self.clf.push(samples, ts):
            self.messages.append(msg)
            if self.writer:
                self.writer.writerow([msg["numero"], msg["t"], msg["classe"],
                                      msg["confiance"], msg["vraie_classe"],
                                      round(self.clf.latency_ms[-1], 2)])
            if self.args.envoi == "changement" and msg["classe"] == self.derniere_classe:
                continue
            self.derniere_classe = msg["classe"]
            sortie.append(msg)
        return sortie

    def terminee(self, attente_max):
        """FIN reçu, lecteur arrêté (plus d'EEG), ou flux mort dès le départ
        (on a retrouvé le flux d'un lecteur en train de se fermer)."""
        maintenant = time.time()
        if self.clf.ended:
            return True
        if self.dernier_eeg:
            return maintenant - self.dernier_eeg > attente_max
        return maintenant - self.debut > attente_max + 5.0

    def bilan(self):
        if self.fichier:
            self.fichier.close()
            self.fichier = None
        m = [x for x in self.messages if x["vraie_classe"]]
        if not self.messages:
            print("Aucune décision.")
            return
        lat = np.array(self.clf.latency_ms)
        print(f"\n--- Bilan : {len(self.messages)} décisions ---")
        if m:
            ok = np.mean([x["classe"] == x["vraie_classe"] for x in m])
            print(f"Décision = vraie classe : {ok:.1%}")
        print("Décisions par classe :", dict(Counter(x["classe"] for x in self.messages)))
        med, p95 = np.median(lat), np.percentile(lat, 95)
        verdict = "OK" if med < config.T12_MEDIAN_MS and p95 < config.T12_P95_MS else "ÉCHEC"
        print(f"T12 (chunk reçu -> décision) : médiane {med:.1f} ms, "
              f"95e centile {p95:.1f} ms -> {verdict}")


async def boucle(bundle, args):
    seance = None
    prochain_etat = 0.0
    flux_precedent = None
    t_fin = time.time() + args.duree if args.duree else None

    while t_fin is None or time.time() < t_fin:
        # 1) connexion au lecteur (dans un thread : la recherche LSL bloque)
        if seance is None:
            try:
                source = await asyncio.to_thread(LSLSource, timeout=1.0)
                seance = Seance(source, bundle, args)
                print("Lecteur trouvé, classifieur prêt "
                      f"(modèle {bundle['model_name']}, config {bundle['config']}, "
                      f"{len(bundle['classes'])} classes)")
            except RuntimeError:
                pass
            except ValueError as err:   # mauvais canaux / fréquence
                raise SystemExit(f"Flux incompatible avec le modèle : {err}")

        # 2) signal -> décisions -> Godot
        if seance is not None:
            for msg in seance.tour():
                envoyer(msg)
                if not args.silencieux:
                    signe = "OK " if msg["classe"] == msg["vraie_classe"] else "FAUX"
                    print(f"[{signe}] n°{msg['numero']} prédite={msg['classe']} "
                          f"vraie={msg['vraie_classe'] or '?'} conf={msg['confiance']}")
            # fin de la lecture, ou lecteur arrêté : bilan puis attente d'un nouveau
            if seance.terminee(args.attente_max):
                if seance.messages:
                    seance.bilan()
                seance = None
                if args.une_seance:
                    break
                # laisser le temps au lecteur qui se ferme de disparaître du
                # réseau LSL, sinon on se reconnecte à son flux mourant
                await asyncio.sleep(2.0)
                print("En attente d'un nouveau lecteur...")

        # 3) message d'état toutes les 0,5 s
        maintenant = time.time()
        if maintenant >= prochain_etat:
            prochain_etat = maintenant + config.WS_STATE_PERIOD_S
            flux_ok = (seance is not None and seance.dernier_eeg > 0 and
                       maintenant - seance.dernier_eeg < config.WS_STREAM_LOST_S)
            envoyer({"type": "etat", "flux": flux_ok, "t": round(maintenant, 4)})
            if flux_ok != flux_precedent:
                print("Flux EEG :", "reçu" if flux_ok else "perdu")
                flux_precedent = flux_ok

        await asyncio.sleep(0.01)

    if seance is not None:
        seance.bilan()


def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="HandMind - pont réel (S6)")
    p.add_argument("--port", type=int, default=config.WS_PORT)
    p.add_argument("--modele", default=None,
                   help="fichier .joblib (défaut : modèle en ligne du choix S4)")
    p.add_argument("--threshold", type=float, default=config.ONLINE_THRESHOLD)
    p.add_argument("--vote", type=int, default=config.ONLINE_N_VOTE)
    p.add_argument("--envoi", choices=["tout", "changement"], default="tout",
                   help="tout = un message toutes les 250 ms ; changement = "
                        "seulement quand la décision change")
    p.add_argument("--log", default=None, help="CSV de toutes les décisions")
    p.add_argument("--duree", type=float, default=0.0,
                   help="arrêt après N secondes (0 = jamais)")
    p.add_argument("--attente-max", type=float, default=3.0,
                   help="secondes sans EEG avant de considérer le lecteur arrêté")
    p.add_argument("--une-seance", action="store_true",
                   help="s'arrêter à la fin de la première lecture")
    p.add_argument("--silencieux", action="store_true")
    return p.parse_args(argv)


async def main(argv=None):
    args = lire_arguments(argv)
    path = chemin_modele(args.modele)
    bundle = joblib.load(path)
    async with websockets.serve(handler, config.WS_HOST, args.port):
        print(f"Pont réel sur ws://{config.WS_HOST}:{args.port} - modèle {path.name}")
        print(f"En attente des flux LSL {config.LSL_EEG_NAME} et {config.LSL_MARKER_NAME}...")
        await boucle(bundle, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Pont arrêté")
