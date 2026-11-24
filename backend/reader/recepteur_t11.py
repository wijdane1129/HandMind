#!/usr/bin/env python3
"""HandMind - Test T11 : verification du lecteur de dataset.

Se branche sur le flux LSL et verifie :
  - la derive : debit mesure (horloge du recepteur) vs 160 Hz nominal  -> < 1 %
  - les echantillons perdus : trous dans les horodatages LSL            -> 0
  - les blocages : plus longue pause entre deux arrivees de donnees

A lancer avec le lecteur en vitesse x1. Procedure des 10 minutes :
  terminal 1 : python lecteur.py --synthetique --boucle --duree 620
               (ou avec PhysioNet : --sujet 1 --runs 4 8 12 --boucle --duree 620)
  terminal 2 : python recepteur_t11.py --duree 600
"""
import argparse
import sys
import time

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

SFREQ = 160.0


def trouver_flux(nom, timeout=15):
    flux = resolve_byprop("name", nom, timeout=timeout)
    if not flux:
        print(f"Flux '{nom}' introuvable. Le lecteur est-il lance ?")
        sys.exit(2)
    return StreamInlet(flux[0], max_buflen=360)


def main():
    p = argparse.ArgumentParser(description="HandMind - test T11")
    p.add_argument("--duree", type=float, default=600, help="duree du test (s)")
    p.add_argument("--seuil-derive", type=float, default=1.0, help="en %%")
    args = p.parse_args()

    inlet = trouver_flux("HandMind_EEG")
    try:
        inlet_m = trouver_flux("HandMind_Markers", timeout=3)
    except SystemExit:
        inlet_m = None
    print(f"Connecte. Mesure pendant {args.duree:.0f} s...", flush=True)

    horodatages = []
    arrivees = []  # (heure d'arrivee, nombre d'echantillons du paquet)
    marqueurs_vus = 0
    fin_recue = None
    debut = local_clock()
    prochain_affichage = 30.0

    while local_clock() - debut < args.duree:
        _, ts = inlet.pull_chunk(timeout=0.05, max_samples=4096)
        if ts:
            arrivees.append((local_clock(), len(ts)))
            horodatages.extend(ts)
        if inlet_m:
            msgs, _ = inlet_m.pull_chunk(timeout=0.0)
            for m in msgs:
                marqueurs_vus += 1
                if m[0].startswith("FIN:"):
                    fin_recue = int(m[0].split(":")[1])
        if fin_recue is not None:
            time.sleep(0.2)  # vider ce qui reste
            _, ts = inlet.pull_chunk(timeout=0.2, max_samples=65536)
            if ts:
                arrivees.append((local_clock(), len(ts)))
                horodatages.extend(ts)
            print("Marqueur de fin recu : le lecteur s'est arrete.")
            break
        ecoule = local_clock() - debut
        if ecoule >= prochain_affichage:
            print(f"  {ecoule:5.0f} s : {len(horodatages)} echantillons recus",
                  flush=True)
            prochain_affichage += 30.0

    # ---------------- Analyse ----------------
    if len(arrivees) < 3:
        print("Pas assez de donnees recues.")
        sys.exit(2)

    ts = np.array(horodatages)
    n = len(ts)
    periode = 1.0 / SFREQ

    # 1) Derive : on ignore le premier paquet (peut contenir un retard initial)
    t_premier = arrivees[0][0]
    t_dernier = arrivees[-1][0]
    n_mesure = sum(k for _, k in arrivees[1:])
    debit = n_mesure / (t_dernier - t_premier)
    derive = (debit - SFREQ) / SFREQ * 100

    # 2) Echantillons perdus : trous dans les horodatages theoriques
    ecarts = np.diff(ts)
    trous = ecarts > 1.5 * periode
    perdus = int(np.sum(np.round(ecarts[trous] / periode) - 1))
    desordre = int(np.sum(ecarts <= 0))

    # 3) Blocages : plus longue pause entre deux arrivees
    pauses = np.diff([a for a, _ in arrivees])
    pause_max = float(pauses.max())

    print("\n=== Resultat T11 ===")
    print(f"Duree mesuree            : {t_dernier - t_premier:.1f} s")
    print(f"Echantillons recus       : {n}")
    print(f"Debit mesure             : {debit:.3f} Hz (nominal {SFREQ:.0f} Hz)")
    print(f"Derive                   : {derive:+.3f} %  (seuil < {args.seuil_derive} %)")
    print(f"Echantillons perdus      : {perdus}")
    print(f"Horodatages desordonnes  : {desordre}")
    print(f"Plus longue pause        : {pause_max * 1000:.0f} ms")
    print(f"Marqueurs recus          : {marqueurs_vus}")
    if fin_recue is not None:
        print(f"Le lecteur annonce avoir envoye {fin_recue} echantillons au total "
              f"(a comparer a {n} si le recepteur etait connecte des le debut).")

    ok = abs(derive) < args.seuil_derive and perdus == 0 and desordre == 0
    print("\nVERDICT :", "REUSSI" if ok else "ECHEC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()