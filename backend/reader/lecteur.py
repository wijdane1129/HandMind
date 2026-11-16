#!/usr/bin/env python3
"""HandMind - Lecteur de dataset (le "casque virtuel").

Rejoue des enregistrements PhysioNet EEGMMIDB comme s'ils arrivaient en direct
d'un casque, via Lab Streaming Layer (LSL).

Deux flux sont publies :
  - HandMind_EEG     : 64 canaux, 160 Hz, float32, en microvolts (uV)
  - HandMind_Markers : flux irregulier de chaines de caracteres
                       ("repos", "main_gauche", ..., "NOUVEAU_RUN:4", "FIN:<n>")

Exemples :
  python lecteur.py --sujet 3 --runs 4 8 12 --boucle
  python lecteur.py --sujet 1 --runs 4 --vitesse 5
  python lecteur.py --synthetique --boucle --duree 610     # sans telechargement
"""
import argparse
import sys
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

SFREQ = 160.0
NB_CANAUX = 64
RUNS_VALIDES = (4, 6, 8, 10, 12, 14)
RUNS_GAUCHE_DROITE = (4, 8, 12)
RUNS_MAINS_PIEDS = (6, 10, 14)

NOM_FLUX_EEG = "HandMind_EEG"
NOM_FLUX_MARQUEURS = "HandMind_Markers"


# --------------------------------------------------------------------------
# 1. Traduction des codes PhysioNet -> noms de classes
# --------------------------------------------------------------------------
def code_vers_classe(code, run):
    """Traduit T0/T1/T2 en nom de classe, selon le numero du run."""
    if code == "T0":
        return "repos"
    if run in RUNS_GAUCHE_DROITE:
        return {"T1": "main_gauche", "T2": "main_droite"}[code]
    if run in RUNS_MAINS_PIEDS:
        return {"T1": "deux_mains", "T2": "deux_pieds"}[code]
    raise ValueError(f"Run {run} non supporte (runs valides : {RUNS_VALIDES})")


# --------------------------------------------------------------------------
# 2. Chargement des donnees
# --------------------------------------------------------------------------
def charger_run(sujet, run, en_volts=False):
    """Charge un run PhysioNet.

    Retourne (signal, noms_canaux, marqueurs)
      signal    : tableau (N, 64) float32, en uV (ou en V si en_volts=True)
      marqueurs : liste de (indice_echantillon, nom_de_classe)
    """
    import mne
    from mne.datasets import eegbci

    chemin = eegbci.load_data(sujet, [run], verbose="ERROR")[0]
    raw = mne.io.read_raw_edf(chemin, preload=True, verbose="ERROR")
    eegbci.standardize(raw)  # noms propres : C3, Cz, C4...

    if abs(raw.info["sfreq"] - SFREQ) > 1e-6:
        raise RuntimeError(f"Frequence inattendue : {raw.info['sfreq']} Hz")

    facteur = 1.0 if en_volts else 1e6  # MNE travaille en volts
    signal = (raw.get_data().T * facteur).astype(np.float32)

    # Indices d'echantillons des annotations (robuste vis-a-vis de first_samp)
    evenements, _ = mne.events_from_annotations(
        raw, event_id={"T0": 1, "T1": 2, "T2": 3}, verbose="ERROR"
    )
    code_de = {1: "T0", 2: "T1", 3: "T2"}
    marqueurs = [
        (int(ev[0] - raw.first_samp), code_vers_classe(code_de[int(ev[2])], run))
        for ev in evenements
    ]
    return signal, list(raw.ch_names), marqueurs


def charger_synthetique(run, duree_s=120, seed=0):
    """Faux enregistrement (bruit) pour tester sans telechargement."""
    rng = np.random.default_rng(seed + run)
    n = int(duree_s * SFREQ)
    signal = rng.normal(0, 10, size=(n, NB_CANAUX)).astype(np.float32)
    noms = ["C3", "Cz", "C4"] + [f"E{i}" for i in range(4, NB_CANAUX + 1)]
    cycle = ["T0", "T1", "T0", "T2"]
    marqueurs = []
    for k in range(int(duree_s // 4)):
        marqueurs.append((int(k * 4 * SFREQ), code_vers_classe(cycle[k % 4], run)))
    return signal, noms, marqueurs


def construire_timeline(sujet, runs, synthetique=False, en_volts=False):
    """Enchaine plusieurs runs en une seule longue sequence.

    Retourne (signal, noms_canaux, marqueurs) ou marqueurs est un dict
    {indice_echantillon: [chaines]}. Un marqueur "NOUVEAU_RUN:<n>" est place au
    debut de chaque run (le signal est discontinu a cet endroit).
    """
    morceaux, marqueurs, noms, offset = [], {}, None, 0
    for run in runs:
        if synthetique:
            sig, noms_run, marq = charger_synthetique(run)
        else:
            print(f"Chargement sujet {sujet}, run {run}...", flush=True)
            sig, noms_run, marq = charger_run(sujet, run, en_volts)
        noms = noms or noms_run
        marqueurs.setdefault(offset, []).append(f"NOUVEAU_RUN:{run}")
        for idx, classe in marq:
            marqueurs.setdefault(offset + idx, []).append(classe)
        morceaux.append(sig)
        offset += len(sig)
    return np.concatenate(morceaux), noms, marqueurs


# --------------------------------------------------------------------------
# 3. Flux LSL
# --------------------------------------------------------------------------
def creer_flux(noms_canaux, sujet, runs, en_volts):
    info = StreamInfo(
        name=NOM_FLUX_EEG,
        type="EEG",
        channel_count=len(noms_canaux),
        nominal_srate=SFREQ,
        channel_format="float32",
        source_id="handmind_replayer_eeg",
    )
    desc = info.desc()
    desc.append_child_value("unit", "volts" if en_volts else "microvolts")
    desc.append_child_value("sujet", str(sujet))
    desc.append_child_value("runs", ",".join(map(str, runs)))
    canaux = desc.append_child("channels")
    for nom in noms_canaux:
        canaux.append_child("channel").append_child_value("label", nom)
    outlet_eeg = StreamOutlet(info)

    info_m = StreamInfo(
        name=NOM_FLUX_MARQUEURS,
        type="Markers",
        channel_count=1,
        nominal_srate=0,  # 0 = flux irregulier
        channel_format="string",
        source_id="handmind_replayer_markers",
    )
    outlet_marq = StreamOutlet(info_m)
    return outlet_eeg, outlet_marq


# --------------------------------------------------------------------------
# 4. Boucle de diffusion
# --------------------------------------------------------------------------
def diffuser(signal, marqueurs, outlet_eeg, outlet_marq, vitesse=1.0,
             boucle=False, duree=0.0):
    """Diffuse le signal a la bonne cadence, sans derive.

    Chaque echantillon g a un instant theorique t0 + g/rate. A chaque tour on
    envoie tous les echantillons "dus" : un retard est donc rattrape tout seul
    (pas d'accumulation, meme si sleep() est imprecis sous Windows).
    Les horodatages LSL sont les instants theoriques (comme une horloge de
    casque), donc un echantillon perdu cote recepteur se voit comme un trou.
    """
    n = len(signal)
    rate = SFREQ * vitesse

    limite = None  # nombre total d'echantillons a envoyer (None = infini)
    if duree > 0:
        limite = int(duree * rate)
    if not boucle:
        limite = n if limite is None else min(limite, n)

    perf0 = time.perf_counter()
    lsl0 = local_clock()
    envoyes = 0
    prochain_affichage = 10.0
    print(f"Diffusion : {n} echantillons ({n / SFREQ:.0f} s), vitesse x{vitesse}"
          f"{', en boucle' if boucle else ''}. Ctrl+C pour arreter.", flush=True)

    try:
        while limite is None or envoyes < limite:
            du = int((time.perf_counter() - perf0) * rate) + 1
            cible = du if limite is None else min(du, limite)
            for g in range(envoyes, cible):
                p = g % n
                ts = lsl0 + g / rate
                for m in marqueurs.get(p, []):
                    outlet_marq.push_sample([m], ts)
                outlet_eeg.push_sample(signal[p].tolist(), ts)
            envoyes = max(envoyes, cible)

            ecoule = time.perf_counter() - perf0
            if ecoule >= prochain_affichage:
                print(f"  {ecoule:6.0f} s reelles, {envoyes} echantillons envoyes",
                      flush=True)
                prochain_affichage += 10.0

            reste = perf0 + envoyes / rate - time.perf_counter()
            if reste > 0:
                time.sleep(min(reste, 0.005))
    except KeyboardInterrupt:
        print("\nArret demande.")

    outlet_marq.push_sample([f"FIN:{envoyes}"], lsl0 + envoyes / rate)
    time.sleep(0.5)  # laisse le temps aux recepteurs de lire le dernier marqueur
    print(f"Termine : {envoyes} echantillons envoyes au total.")
    return envoyes


# --------------------------------------------------------------------------
# 5. Programme principal
# --------------------------------------------------------------------------
def lire_arguments(argv=None):
    p = argparse.ArgumentParser(description="HandMind - lecteur de dataset (LSL)")
    p.add_argument("--sujet", type=int, default=1, help="sujet 1 a 109")
    p.add_argument("--runs", type=int, nargs="+", default=[4, 8, 12],
                   help="runs enchaines, parmi 4 6 8 10 12 14")
    p.add_argument("--vitesse", type=float, default=1.0,
                   help="1 = temps reel, 5 = 5x plus vite (dev uniquement)")
    p.add_argument("--boucle", action="store_true", help="rejouer sans fin")
    p.add_argument("--duree", type=float, default=0.0,
                   help="arret apres N secondes de signal (0 = pas de limite)")
    p.add_argument("--volts", action="store_true",
                   help="envoyer en volts au lieu de microvolts")
    p.add_argument("--attendre", action="store_true",
                   help="attendre qu'un recepteur se connecte avant de demarrer")
    p.add_argument("--synthetique", action="store_true",
                   help="bruit au lieu de PhysioNet (test sans telechargement)")
    args = p.parse_args(argv)
    for r in args.runs:
        if r not in RUNS_VALIDES:
            p.error(f"run {r} invalide, choisir parmi {RUNS_VALIDES}")
    if not 1 <= args.sujet <= 109:
        p.error("le sujet doit etre entre 1 et 109")
    return args


def main():
    args = lire_arguments()
    signal, noms, marqueurs = construire_timeline(
        args.sujet, args.runs, args.synthetique, args.volts)
    outlet_eeg, outlet_marq = creer_flux(noms, args.sujet, args.runs, args.volts)
    print(f"Flux '{NOM_FLUX_EEG}' et '{NOM_FLUX_MARQUEURS}' publies.", flush=True)

    if args.attendre:
        print("En attente d'un recepteur...", flush=True)
        if not outlet_eeg.wait_for_consumers(120):
            print("Aucun recepteur apres 120 s, abandon.")
            sys.exit(1)

    diffuser(signal, marqueurs, outlet_eeg, outlet_marq,
             args.vitesse, args.boucle, args.duree)


if __name__ == "__main__":
    main()