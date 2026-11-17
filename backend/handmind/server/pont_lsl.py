"""
HandMind - Semaine 5 (Étudiant B)
Pont : flux LSL du lecteur -> classifieur factice -> serveur WebSocket -> Godot.

Le pont lit le flux de marqueurs (HandMind_Markers) publié par lecteur.py,
fabrique une "prédiction" avec un classifieur factice (la vraie classe, ou une
classe au hasard avec une probabilité --taux-erreur), puis diffuse un message JSON
à tous les clients WebSocket (l'avatar Godot).

Il lit aussi le flux EEG (HandMind_EEG) pour savoir si le flux est vivant, et
envoie toutes les 0,5 s un message d'état {"type": "etat", "flux": true/false}.

Le champ "t" est l'heure Unix (secondes) de l'instant où le marqueur a été émis
par le lecteur, ce qui permet à Godot de mesurer la latence (test T13).
"""
import argparse
import asyncio
import json
import random
import time

import pylsl
import websockets

CLASSES = ["repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"]
CLIENTS = set()

NOM_FLUX_MARQUEURS = "HandMind_Markers"
NOM_FLUX_EEG = "HandMind_EEG"
DELAI_FLUX_PERDU = 1.5   # secondes sans échantillon EEG -> flux considéré comme perdu
PERIODE_ETAT = 0.5       # secondes entre deux messages d'état


async def handler(ws):
    CLIENTS.add(ws)
    print(f"Client connecté ({len(CLIENTS)} au total)")
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)
        print(f"Client déconnecté ({len(CLIENTS)} restant)")


def classifieur_factice(vraie, taux_erreur):
    """Renvoie (classe prédite, confiance). Se trompe avec la probabilité taux_erreur."""
    if random.random() < taux_erreur:
        fausse = random.choice([c for c in CLASSES if c != vraie])
        return fausse, round(random.uniform(0.40, 0.65), 2)
    return vraie, round(random.uniform(0.70, 0.95), 2)


async def ouvrir_flux(nom):
    """Cherche un flux LSL par son nom. Renvoie un StreamInlet, ou None si introuvable."""
    infos = await asyncio.to_thread(pylsl.resolve_byprop, "name", nom, 1, 1.0)
    if not infos:
        return None
    inlet = pylsl.StreamInlet(infos[0], max_buflen=60)
    print(f"Flux trouvé : {nom}")
    return inlet


def decalage_horloge(inlet):
    """Écart entre l'horloge du lecteur et la nôtre (nul si tout tourne sur la même machine)."""
    try:
        return inlet.time_correction(0.5)
    except Exception:
        return 0.0


async def boucle_pont(taux_erreur):
    inlet_m = None
    inlet_e = None
    corr_m = 0.0
    dernier_eeg = 0.0
    numero = 0
    flux_precedent = None
    prochain_etat = 0.0

    while True:
        # 1) (re)connexion aux flux si nécessaire
        if inlet_m is None:
            inlet_m = await ouvrir_flux(NOM_FLUX_MARQUEURS)
            if inlet_m is not None:
                corr_m = decalage_horloge(inlet_m)
        if inlet_e is None:
            inlet_e = await ouvrir_flux(NOM_FLUX_EEG)

        # 2) marqueurs -> prédiction -> JSON
        if inlet_m is not None:
            while True:
                echantillon, ts = inlet_m.pull_sample(timeout=0.0)
                if echantillon is None:
                    break
                texte = echantillon[0]
                if texte not in CLASSES:
                    print("Marqueur technique :", texte)
                    continue
                # horodatage LSL du marqueur -> heure Unix
                t_unix = time.time() - (pylsl.local_clock() - (ts + corr_m))
                predite, confiance = classifieur_factice(texte, taux_erreur)
                message = json.dumps({
                    "numero": numero,
                    "classe": predite,
                    "confiance": confiance,
                    "vraie_classe": texte,
                    "t": t_unix,
                })
                websockets.broadcast(CLIENTS, message)
                signe = "OK " if predite == texte else "FAUX"
                print(f"[{signe}] n°{numero} prédite={predite} vraie={texte} conf={confiance}")
                numero += 1

        # 3) flux EEG : y a-t-il encore des échantillons qui arrivent ?
        if inlet_e is not None:
            _, horodatages = inlet_e.pull_chunk(timeout=0.0, max_samples=4096)
            if horodatages:
                dernier_eeg = time.time()

        # 4) message d'état régulier
        maintenant = time.time()
        if maintenant >= prochain_etat:
            prochain_etat = maintenant + PERIODE_ETAT
            flux_ok = dernier_eeg > 0 and (maintenant - dernier_eeg) < DELAI_FLUX_PERDU
            websockets.broadcast(CLIENTS, json.dumps(
                {"type": "etat", "flux": flux_ok, "t": maintenant}))
            if flux_ok != flux_precedent:
                print("Flux EEG :", "reçu" if flux_ok else "perdu")
                flux_precedent = flux_ok

        await asyncio.sleep(0.01)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--taux-erreur", type=float, default=0.2,
                        help="probabilité que le classifieur factice se trompe (0 = toujours juste)")
    args = parser.parse_args()

    async with websockets.serve(handler, "localhost", args.port):
        print(f"Pont démarré sur ws://localhost:{args.port} (taux d'erreur {args.taux_erreur})")
        print("En attente des flux LSL HandMind_Markers et HandMind_EEG...")
        await boucle_pont(args.taux_erreur)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Pont arrêté")