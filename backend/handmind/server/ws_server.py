import argparse
import asyncio
import json
import time

import websockets

CLASSES = ["repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"]
CLIENTS = set()


async def handler(ws):
    CLIENTS.add(ws)
    print(f"Client connecté ({len(CLIENTS)} au total)")
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)
        print(f"Client déconnecté ({len(CLIENTS)} restant)")


async def emetteur(cadence, nombre):
    n = 0
    while nombre == 0 or n < nombre:
        await asyncio.sleep(1 / cadence)
        if not CLIENTS:
            continue
        classe = CLASSES[n % len(CLASSES)]
        message = json.dumps({
            "numero": n,
            "classe": classe,
            "confiance": 0.87,
            "vraie_classe": classe,
            "t": time.time(),
        })
        websockets.broadcast(CLIENTS, message)
        print("Envoyé :", message)
        n += 1

    # Laisse le temps au dernier message d'arriver, puis ferme proprement
    await asyncio.sleep(1.0)
    for ws in list(CLIENTS):
        await ws.close(code=1000, reason="fin de la séquence")
    print(f"Séquence terminée : {n} messages envoyés")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cadence", type=float, default=1.0,
                        help="messages par seconde")
    parser.add_argument("--nombre", type=int, default=0,
                        help="nombre de messages (0 = infini)")
    args = parser.parse_args()

    async with websockets.serve(handler, "localhost", args.port):
        print(f"Serveur démarré sur ws://localhost:{args.port}")
        await emetteur(args.cadence, args.nombre)


asyncio.run(main())