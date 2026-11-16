import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        print("Connecté")
        async for message in ws:
            print("Reçu :", message)

asyncio.run(main())