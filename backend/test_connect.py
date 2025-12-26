import socket
import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

hostname = "generativelanguage.googleapis.com"

print(f"Testing DNS resolution for {hostname}...")
try:
    addr = socket.gethostbyname(hostname)
    print(f"SUCCESS: Resolved to {addr}")
except Exception as e:
    print(f"FAILURE: DNS resolution failed: {e}")

print("\nTesting HTTPX (Sync)...")
try:
    with httpx.Client() as client:
        resp = client.get(f"https://{hostname}", timeout=5)
        print(f"SUCCESS: Status Code {resp.status_code}")
except Exception as e:
    print(f"FAILURE: HTTPX Sync failed: {e}")

print("\nTesting HTTPX (Async)...")
async def test_async():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://{hostname}", timeout=5)
            print(f"SUCCESS: Status Code {resp.status_code}")
    except Exception as e:
        print(f"FAILURE: HTTPX Async failed: {e}")

asyncio.run(test_async())
