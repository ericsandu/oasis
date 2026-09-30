import httpx
import asyncio
import json
import time
import uuid
import random
import os

BASE_URL = "http://127.0.0.1:8088"

async def test_setup():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Check health
        health = await client.get("/api/dashboard/stats")
        print("Gorse stats:", health.status_code, health.text[:100] if health.status_code == 200 else "")

if __name__ == "__main__":
    asyncio.run(test_setup())
