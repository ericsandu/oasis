import logging
from typing import Any, Dict, List
import httpx
import asyncio

gorse_log = logging.getLogger("social.gorse")
gorse_log.setLevel(logging.DEBUG)

class GorseClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8088"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def _post(self, endpoint: str, data: Any):
        try:
            response = await self.client.post(f"/api{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            gorse_log.error(f"Gorse API error on {endpoint}: {e}")
            return None

    async def _get(self, endpoint: str):
        try:
            response = await self.client.get(f"/api{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            gorse_log.error(f"Gorse API error on {endpoint}: {e}")
            return None

    async def bulk_insert_users(self, user_table: List[Dict[str, Any]]):
        users = []
        for u in user_table:
            users.append({
                "UserId": str(u["user_id"]),
                "Labels": [u.get("name", "agent")]
            })
        if users:
            await self._post("/user", users)

    async def bulk_insert_items(self, post_table: List[Dict[str, Any]]):
        items = []
        for p in post_table:
            items.append({
                "ItemId": str(p["post_id"]),
                "Timestamp": p.get("created_at", "2020-01-01T00:00:00Z"),
                "Labels": [] # Could map NLP embeddings here later
            })
        if items:
            await self._post("/item", items)

    async def bulk_insert_feedback(self, trace_table: List[Dict[str, Any]]):
        feedback = []
        for t in trace_table:
            if t["action"] == "like_post":
                action_info = eval(t["action_info"]) if isinstance(t["action_info"], str) else t["action_info"]
                post_id = action_info.get("like_id") or action_info.get("post_id")
                if post_id:
                    feedback.append({
                        "FeedbackType": "like",
                        "UserId": str(t["user_id"]),
                        "ItemId": str(post_id),
                        "Timestamp": t.get("created_at", "2020-01-01T00:00:00Z")
                    })
        if feedback:
            await self._post("/feedback", feedback)

    async def update_rec_table(self, user_table, post_table, trace_table, rec_matrix, max_rec_post_len):
        gorse_log.info("Syncing state to Gorse...")
        # In a real dynamic system we'd only sync diffs, but for the simulation step we bulk sync
        await self.bulk_insert_users(user_table)
        await self.bulk_insert_items(post_table)
        await self.bulk_insert_feedback(trace_table)

        # Force a recommendation generation update
        # Wait for Gorse to process (it runs on background cron natively, but we can trigger a fast recommend sync if needed)
        
        gorse_log.info("Fetching recommendations from Gorse...")
        new_rec_matrix = []
        # Query recommendations for all users
        for u in user_table:
            uid = str(u["user_id"])
            recs = await self._get(f"/recommend/{uid}?n={max_rec_post_len}")
            # Gorse returns a list of item IDs
            if recs:
                for item_id in recs:
                    new_rec_matrix.append({
                        "user_id": int(uid),
                        "post_id": int(item_id)
                    })
        return new_rec_matrix
