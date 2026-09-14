# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import logging
from typing import Any, Dict, List

import httpx

gorse_log = logging.getLogger("social.gorse")
gorse_log.setLevel(logging.DEBUG)

class GorseClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8088"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url)
        self.last_user_count = 0
        self.last_post_count = 0
        self.last_trace_count = 0

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
            await self._post("/users", users)

    async def bulk_insert_items(self, post_table: List[Dict[str, Any]]):
        items = []
        for p in post_table:
            text = str(p.get("content", ""))
            # Extract basic NLP labels for the Semantic engine (words > 3 chars)
            keywords = [w.lower() for w in text.split() if len(w) > 3]

            items.append({
                "ItemId": str(p["post_id"]),
                "Timestamp": "2026-01-01T12:00:00Z",
                "Labels": keywords
            })
        if items:
            await self._post("/items", items)

    async def bulk_insert_feedback(self, trace_table: List[Dict[str, Any]]):
        feedback = []
        for t in trace_table:
            if t["action"] == "like_post":
                action_info_str = t.get("action_info") or t.get("info", "{}")
                action_info = eval(action_info_str) if isinstance(action_info_str, str) else action_info_str
                post_id = action_info.get("like_id") or action_info.get("post_id")
                if post_id:
                    feedback.append({
                        "FeedbackType": "like",
                        "UserId": str(t["user_id"]),
                        "ItemId": str(post_id),
                        "Timestamp": "2026-01-01T12:00:00Z"
                    })
        if feedback:
            await self._post("/feedback", feedback)

    async def update_rec_table(self, user_table, post_table, trace_table, rec_matrix, max_rec_post_len):
        gorse_log.info("Delta-Syncing state to Gorse...")
        # 1. Delta Encoding: Slice the arrays to ONLY grab new entries
        new_users = user_table[self.last_user_count:]
        new_posts = post_table[self.last_post_count:]
        new_traces = trace_table[self.last_trace_count:]

        # 2. Push only the micro-payloads over REST
        await self.bulk_insert_users(new_users)
        await self.bulk_insert_items(new_posts)
        await self.bulk_insert_feedback(new_traces)

        # 3. Update the delta trackers
        self.last_user_count = len(user_table)
        self.last_post_count = len(post_table)
        self.last_trace_count = len(trace_table)

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
