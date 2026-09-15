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
import ast
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any, Dict, List

import httpx

gorse_log = logging.getLogger("social.gorse")
gorse_log.setLevel(logging.DEBUG)


def format_iso_timestamp(ts: Any) -> str:
    """Convert any timestep, unix timestamp, or datetime into RFC3339/ISO8601 string for Gorse."""
    if ts is None:
        return "2026-01-01T12:00:00Z"
    if isinstance(ts, (int, float)):
        # If timestamp >= 1e9, treat as unix epoch seconds; otherwise treat as simulation minute offset
        if ts >= 1e9:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            base_dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            dt = base_dt + timedelta(minutes=float(ts))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(ts, str):
        ts_str = ts.strip()
        if ts_str.isdigit():
            return format_iso_timestamp(int(ts_str))
        try:
            val = float(ts_str)
            return format_iso_timestamp(val)
        except ValueError:
            pass
        if "T" in ts_str:
            if not ts_str.endswith("Z") and "+" not in ts_str and "-" not in ts_str[10:]:
                ts_str += "Z"
            return ts_str
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return "2026-01-01T12:00:00Z"
    return "2026-01-01T12:00:00Z"


def extract_action_info(t: Dict[str, Any]) -> Dict[str, Any]:
    """Safely parse action_info from trace dictionary or JSON/repr string."""
    action_info_val = t.get("action_info")
    if action_info_val is None:
        action_info_val = t.get("info", {})
    if isinstance(action_info_val, str):
        try:
            return json.loads(action_info_val)
        except Exception:
            try:
                return ast.literal_eval(action_info_val)
            except Exception:
                return {}
    elif isinstance(action_info_val, dict):
        return action_info_val
    return {}


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
                "Timestamp": format_iso_timestamp(p.get("created_at")),
                "Labels": keywords
            })
        if items:
            await self._post("/items", items)

    async def bulk_insert_feedback(self, trace_table: List[Dict[str, Any]]):
        """Map OASIS multi-modal engagement traces into Gorse feedback types:
        - like_post -> 'like' (positive)
        - repost / quote_post -> 'repost' (positive)
        - create_comment -> 'comment' (positive)
        - dislike_post / report_post -> 'dislike' (negative)
        - refresh -> 'read' (impression for all viewed posts)
        """
        feedback = []
        for t in trace_table:
            action = t.get("action")
            user_id = str(t.get("user_id"))
            action_info = extract_action_info(t)
            ts = format_iso_timestamp(t.get("created_at"))

            if action == "like_post":
                post_id = action_info.get("post_id") or action_info.get("like_id")
                if post_id is not None:
                    feedback.append({
                        "FeedbackType": "like",
                        "UserId": user_id,
                        "ItemId": str(post_id),
                        "Timestamp": ts
                    })
            elif action == "repost":
                post_id = action_info.get("reposted_id") or action_info.get("post_id")
                if post_id is not None:
                    feedback.append({
                        "FeedbackType": "repost",
                        "UserId": user_id,
                        "ItemId": str(post_id),
                        "Timestamp": ts
                    })
            elif action == "quote_post":
                post_id = action_info.get("quoted_id") or action_info.get("post_id")
                if post_id is not None:
                    feedback.append({
                        "FeedbackType": "repost",
                        "UserId": user_id,
                        "ItemId": str(post_id),
                        "Timestamp": ts
                    })
            elif action == "create_comment":
                post_id = action_info.get("post_id")
                if post_id is not None:
                    feedback.append({
                        "FeedbackType": "comment",
                        "UserId": user_id,
                        "ItemId": str(post_id),
                        "Timestamp": ts
                    })
            elif action in ("dislike_post", "report_post"):
                post_id = action_info.get("post_id") or action_info.get("dislike_id")
                if post_id is not None:
                    feedback.append({
                        "FeedbackType": "dislike",
                        "UserId": user_id,
                        "ItemId": str(post_id),
                        "Timestamp": ts
                    })
            elif action == "refresh":
                posts = action_info.get("posts", [])
                if isinstance(posts, list):
                    for post in posts:
                        if isinstance(post, dict):
                            post_id = post.get("post_id")
                        else:
                            post_id = post
                        if post_id is not None:
                            feedback.append({
                                "FeedbackType": "read",
                                "UserId": user_id,
                                "ItemId": str(post_id),
                                "Timestamp": ts
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
