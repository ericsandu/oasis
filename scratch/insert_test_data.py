import httpx
import uuid
import random

GORSE_URL = "http://127.0.0.1:8088"

run_id = "test_boot"
# 1. Setup Base Items
sports_items = [{"ItemId": f"{run_id}_s_{i}", "IsHidden": False, "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
tech_items = [{"ItemId": f"{run_id}_t_{i}", "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
httpx.post(f"{GORSE_URL}/api/items", json=sports_items + tech_items)

# 2. Setup Base Users
sports_users = [f"{run_id}_u_sport_{i}" for i in range(10)]
feedbacks = []
for u in sports_users:
    user_likes = random.sample(sports_items, 5)
    for item in user_likes: 
        feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item["ItemId"], "Timestamp": "2026-01-03T12:00:00Z"})

httpx.post(f"{GORSE_URL}/api/feedback", json=feedbacks)
print("Data inserted.")
