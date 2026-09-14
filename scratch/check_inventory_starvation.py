import httpx
import time
import uuid

GORSE_URL = "http://127.0.0.1:8088"

def test_inventory():
    run_id = uuid.uuid4().hex[:6]
    print(f"\n--- Diagnostic: Inventory Starvation vs Recency Bias [{run_id}] ---")
    
    # 1. Setup Base Items
    # 10 Sports, 10 Tech (Day 1)
    sports_items = [{"ItemId": f"{run_id}_s_{i}", "IsHidden": False, "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    tech_items = [{"ItemId": f"{run_id}_t_{i}", "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    # 1 Payload (Day 2)
    payload_id = f"{run_id}_PAYLOAD"
    payload_item = [{"ItemId": payload_id, "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-02T00:00:00Z"}]
    
    httpx.post(f"{GORSE_URL}/api/items", json=sports_items + tech_items + payload_item)

    # 2. Setup 1 User who likes 5 sports items
    user_id = f"{run_id}_diagnostic_user"
    ts = "2026-01-03T12:00:00Z"
    
    feedbacks = []
    for item in sports_items[:5]: 
        feedbacks.append({"FeedbackType": "like", "UserId": user_id, "ItemId": item["ItemId"], "Timestamp": ts})
        
    httpx.post(f"{GORSE_URL}/api/feedback", json=feedbacks)
    
    print("Waiting 15 seconds for Gorse to generate recommendations...")
    time.sleep(15)
    
    # 3. Test Feed Size 10
    recs_10 = httpx.get(f"{GORSE_URL}/api/recommend/{user_id}?n=10").json()
    print("\n--- Feed Composition (n=10) ---")
    sports_count = 0
    tech_count = 0
    has_payload = False
    
    if not recs_10:
        print("No recommendations returned yet. Gorse may need more time.")
    else:
        for i, r in enumerate(recs_10):
            item_type = "Sports (Day 1)" if "_s_" in r else ("Payload (Day 2)" if "PAYLOAD" in r else "Tech (Day 1)")
            print(f"Rank {i+1}: {r} -> {item_type}")
            if "_s_" in r: sports_count += 1
            if "_t_" in r: tech_count += 1
            if "PAYLOAD" in r: has_payload = True
            
        print(f"\nSummary (n=10): Sports={sports_count}, Tech={tech_count}, Payload={has_payload}")

    # 4. Test Feed Size 3 (No Inventory Starvation)
    recs_3 = httpx.get(f"{GORSE_URL}/api/recommend/{user_id}?n=3").json()
    print("\n--- Feed Composition (n=3) ---")
    if not recs_3:
        pass
    else:
        for i, r in enumerate(recs_3):
            item_type = "Sports (Day 1)" if "_s_" in r else ("Payload (Day 2)" if "PAYLOAD" in r else "Tech (Day 1)")
            print(f"Rank {i+1}: {r} -> {item_type}")

if __name__ == "__main__":
    test_inventory()
