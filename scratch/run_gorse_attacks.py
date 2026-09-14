import httpx
import time
import uuid

GORSE_URL = "http://127.0.0.1:8088"

def check_gorse_ready():
    try:
        httpx.get(f"{GORSE_URL}/api/health", timeout=2)
        return True
    except:
        return False

def insert_data(run_id, is_organic_bridger):
    print(f"\n[{run_id}] Injecting data into Gorse (Organic Bridger = {is_organic_bridger})...")
    
    # 1. Define Items
    # 10 Sports Items
    sports_items = [{"ItemId": f"item_{run_id}_sport_{i}", "IsHidden": False, "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    # 10 Tech Items
    tech_items = [{"ItemId": f"item_{run_id}_tech_{i}", "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    # The Payload
    payload_item_id = f"item_{run_id}_tech_PAYLOAD"
    payload_item = [{"ItemId": payload_item_id, "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-02T00:00:00Z"}]
    
    httpx.post(f"{GORSE_URL}/api/items", json=sports_items + tech_items + payload_item)

    # 2. Define Users & Feedback
    feedbacks = []
    timestamp = "2026-01-03T12:00:00Z"
    
    # 100 Sports Users (Organic Base)
    sports_users = [f"user_{run_id}_sport_{i}" for i in range(100)]
    for u in sports_users:
        # Each likes 5 random sports items
        for item in sports_items[:5]: 
            feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item["ItemId"], "Timestamp": timestamp})
            
    # 20 Bots
    bots = [f"user_{run_id}_bot_{i}" for i in range(20)]
    for b in bots:
        if is_organic_bridger:
            # Bots pretend to be sports fans first
            for item in sports_items[:5]:
                feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": item["ItemId"], "Timestamp": timestamp})
        
        # Bots execute payload
        feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": payload_item_id, "Timestamp": timestamp})
        
    # Insert users implicitly by inserting feedback
    print(f"[{run_id}] Pushing {len(feedbacks)} interactions to Gorse...")
    # Chunk feedbacks because Gorse has a limit per request
    chunk_size = 500
    for i in range(0, len(feedbacks), chunk_size):
        resp = httpx.post(f"{GORSE_URL}/api/feedback", json=feedbacks[i:i+chunk_size])
        if resp.status_code != 200:
            print(f"Error inserting feedback: {resp.text}")

    return sports_users, payload_item_id

def evaluate_breach(run_id, sports_users, payload_item_id):
    print(f"[{run_id}] Waiting for Gorse Worker to compute Matrix Factorization (approx 20-30s)...")
    
    # We will poll the recommendation endpoint for a subset of users until we see personalized recommendations 
    # instead of just generic 'latest' items. Actually, let's just poll one user until the payload appears, 
    # or timeout after 45 seconds.
    
    breached_users = 0
    
    # Give it a baseline 15 seconds to run tasks
    time.sleep(15)
    
    for _ in range(6):
        # Sample check
        sample_recs = httpx.get(f"{GORSE_URL}/api/recommend/{sports_users[0]}?n=10").json()
        if sample_recs:
            # Has data
            break
        print("Still computing...")
        time.sleep(10)
        
    print(f"[{run_id}] Evaluating exposures...")
    for u in sports_users:
        recs = httpx.get(f"{GORSE_URL}/api/recommend/{u}?n=20").json()
        rec_ids = [r for r in recs]
        if payload_item_id in rec_ids:
            breached_users += 1
            
    reach_percentage = (breached_users / len(sports_users)) * 100
    print(f"[{run_id}] RESULT: Payload reached {breached_users}/{len(sports_users)} ({reach_percentage}%) of target cluster.")
    return breached_users

if __name__ == "__main__":
    while not check_gorse_ready():
        print("Waiting for Gorse to start...")
        time.sleep(2)
        
    print("Gorse is online. Running simulations...")
    
    for iteration in range(1, 3):
        # 1. Blunt Force
        run_id_1 = f"blunt_iter_{iteration}_{uuid.uuid4().hex[:4]}"
        su1, p1 = insert_data(run_id_1, is_organic_bridger=False)
        evaluate_breach(run_id_1, su1, p1)
        
        # 2. Organic Bridger
        run_id_2 = f"bridge_iter_{iteration}_{uuid.uuid4().hex[:4]}"
        su2, p2 = insert_data(run_id_2, is_organic_bridger=True)
        evaluate_breach(run_id_2, su2, p2)

