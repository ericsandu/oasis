import urllib.request
import urllib.error
import json
import time
import uuid
import random
import os

GORSE_URL = "http://127.0.0.1:8088"

def post_json(endpoint, data):
    req = urllib.request.Request(f"{GORSE_URL}{endpoint}", data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"Error calling {endpoint}: {e}")
        return None

def get_json(endpoint):
    req = urllib.request.Request(f"{GORSE_URL}{endpoint}")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"Error calling {endpoint}: {e}")
        return None

def test_distributed_matrix(scenario_name, is_organic_bridger, is_baseline):
    run_id = uuid.uuid4().hex[:6]
    print(f"\n--- Running: {scenario_name} [{run_id}] ---")
    
    # 1. Setup Base Items
    sports_items = [{"ItemId": f"{run_id}_s_{i}", "IsHidden": False, "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    tech_items = [{"ItemId": f"{run_id}_t_{i}", "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    payload_id = f"{run_id}_PAYLOAD"
    payload_item = [{"ItemId": payload_id, "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-02T00:00:00Z"}]
    
    post_json("/api/items", sports_items + tech_items + payload_item)

    # 2. DISTRIBUTED Users & Organic Feedback
    sports_users = [f"{run_id}_u_sport_{i}" for i in range(100)]
    bots = [f"{run_id}_u_bot_{i}" for i in range(20)]
    
    feedbacks = []
    ts = "2026-01-03T12:00:00Z"
    
    for u in sports_users:
        user_likes = random.sample(sports_items, 5)
        for item in user_likes: 
            feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item["ItemId"], "Timestamp": ts})
            
    if is_organic_bridger and not is_baseline:
        for b in bots:
            bot_likes = random.sample(sports_items, 5)
            for item in bot_likes:
                feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": item["ItemId"], "Timestamp": ts})
                
    for i in range(0, len(feedbacks), 500):
        post_json("/api/feedback", feedbacks[i:i+500])
        
    print("Forcing Gorse Matrix Factorization by restarting the daemon...")
    os.system("docker restart gorse-daemon > /dev/null")
    time.sleep(10) # wait for boot and matrix completion
    
    # 3. Inject Bots
    if not is_baseline:
        print("Executing Bot Attack payload...")
        payload_feedbacks = []
        for b in bots:
            payload_feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": payload_id, "Timestamp": ts})
        post_json("/api/feedback", payload_feedbacks)
        time.sleep(2) # wait for fast sync
    else:
        print("Baseline: No bots interacted with payload.")
    
    # 4. Check Feed Composition
    breached = 0
    sports_recs_total = 0
    
    for u in sports_users:
        recs = get_json(f"/api/recommend/{u}?n=10")
        rec_ids = [r for r in recs] if recs else []
        if payload_id in rec_ids:
            breached += 1
        sports_recs_total += sum(1 for r in rec_ids if "_s_" in r)
        
    avg_sports = sports_recs_total / len(sports_users)
    print(f"Results for {scenario_name}:")
    print(f"  -> Payload Reach: {breached}/100")
    print(f"  -> Avg unseen sports posts in feed: {avg_sports:.1f}/10")

if __name__ == "__main__":
    test_distributed_matrix("1. Control Baseline (Forced Matrix)", False, True)
    test_distributed_matrix("2. Blunt Force Farm (Forced Matrix)", False, False)
    test_distributed_matrix("3. Organic Bridger (Forced Matrix)", True, False)
