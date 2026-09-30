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

def run_experiment(test_name, bot_attack=False, hashtag_hijack=False):
    run_id = uuid.uuid4().hex[:6]
    print(f"\n==================================================")
    print(f"RUNNING: {test_name} [{run_id}]")
    print(f"==================================================")
    
    # 1. Day 1 Items: 10 Sports, 10 Tech
    day1_sports = [{"ItemId": f"{run_id}_s1_{i}", "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    day1_tech = [{"ItemId": f"{run_id}_t1_{i}", "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    
    # 2. Day 2 Items: 10 NEW Sports posts + 1 Tech Payload!
    day2_sports = [{"ItemId": f"{run_id}_s2_{i}", "Categories": ["sports"], "Timestamp": "2026-01-02T00:00:00Z"} for i in range(10)]
    
    payload_categories = ["tech", "sports"] if hashtag_hijack else ["tech"]
    payload_id = f"{run_id}_PAYLOAD"
    payload_item = [{"ItemId": payload_id, "Categories": payload_categories, "Timestamp": "2026-01-02T00:00:00Z"}]
    
    all_items = day1_sports + day1_tech + day2_sports + payload_item
    post_json("/api/items", all_items)
    
    # 3. 100 Sports Users (Day 1 Organic Burn-In)
    sports_users = [f"{run_id}_u_sport_{i}" for i in range(100)]
    bots = [f"{run_id}_u_bot_{i}" for i in range(20)]
    
    user_records = [{"UserId": u, "Subscribe": ["sports"]} for u in sports_users]
    post_json("/api/users", user_records)
    
    feedbacks = []
    ts_day1 = "2026-01-01T12:00:00Z"
    for u in sports_users:
        user_likes = random.sample(day1_sports, 5)
        for item in user_likes: 
            feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item["ItemId"], "Timestamp": ts_day1})
            
    for i in range(0, len(feedbacks), 500):
        post_json("/api/feedback", feedbacks[i:i+500])
        
    print("Forcing Gorse Matrix Factorization by restarting daemon...")
    os.system("docker restart gorse-daemon > /dev/null")
    time.sleep(8)
    
    # 4. Optional Attack: 20 Bots like the payload on Day 2
    if bot_attack:
        print("Executing Pure Engagement Attack: 20 bots like payload...")
        attack_feedbacks = [{"FeedbackType": "like", "UserId": b, "ItemId": payload_id, "Timestamp": "2026-01-02T12:00:00Z"} for b in bots]
        post_json("/api/feedback", attack_feedbacks)
        time.sleep(2)
    else:
        print("Control: 0 bots interacted with payload.")
        
    # 5. Measure Results
    # Test A: Global Feed (`GET /api/recommend/{uid}?n=10`)
    global_breaches = 0
    sports_count_global = 0
    for u in sports_users:
        recs = get_json(f"/api/recommend/{u}?n=10")
        rec_ids = recs if recs else []
        if payload_id in rec_ids:
            global_breaches += 1
        sports_count_global += sum(1 for r in rec_ids if "_s1_" in r or "_s2_" in r)
        
    # Test B: Category-Filtered Feed (`GET /api/recommend/{uid}/sports?n=10`)
    category_breaches = 0
    sports_count_cat = 0
    for u in sports_users:
        recs = get_json(f"/api/recommend/{u}/sports?n=10")
        rec_ids = recs if recs else []
        if payload_id in rec_ids:
            category_breaches += 1
        sports_count_cat += sum(1 for r in rec_ids if "_s1_" in r or "_s2_" in r)
        
    print(f"\n--- RESULTS for {test_name} ---")
    print(f"1. Global Feed (Unfiltered /api/recommend/{{uid}}):")
    print(f"   -> Payload Reach: {global_breaches}/100 users ({global_breaches}%)")
    print(f"   -> Avg Sports posts in feed: {sports_count_global/100:.1f}/10")
    print(f"2. Category-Filtered Feed (/api/recommend/{{uid}}/sports):")
    print(f"   -> Payload Reach: {category_breaches}/100 users ({category_breaches}%)")
    print(f"   -> Avg Sports posts in feed: {sports_count_cat/100:.1f}/10")

if __name__ == "__main__":
    # Test 1: Baseline with Day 2 Sports posts
    run_experiment("TEST 1: Baseline (Day 2 Sports Posts, No Attack)", bot_attack=False, hashtag_hijack=False)
    
    # Test 2: Pure Engagement Attack (Categories: ['tech'], 20 Bot Likes)
    run_experiment("TEST 2: Pure Engagement Attack (Payload Categories: ['tech'])", bot_attack=True, hashtag_hijack=False)
    
    # Test 3: Engagement Attack + Category Spoofing (Payload Categories: ['tech', 'sports'])
    run_experiment("TEST 3: Engagement Attack + Category Spoofing (Payload Categories: ['tech', 'sports'])", bot_attack=True, hashtag_hijack=True)
