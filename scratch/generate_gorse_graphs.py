import httpx
import time
import uuid
import matplotlib.pyplot as plt

GORSE_URL = "http://127.0.0.1:8088"

def run_timeseries(scenario_name, is_organic_bridger, feed_size=10, is_baseline=False):
    run_id = uuid.uuid4().hex[:6]
    print(f"\n--- Starting {scenario_name} [{run_id}] ---")
    
    # 1. Setup Base Items
    sports_items = [{"ItemId": f"{run_id}_s_{i}", "IsHidden": False, "Categories": ["sports"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    tech_items = [{"ItemId": f"{run_id}_t_{i}", "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-01T00:00:00Z"} for i in range(10)]
    payload_id = f"{run_id}_PAYLOAD"
    payload_item = [{"ItemId": payload_id, "IsHidden": False, "Categories": ["tech"], "Timestamp": "2026-01-02T00:00:00Z"}]
    
    httpx.post(f"{GORSE_URL}/api/items", json=sports_items + tech_items + payload_item)

    # 2. Setup Base Users & Organic Feedback
    sports_users = [f"{run_id}_u_sport_{i}" for i in range(100)]
    bots = [f"{run_id}_u_bot_{i}" for i in range(20)]
    
    feedbacks = []
    ts = "2026-01-03T12:00:00Z"
    
    for u in sports_users:
        for item in sports_items[:5]: 
            feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item["ItemId"], "Timestamp": ts})
            
    if is_organic_bridger and not is_baseline:
        for b in bots:
            for item in sports_items[:5]:
                feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": item["ItemId"], "Timestamp": ts})
                
    # Chunk insert
    for i in range(0, len(feedbacks), 500):
        httpx.post(f"{GORSE_URL}/api/feedback", json=feedbacks[i:i+500])
        
    print(f"Burn-in complete. Waiting 20s for Gorse to compute baseline Matrix Factorization...")
    time.sleep(20)
    
    reach_history = []
    
    # 3. Step-by-Step Attack
    for step in range(1, 11):
        step_feedbacks = []
        
        if not is_baseline:
            acting_bots = bots[(step-1)*2 : step*2]
            for b in acting_bots:
                step_feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": payload_id, "Timestamp": ts})
                
            httpx.post(f"{GORSE_URL}/api/feedback", json=step_feedbacks)
            print(f"Step {step}: 2 bots liked payload. Waiting 5s for internal sync...")
        else:
            print(f"Step {step}: Control Group (No bot action). Waiting 5s for internal sync...")
            
        time.sleep(5)
        
        # Measure Reach
        breached_users = 0
        for u in sports_users:
            recs = httpx.get(f"{GORSE_URL}/api/recommend/{u}?n={feed_size}").json()
            rec_ids = [r for r in recs] if recs else []
            if payload_id in rec_ids:
                breached_users += 1
                
        reach_history.append(breached_users)
        print(f"   -> Reach: {breached_users}/100")
        
    return reach_history

if __name__ == "__main__":
    reach_baseline = run_timeseries("Control Baseline (No Bots)", is_organic_bridger=False, is_baseline=True)
    reach_blunt = run_timeseries("Blunt-Force Farm", is_organic_bridger=False, is_baseline=False)
    reach_bridge = run_timeseries("Organic Bridger", is_organic_bridger=True, is_baseline=False)
    
    steps = list(range(1, 11))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, reach_blunt, marker='o', label='Blunt-Force Farm (Pure Popularity)', color='red', linewidth=2)
    plt.plot(steps, reach_bridge, marker='s', label='Organic Bridger (CF Matrix Corruption)', color='purple', linewidth=2)
    plt.plot(steps, reach_baseline, marker='^', label='Control Baseline (0 Bot Engagement)', color='green', linewidth=2, linestyle='--')
    
    plt.title('Gorse Recommender: CIB Propagation vs Control (Feed Size = 10)', fontsize=14)
    plt.xlabel('Simulation Steps (2 Bot Actions per Step)', fontsize=12)
    plt.ylabel('Organic Users Reached (Out of 100)', fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('scratch/gorse_propagation.png', dpi=300)
    print("Graph saved to scratch/gorse_propagation.png")
