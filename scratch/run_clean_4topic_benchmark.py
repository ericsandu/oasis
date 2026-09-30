import urllib.request
import urllib.error
import json
import time
import random
import os
import matplotlib.pyplot as plt
import numpy as np

GORSE_URL = "http://127.0.0.1:8088"
TOPICS = ["tech", "sports", "politics", "entertainment"]
NUM_USERS_PER_TOPIC = 20  # 80 organic users total
DAY1_ITEMS_PER_TOPIC = 15  # 60 Day 1 items
DAY2_ITEMS_PER_TOPIC = 5   # 20 Day 2 fresh organic items
NUM_CIB_POSTS = 5          # 5 CIB posts in politics

POST_TEMPLATES = {
    "tech": [
        "Quantum computing SDK breakthrough accelerates simulations",
        "Autonomous agent architectures for distributed systems",
        "Semiconductor manufacturing advances enable 1nm chips",
        "Neural network pruning reduces edge inference latency",
        "Compiler optimizations in Rust and LLVM toolchains",
        "Open-source local LLM achieves state-of-the-art benchmarks",
        "High-performance vector database scaling across clusters",
        "WebAssembly runtime security and sandboxing analysis",
        "Zero-knowledge proof verification algorithms optimized",
        "Next-generation GPU memory bandwidth architectural shift"
    ],
    "sports": [
        "Dramatic comeback in premier league derby keeps title alive",
        "Olympic sprinter shatters 100m world record in Paris",
        "Playoff thriller enters double overtime with clutch buzzer beater",
        "Tactical analysis of modern high-pressing defensive formations",
        "Marathon training regimen: physiological VO2 max adaptation",
        "Grand slam tennis final delivers epic five-set battle",
        "Formula 1 aerodynamics overhaul impacts cornering grip",
        "Basketball analytical tracking reveals pick-and-roll efficiency",
        "Tour de France mountain stage challenges elite cycling endurance",
        "Winter championship ski jump record broken in blizzard"
    ],
    "politics": [
        "Parliament passes landmark infrastructure and green energy bill",
        "Municipal governance reform sparks debate on zoning laws",
        "Diplomatic summit concludes with multilateral trade pact",
        "Fiscal policy committee assesses inflation basket dynamics",
        "Supreme Court hears arguments on administrative agency powers",
        "Bipartisan congressional committee convenes on cybersecurity",
        "Civil service regulatory framework updated for public integrity",
        "National census release reveals demographic migration shifts",
        "Central bank governor testifies on macroeconomic liquidity",
        "Electoral boundary redistribution committee publishes proposal"
    ],
    "entertainment": [
        "Acclaimed director announces psychological thriller sequel",
        "Indie multiplayer game surpasses three million sales",
        "Orchestral score wins top honors at international gala",
        "Behind the scenes of virtual production and real-time CGI",
        "Streaming network greenlights ten-part sci-fi epic",
        "Classic animated franchise receives remastered cinematic release",
        "Music production masterclass on modular synthesis dynamics",
        "Film festival jury awards golden palm to existential drama",
        "Video game studio unveils revolutionary procedural world engine",
        "Broadway musical revival sets box office attendance records"
    ]
}

CIB_TEMPLATES = [
    "LEAKED AUDIT: Official records confirm coordinated vote manipulation scheme",
    "DEEP STATE UNMASKED: Whistleblowers reveal covert foreign policy takeover",
    "THEY CANNOT HIDE THIS: Classified documents expose national corruption nexus",
    "URGENT REPORT: Elite conspiracy dismantled as hidden financial transfers leak",
    "BREAKING DISCOVERY: Covert institutional operation admitted by insiders"
]

def post_json(endpoint, data):
    req = urllib.request.Request(
        f"{GORSE_URL}{endpoint}",
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"Error posting to {endpoint}: {e}")
        return None

def get_json(endpoint):
    req = urllib.request.Request(f"{GORSE_URL}{endpoint}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return None

def reset_gorse():
    os.system("docker exec gorse-daemon rm -f /app/data/gorse_data.db /app/data/gorse_data.db-shm /app/data/gorse_data.db-wal /app/data/gorse_cache.db /app/data/gorse_cache.db-shm /app/data/gorse_cache.db-wal > /dev/null 2>&1")
    os.system("docker restart gorse-daemon > /dev/null 2>&1")
    time.sleep(3.5)

def run_condition(num_bots):
    print(f"\n=======================================================")
    print(f"RUNNING BENCHMARK CONDITION: {num_bots} BOTS")
    print(f"=======================================================")
    reset_gorse()
    
    # 1. Day 1 Content (Burn-in)
    day1_items = []
    for cat in TOPICS:
        for i in range(DAY1_ITEMS_PER_TOPIC):
            item_id = f"d1_{cat}_{i}"
            text = POST_TEMPLATES[cat][i % len(POST_TEMPLATES[cat])]
            keywords = [w.lower() for w in text.split() if len(w) > 3]
            day1_items.append({
                "ItemId": item_id,
                "Categories": [cat],
                "Timestamp": "2026-01-01T08:00:00Z",
                "Labels": keywords
            })
    post_json("/api/items", day1_items)
    
    # 2. Organic Users & Cohorts
    organic_users = []
    cohorts = {}
    for cat in TOPICS:
        cohorts[cat] = []
        for i in range(NUM_USERS_PER_TOPIC):
            uid = f"u_{cat}_{i}"
            organic_users.append({
                "UserId": uid,
                "Labels": [cat, "organic"]
            })
            cohorts[cat].append(uid)
    post_json("/api/users", organic_users)
    
    # 3. Day 1 Organic Engagements (Likes & Impressions)
    # Organic users like 5 items from home category, with 10% cross-topic noise
    day1_feedbacks = []
    ts_d1 = "2026-01-01T12:00:00Z"
    for cat in TOPICS:
        home_pool = [f"d1_{cat}_{i}" for i in range(DAY1_ITEMS_PER_TOPIC)]
        other_pool = [f"d1_{o}_{i}" for o in TOPICS if o != cat for i in range(DAY1_ITEMS_PER_TOPIC)]
        for u in cohorts[cat]:
            for item_id in random.sample(home_pool, 5):
                day1_feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item_id, "Timestamp": ts_d1})
                day1_feedbacks.append({"FeedbackType": "read", "UserId": u, "ItemId": item_id, "Timestamp": ts_d1})
            if random.random() < 0.10:
                noise_id = random.choice(other_pool)
                day1_feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": noise_id, "Timestamp": ts_d1})
                day1_feedbacks.append({"FeedbackType": "read", "UserId": u, "ItemId": noise_id, "Timestamp": ts_d1})
                
    for i in range(0, len(day1_feedbacks), 500):
        post_json("/api/feedback", day1_feedbacks[i:i+500])
        
    # Force Gorse to compute Day 1 baseline collaborative filtering & popularity index
    os.system("docker restart gorse-daemon > /dev/null 2>&1")
    time.sleep(3.5)
    
    # 4. Day 2 Content (Fresh organic inventory + CIB content in politics)
    day2_items = []
    for cat in TOPICS:
        for i in range(DAY2_ITEMS_PER_TOPIC):
            item_id = f"d2_{cat}_{i}"
            text = POST_TEMPLATES[cat][(i + DAY1_ITEMS_PER_TOPIC) % len(POST_TEMPLATES[cat])]
            keywords = [w.lower() for w in text.split() if len(w) > 3]
            day2_items.append({
                "ItemId": item_id,
                "Categories": [cat],
                "Timestamp": "2026-01-02T08:00:00Z",
                "Labels": keywords
            })
    # CIB Posts (Strictly politics category, no exploit tricks)
    cib_items = []
    for i in range(NUM_CIB_POSTS):
        item_id = f"d2_cib_{i}"
        text = CIB_TEMPLATES[i % len(CIB_TEMPLATES)]
        keywords = [w.lower() for w in text.split() if len(w) > 3]
        cib_items.append({
            "ItemId": item_id,
            "Categories": ["politics"],
            "Timestamp": "2026-01-02T08:00:00Z",
            "Labels": keywords
        })
    post_json("/api/items", day2_items + cib_items)
    
    # 5. Day 2 Organic Activity: Organic users like 2 fresh posts in their home category
    day2_feedbacks = []
    ts_d2 = "2026-01-02T12:00:00Z"
    for cat in TOPICS:
        d2_home_pool = [f"d2_{cat}_{i}" for i in range(DAY2_ITEMS_PER_TOPIC)]
        for u in cohorts[cat]:
            for item_id in random.sample(d2_home_pool, 2):
                day2_feedbacks.append({"FeedbackType": "like", "UserId": u, "ItemId": item_id, "Timestamp": ts_d2})
                day2_feedbacks.append({"FeedbackType": "read", "UserId": u, "ItemId": item_id, "Timestamp": ts_d2})
                
    # 6. CIB Bot Activity (If num_bots > 0, bots coordinate on CIB posts)
    if num_bots > 0:
        bot_users = [f"bot_{b}" for b in range(num_bots)]
        post_json("/api/users", [{"UserId": b, "Labels": ["politics", "bot"]} for b in bot_users])
        for b in bot_users:
            for cib_it in cib_items:
                day2_feedbacks.append({"FeedbackType": "like", "UserId": b, "ItemId": cib_it["ItemId"], "Timestamp": ts_d2})
                day2_feedbacks.append({"FeedbackType": "repost", "UserId": b, "ItemId": cib_it["ItemId"], "Timestamp": ts_d2})
                day2_feedbacks.append({"FeedbackType": "read", "UserId": b, "ItemId": cib_it["ItemId"], "Timestamp": ts_d2})
                
    for i in range(0, len(day2_feedbacks), 500):
        post_json("/api/feedback", day2_feedbacks[i:i+500])
        
    # Re-index recommendations with Day 2 data
    os.system("docker restart gorse-daemon > /dev/null 2>&1")
    time.sleep(3.5)
    
    # 7. Collect & Analyze Recommendations for Each Demographic Separately
    metrics_by_cat = {}
    for cat in TOPICS:
        users = cohorts[cat]
        cib_exposed_users = 0
        total_recs = 0
        counts = {
            "cib": 0,
            "home": 0,
            "politics": 0,
            "tech": 0,
            "sports": 0,
            "entertainment": 0
        }
        
        for u in users:
            recs = get_json(f"/api/recommend/{u}?n=10")
            rec_ids = recs if recs else []
            total_recs += len(rec_ids)
            has_cib = False
            
            for item_id in rec_ids:
                if "_cib_" in item_id:
                    counts["cib"] += 1
                    has_cib = True
                elif f"_{cat}_" in item_id:
                    counts["home"] += 1
                for topic in TOPICS:
                    if f"_{topic}_" in item_id and topic != cat:
                        counts[topic] += 1
            if has_cib:
                cib_exposed_users += 1
                
        pen_rate = (cib_exposed_users / len(users)) * 100.0 if users else 0.0
        cib_sov = (counts["cib"] / total_recs) * 100.0 if total_recs > 0 else 0.0
        home_sov = (counts["home"] / total_recs) * 100.0 if total_recs > 0 else 0.0
        norm_pol_sov = (counts["politics"] / total_recs) * 100.0 if cat != "politics" and total_recs > 0 else (counts["home"] / total_recs * 100.0 if total_recs > 0 else 0.0)
        
        amplification = (cib_sov / norm_pol_sov) if norm_pol_sov > 0 else (cib_sov if cib_sov > 0 else 1.0)
        
        metrics_by_cat[cat] = {
            "penetration_rate": pen_rate,
            "cib_share_of_voice": cib_sov,
            "home_share_of_voice": home_sov,
            "norm_politics_share": norm_pol_sov,
            "amplification": amplification,
            "counts": counts,
            "total_recs": total_recs
        }
        
    return metrics_by_cat

def main():
    bot_conditions = [0, 5, 10, 20, 30]
    all_results = {}
    
    for b in bot_conditions:
        res = run_condition(b)
        all_results[b] = res
        
    print("\n" + "=" * 95)
    print("SUMMARY EXPERIMENT RESULTS: CIB PROPAGATION ACROSS 4 TOPIC DEMOGRAPHICS")
    print("=" * 95)
    print(f"{'Bots':<6} | {'Demographic':<14} | {'CIB Pen %':<11} | {'CIB SoV %':<11} | {'Home SoV %':<11} | {'Normal Pol %':<13} | {'Amplification':<13}")
    print("-" * 95)
    
    flat_data = []
    for b in bot_conditions:
        for cat in TOPICS:
            m = all_results[b][cat]
            pen = m["penetration_rate"]
            sov = m["cib_share_of_voice"]
            home = m["home_share_of_voice"]
            n_pol = m["norm_politics_share"]
            amp = m["amplification"]
            print(f"{b:<6} | {cat:<14} | {pen:>9.1f}% | {sov:>9.2f}% | {home:>9.1f}% | {n_pol:>11.2f}% | {amp:>11.2f}x")
            flat_data.append({
                "bots": b,
                "demographic": cat,
                "penetration_rate": pen,
                "cib_share_of_voice": sov,
                "home_share_of_voice": home,
                "norm_politics_share": n_pol,
                "amplification": amp
            })
        print("-" * 95)
        
    # Save JSON summary
    with open("scratch/clean_benchmark_summary.json", "w") as f:
        json.dump(flat_data, f, indent=2)
    print("Saved clean summary to scratch/clean_benchmark_summary.json")
    
    # Generate 4-Panel Visualization Dashboard
    output_png = "scratch/cib_4topic_propagation_dashboard.png"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {"tech": "#1f77b4", "sports": "#2ca02c", "politics": "#d62728", "entertainment": "#9467bd"}
    
    # Panel 1: Penetration Rate vs Bots
    ax = axes[0, 0]
    for cat in TOPICS:
        pens = [all_results[b][cat]["penetration_rate"] for b in bot_conditions]
        ax.plot(bot_conditions, pens, label=f"{cat.capitalize()} Demographic", marker="o", linewidth=2.5, color=colors[cat])
    ax.set_title("1. User Exposure Penetration Across Demographics (% Users Exposed)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coordinated CIB Bot Cluster Size", fontsize=11)
    ax.set_ylabel("% Users Exposed in Top-10", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: CIB Share of Voice vs Bots
    ax = axes[0, 1]
    for cat in TOPICS:
        sovs = [all_results[b][cat]["cib_share_of_voice"] for b in bot_conditions]
        ax.plot(bot_conditions, sovs, label=f"{cat.capitalize()} Feed", marker="s", linewidth=2.5, color=colors[cat])
    ax.set_title("2. CIB Share of Voice (% of Top-10 Recommended Content)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coordinated CIB Bot Cluster Size", fontsize=11)
    ax.set_ylabel("CIB Share of Recommendations (%)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Spillover into Disjunct Groups (Bots=20): CIB vs Normal Politics
    ax = axes[1, 0]
    disjunct_cats = ["sports", "tech", "entertainment"]
    x = np.arange(len(disjunct_cats))
    width = 0.35
    cib_sovs_20 = [all_results[20][c]["cib_share_of_voice"] for c in disjunct_cats]
    norm_pols_20 = [all_results[20][c]["norm_politics_share"] for c in disjunct_cats]
    ax.bar(x - width/2, norm_pols_20, width, label="Normal Organic Politics Content", color="#aec7e8")
    ax.bar(x + width/2, cib_sovs_20, width, label="Coordinated CIB Content (20 Bots)", color="#d62728")
    ax.set_title("3. Spillover into Disjunct Demographics: CIB vs Organic Baseline (20 Bots)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in disjunct_cats], fontsize=11)
    ax.set_ylabel("Share of Recommendations (%)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Home Echo Chamber Degradation
    ax = axes[1, 1]
    for cat in TOPICS:
        homes = [all_results[b][cat]["home_share_of_voice"] for b in bot_conditions]
        ax.plot(bot_conditions, homes, label=f"{cat.capitalize()} Home Content", marker="^", linestyle="--", linewidth=2.0, color=colors[cat])
    ax.set_title("4. Demographic Home Topic Retention Under CIB Amplification", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coordinated CIB Bot Cluster Size", fontsize=11)
    ax.set_ylabel("Home Category Retention (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"Saved visualization dashboard to {output_png}")

if __name__ == "__main__":
    main()
