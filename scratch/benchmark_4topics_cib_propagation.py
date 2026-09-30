import asyncio
import httpx
import json
import time
import uuid
import random
import os
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "http://127.0.0.1:8088"

TOPICS = ["tech", "sports", "politics", "entertainment"]
NUM_USERS_PER_TOPIC = 25  # 100 organic users total
NUM_POSTS_PER_TOPIC = 20  # 80 organic posts total
NUM_CIB_POSTS = 5         # 5 CIB posts in politics
CIB_TOPIC = "politics"

POST_TEMPLATES = {
    "tech": [
        "New release of quantum computing SDK brings 100x simulation speedups",
        "Autonomous agent architectures for distributed software verification",
        "Semiconductor manufacturing advances enable 1nm chip fabrication",
        "Neural network pruning techniques reduce edge inference memory footprint",
        "Next-generation compiler optimizations in Rust and LLVM toolchains"
    ],
    "sports": [
        "Historic comeback in premier league derby leaves title race wide open",
        "Olympic sprinter breaks world record with unprecedented 100m sprint",
        "Championship playoffs enter double overtime in thrilling tactical showdown",
        "Tactical analysis of modern high-pressing football defensive schemes",
        "Marathon training regimen: physiological adaptations to endurance pacing"
    ],
    "politics": [
        "Parliament passes landmark infrastructure investment and taxation bill",
        "Local governance reform sparks debate over municipal zoning laws",
        "Diplomatic summit concludes with multilateral trade agreement framework",
        "Fiscal policy review assesses inflation impacts across consumer baskets",
        "Supreme Court hears oral arguments on constitutional regulatory power"
    ],
    "entertainment": [
        "Acclaimed director announces psychological thriller cinematic sequel",
        "Indie game reaches two million downloads through emergent multiplayer",
        "Original sound track wins prestigious award at international music gala",
        "Behind the scenes of modern virtual production and real-time CGI sets",
        "Streaming studio greenlights ten-episode adaptation of sci-fi novel"
    ]
}

CIB_TEMPLATES = [
    "LEAKED DOCUMENTS: Official election audit confirms secret manipulation scheme",
    "URGENT ALERT: Unchecked deep-state agenda exposed by whistleblowers today",
    "THE TRUTH THEY HIDE: Classified memo proves foreign interference in our votes",
    "CRITICAL EVIDENCE: Coordinated cover-up revealed behind national policy collapse",
    "SHOCKING CORRUPTION: Insiders blow the lid on covert institutional conspiracy"
]

def make_items(run_id):
    items = []
    # 1. Organic items
    for cat in TOPICS:
        for i in range(NUM_POSTS_PER_TOPIC):
            item_id = f"{run_id}_{cat}_{i}"
            text = POST_TEMPLATES[cat][i % len(POST_TEMPLATES[cat])]
            keywords = [w.lower() for w in text.split() if len(w) > 3]
            items.append({
                "ItemId": item_id,
                "Categories": [cat],
                "Timestamp": "2026-01-01T10:00:00Z",
                "Labels": keywords
            })
    # 2. CIB items (Strictly in CIB_TOPIC, no cross-tagging, pure baseline)
    for i in range(NUM_CIB_POSTS):
        item_id = f"{run_id}_cib_{i}"
        text = CIB_TEMPLATES[i % len(CIB_TEMPLATES)]
        keywords = [w.lower() for w in text.split() if len(w) > 3]
        items.append({
            "ItemId": item_id,
            "Categories": [CIB_TOPIC],
            "Timestamp": "2026-01-01T10:00:00Z",
            "Labels": keywords
        })
    return items

def make_users(run_id, num_bots):
    users = []
    user_cohorts = {}
    
    for cat in TOPICS:
        cohort_users = []
        for i in range(NUM_USERS_PER_TOPIC):
            uid = f"{run_id}_u_{cat}_{i}"
            users.append({
                "UserId": uid,
                "Labels": [cat, "organic"]
            })
            cohort_users.append(uid)
        user_cohorts[cat] = cohort_users
        
    bots = []
    for b in range(num_bots):
        bot_id = f"{run_id}_bot_{b}"
        users.append({
            "UserId": bot_id,
            "Labels": [CIB_TOPIC, "bot"]
        })
        bots.append(bot_id)
    user_cohorts["bot"] = bots
    return users, user_cohorts

def make_feedbacks(run_id, user_cohorts, num_bots, bot_intensity=1.0):
    feedbacks = []
    ts_organic = "2026-01-01T12:00:00Z"
    
    # 1. Organic Interactions
    # Each organic user likes ~5 items in their home category, plus 0-1 random item from other categories
    for cat in TOPICS:
        home_item_ids = [f"{run_id}_{cat}_{i}" for i in range(NUM_POSTS_PER_TOPIC)]
        other_item_ids = [f"{run_id}_{o_cat}_{i}" for o_cat in TOPICS if o_cat != cat for i in range(NUM_POSTS_PER_TOPIC)]
        
        for u in user_cohorts[cat]:
            # Home category likes
            chosen_home = random.sample(home_item_ids, k=min(5, len(home_item_ids)))
            for item_id in chosen_home:
                feedbacks.append({
                    "FeedbackType": "like",
                    "UserId": u,
                    "ItemId": item_id,
                    "Timestamp": ts_organic
                })
                # Read impression
                feedbacks.append({
                    "FeedbackType": "read",
                    "UserId": u,
                    "ItemId": item_id,
                    "Timestamp": ts_organic
                })
            # Exploration / noise (10% chance to like an item from another category)
            if random.random() < 0.15:
                noise_item = random.choice(other_item_ids)
                feedbacks.append({
                    "FeedbackType": "like",
                    "UserId": u,
                    "ItemId": noise_item,
                    "Timestamp": ts_organic
                })

    # 2. CIB Bot Interactions (Pure coordination: bots like CIB items)
    cib_item_ids = [f"{run_id}_cib_{i}" for i in range(NUM_CIB_POSTS)]
    ts_cib = "2026-01-01T12:05:00Z"
    for b in user_cohorts["bot"]:
        # Each bot likes all or subset of CIB posts
        num_likes = max(1, int(len(cib_item_ids) * bot_intensity))
        liked_cib = random.sample(cib_item_ids, k=num_likes)
        for item_id in liked_cib:
            feedbacks.append({
                "FeedbackType": "like",
                "UserId": b,
                "ItemId": item_id,
                "Timestamp": ts_cib
            })
            feedbacks.append({
                "FeedbackType": "repost",
                "UserId": b,
                "ItemId": item_id,
                "Timestamp": ts_cib
            })
            feedbacks.append({
                "FeedbackType": "read",
                "UserId": b,
                "ItemId": item_id,
                "Timestamp": ts_cib
            })
    return feedbacks

async def run_trial(client, num_bots, trial_id=0):
    run_id = f"t{trial_id}_{uuid.uuid4().hex[:5]}"
    items = make_items(run_id)
    users, user_cohorts = make_users(run_id, num_bots)
    feedbacks = make_feedbacks(run_id, user_cohorts, num_bots)
    
    # 1. Ingest into Gorse
    await client.post("/api/items", json=items)
    await client.post("/api/users", json=users)
    
    # Batch feedbacks in chunks of 500
    for i in range(0, len(feedbacks), 500):
        await client.post("/api/feedback", json=feedbacks[i:i+500])
        
    # Allow background ranking cycle
    await asyncio.sleep(1.0)
    
    # 2. Measure Recommendations across each demographic separately
    results_by_demographic = {}
    
    for cat in TOPICS:
        cohort_users = user_cohorts[cat]
        cib_exposed_users = 0
        total_recs = 0
        cib_recs_count = 0
        home_recs_count = 0
        disjunct_recs_count = {other: 0 for other in TOPICS if other != cat}
        
        for u in cohort_users:
            resp = await client.get(f"/api/recommend/{u}?n=10")
            rec_ids = resp.json() if resp.status_code == 200 and resp.json() else []
            total_recs += len(rec_ids)
            
            has_cib = False
            for item_id in rec_ids:
                if "_cib_" in item_id:
                    cib_recs_count += 1
                    has_cib = True
                elif f"_{cat}_" in item_id:
                    home_recs_count += 1
                else:
                    for o_cat in disjunct_recs_count:
                        if f"_{o_cat}_" in item_id:
                            disjunct_recs_count[o_cat] += 1
                            break
            if has_cib:
                cib_exposed_users += 1
                
        user_penetration_rate = (cib_exposed_users / len(cohort_users)) if cohort_users else 0.0
        cib_share_of_voice = (cib_recs_count / total_recs) if total_recs > 0 else 0.0
        home_share_of_voice = (home_recs_count / total_recs) if total_recs > 0 else 0.0
        
        # Average non-cib politics penetration into non-politics cohorts
        non_cib_politics_share = 0.0
        if cat != "politics":
            politics_count = disjunct_recs_count.get("politics", 0)
            non_cib_politics_share = (politics_count / total_recs) if total_recs > 0 else 0.0
            
        results_by_demographic[cat] = {
            "penetration_rate": user_penetration_rate,
            "cib_share_of_voice": cib_share_of_voice,
            "home_share_of_voice": home_share_of_voice,
            "disjunct_shares": {k: (v / total_recs if total_recs > 0 else 0.0) for k, v in disjunct_recs_count.items()},
            "non_cib_politics_share": non_cib_politics_share
        }
        
    return results_by_demographic

async def main_experiment():
    print("==========================================================================")
    print("STARTING 4-TOPIC GORSE CIB PROPAGATION BENCHMARK (NO EXPLOIT TRICKS)")
    print("Topics: Tech, Sports, Politics (CIB origin), Entertainment")
    print(f"Demographics: {NUM_USERS_PER_TOPIC} users per category (Total {NUM_USERS_PER_TOPIC * 4} organic)")
    print("==========================================================================")
    
    bot_counts = [0, 5, 10, 20, 40]
    trials_per_bot_count = 3  # 15 complete simulation runs
    
    aggregated_results = {b: {cat: {"pen": [], "sov": [], "home": [], "norm_pol": []} for cat in TOPICS} for b in bot_counts}
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        trial_counter = 0
        for b in bot_counts:
            print(f"\n>>> Running Condition: Bot Cluster Size = {b} ({trials_per_bot_count} independent trials)...")
            for t in range(trials_per_bot_count):
                trial_counter += 1
                res = await run_trial(client, num_bots=b, trial_id=trial_counter)
                for cat in TOPICS:
                    aggregated_results[b][cat]["pen"].append(res[cat]["penetration_rate"])
                    aggregated_results[b][cat]["sov"].append(res[cat]["cib_share_of_voice"])
                    aggregated_results[b][cat]["home"].append(res[cat]["home_share_of_voice"])
                    aggregated_results[b][cat]["norm_pol"].append(res[cat]["non_cib_politics_share"])
                print(f"  [Trial {t+1}/{trials_per_bot_count}] Done.")

    print("\n==========================================================================")
    print("BENCHMARK RESULTS BY DEMOGRAPHIC:")
    print("==========================================================================")
    
    print(f"{'Bots':<6} | {'Demographic':<14} | {'CIB Pen %':<10} | {'CIB SoV %':<10} | {'Home SoV %':<11} | {'Organic Pol %':<14} | {'Amplification':<13}")
    print("-" * 88)
    
    table_data = []
    for b in bot_counts:
        for cat in TOPICS:
            m_pen = np.mean(aggregated_results[b][cat]["pen"]) * 100
            m_sov = np.mean(aggregated_results[b][cat]["sov"]) * 100
            m_home = np.mean(aggregated_results[b][cat]["home"]) * 100
            m_norm_pol = np.mean(aggregated_results[b][cat]["norm_pol"]) * 100
            amplification = (m_sov / m_norm_pol) if m_norm_pol > 0 else (m_sov if m_sov > 0 else 1.0)
            
            print(f"{b:<6} | {cat:<14} | {m_pen:>8.1f}% | {m_sov:>8.2f}% | {m_home:>9.1f}% | {m_norm_pol:>12.2f}% | {amplification:>11.2f}x")
            table_data.append({
                "bots": b,
                "demographic": cat,
                "pen": m_pen,
                "sov": m_sov,
                "home": m_home,
                "norm_pol": m_norm_pol,
                "amplification": amplification
            })
        print("-" * 88)

    # Generate Visualization Dashboard
    output_image_path = "scratch/cib_4topic_propagation_dashboard.png"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Penetration Rate vs Bot Size for Each Demographic
    ax = axes[0, 0]
    colors = {"tech": "#1f77b4", "sports": "#2ca02c", "politics": "#d62728", "entertainment": "#9467bd"}
    for cat in TOPICS:
        pens = [np.mean(aggregated_results[b][cat]["pen"]) * 100 for b in bot_counts]
        stds = [np.std(aggregated_results[b][cat]["pen"]) * 100 for b in bot_counts]
        ax.errorbar(bot_counts, pens, yerr=stds, label=f"Demographic: {cat.capitalize()}",
                    marker="o", linewidth=2.5, capsize=4, color=colors[cat])
    ax.set_title("CIB Exposure Penetration Across Demographics (% Users Exposed)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("Penetration Rate (%)", fontsize=11)
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. CIB Share of Voice vs Bot Size
    ax = axes[0, 1]
    for cat in TOPICS:
        sovs = [np.mean(aggregated_results[b][cat]["sov"]) * 100 for b in bot_counts]
        ax.plot(bot_counts, sovs, label=f"{cat.capitalize()} Feed", marker="s", linewidth=2.5, color=colors[cat])
    ax.set_title("CIB Share of Voice in Recommendations (% Feed Items)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("Share of Voice (%)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Disjunct Spillover Comparison (CIB vs Organic Politics into Disjunct Groups at Bots=20)
    ax = axes[1, 0]
    disjunct_cats = ["sports", "tech", "entertainment"]
    x = np.arange(len(disjunct_cats))
    width = 0.35
    cib_vals_20 = [np.mean(aggregated_results[20][c]["sov"]) * 100 for c in disjunct_cats]
    norm_vals_20 = [np.mean(aggregated_results[20][c]["norm_pol"]) * 100 for c in disjunct_cats]
    
    ax.bar(x - width/2, norm_vals_20, width, label="Normal Organic Politics Posts", color="#aec7e8")
    ax.bar(x + width/2, cib_vals_20, width, label="Coordinated CIB Posts (20 Bots)", color="#d62728")
    ax.set_title("Spillover into Disjunct Demographics: CIB vs Organic Baseline (20 Bots)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in disjunct_cats], fontsize=11)
    ax.set_ylabel("Share of Recommendations (%)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Echo Chamber Resistance / Home Share of Voice under CIB Pressure
    ax = axes[1, 1]
    for cat in TOPICS:
        homes = [np.mean(aggregated_results[b][cat]["home"]) * 100 for b in bot_counts]
        ax.plot(bot_counts, homes, label=f"{cat.capitalize()} Home Content", marker="^", linewidth=2, linestyle="--", color=colors[cat])
    ax.set_title("Home Category Retention Under Increasing CIB Bot Volume", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("Home Share of Voice (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    print(f"\nDashboard saved to: {output_image_path}")

    # Save summary JSON for analysis
    summary_path = "scratch/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"Summary data saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main_experiment())
