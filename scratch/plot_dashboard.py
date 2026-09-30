import json
import matplotlib.pyplot as plt
import numpy as np

def plot_dashboard():
    with open("scratch/clean_benchmark_summary.json", "r") as f:
        data = json.load(f)

    bot_conditions = sorted(list(set(d["bots"] for d in data)))
    topics = ["tech", "sports", "politics", "entertainment"]
    colors = {"tech": "#1f77b4", "sports": "#2ca02c", "politics": "#d62728", "entertainment": "#9467bd"}
    
    # Organize data by demographic
    by_demo = {t: {b: None for b in bot_conditions} for t in topics}
    for row in data:
        by_demo[row["demographic"]][row["bots"]] = row

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Penetration Rate (% of Demographic Users Exposed to CIB in Top-10)
    ax = axes[0, 0]
    for t in topics:
        pens = [by_demo[t][b]["penetration_rate"] for b in bot_conditions]
        ax.plot(bot_conditions, pens, label=f"{t.capitalize()} Demographic", marker="o", linewidth=2.5, color=colors[t])
    ax.set_title("1. CIB Demographic Penetration (% Users with CIB in Feed)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("% Users Exposed", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. CIB Share of Voice (% of Top-10 Recommendations that are CIB)
    ax = axes[0, 1]
    for t in topics:
        sovs = [by_demo[t][b]["cib_share_of_voice"] for b in bot_conditions]
        ax.plot(bot_conditions, sovs, label=f"{t.capitalize()} Feed", marker="s", linewidth=2.5, color=colors[t])
    ax.set_title("2. CIB Share of Voice in Recommendations (% Feed Items)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("CIB Share of Voice (%)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Disjunct Group Spillover (Bots=20): CIB vs Organic Politics
    ax = axes[1, 0]
    disjunct = ["tech", "sports", "entertainment"]
    x = np.arange(len(disjunct))
    width = 0.35
    target_bots = 20 if 20 in bot_conditions else bot_conditions[-1]
    cib_vals = [by_demo[d][target_bots]["cib_share_of_voice"] for d in disjunct]
    norm_vals = [by_demo[d][target_bots]["norm_politics_share"] for d in disjunct]
    
    ax.bar(x - width/2, norm_vals, width, label="Organic Politics Content", color="#aec7e8", edgecolor="#1f77b4")
    ax.bar(x + width/2, cib_vals, width, label=f"Coordinated CIB Content ({target_bots} Bots)", color="#d62728", edgecolor="#7f0000")
    ax.set_title(f"3. Spillover into Disjunct Cohorts: CIB vs Organic Politics ({target_bots} Bots)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in disjunct], fontsize=11)
    ax.set_ylabel("Share of Recommendations (%)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Echo Chamber Erosion: Home Topic Share of Voice
    ax = axes[1, 1]
    for t in topics:
        homes = [by_demo[t][b]["home_share_of_voice"] for b in bot_conditions]
        ax.plot(bot_conditions, homes, label=f"{t.capitalize()} Home Content", marker="^", linestyle="--", linewidth=2.2, color=colors[t])
    ax.set_title("4. Demographic Echo Chamber Retention vs CIB Intrusion", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coordinated Bot Seed Size", fontsize=11)
    ax.set_ylabel("Home Topic Retention (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_png = "scratch/cib_4topic_propagation_dashboard.png"
    plt.savefig(output_png, dpi=150)
    print(f"Visualization dashboard saved to: {output_png}")

if __name__ == "__main__":
    plot_dashboard()
