#!/usr/bin/env python3
"""Compare classic-OASIS vs JEV simulation DBs for parity + similarity.

Usage (run inside the SIF so prop_graph deps are available):
    python3 compare_jev_parity.py <classic.db> <jev.db> <topic_name>

Reports, for each engine:
  - action counts (like/repost/follow/comment/refresh)
  - propagation scale / depth / max_breadth curves (paper metrics via prop_graph)
  - NRMSE of each engine vs the real-world ground-truth pkl
  - JEV-vs-classic similarity (NRMSE between the two engines) -> the parity number
"""
from __future__ import annotations
import os
import sys
import pickle
import sqlite3
import numpy as np

# prop_graph lives in the align visualization code
VIZ = os.path.join(os.path.dirname(__file__),
                   "../../visualization/twitter_simulation/align_with_real_world/code")
sys.path.insert(0, os.path.abspath(VIZ))
import pandas as pd  # noqa: E402
from graph import prop_graph  # noqa: E402

HORIZON = 150  # paper compares first 150 minutes (50 steps x 3 min)


def action_counts(db):
    c = sqlite3.connect(db).cursor()
    try:
        rows = c.execute(
            "SELECT action, COUNT(*) FROM trace GROUP BY action").fetchall()
    except Exception:
        rows = []
    d = {a: n for a, n in rows}
    # follows live in the follow table, not trace
    try:
        d["follow(table)"] = c.execute("SELECT COUNT(*) FROM follow").fetchone()[0]
    except Exception:
        pass
    return d


def stats(db, content):
    pg = prop_graph(content, db, viz=False)
    pg.build_graph()
    _, scale = pg.plot_scale_time()
    _, depth = pg.plot_depth_time()
    _, mb = pg.plot_max_breadth_time()

    def pad(x):
        x = list(x)
        x += [x[-1] if x else 0] * (HORIZON - len(x))
        return np.array(x[:HORIZON], dtype=float)
    return pad(scale), pad(depth), pad(mb)


def nrmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = (b.max() - b.min()) or 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def real_curve(topic, stat):
    p = os.path.join(
        os.path.dirname(__file__),
        f"../../data/twitter_dataset/real_world_prop_data/real_data_{stat}/{topic}.pkl")
    y = pickle.load(open(os.path.abspath(p), "rb"))
    y = list(y) + [y[-1]] * (HORIZON - len(y))
    return np.array(y[:HORIZON], dtype=float)


def main():
    classic_db, jev_db, topic = sys.argv[1], sys.argv[2], sys.argv[3]
    topics = pd.read_csv(os.path.join(
        os.path.dirname(__file__), "../../data/twitter_dataset/all_topics.csv"))
    content = topics[topics["topic_name"] == topic]["source_tweet"].item()

    print("=== ACTION COUNTS ===")
    print("classic:", action_counts(classic_db))
    print("jev    :", action_counts(jev_db))

    cs, cd, cmb = stats(classic_db, content)
    js, jd, jmb = stats(jev_db, content)

    print("\n=== FINAL VALUES (scale / depth / max_breadth) ===")
    print(f"classic: {cs[-1]:.0f} / {cd.max():.0f} / {cmb[-1]:.0f}")
    print(f"jev    : {js[-1]:.0f} / {jd.max():.0f} / {jmb[-1]:.0f}")

    print("\n=== JEV vs CLASSIC similarity (NRMSE, lower=closer) -> PARITY ===")
    print(f"scale       : {nrmse(js, cs):.3f}")
    print(f"depth       : {nrmse(jd, cd):.3f}")
    print(f"max_breadth : {nrmse(jmb, cmb):.3f}")

    print("\n=== vs REAL-WORLD ground truth (NRMSE; paper reports ~0.30) ===")
    for name, (c_, j_) in {
        "scale": (cs, js), "depth": (cd, jd), "max_breadth": (cmb, jmb)
    }.items():
        r = real_curve(topic, name)
        print(f"{name:12s} classic={nrmse(c_, r):.3f}  jev={nrmse(j_, r):.3f}")


if __name__ == "__main__":
    main()
