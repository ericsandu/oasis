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

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
import pandas as pd  # noqa: E402

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


def _cascade_curves(db):
    """Build the repost cascade DIRECTLY from the post table and return
    (scale, depth, max_breadth) curves over the 150-minute horizon.

    This replaces visualization/.../graph.py:prop_graph, which crashes on a
    sparse/degenerate cascade (NetworkX 'node not in digraph' when the root was
    never added, 'max() on empty sequence' when there are no reposts, and
    pdb.set_trace() in its except blocks -- fatal in a batch job). The classic
    tool_choice=auto run produces ZERO reposts, which hit every one of those.

    Cascade model (paper F.2.2): each row in `post` is a node.
      - original_post_id IS NULL  -> source/root post (depth 0)
      - original_post_id = P      -> a repost of post P (edge P -> this)
    created_at is minutes since the source post.
    scale(t)       = number of posts in the cascade with created_at <= t
    depth(t)       = longest root->leaf path length among those posts
    max_breadth(t) = largest number of posts at any single depth level
    All three are monotonic step curves padded/truncated to HORIZON.
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        rows = cur.execute(
            "SELECT post_id, original_post_id, created_at FROM post "
            "ORDER BY created_at, post_id").fetchall()
    except Exception:
        rows = []
    finally:
        con.close()

    zero = np.zeros(HORIZON, dtype=float)
    if not rows:
        return zero.copy(), zero.copy(), zero.copy()

    # parent map + per-post arrival time (clamped to >=0)
    parent = {}
    created = {}
    roots = []
    for pid, opid, t in rows:
        t = max(0, int(t or 0))
        created[pid] = t
        if opid is None:
            roots.append(pid)
        else:
            parent[pid] = opid

    # depth of each post within the cascade tree (root depth = 0); any post
    # whose parent chain does not terminate at a known root is skipped.
    def post_depth(pid, _seen=None):
        _seen = _seen or set()
        d = 0
        cur_id = pid
        while cur_id in parent:
            if cur_id in _seen:        # cycle guard (should not happen)
                return None
            _seen.add(cur_id)
            cur_id = parent[cur_id]
            d += 1
            if cur_id not in created:  # dangling parent
                return None
        return d  # cur_id is a root

    depth_of = {}
    for pid in created:
        dd = post_depth(pid)
        if dd is not None:
            depth_of[pid] = dd

    scale = np.zeros(HORIZON, dtype=float)
    depth = np.zeros(HORIZON, dtype=float)
    mb = np.zeros(HORIZON, dtype=float)
    for t in range(HORIZON):
        present = [p for p in depth_of if created[p] <= t]
        scale[t] = len(present)
        if present:
            depth[t] = max(depth_of[p] for p in present)
            # breadth per level, max over levels
            counts = {}
            for p in present:
                counts[depth_of[p]] = counts.get(depth_of[p], 0) + 1
            mb[t] = max(counts.values())
    return scale, depth, mb


def stats(db, content=None):
    # content kept for signature compatibility; no longer needed.
    return _cascade_curves(db)


def nrmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = (b.max() - b.min()) or 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def real_curve(topic, stat):
    p = os.path.join(
        _REPO,
        f"data/twitter_dataset/real_world_prop_data/real_data_{stat}/{topic}.pkl")
    try:
        y = pickle.load(open(os.path.abspath(p), "rb"))
    except Exception as e:
        print(f"  (real-world {stat} pkl unavailable: {e})")
        return None
    y = list(y) + [y[-1] if len(y) else 0] * (HORIZON - len(y))
    return np.array(y[:HORIZON], dtype=float)


def main():
    classic_db, jev_db, topic = sys.argv[1], sys.argv[2], sys.argv[3]

    print("=== ACTION COUNTS ===")
    print("classic:", action_counts(classic_db))
    print("jev    :", action_counts(jev_db))

    cs, cd, cmb = stats(classic_db)
    js, jd, jmb = stats(jev_db)

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
        if r is None:
            continue
        print(f"{name:12s} classic={nrmse(c_, r):.3f}  jev={nrmse(j_, r):.3f}")


if __name__ == "__main__":
    main()
