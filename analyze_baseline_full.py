import sqlite3
import json
import re
import math
from collections import Counter, defaultdict

DB_PATH = "baseline_20260917_165655/baseline_simulation.db"

def tokenize(text):
    if not text or text == "nan":
        return []
    text = re.sub(r"[^\w\s#@]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2]

def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def gini(x):
    sorted_x = sorted(x)
    n = len(x)
    if sum(x) == 0:
        return 0.0
    num = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_x))
    den = n * sum(sorted_x)
    return num / den

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Macro Platform Metrics
    total_users = c.execute("SELECT COUNT(*) FROM user").fetchone()[0]
    total_posts = c.execute("SELECT COUNT(*) FROM post").fetchone()[0]
    orig_posts = c.execute("SELECT COUNT(*) FROM post WHERE original_post_id IS NULL").fetchone()[0]
    quote_posts = c.execute("SELECT COUNT(*) FROM post WHERE original_post_id IS NOT NULL").fetchone()[0]
    total_likes = c.execute("SELECT COUNT(*) FROM like").fetchone()[0]
    total_traces = c.execute("SELECT COUNT(*) FROM trace").fetchone()[0]

    # 2. Action distribution by step
    step_actions = defaultdict(Counter)
    for step, act in c.execute("SELECT created_at, action FROM trace").fetchall():
        step_actions[step][act] += 1

    # 3. User activity profiles
    user_action_counts = Counter(row[0] for row in c.execute("SELECT user_id FROM trace WHERE action != 'sign_up'").fetchall())
    action_counts_list = [user_action_counts[u] for u in range(total_users)]
    user_orig_posts = Counter(row[0] for row in c.execute("SELECT user_id FROM post WHERE original_post_id IS NULL AND user_id > 0").fetchall())
    user_quote_posts = Counter(row[0] for row in c.execute("SELECT user_id FROM post WHERE original_post_id IS NOT NULL").fetchall())

    gini_actions = gini(action_counts_list)
    gini_posts = gini([user_orig_posts[u] for u in range(total_users)])
    gini_quotes = gini([user_quote_posts[u] for u in range(total_users)])

    # 4. Cascade Tree & Diffusion Metrics
    cascades = c.execute("""
        SELECT original_post_id, COUNT(*) as cnt, COUNT(DISTINCT user_id) as uniq_users
        FROM post
        WHERE original_post_id IS NOT NULL
        GROUP BY original_post_id
        ORDER BY cnt DESC
    """).fetchall()

    cascade_details = []
    for orig_id, quote_cnt, uniq_users in cascades:
        root = c.execute("SELECT user_id, content, created_at FROM post WHERE post_id = ?", (orig_id,)).fetchone()
        author_bio = c.execute("SELECT bio FROM user WHERE user_id = ?", (root[0],)).fetchone()[0]
        
        quotes = c.execute("""
            SELECT p.post_id, p.user_id, p.created_at, p.quote_content, u.bio
            FROM post p
            JOIN user u ON p.user_id = u.user_id
            WHERE p.original_post_id = ?
            ORDER BY p.post_id
        """, (orig_id,)).fetchall()

        token_sets = [set(tokenize(q[3])) for q in quotes if q[3] and q[3] != "nan"]
        pairwise_jaccard = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                pairwise_jaccard.append(jaccard_similarity(token_sets[i], token_sets[j]))
        avg_pairwise_jaccard = sum(pairwise_jaccard) / len(pairwise_jaccard) if pairwise_jaccard else 0.0

        all_vocab = set()
        for ts in token_sets:
            all_vocab.update(ts)

        step_dist = Counter(q[2] for q in quotes)

        cascade_details.append({
            "post_id": orig_id,
            "author_id": root[0],
            "author_bio": author_bio,
            "content": root[1],
            "created_step": root[2],
            "quote_count": quote_cnt,
            "unique_quoting_users": uniq_users,
            "step_distribution": dict(step_dist),
            "lexical_diversity": {
                "vocab_size": len(all_vocab),
                "avg_pairwise_jaccard": round(avg_pairwise_jaccard, 4)
            },
            "sample_quotes": [
                {"user_id": q[1], "bio": q[4][:50] if q[4] else "", "quote": q[3]}
                for q in quotes[:5]
            ]
        })

    # 5. Thematic Clustering of Original Posts
    categories = {
        "Tech / AI / Software / Data": [
            "ai", "software", "tech", "data", "developer", "cloud", "code", "multi-agent",
            "systems", "algorithm", "python", "dev", "model", "cyber", "security", "saas"
        ],
        "Business / Marketing / CX / E-commerce": [
            "marketing", "cx", "customer", "retail", "ebiz", "brand", "conversion",
            "sales", "business", "seo", "ecommerce", "strategy", "funnel", "media"
        ],
        "Daily Life / Travel / Cats / Coffee / Wellness": [
            "coffee", "tea", "cat", "adventure", "travel", "morning", "park",
            "books", "run", "weather", "quiet", "beach", "vegan"
        ],
        "Culture / Arts / Music / Philosophy": [
            "music", "composer", "art", "film", "philosophy", "literature", "theatre",
            "human", "stories", "history"
        ],
        "Sports / Entertainment": [
            "game", "finish", "extra time", "marines", "sports", "football", "basketball", "tennis"
        ]
    }

    all_orig_posts = c.execute("SELECT post_id, user_id, content, created_at FROM post WHERE original_post_id IS NULL").fetchall()
    cat_counts = Counter()
    for p in all_orig_posts:
        text = (p[2] or "").lower()
        matched = False
        for cat, keywords in categories.items():
            if any(k in text for k in keywords):
                cat_counts[cat] += 1
                matched = True
                break
        if not matched:
            cat_counts["General / Conversational / Reflections"] += 1

    # 6. Recommender Dynamics (Gorse Cache vs Activity)
    rec_entries = c.execute("SELECT user_id, post_id FROM rec").fetchall()
    rec_post_dist = Counter(r[1] for r in rec_entries)

    # 7. Package everything into output
    results = {
        "macro_metrics": {
            "total_users": total_users,
            "total_posts": total_posts,
            "original_posts": orig_posts,
            "quote_posts": quote_posts,
            "total_likes": total_likes,
            "total_traces": total_traces,
            "interaction_ratio_quotes_to_likes": f"{quote_posts}:{total_likes} ({quote_posts/(quote_posts+total_likes)*100:.1f}% quotes)"
        },
        "step_actions": {str(k): dict(v) for k, v in sorted(step_actions.items())},
        "user_inequality_metrics": {
            "gini_all_actions": round(gini_actions, 4),
            "gini_original_posts": round(gini_posts, 4),
            "gini_quote_posts": round(gini_quotes, 4),
            "action_count_distribution": dict(Counter(action_counts_list)),
            "lurkers_count": action_counts_list.count(0),
            "lurker_percentage": round(action_counts_list.count(0) / total_users * 100, 2)
        },
        "content_categories": dict(cat_counts),
        "recommender_status": {
            "total_recommendations": len(rec_entries),
            "recommended_post_distribution": dict(rec_post_dist)
        },
        "cascades": cascade_details
    }

    with open("baseline_20260917_165655/deep_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Successfully generated deep_analysis_results.json")

if __name__ == "__main__":
    main()
