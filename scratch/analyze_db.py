# flake8: noqa
# ruff: noqa
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import sqlite3
os.environ["OPENAI_API_KEY"] = "sk-mock-key"


db_path = "./data/large_scale_v2.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Total Users
c.execute("SELECT COUNT(DISTINCT user_id) FROM user")
total_users = c.fetchone()[0]

# Total Posts
c.execute("SELECT COUNT(*) FROM post")
total_posts = c.fetchone()[0]

# Find Target Post (created by Agent 0)
c.execute("SELECT post_id, num_likes FROM post WHERE user_id = 0")
target_post = c.fetchone()
target_post_id = target_post[0] if target_post else None
target_likes = target_post[1] if target_post else 0

# Reach of Target Post
if target_post_id:
    c.execute("SELECT DISTINCT user_id FROM rec WHERE post_id = ?", (target_post_id,))
    reached_users = [r[0] for r in c.fetchall()]
else:
    reached_users = []

# Top Recommended Posts overall
c.execute("SELECT post_id, COUNT(user_id) as reach FROM rec GROUP BY post_id ORDER BY reach DESC LIMIT 5")
top_posts = c.fetchall()

print("--- RAW DATA ANALYSIS ---")
print(f"Total Users: {total_users}")
print(f"Total Organic Posts Generated (Phase 1): {total_posts}")
print(f"Target Post ID: {target_post_id}")
print(f"Total Bot Likes on Target Post: {target_likes}")
print(f"Target Post Reach (Unique Users): {len(reached_users)}")
print(f"Users who saw the Target Post: {reached_users}")
print(f"Top 5 Recommended Posts across all feeds: {top_posts}")

conn.close()
