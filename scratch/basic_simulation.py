import asyncio
import os
import random

import oasis
from oasis import ActionType, ManualAction, generate_reddit_agent_graph
from oasis.social_platform.typing import DefaultPlatformType


async def main():
    # We use a mock configuration by not passing an LLM model
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    # and using the pre-existing user data.
    print("Generating agent graph without LLM...")
    agent_graph = await generate_reddit_agent_graph(
        profile_path="./data/reddit/user_data_36.json",
        model=None,
        available_actions=ActionType.get_default_reddit_actions(),
    )

    db_path = os.path.abspath("./data/reddit_simulation.db")
    os.environ["OASIS_DB_PATH"] = db_path
    if os.path.exists(db_path):
        os.remove(db_path)

    print("Initializing environment (Mock Embedder / Reddit RecSys)...")
    env = oasis.make(
        agent_graph=agent_graph,
        platform=DefaultPlatformType.REDDIT, # Bypasses twhin-bert, uses hot-score
        database_path=db_path,
    )

    await env.reset()

    # Step 1: Initial organic seeding (agents 0-5 create posts)
    print("Step 1: Creating initial posts...")
    actions_step1 = {}
    for i in range(5):
        agent = env.agent_graph.get_agent(i)
        actions_step1[agent] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": f"Organic post {i} about everyday life."}
        )
    await env.step(actions_step1)

    # We can inspect the DB by querying it
    import sqlite3
    from oasis.social_platform.database import fetch_table_from_db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    posts = fetch_table_from_db(cursor, "post")
    print(f"Total posts in DB: {len(posts)}")
    
    # Step 2: Systematic behavior 
    # Let's make agents 5-15 systematically like all posts that mention "Organic"
    print("Step 2: Systematic behavior (Liking specific posts)...")
    target_post_ids = [str(p['post_id']) for p in posts if "Organic" in p['content']]
    
    actions_step2 = {}
    for i in range(5, 15):
        agent = env.agent_graph.get_agent(i)
        # Choose a random post from the targets to like
        if target_post_ids:
            chosen_post = random.choice(target_post_ids)
            actions_step2[agent] = ManualAction(
                action_type=ActionType.LIKE_POST,
                action_args={"post_id": chosen_post}
            )
            
    if actions_step2:
        await env.step(actions_step2)

    # Let's check likes in DB
    posts_after = fetch_table_from_db(cursor, "post")
    print("Post likes after Step 2:")
    for p in posts_after:
        print(f" Post {p['post_id']}: {p['num_likes']} likes")

    # Step 3: Comment raid
    print("Step 3: Systematic comment raid on the most popular post...")
    most_liked_post = max(posts_after, key=lambda x: x['num_likes'])
    actions_step3 = {}
    for i in range(15, 25):
        agent = env.agent_graph.get_agent(i)
        actions_step3[agent] = ManualAction(
            action_type=ActionType.CREATE_COMMENT,
            action_args={"post_id": str(most_liked_post['post_id']), "content": f"Bot comment from {agent.user_info.name}"}
        )
    await env.step(actions_step3)
    
    comments = fetch_table_from_db(cursor, "comment")
    print(f"Total comments on post {most_liked_post['post_id']}: {len(comments)}")

    await env.close()
    conn.close()
    print("Simulation complete.")

if __name__ == "__main__":
    asyncio.run(main())
