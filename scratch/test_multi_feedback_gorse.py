import asyncio
import httpx
from oasis.social_platform.gorse_client import GorseClient, format_iso_timestamp, extract_action_info

async def test_multi_feedback():
    print("--- 1. Testing timestamp formatting ---")
    ts_int = format_iso_timestamp(15)
    ts_epoch = format_iso_timestamp(1789480532)
    print(f"Int timestep (15): {ts_int}")
    print(f"Epoch timestamp (1789480532): {ts_epoch}")
    assert ts_int.endswith("Z") and "2026" in ts_int
    assert ts_epoch.endswith("Z")

    print("\n--- 2. Testing extract_action_info ---")
    info1 = extract_action_info({"action_info": '{"post_id": 42, "comment_id": 7}'})
    info2 = extract_action_info({"action_info": {"post_id": 99}})
    assert info1["post_id"] == 42
    assert info2["post_id"] == 99

    print("\n--- 3. Testing GorseClient multi-modal feedback insertion ---")
    client = GorseClient()
    test_user_id = "test_verify_user_99"
    test_posts = [
        {"post_id": 1001, "content": "Technology breakthrough in AI agents", "created_at": 1},
        {"post_id": 1002, "content": "Sports news and championship match", "created_at": 2},
        {"post_id": 1003, "content": "Cooking recipes and Italian pasta dishes", "created_at": 3},
        {"post_id": 1004, "content": "Negative content to be disliked", "created_at": 4},
        {"post_id": 1005, "content": "Reposted discussion topic", "created_at": 5},
    ]

    # Insert items
    await client.bulk_insert_users([{"user_id": test_user_id, "name": "verification_agent"}])
    await client.bulk_insert_items(test_posts)

    # Prepare trace records representing all engagement types:
    traces = [
        # Like
        {"user_id": test_user_id, "action": "like_post", "action_info": {"post_id": 1001, "like_id": 1}, "created_at": 10},
        # Repost
        {"user_id": test_user_id, "action": "repost", "action_info": {"reposted_id": 1005, "new_post_id": 2001}, "created_at": 11},
        # Comment
        {"user_id": test_user_id, "action": "create_comment", "action_info": {"post_id": 1003, "comment_id": 501, "content": "Great recipe!"}, "created_at": 12},
        # Dislike
        {"user_id": test_user_id, "action": "dislike_post", "action_info": {"post_id": 1004, "dislike_id": 1}, "created_at": 13},
        # Refresh (Impressions/Reads)
        {"user_id": test_user_id, "action": "refresh", "action_info": {"posts": [{"post_id": 1001}, {"post_id": 1002}, {"post_id": 1003}]}, "created_at": 14},
    ]

    await client.bulk_insert_feedback(traces)

    # Verify feedbacks in Gorse
    resp = await client._get(f"/user/{test_user_id}/feedback")
    print(f"\nRetrieved {len(resp)} feedback items for {test_user_id}:")
    feedback_types = set()
    for fb in resp:
        print(f"  - Item: {fb['ItemId']}, Type: {fb['FeedbackType']}, Timestamp: {fb['Timestamp']}")
        feedback_types.add(fb['FeedbackType'])

    expected_types = {"like", "repost", "comment", "dislike", "read"}
    missing = expected_types - feedback_types
    assert not missing, f"Missing expected feedback types in Gorse: {missing}"
    print(f"\nSUCCESS: All expected feedback types present: {feedback_types}")

    # Cleanup test user
    async with httpx.AsyncClient(base_url=client.base_url) as hc:
        await hc.delete(f"/api/user/{test_user_id}")
    print("Test user cleaned up successfully.")

if __name__ == "__main__":
    asyncio.run(test_multi_feedback())
