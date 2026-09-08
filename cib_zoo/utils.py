from typing import List, Dict, Any, Callable
import sqlite3

def get_target_posts(db_path: str, target_type: str, target_val: str) -> List[Dict[str, Any]]:
    """
    Query the sqlite db to find target posts based on a criteria.
    target_type can be 'post_id', 'user_id', or 'tag/category'.
    Returns a list of matching post rows.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM post")
    columns = [description[0] for description in cursor.description]
    all_posts = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()

    if target_type == 'post_id':
        return [p for p in all_posts if str(p['post_id']) == str(target_val)]
    elif target_type == 'user_id':
        return [p for p in all_posts if str(p['user_id']) == str(target_val)]
    elif target_type in ('tag', 'category'):
        # Just simple keyword matching in content
        return [p for p in all_posts if target_val.lower() in p['content'].lower()]
    
    return []
