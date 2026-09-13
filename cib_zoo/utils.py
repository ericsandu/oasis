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
from typing import Any, Dict, List


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
