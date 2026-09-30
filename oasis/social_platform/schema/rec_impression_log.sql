-- This is the schema definition for the rec_impression_log table
CREATE TABLE IF NOT EXISTS rec_impression_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_index INTEGER,
    user_id INTEGER,
    post_id INTEGER,
    rank INTEGER,
    score REAL,
    timestamp TEXT
);

CREATE INDEX IF NOT EXISTS idx_rec_log_post ON rec_impression_log(post_id);
CREATE INDEX IF NOT EXISTS idx_rec_log_step ON rec_impression_log(step_index);
CREATE INDEX IF NOT EXISTS idx_rec_log_user ON rec_impression_log(user_id);
