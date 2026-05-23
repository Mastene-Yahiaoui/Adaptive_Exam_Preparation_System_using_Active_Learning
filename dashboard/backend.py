import sqlite3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(db_path=None):
    db_path = Path(db_path) if db_path is not None else Path('data') / 'al_sessions.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            student_id TEXT,
            first_name TEXT,
            last_name TEXT,
            qid TEXT,
            question_text TEXT,
            answer INTEGER,
            features TEXT,
            ts TEXT
        )
        '''
    )
    existing_columns = {row[1] for row in cur.execute('PRAGMA table_info(answers)').fetchall()}
    if 'first_name' not in existing_columns:
        cur.execute('ALTER TABLE answers ADD COLUMN first_name TEXT')
    if 'last_name' not in existing_columns:
        cur.execute('ALTER TABLE answers ADD COLUMN last_name TEXT')
    conn.commit()
    conn.close()
    return db_path


def save_answer(db_path, session_id, student_id, qid, question_text, answer, features=None, first_name='', last_name=''):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO answers (session_id, student_id, first_name, last_name, qid, question_text, answer, features, ts) VALUES (?,?,?,?,?,?,?,?,?)',
        (
            session_id,
            student_id,
            first_name,
            last_name,
            str(qid),
            question_text,
            int(answer),
            json.dumps(features) if features is not None else None,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_session_answers(db_path, session_id):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute('SELECT qid, question_text, answer, features, ts FROM answers WHERE session_id=? ORDER BY id', (session_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def student_has_attempted(db_path, student_id, exclude_session_id=None):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    if exclude_session_id:
        cur.execute(
            "SELECT 1 FROM answers WHERE student_id=? AND qid <> 'INITIAL' AND session_id <> ? LIMIT 1",
            (str(student_id), str(exclude_session_id)),
        )
    else:
        cur.execute(
            "SELECT 1 FROM answers WHERE student_id=? AND qid <> 'INITIAL' LIMIT 1",
            (str(student_id),),
        )
    row = cur.fetchone()
    conn.close()
    return row is not None


def list_sessions(db_path):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT session_id FROM answers')
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def export_csv(db_path, out_path):
    import csv

    rows = []
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute('SELECT session_id, student_id, first_name, last_name, qid, question_text, answer, features, ts FROM answers ORDER BY id')
    rows = cur.fetchall()
    conn.close()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'student_id', 'first_name', 'last_name', 'qid', 'question_text', 'answer', 'features', 'ts'])
        for r in rows:
            writer.writerow(r)
    return out_path


def get_all_answers_df(db_path):
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(
            'SELECT session_id, student_id, first_name, last_name, qid, question_text, answer, features, ts FROM answers ORDER BY id',
            conn,
        )
    finally:
        conn.close()


def list_session_summaries(db_path):
    answers = get_all_answers_df(db_path)
    if answers.empty:
        return []

    summaries = []
    for session_id, group in answers.groupby('session_id', sort=False):
        real_rows = group[group['qid'] != 'INITIAL'].copy()
        student_ids = [sid for sid in group['student_id'].dropna().astype(str).tolist() if sid]
        first_names = [name for name in group.get('first_name', pd.Series(dtype=str)).dropna().astype(str).tolist() if name]
        last_names = [name for name in group.get('last_name', pd.Series(dtype=str)).dropna().astype(str).tolist() if name]
        student_id = student_ids[0] if student_ids else ''
        first_name = first_names[0] if first_names else ''
        last_name = last_names[0] if last_names else ''

        total_answers = int(len(real_rows))
        accuracy = float(real_rows['answer'].mean()) if total_answers else 0.0

        # Try to read estimated_score from the last answer's features
        estimated_score = None
        simulation_score = None
        total_questions = None
        if total_answers:
            try:
                last_features = json.loads(real_rows.iloc[-1]['features']) if real_rows.iloc[-1]['features'] else {}
                estimated_score = last_features.get('estimated_score')
                simulation_score = last_features.get('simulation_score')
                total_questions = last_features.get('total_questions')
            except Exception:
                pass

        summaries.append({
            'session_id': session_id,
            'student_id': student_id,
            'first_name': first_name,
            'last_name': last_name,
            'full_name': ' '.join(part for part in [first_name, last_name] if part).strip(),
            'answers': total_answers,
            'accuracy': accuracy,
            'estimated_score': estimated_score,
            'simulation_score': simulation_score,
            'total_questions': total_questions,
        })

    return summaries


def get_global_session_stats(db_path):
    summaries = list_session_summaries(db_path)
    if not summaries:
        return {
            'sessions': 0,
            'students': 0,
            'answers': 0,
            'mean_accuracy': 0.0,
        }

    df = pd.DataFrame(summaries)
    return {
        'sessions': int(len(df)),
        'students': int(df['student_id'].nunique()),
        'answers': int(df['answers'].sum()),
        'mean_accuracy': float(df['accuracy'].mean()),
    }
