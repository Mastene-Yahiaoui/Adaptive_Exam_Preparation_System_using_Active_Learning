import sqlite3
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
try:
    import textstat
except Exception:
    textstat = None

try:
    from typing import Optional
except Exception:
    pass


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


def get_question_bank_df(base_dir=None):
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent / 'Dataset'
    candidates = [
        base / 'questions.csv',
        base / 'clean' / 'question_bank.csv',
        base / 'clean' / 'question_bank_app.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame(columns=['Question ID', 'difficulty'])


def _difficulty_to_level(value):
    mapping = {'easy': 1, 'medium': 2, 'hard': 3}
    if pd.isna(value):
        return None
    try:
        return int(mapping.get(str(value).strip().lower(), 2))
    except Exception:
        return 2


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

        difficulty_levels = []
        complexity_levels = []
        response_times = []
        for _, row in real_rows.iterrows():
            try:
                features = json.loads(row['features']) if row['features'] else {}
            except Exception:
                features = {}
            try:
                complexity = features.get('qstn_complexity', None)
                if complexity is not None:
                    complexity_levels.append(float(complexity))
            except Exception:
                pass
            try:
                level = features.get('difficulty_encoded', None)
                if level is not None:
                    difficulty_levels.append(float(level))
            except Exception:
                pass
            try:
                response_times.append(float(features.get('response_time', 0.0)))
            except Exception:
                pass

        total_answers = int(len(real_rows))
        accuracy = float(real_rows['answer'].mean()) if total_answers else 0.0
        mean_level = float(np.mean(difficulty_levels)) if difficulty_levels else float(np.mean(complexity_levels)) if complexity_levels else 0.0
        mean_response_time = float(np.mean(response_times)) if response_times else 0.0

        summaries.append({
            'session_id': session_id,
            'student_id': student_id,
            'first_name': first_name,
            'last_name': last_name,
            'full_name': ' '.join(part for part in [first_name, last_name] if part).strip(),
            'answers': total_answers,
            'accuracy': accuracy,
            'mean_quiz_level': mean_level,
            'mean_response_time': mean_response_time,
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
            'mean_quiz_level': 0.0,
            'mean_response_time': 0.0,
        }

    df = pd.DataFrame(summaries)
    return {
        'sessions': int(len(df)),
        'students': int(df['student_id'].nunique()),
        'answers': int(df['answers'].sum()),
        'mean_accuracy': float(df['accuracy'].mean()),
        'mean_quiz_level': float(df['mean_quiz_level'].mean()),
        'mean_response_time': float(df['mean_response_time'].mean()),
    }


class AlClass:
    """Active Learning engine moved into backend for single-module usage.
    This class keeps most of the original AlClass behavior but uses Path
    objects for directories and is intended to be called from the Streamlit app.
    """
    def __init__(self, model_dir: Optional[str] = None, data_dir: Optional[str] = None):
        base = Path(__file__).resolve().parent
        self.models_dir = Path(model_dir) if model_dir is not None else base.parent / 'models'
        self.data_dir = Path(data_dir) if data_dir is not None else base.parent / 'Dataset'
        # database path for answers (SQLite)
        self.db_path = base.parent / 'data' / 'al_sessions.db'
        # ensure DB exists
        init_db(self.db_path)

        # load artifacts (best-effort)
        try:
            self.cp_model = joblib.load(self.models_dir / 'correctness_predictor.pkl')
        except Exception:
            self.cp_model = None
        try:
            self.scaler = joblib.load(self.models_dir / 'scaler.pkl')
        except Exception:
            self.scaler = None
        try:
            self.le_topic = joblib.load(self.models_dir / 'al_topic_label_encoder.pkl')
        except Exception:
            self.le_topic = None
        try:
            with open(self.models_dir / 'features.json') as f:
                self.features = json.load(f)
        except Exception:
            self.features = None
        try:
            with open(self.models_dir / 'config.json') as f:
                self.config = json.load(f)
        except Exception:
            # defaults
            self.config = {
                'difficulty_map': {'easy': 1, 'medium': 2, 'hard': 3},
                'uncertain_low': 0.3,
                'uncertain_high': 0.7,
                'max_questions': 20,
            }

        self.difficulty_map = self.config.get('difficulty_map', {'easy': 1, 'medium': 2, 'hard': 3})
        self.uncertain_low = self.config.get('uncertain_low', 0.3)
        self.uncertain_high = self.config.get('uncertain_high', 0.7)
        self.max_questions = self.config.get('max_questions', 20)

        self.questions_file = self.data_dir / 'questions.csv'
        self.answers_file = self.data_dir / 'answers.csv'

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.init_csv_files()

    def init_csv_files(self):
        cols = ['Question ID', 'question_text', 'options', 'correct_answer', 'topic', 'difficulty', 'num_words', 'qstn_complexity']
        if not self.questions_file.exists():
            qb_path = self.data_dir / 'clean' / 'question_bank.csv'
            if qb_path.exists():
                df = pd.read_csv(qb_path)
                df = df.apply(lambda row: pd.Series(self.compute_question_features(row.to_dict())), axis=1)
                try:
                    df[cols].to_csv(self.questions_file, index=False)
                except Exception:
                    df.to_csv(self.questions_file, index=False)
            else:
                pd.DataFrame(columns=cols).to_csv(self.questions_file, index=False)

        # answers are stored in SQLite (see init_db). No CSV answers file needed.

    def compute_question_features(self, question: dict):
        text = str(question.get('question_text', ''))
        question['num_words'] = len(text.split())
        try:
            question['qstn_complexity'] = float(textstat.flesch_kincaid_grade(text))
        except Exception:
            question['qstn_complexity'] = 0.0
        return question

    def get_question_details(self, question_id: str):
        df = pd.read_csv(self.questions_file)
        row = df[df['Question ID'] == question_id]
        if row.empty:
            raise ValueError(f'Question ID {question_id} not found.')
        q = row.iloc[0]
        return q['topic'], q['difficulty'], q['num_words'], q['qstn_complexity'], q.get('correct_answer', '')

    def _get_metrics_from_history(self, history, current_topic=None):
        if not history:
            return {
                'global_correctness': 0.5, 'easy_correct_avg': 0.5, 'medium_correct_avg': 0.5,
                'hard_correct_avg': 0.5, 'avg_response_time': 0.0, 'questions_answered_so_far': 0,
                'topic_correctness': 0.5
            }
        df = pd.DataFrame(history)
        metrics = {
            'global_correctness': float(df['correct'].mean()),
            'avg_response_time': float(df['response_time'].mean()),
            'questions_answered_so_far': len(df),
            'easy_correct_avg': float(df[df['diff_enc'] == 1]['correct'].mean()) if not df[df['diff_enc'] == 1].empty else 0.5,
            'medium_correct_avg': float(df[df['diff_enc'] == 2]['correct'].mean()) if not df[df['diff_enc'] == 2].empty else 0.5,
            'hard_correct_avg': float(df[df['diff_enc'] == 3]['correct'].mean()) if not df[df['diff_enc'] == 3].empty else 0.5,
        }
        if current_topic:
            topic_df = df[df['topic'] == current_topic]
            metrics['topic_correctness'] = float(topic_df['correct'].mean()) if not topic_df.empty else 0.5
        return metrics

    def _build_vector_internal(self, metrics, question_row):
        diff_enc = int(question_row.get('diff_enc', self.difficulty_map.get(question_row.get('difficulty', 'medium'), 2)))
        topic = question_row.get('topic', 'N/A')
        known_topics = set(self.le_topic.classes_) if self.le_topic is not None else set()
        topic_enc = int(self.le_topic.transform([topic])[0]) if topic in known_topics else -1
        return [
            metrics.get('global_correctness', 0.5),
            metrics.get('topic_correctness', 0.5),
            metrics.get('easy_correct_avg', 0.5),
            metrics.get('medium_correct_avg', 0.5),
            metrics.get('hard_correct_avg', 0.5),
            metrics.get('avg_response_time', 0.0),
            metrics.get('questions_answered_so_far', 0),
            diff_enc,
            topic_enc,
            float(question_row.get('num_words', 10)),
            float(question_row.get('qstn_complexity', 5.0))
        ]

    def get_question_pool(self):
        pool = pd.read_csv(self.questions_file)
        pool['diff_enc'] = pool['difficulty'].map(self.difficulty_map).fillna(2).astype(int)
        if self.le_topic is not None:
            known_topics = set(self.le_topic.classes_)
            pool['topic_enc'] = pool['topic'].apply(lambda t: int(self.le_topic.transform([t])[0]) if t in known_topics else -1)
        else:
            pool['topic_enc'] = -1
        return pool

    def build_feature_vector(self, student_id, history, question_row):
        metrics = self._get_metrics_from_history(history, current_topic=question_row.get('topic'))
        return self._build_vector_internal(metrics, question_row)

    def score_pool(self, student_id, history, pool):
        if len(pool) == 0:
            return np.array([])
        rows = [self.build_feature_vector(student_id, history, pool.iloc[i]) for i in range(len(pool))]
        X_scaled = self.scaler.transform(np.array(rows)) if self.scaler is not None else np.array(rows)
        return self.cp_model.predict_proba(X_scaled)[:, 1] if self.cp_model is not None else np.full(len(rows), 0.5)

    def select_next_question(self, strategy, probas, pool):
        if len(pool) == 0:
            return None
        if strategy == 'uncertainty':
            idx = int(np.argmin(np.abs(probas - 0.5)))
        elif strategy == 'entropy':
            p = np.clip(probas, 1e-9, 1 - 1e-9)
            idx = int(np.argmax(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))
        else:
            idx = int(np.random.choice(len(pool)))
        return int(idx)

    def should_stop(self, probas):
        if len(probas) == 0:
            return True
        uncertain = (probas > self.uncertain_low) & (probas < self.uncertain_high)
        return not uncertain.any()

    def prepare_the_query(self, student_id):
        df_questions = self.get_question_pool()
        # Fetch student's answered qids from SQLite
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute('SELECT qid, answer, features FROM answers WHERE student_id = ? ORDER BY id', (student_id,))
        rows = cur.fetchall()
        conn.close()

        answered_ids = [r[0] for r in rows]
        df_pool = df_questions[~df_questions['Question ID'].isin(answered_ids)].copy()
        if df_pool.empty:
            return pd.DataFrame()

        # Build history list for metrics
        history = []
        for qid, ans, features_json in rows:
            try:
                features = json.loads(features_json) if features_json else {}
            except Exception:
                features = {}
            matched = df_questions[df_questions['Question ID'] == qid]
            if matched.empty:
                continue
            q = matched.iloc[0]
            diff = str(q.get('difficulty', 'medium')).lower()
            diff_enc = int(self.difficulty_map.get(diff, 2))
            history.append({
                'correct': int(ans),
                'response_time': float(features.get('response_time', 0.0)),
                'diff_enc': diff_enc,
                'topic': q.get('topic', '')
            })

        base_metrics = {
            'global_correctness': 0.5, 'easy_correct_avg': 0.5, 'medium_correct_avg': 0.5,
            'hard_correct_avg': 0.5, 'avg_response_time': 0.0, 'questions_answered_so_far': 0
        }
        if history:
            metrics = self._get_metrics_from_history(history)
            base_metrics.update(metrics)

        feature_rows = []
        for _, q_row in df_pool.iterrows():
            topic_hist = [h for h in history if h.get('topic') == q_row.get('topic')]
            m = base_metrics.copy()
            m['topic_correctness'] = topic_hist[-1]['correct'] if topic_hist else 0.5
            feature_rows.append(self._build_vector_internal(m, q_row))

        X_scaled = self.scaler.transform(np.array(feature_rows)) if self.scaler is not None else np.array(feature_rows)
        proba = self.cp_model.predict_proba(X_scaled)[:, 1] if self.cp_model is not None else np.full(len(feature_rows), 0.5)
        df_pool['proba_correct'] = proba
        return df_pool

    def record_answer(self, student_id, question_id, student_answer, response_time):
        # Determine correctness
        topic, difficulty, _, _, correct_answer = self.get_question_details(question_id)
        is_correct = 1 if student_answer in [opt.strip() for opt in str(correct_answer).split('|')] else 0

        # Build features payload (include response_time)
        features = {'response_time': float(response_time)}

        # Use existing save_answer to persist to SQLite
        try:
            save_answer(self.db_path, str(uuid.uuid4()), student_id, question_id, '', is_correct, features)
        except Exception:
            # fallback: direct sqlite insert
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute('INSERT INTO answers (session_id, student_id, qid, question_text, answer, features, ts) VALUES (?,?,?,?,?,?,?)',
                        (str(uuid.uuid4()), student_id, str(question_id), '', int(is_correct), json.dumps(features), datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
        return is_correct

    def create_student_instance(self):
        student_id = str(uuid.uuid4())
        # Insert an INITIAL marker row into SQLite answers table
        init_features = json.dumps({'note': 'initial'})
        try:
            save_answer(self.db_path, str(uuid.uuid4()), student_id, 'INITIAL', '', 0, init_features)
        except Exception:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute('INSERT INTO answers (session_id, student_id, qid, question_text, answer, features, ts) VALUES (?,?,?,?,?,?,?)',
                        (str(uuid.uuid4()), student_id, 'INITIAL', '', 0, init_features, datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
        return student_id
