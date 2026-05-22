"""
Active Learning Engine — Core Module
Adaptive Exam Preparation System | ENSIA ML | Spring 2025-2026

Stateless engine: receives session state, returns next action.
Used by both the Streamlit dashboard and the FastAPI service.
"""

import json
import os
import uuid
import numpy as np
import pandas as pd
import joblib
import textstat


class ALEngine:
    """
    Stateless Active Learning engine.

    All session state (history, pool, etc.) is managed by the caller.
    This engine just loads artifacts and exposes pure functions.
    """

    def __init__(self, models_dir=None, data_dir=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = models_dir or os.path.join(base, "..", "models")
        self.data_dir = data_dir or os.path.join(base, "..", "Dataset", "clean")

        # ── Load artifacts ───────────────────────────────────────────────
        self.cp_model = joblib.load(os.path.join(self.models_dir, "al_correctness_predictor.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "al_scaler.pkl"))
        self.le_topic = joblib.load(os.path.join(self.models_dir, "al_topic_label_encoder.pkl"))
        self.le_student = joblib.load(os.path.join(self.models_dir, "al_student_label_encoder.pkl"))

        with open(os.path.join(self.models_dir, "al_features.json")) as f:
            self.features = json.load(f)

        with open(os.path.join(self.models_dir, "al_config.json")) as f:
            self.config = json.load(f)

        self.difficulty_map = self.config["difficulty_map"]
        self.uncertain_low = self.config["uncertain_low"]
        self.uncertain_high = self.config["uncertain_high"]
        self.max_questions = self.config["max_questions"]

        # ── Load question bank ───────────────────────────────────────────
        qb_path = os.path.join(self.data_dir, "question_bank.csv")
        self.question_bank = pd.read_csv(qb_path)

        # Pre-compute question-level features if not already present
        if "num_words" not in self.question_bank.columns:
            self.question_bank["num_words"] = self.question_bank["question_text"].apply(
                lambda x: len(str(x).split())
            )
        if "qstn_complexity" not in self.question_bank.columns:
            self.question_bank["qstn_complexity"] = self.question_bank["question_text"].apply(
                lambda x: textstat.flesch_kincaid_grade(str(x))
            )

        # Also try to load pre-computed question features
        qf_path = os.path.join(self.models_dir, "al_question_features.csv")
        if os.path.exists(qf_path):
            self.qstn_features = pd.read_csv(qf_path)
        else:
            self.qstn_features = self.question_bank[
                ["Question ID", "num_words", "qstn_complexity"]
            ].copy()

        # ── Load interactions for topic/difficulty mapping ────────────────
        interactions_path = os.path.join(self.data_dir, "interactions_clean.csv")
        interactions = pd.read_csv(interactions_path)
        # Build question → (difficulty, topic) mapping from interactions
        q_meta = (
            interactions[["Question ID", "difficulty", "topic"]]
            .drop_duplicates("Question ID")
            .reset_index(drop=True)
        )
        self.question_meta = q_meta

        # ── Build full question bank ─────────────────────────────────────
        # The question_bank.csv may already have difficulty/topic columns
        # (from the enrichment script). Only add what's missing.
        self.full_question_bank = self.question_bank.copy()

        # Add difficulty from interactions if missing
        if "difficulty" not in self.full_question_bank.columns:
            diff_map = q_meta.set_index("Question ID")["difficulty"]
            self.full_question_bank["difficulty"] = (
                self.full_question_bank["Question ID"].map(diff_map)
            )

        # Add topic from interactions if missing
        if "topic" not in self.full_question_bank.columns:
            topic_map = q_meta.set_index("Question ID")["topic"]
            self.full_question_bank["topic"] = (
                self.full_question_bank["Question ID"].map(topic_map)
            )

        # Ensure num_words and qstn_complexity exist (from qstn_features or bank)
        for col in ["num_words", "qstn_complexity"]:
            if col not in self.full_question_bank.columns:
                if col in self.qstn_features.columns:
                    feat_map = self.qstn_features.set_index("Question ID")[col]
                    self.full_question_bank[col] = (
                        self.full_question_bank["Question ID"].map(feat_map)
                    )

        # ── Existing student IDs (to avoid collisions) ───────────────────
        self._existing_student_ids = set(self.le_student.classes_)

    # ─────────────────────────────────────────────────────────────────────
    # Student ID Management
    # ─────────────────────────────────────────────────────────────────────
    def resolve_student_id(self, raw_id=None):
        """
        Resolve a student ID for the session.

        - If raw_id is provided and exists in training data → use its encoded value.
        - If raw_id is provided but new → assign a new encoded value.
        - If raw_id is None → generate a new unique ID.

        Returns: (raw_id: str, encoded_id: int)
        """
        if raw_id is None:
            raw_id = f"new_{uuid.uuid4().hex[:8]}"

        if raw_id in self._existing_student_ids:
            encoded_id = int(self.le_student.transform([raw_id])[0])
        else:
            # Use a default encoded value (median of known IDs)
            encoded_id = int(np.median(range(len(self.le_student.classes_))))

        return raw_id, encoded_id

    # ─────────────────────────────────────────────────────────────────────
    # Question Pool
    # ─────────────────────────────────────────────────────────────────────
    def get_question_pool(self):
        """Return the full question pool with all metadata."""
        pool = self.full_question_bank.copy()
        pool["diff_enc"] = pool["difficulty"].map(self.difficulty_map).fillna(2).astype(int)

        # Encode topic
        known_topics = set(self.le_topic.classes_)
        pool["topic_enc"] = pool["topic"].apply(
            lambda t: int(self.le_topic.transform([t])[0]) if t in known_topics else -1
        )
        return pool

    def get_initial_question(self, pool):
        """
        Select the first question randomly from intermediate (medium) difficulty.
        Fallback to any question if no medium questions exist.
        """
        medium_pool = pool[pool["difficulty"] == "medium"]
        if len(medium_pool) > 0:
            return medium_pool.sample(1).iloc[0]
        return pool.sample(1).iloc[0]

    # ─────────────────────────────────────────────────────────────────────
    # Feature Vector Construction
    # ─────────────────────────────────────────────────────────────────────
    def build_feature_vector(self, student_id_enc, history, question_row):
        """
        Build the 12-feature vector matching the training feature order.

        Features: Student ID, global_correctness, topic_correctness,
                  easy_correct_avg, medium_correct_avg, hard_correct_avg,
                  avg_response_time, questions_answered_so_far,
                  difficulty_encoded, topic, num_words, qstn_complexity
        """
        diff_enc = int(question_row.get("diff_enc", self.difficulty_map.get(question_row.get("difficulty", "medium"), 2)))
        topic_enc = int(question_row.get("topic_enc", -1))
        num_w = float(question_row.get("num_words", 10))
        complexity = float(question_row.get("qstn_complexity", 5.0))

        n = len(history)

        if n == 0:
            return [
                student_id_enc,
                0.5, 0.5, 0.5, 0.5, 0.5,  # correctness features (neutral)
                0.0,                         # avg_response_time
                0,                           # questions_answered_so_far
                diff_enc, topic_enc,
                num_w, complexity,
            ]

        hist_df = pd.DataFrame(history)

        global_corr = float(hist_df["correct"].mean())

        topic_hist = hist_df[hist_df["topic_enc"] == topic_enc]
        topic_corr = float(topic_hist["correct"].mean()) if len(topic_hist) > 0 else 0.5

        easy_h = hist_df[hist_df["diff_enc"] == 1]
        easy_corr = float(easy_h["correct"].mean()) if len(easy_h) > 0 else 0.5

        med_h = hist_df[hist_df["diff_enc"] == 2]
        med_corr = float(med_h["correct"].mean()) if len(med_h) > 0 else 0.5

        hard_h = hist_df[hist_df["diff_enc"] == 3]
        hard_corr = float(hard_h["correct"].mean()) if len(hard_h) > 0 else 0.5

        avg_rt = float(hist_df["response_time"].mean())

        return [
            student_id_enc,
            global_corr, topic_corr,
            easy_corr, med_corr, hard_corr,
            avg_rt, n,
            diff_enc, topic_enc,
            num_w, complexity,
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Pool Scoring & Stopping
    # ─────────────────────────────────────────────────────────────────────
    def score_pool(self, student_id_enc, history, pool):
        """Return P(correct) for every question in the pool."""
        if len(pool) == 0:
            return np.array([])

        rows = [
            self.build_feature_vector(student_id_enc, history, pool.iloc[i])
            for i in range(len(pool))
        ]
        X = np.array(rows)
        X_scaled = self.scaler.transform(X)
        probas = self.cp_model.predict_proba(X_scaled)[:, 1]
        return probas

    def should_stop(self, probas):
        """True if NO question has P(correct) in the uncertain zone (0.3, 0.7)."""
        if len(probas) == 0:
            return True
        uncertain = (probas > self.uncertain_low) & (probas < self.uncertain_high)
        return not uncertain.any()

    # ─────────────────────────────────────────────────────────────────────
    # Strategy Selection
    # ─────────────────────────────────────────────────────────────────────
    def select_next_question(self, strategy, probas, pool):
        """
        Select the next question using the given strategy.

        Parameters
        ----------
        strategy : str — "uncertainty", "entropy", or "qbc"
        probas   : np.array — P(correct) for each question in pool
        pool     : DataFrame — remaining questions

        Returns
        -------
        int — index into pool of the selected question
        """
        if len(pool) == 0:
            return None

        if strategy == "uncertainty":
            # Least confident: closest to 0.5
            return int(np.argmin(np.abs(probas - 0.5)))

        elif strategy == "entropy":
            # Highest binary entropy
            p = np.clip(probas, 1e-9, 1 - 1e-9)
            H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            return int(np.argmax(H))

        elif strategy == "qbc":
            # Use the BaggingClassifier's individual estimators
            try:
                ensemble = joblib.load(
                    os.path.join(self.models_dir, "al_ensemble_qbc.pkl")
                )
                rows = []
                for i in range(len(pool)):
                    rows.append(
                        self.build_feature_vector(
                            0, [], pool.iloc[i]  # doesn't matter for QBC scoring
                        )
                    )
                X = self.scaler.transform(np.array(rows))

                # Get individual predictions from each estimator
                votes = np.array([est.predict(X) for est in ensemble.estimators_])
                # Vote entropy per question
                p_vote = np.clip(votes.mean(axis=0), 1e-9, 1 - 1e-9)
                H_vote = -p_vote * np.log2(p_vote) - (1 - p_vote) * np.log2(1 - p_vote)
                return int(np.argmax(H_vote))
            except Exception:
                # Fallback to uncertainty
                return int(np.argmin(np.abs(probas - 0.5)))

        else:
            # Random fallback
            return int(np.random.randint(len(pool)))

    # ─────────────────────────────────────────────────────────────────────
    # Session Summary
    # ─────────────────────────────────────────────────────────────────────
    def compute_score(self, history):
        """Compute the quiz score: correct answers / total questions asked."""
        if not history:
            return 0.0
        correct = sum(1 for h in history if h["correct"] == 1)
        return correct / len(history)

    def get_session_summary(self, history, pool_remaining):
        """Return a summary dict after the quiz finishes."""
        total_asked = len(history)
        correct_count = sum(1 for h in history if h["correct"] == 1)
        score = correct_count / total_asked if total_asked > 0 else 0.0

        return {
            "total_questions_asked": total_asked,
            "correct_answers": correct_count,
            "score": round(score, 4),
            "score_percent": f"{score:.0%}",
            "remaining_pool_size": pool_remaining,
        }
