import os
import json
import numpy as np
import pandas as pd
import joblib
import re

try:
    import textstat
except Exception:
    textstat = None


class AlClass:
    """
    Active-learning quiz engine. 
    Serves as the backend for the Adaptive Exam Dashboard.
    """

    def __init__(self, model_path=None, config_path=None):
        base = os.path.dirname(os.path.abspath(__file__))
        
        # Determine paths if not provided
        if model_path is None:
            model_path = os.path.join(base, "..", "models", "xgboost.pkl")
        if config_path is None:
            config_path = os.path.join(base, "..", "models", "simulation_config.json")
            
        features_path = os.path.join(base, "..", "models", "features.json")

        self.model = joblib.load(model_path)

        with open(config_path) as f:
            self.config = json.load(f)

        if os.path.exists(features_path):
            with open(features_path) as f:
                self.features = json.load(f)
        else:
            self.features = self.config.get('features', [])

        self.state_columns = self.config.get('state_columns', [
            "accuracy_so_far", "number_of_questions_answered_so_far", 
            "accuracy_easy_so_far", "accuracy_medium_so_far", 
            "accuracy_hard_so_far", "average_response_time_so_far"
        ])
        
        self.difficulty_map = self.config.get('difficulty_map', {'easy': 0, 'medium': 1, 'hard': 2})
        self.lower_threshold = self.config.get('lower_threshold', 0.25)
        self.upper_threshold = self.config.get('upper_threshold', 0.75)
        self.weight1 = self.config.get('weight1', 1.0)
        self.weight2 = self.config.get('weight2', 1.0)
        self.prior_rate = self.config.get('prior_rate', 0.5)
        self.prior_by_difficulty = {
            int(k): float(v) for k, v in self.config.get('prior_by_difficulty', {}).items()
        }
        self.prior_response_time = self.config.get('prior_response_time', 0.0)
        self.strategy = self.config.get('strategy', 'uncertainty').lower()
        self.max_quiz_length = int(self.config.get('max_quiz_length', 20))

    @staticmethod
    def _safe_accuracy(values, default):
        return float(np.mean(values)) if len(values) else float(default)

    def compute_state(self, history_labels, history_difficulty, history_response_time):
        attempts = len(history_labels)
        accuracy_so_far = self._safe_accuracy(history_labels, self.prior_rate)
        accuracy_easy = self._safe_accuracy(
            [y for y, d in zip(history_labels, history_difficulty) if d == 0],
            self.prior_by_difficulty.get(0, self.prior_rate),
        )
        accuracy_medium = self._safe_accuracy(
            [y for y, d in zip(history_labels, history_difficulty) if d == 1],
            self.prior_by_difficulty.get(1, self.prior_rate),
        )
        accuracy_hard = self._safe_accuracy(
            [y for y, d in zip(history_labels, history_difficulty) if d == 2],
            self.prior_by_difficulty.get(2, self.prior_rate),
        )
        avg_rt = self._safe_accuracy(history_response_time, self.prior_response_time)
        return {
            'accuracy_so_far': accuracy_so_far,
            'number_of_questions_answered_so_far': attempts,
            'accuracy_easy_so_far': accuracy_easy,
            'accuracy_medium_so_far': accuracy_medium,
            'accuracy_hard_so_far': accuracy_hard,
            'average_response_time_so_far': avg_rt,
        }

    def build_feature_row(self, question_row, state):
        row = {}
        for col in self.state_columns:
            row[col] = state[col]
        question_features = ['difficulty', 'num_words', 'qstn_complexity', 'num_options', 'avg_option_length']
        for col in question_features:
            row[col] = question_row.get(col, 0)
        return row

    def predict_pool(self, remaining_questions, state):
        if not remaining_questions:
            return np.array([])
        feature_rows = []
        for q in remaining_questions:
            feature_rows.append(self.build_feature_row(q, state))
        X = pd.DataFrame(feature_rows, columns=self.features)
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def should_stop(self, probs):
        if len(probs) == 0:
            return True
            
        # In random selection strategy, do not stop early based on model confidence.
        # It will only stop when max_quiz_length is reached in streamlit_app.py.
        if self.strategy == 'random':
            return False
            
        confident = (probs >= self.upper_threshold) | (probs <= self.lower_threshold)
        return confident.all()

    def pick_next_question_idx(self, remaining_questions, probs, state):
        if len(remaining_questions) == 0:
            return 0
        strategy = self.strategy
        
        # Use a deterministic random state tied to the number of answered questions
        # so that Streamlit reruns (e.g. clicking a radio button) don't change the question.
        attempts = state.get('number_of_questions_answered_so_far', 0)
        rng = np.random.RandomState(int(attempts))

        if strategy == 'random':
            return int(rng.choice(len(remaining_questions)))
        elif strategy == 'entropy':
            p = np.clip(probs, 1e-9, 1 - 1e-9)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            return int(np.argmax(entropy))
        elif strategy == 'qbc':
            n_committee = 5
            committee_preds = []
            for _ in range(n_committee):
                noise = rng.normal(0, 0.1, size=probs.shape)
                perturbed = np.clip(probs + noise, 0, 1)
                committee_preds.append(perturbed)
            
            votes = np.array(committee_preds)
            p_vote = np.clip(votes.mean(axis=0), 1e-9, 1 - 1e-9)
            qbc_entropy = -p_vote * np.log2(p_vote) - (1 - p_vote) * np.log2(1 - p_vote)
            return int(np.argmax(qbc_entropy))
        else:
            return int(np.argmin(np.abs(probs - 0.5)))

    def estimate_score(self, history_labels, probs_remaining, total_questions):
        answered_count = len(history_labels)
        actual_correct_answered = int(sum(history_labels))
        
        if answered_count == 0:
            return 0.0
            
        return (actual_correct_answered * self.max_quiz_length) / answered_count

    def simulation_score(self, history_labels, probs_remaining, total_questions):
        actual_correct_answered = int(sum(history_labels))
        if len(probs_remaining) > 0:
            predicted_true_unanswered = float((probs_remaining * (probs_remaining >= 0.5)).sum())
        else:
            predicted_true_unanswered = 0.0

        if total_questions == 0:
            return 0.0
            
        return (self.weight1 * actual_correct_answered + self.weight2 * predicted_true_unanswered) / total_questions


# Functions to compute question features from a given dataframe
def _detect_num_words(question_text):
    return len(re.findall(r"\b\w+\b", str(question_text)))

def _detect_complexity(question_text):
    try:
        if textstat is not None:
            return float(textstat.flesch_kincaid_grade(str(question_text)))
    except Exception:
        pass
    return 0.0

def _count_options(row):
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e', 'option_f', 'option_g']
    count = 0
    for col in option_cols:
        val = row.get(col, None)
        if val is not None and pd.notna(val) and str(val).strip():
            count += 1
    if count > 0:
        return count
    options_str = row.get('options', '')
    if options_str and pd.notna(options_str) and str(options_str).strip():
        parts = [p.strip() for p in str(options_str).split('|') if p.strip()]
        return len(parts)
    return 4

def _avg_option_length(row):
    option_texts = []
    option_cols = ['option_a', 'option_b', 'option_c', 'option_d', 'option_e', 'option_f', 'option_g']
    for col in option_cols:
        val = row.get(col, None)
        if val is not None and pd.notna(val) and str(val).strip():
            option_texts.append(str(val).strip())
    if not option_texts:
        options_str = row.get('options', '')
        if options_str and pd.notna(options_str) and str(options_str).strip():
            option_texts = [p.strip() for p in str(options_str).split('|') if p.strip()]
    if option_texts:
        return float(np.mean([len(t) for t in option_texts]))
    return 10.0

def compute_question_features(df, difficulty_map=None):
    if difficulty_map is None:
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
    df = df.copy()
    if 'difficulty' in df.columns:
        df['difficulty'] = (
            df['difficulty']
            .astype(str)
            .str.lower()
            .map(difficulty_map)
            .fillna(1)
            .astype(int)
        )
    else:
        df['difficulty'] = 1
    if 'question_text' in df.columns:
        df['num_words'] = df['question_text'].apply(_detect_num_words)
    elif 'num_words' not in df.columns:
        df['num_words'] = 10
    if 'question_text' in df.columns:
        df['qstn_complexity'] = df['question_text'].apply(_detect_complexity)
    elif 'qstn_complexity' not in df.columns:
        df['qstn_complexity'] = 5.0
    if 'num_options' not in df.columns:
        df['num_options'] = df.apply(_count_options, axis=1)
    if 'avg_option_length' not in df.columns:
        df['avg_option_length'] = df.apply(_avg_option_length, axis=1)
    return df