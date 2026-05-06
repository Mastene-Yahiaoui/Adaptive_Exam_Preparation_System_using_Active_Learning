import json
import os
import uuid
import numpy as np
import pandas as pd
import joblib
import textstat

class AlClass :

    def __init__(self,model_dir=None,data_dir=None):
        base=os.path.dirname(os.path.abspath(__file__))
        self.models_dir=model_dir or os.path.join(base,"..","models")
        self.data_dir=data_dir or os.path.join(base,"..","Dataset")
        #loading the artifacts
        self.cp_model = joblib.load(os.path.join(self.models_dir, "correctness_predictor.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.le_topic = joblib.load(os.path.join(self.models_dir, "al_topic_label_encoder.pkl"))
        with open(os.path.join(self.models_dir, "features.json")) as f:
            self.features = json.load(f)
        with open(os.path.join(self.models_dir, "config.json")) as f:
            self.config = json.load(f)


        

        #initialize needed variables 
        self.difficulty_map = self.config["difficulty_map"]
        self.uncertain_low = self.config["uncertain_low"]
        self.uncertain_high = self.config["uncertain_high"]
        self.max_questions = self.config["max_questions"]
        
        # Define CSV file paths
        self.questions_file = os.path.join(self.data_dir, "questions.csv")
        self.answers_file = os.path.join(self.data_dir, "answers.csv")
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Initialize CSV files
        self.init_csv_files()

    def init_csv_files(self):
        """Initialize CSV files if they don't exist."""
        cols = ['Question ID', 'question_text', 'options', 'correct_answer', 'topic', 'difficulty', 'num_words', 'qstn_complexity']
        if not os.path.exists(self.questions_file):
            qb_path = os.path.join(self.data_dir, "clean", "question_bank.csv")
            if os.path.exists(qb_path):
                df = pd.read_csv(qb_path)
                df = df.apply(lambda row: pd.Series(self.compute_question_features(row.to_dict())), axis=1)
                df[cols].to_csv(self.questions_file, index=False)
            else:
                pd.DataFrame(columns=cols).to_csv(self.questions_file, index=False)
        
        if not os.path.exists(self.answers_file):
            pd.DataFrame(columns=[
                'studentID', 'questionID', 'topic', 'difficulty', 'is_correct', 'response_time',
                'global_correctness', 'topic_correctness', 'easy_correct_avg', 'medium_correct_avg',
                'hard_correct_avg', 'avg_response_time', 'questions_answered_so_far'
            ]).to_csv(self.answers_file, index=False)

    def compute_question_features(self, question):
        """Compute features for a question."""
        text = str(question.get("question_text", ""))
        question["num_words"] = len(text.split())
        question["qstn_complexity"] = textstat.flesch_kincaid_grade(text)
        return question

    def get_question_details(self, question_id):
        """Helper to get question metadata."""
        if not hasattr(self, '_questions_df') or self._questions_df is None:
            self._questions_df = pd.read_csv(self.questions_file)
        row = self._questions_df[self._questions_df['Question ID'] == question_id]
        if row.empty:
            self._questions_df = pd.read_csv(self.questions_file)
            row = self._questions_df[self._questions_df['Question ID'] == question_id]
            if row.empty: raise ValueError(f"Question ID {question_id} not found.")
        q = row.iloc[0]
        return q['topic'], q['difficulty'], q['num_words'], q['qstn_complexity'], q['correct_answer']

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
        diff_enc = int(question_row.get("diff_enc", self.difficulty_map.get(question_row.get("difficulty", "medium"), 2)))
        topic = question_row.get("topic", "N/A")
        known_topics = set(self.le_topic.classes_)
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
            float(question_row.get("num_words", 10)),
            float(question_row.get("qstn_complexity", 5.0))
        ]

    def get_question_pool(self):
        pool = pd.read_csv(self.questions_file)
        pool["diff_enc"] = pool["difficulty"].map(self.difficulty_map).fillna(2).astype(int)
        known_topics = set(self.le_topic.classes_)
        pool["topic_enc"] = pool["topic"].apply(lambda t: int(self.le_topic.transform([t])[0]) if t in known_topics else -1)
        return pool

    def build_feature_vector(self, student_id, history, question_row):
        metrics = self._get_metrics_from_history(history, current_topic=question_row.get("topic"))
        return self._build_vector_internal(metrics, question_row)

    def score_pool(self, student_id, history, pool):
        if len(pool) == 0: return np.array([])
        rows = [self.build_feature_vector(student_id, history, pool.iloc[i]) for i in range(len(pool))]
        X_scaled = self.scaler.transform(np.array(rows))
        return self.cp_model.predict_proba(X_scaled)[:, 1]

    def select_next_question(self, strategy, probas, pool):
        if len(pool) == 0: return None
        if strategy == "uncertainty": idx = np.argmin(np.abs(probas - 0.5))
        elif strategy == "entropy":
            p = np.clip(probas, 1e-9, 1 - 1e-9)
            idx = np.argmax(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
        elif strategy == "qbc":
            rows = [self.build_feature_vector(0, [], pool.iloc[i]) for i in range(len(pool))]
            X = self.scaler.transform(np.array(rows))
            votes = np.array([est.predict(X) for est in self.qbc_model.estimators_])
            p_vote = np.clip(votes.mean(axis=0), 1e-9, 1 - 1e-9)
            idx = np.argmax(-p_vote * np.log2(p_vote) - (1 - p_vote) * np.log2(1 - p_vote))
        else: idx = np.random.choice(len(pool))
        return int(idx)

    def should_stop(self, probas):
        if len(probas) == 0: return True
        uncertain = (probas > self.uncertain_low) & (probas < self.uncertain_high)
        return not uncertain.any()

    def get_initial_question(self, pool):
        medium_pool = pool[pool["difficulty"] == "medium"]
        return medium_pool.sample(1).iloc[0] if not medium_pool.empty else pool.sample(1).iloc[0]

    def compute_score(self, history):
        if not history: return 0.0
        return sum(1 for h in history if h["correct"] == 1) / len(history)

  
    def prepare_the_query(self, student_id):
        df_questions = self.get_question_pool()
        df_answers = pd.read_csv(self.answers_file)
        answered_ids = df_answers[df_answers['studentID'] == student_id]['questionID'].unique()
        df_pool = df_questions[~df_questions['Question ID'].isin(answered_ids)].copy()
        if df_pool.empty: return pd.DataFrame()
            
        student_hist = df_answers[df_answers['studentID'] == student_id]
        last_row = student_hist.iloc[-1]
        base_metrics = {
            'global_correctness': last_row['global_correctness'], 
            'easy_correct_avg': last_row['easy_correct_avg'],
            'medium_correct_avg': last_row['medium_correct_avg'], 
            'hard_correct_avg': last_row['hard_correct_avg'],
            'avg_response_time': last_row['avg_response_time'], 
            'questions_answered_so_far': last_row['questions_answered_so_far']
        }
        
        feature_rows = []
        for _, q_row in df_pool.iterrows():
            topic_hist = student_hist[student_hist['topic'] == q_row['topic']]
            metrics = base_metrics.copy()
            metrics['topic_correctness'] = topic_hist.iloc[-1]['topic_correctness'] if not topic_hist.empty else 0.5
            feature_rows.append(self._build_vector_internal(metrics, q_row))
            
        X_scaled = self.scaler.transform(np.array(feature_rows))
        df_pool['proba_correct'] = self.cp_model.predict_proba(X_scaled)[:, 1]
        
        votes = np.array([est.predict(X_scaled) for est in self.qbc_model.estimators_])
        p_vote = np.clip(votes.mean(axis=0), 1e-9, 1 - 1e-9)
        df_pool['qbc_entropy'] = -p_vote * np.log2(p_vote) - (1 - p_vote) * np.log2(1 - p_vote)
        return df_pool

    def record_answer(self, student_id, question_id, student_answer, response_time):
        df_answers = pd.read_csv(self.answers_file)
        topic, difficulty, _, _, correct_answer = self.get_question_details(question_id)
        is_correct = 1 if student_answer in [opt.strip() for opt in str(correct_answer).split('|')] else 0
        
        student_hist = df_answers[df_answers['studentID'] == student_id]
        real_hist = student_hist[student_hist['questionID'] != 'INITIAL']
        n = len(real_hist) + 1
        
        global_corr = (real_hist['is_correct'].sum() + is_correct) / n
        avg_rt = (real_hist['response_time'].sum() + response_time) / n
        topic_h = real_hist[real_hist['topic'] == topic]
        topic_corr = (topic_h['is_correct'].sum() + is_correct) / (len(topic_h) + 1)
        
        def get_diff_avg(d):
            h = real_hist[real_hist['difficulty'] == d]
            c = is_correct if difficulty == d else 0
            cnt = 1 if difficulty == d else 0
            return (h['is_correct'].sum() + c) / (len(h) + cnt) if (len(h) + cnt) > 0 else 0.5

        new_row = {
            'studentID': student_id, 'questionID': question_id, 'topic': topic, 'difficulty': difficulty,
            'is_correct': is_correct, 'response_time': response_time, 'global_correctness': global_corr,
            'topic_correctness': topic_corr, 'easy_correct_avg': get_diff_avg('easy'),
            'medium_correct_avg': get_diff_avg('medium'), 'hard_correct_avg': get_diff_avg('hard'),
            'avg_response_time': avg_rt, 'questions_answered_so_far': n
        }
        pd.concat([df_answers, pd.DataFrame([new_row])], ignore_index=True).to_csv(self.answers_file, index=False)
        return is_correct

    def create_student_instance(self):
        student_id = str(uuid.uuid4())
        initial = {
            'studentID': student_id, 'questionID': 'INITIAL', 'topic': 'N/A', 'difficulty': 'N/A',
            'global_correctness': 0.5, 'topic_correctness': 0.5, 'easy_correct_avg': 0.5,
            'medium_correct_avg': 0.5, 'hard_correct_avg': 0.5, 'avg_response_time': 0.0, 'questions_answered_so_far': 0
        }
        df = pd.read_csv(self.answers_file)
        pd.concat([df, pd.DataFrame([initial])], ignore_index=True).to_csv(self.answers_file, index=False)
        return student_id

        