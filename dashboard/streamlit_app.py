import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
from pathlib import Path
import uuid
from datetime import datetime
import sys
import os
from pathlib import Path as _P

# Resolve project root relative to this file so paths work
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import backend reliably whether running from project root or inside dashboard
try:
    import dashboard.backend as backend
except Exception:
    import importlib.util
    backend_path = HERE / 'backend.py'
    spec = importlib.util.spec_from_file_location("dashboard.backend", str(backend_path))
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)

# AlClass is provided inside `backend.py` as `backend.AlClass` (if present)
AlClass = getattr(backend, 'AlClass', None)

BASE_DIR = ROOT_DIR
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'Dataset' / 'clean'
DB_PATH = BASE_DIR / 'data' / 'al_sessions.db'

DIFFICULTY_OPTIONS = ['easy', 'medium', 'hard']
# Teacher still uses a simple access code; students log in with their Student ID
ROLE_CODES = {
    'teacher': 'teacher123',
}
CERTAINTY_LOW = 0.30
CERTAINTY_HIGH = 0.70


def load_artifacts():
    artifacts = {}
    try:
        artifacts['cp'] = joblib.load(MODELS_DIR / 'correctness_predictor.pkl')
    except Exception:
        artifacts['cp'] = None
    try:
        artifacts['scaler'] = joblib.load(MODELS_DIR / 'al_scaler.pkl')
    except Exception:
        artifacts['scaler'] = None
    try:
        # Prefer feature list saved by CP.ipynb (pickle), fallback to al_features.json
        feat_pkl = MODELS_DIR / 'correctness_predictor_features.pkl'
        if feat_pkl.exists():
            artifacts['features'] = joblib.load(feat_pkl)
        else:
            with open(MODELS_DIR / 'al_features.json') as f:
                artifacts['features'] = json.load(f)
    except Exception:
        artifacts['features'] = None
    # load label encoders if present
    try:
        artifacts['le_topic'] = joblib.load(MODELS_DIR / 'al_topic_label_encoder.pkl')
    except Exception:
        artifacts['le_topic'] = None
    try:
        artifacts['le_student'] = joblib.load(MODELS_DIR / 'al_student_label_encoder.pkl')
    except Exception:
        artifacts['le_student'] = None
    # initialize AlClass engine if available
    try:
        if AlClass is not None:
            artifacts['al'] = AlClass(model_dir=str(MODELS_DIR), data_dir=str(BASE_DIR / 'Dataset'))
        else:
            artifacts['al'] = None
    except Exception:
        artifacts['al'] = None
    return artifacts


def compute_placeholder_features(q_row, session_stats, features_list):
    # Build a feature vector aligned to features_list with sensible defaults
    vals = {}
    def get(k, default=0):
        return q_row.get(k, default)

    vals['global_correctness'] = session_stats.get('global_correctness', 0.5)
    vals['topic_correctness'] = session_stats.get('topic_correctness', 0.5)
    vals['easy_correct_avg'] = session_stats.get('easy_correct_avg', 0.5)
    vals['medium_correct_avg'] = session_stats.get('medium_correct_avg', 0.5)
    vals['hard_correct_avg'] = session_stats.get('hard_correct_avg', 0.5)
    vals['avg_response_time'] = session_stats.get('avg_response_time', 0.0)
    vals['questions_answered_so_far'] = session_stats.get('questions_answered_so_far', 0)
    # map difficulty
    diff_map = {"easy": 1, "medium": 2, "hard": 3}
    vals['difficulty_encoded'] = diff_map.get(str(q_row.get('difficulty', '')).lower(), 2)
    vals['topic'] = q_row.get('topic', 0)
    vals['num_words'] = q_row.get('num_words', len(str(q_row.get('question_text','')).split()))
    vals['qstn_complexity'] = q_row.get('qstn_complexity', 0.0)

    # return ordered vector
    if features_list:
        out = []
        for f in features_list:
            v = vals.get(f, 0.0)
            # if topic is string and label encoder provided, leave encoding to caller
            out.append(v)
        return out
    else:
        # fallback ordering
        return [
            vals['global_correctness'],
            vals['topic_correctness'],
            vals['easy_correct_avg'],
            vals['medium_correct_avg'],
            vals['hard_correct_avg'],
            vals['avg_response_time'],
            vals['questions_answered_so_far'],
            vals['difficulty_encoded'],
            vals['topic'],
            vals['num_words'],
            vals['qstn_complexity'],
        ]


def detect_num_words(question_text):
    return len(re.findall(r"\b\w+\b", str(question_text)))


def detect_complexity(question_text):
    try:
        import textstat
        return float(textstat.flesch_kincaid_grade(str(question_text)))
    except Exception:
        return 0.0


def normalize_question_frame(df):
    df = df.copy()
    column_map = {
        'question': 'question_text',
        'text': 'question_text',
        'topic_name': 'topic',
        'level': 'difficulty',
        'difficulty_level': 'difficulty',
        'question id': 'Question ID',
        'question_id': 'Question ID',
        'option_a': 'option_a',
        'option_b': 'option_b',
        'option_c': 'option_c',
        'option_d': 'option_d',
        'correct': 'correct_answer',
        'answer': 'correct_answer',
    }
    df.columns = [column_map.get(str(c).strip().lower(), c) for c in df.columns]

    if 'question_text' not in df.columns:
        raise ValueError('Input file must contain a question text column.')

    if 'Question ID' not in df.columns:
        df['Question ID'] = [f'Q{i+1}' for i in range(len(df))]

    if 'topic' not in df.columns:
        df['topic'] = ''

    if 'difficulty' not in df.columns:
        df['difficulty'] = 'medium'

    df['difficulty'] = df['difficulty'].astype(str).str.lower().where(df['difficulty'].astype(str).str.lower().isin(DIFFICULTY_OPTIONS), 'medium')
    df['num_words'] = df['question_text'].apply(detect_num_words)
    df['qstn_complexity'] = df['question_text'].apply(detect_complexity)
    base_cols = ['Question ID', 'question_text', 'difficulty', 'topic', 'num_words', 'qstn_complexity']
    mc_cols = [c for c in ['option_a', 'option_b', 'option_c', 'option_d', 'correct_answer'] if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols and c not in mc_cols]
    return df[base_cols + mc_cols + other_cols]


def parse_pdf_questions(uploaded_file):
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError('PDF support requires pypdf. Install dependencies again.') from exc

    reader = PdfReader(uploaded_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ''
        chunks.append(text)
    full_text = '\n'.join(chunks)
    lines = [line.strip() for line in re.split(r'\n+', full_text) if line.strip()]
    return lines


def parse_csv_or_text_file(uploaded_file):
    raw = pd.read_csv(uploaded_file)
    return normalize_question_frame(raw)


def build_question_bank_from_manual(question_text, topic, difficulty):
    if not str(question_text).strip():
        raise ValueError('Question text cannot be empty.')
    row = {
        'Question ID': f'Q{uuid.uuid4().hex[:8]}',
        'question_text': question_text.strip(),
        'difficulty': difficulty,
        'topic': topic.strip(),
        'num_words': detect_num_words(question_text),
        'qstn_complexity': detect_complexity(question_text),
    }
    return pd.DataFrame([row])


def is_multiple_choice(row):
    return pd.notna(row.get('option_a')) and str(row.get('option_a')).strip()


def predict_question_probability(row, artifacts, session_stats):
    features = artifacts.get('features')
    cp = artifacts.get('cp')
    scaler = artifacts.get('scaler')

    row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    le_topic = artifacts.get('le_topic')
    if le_topic is not None:
        try:
            topic_val = row_dict.get('topic')
            if topic_val is not None and not isinstance(topic_val, (int, float)):
                row_dict['topic'] = int(le_topic.transform([str(topic_val)])[0])
        except Exception:
            pass

    x = compute_placeholder_features(row_dict, session_stats, features)
    try:
        x_arr = np.array(x).reshape(1, -1)
        if scaler is not None:
            x_arr = scaler.transform(x_arr)
        if cp is not None:
            return float(cp.predict_proba(x_arr)[0, 1])
    except Exception:
        pass
    return 0.5


def level_from_probability(probability):
    if probability >= CERTAINTY_HIGH:
        return 'Strong'
    if probability <= CERTAINTY_LOW:
        return 'Needs support'
    return 'Developing'


def confidence_band_label(probability):
    return f'{probability:.2f}'


def student_level_summary(session_stats):
    accuracy = session_stats.get('global_correctness', 0.5)
    answered = session_stats.get('questions_answered_so_far', 0)
    if answered == 0:
        return 'Not started'
    if accuracy >= 0.80:
        return 'Strong'
    if accuracy >= 0.55:
        return 'Developing'
    return 'Needs support'


def build_multiple_choice_question(question_text, topic, difficulty, option_a, option_b, option_c, option_d, correct_answer):
    if not str(question_text).strip():
        raise ValueError('Question text cannot be empty.')
    if not all([str(option_a).strip(), str(option_b).strip(), str(option_c).strip(), str(option_d).strip()]):
        raise ValueError('All options must be filled.')
    if correct_answer not in ['A', 'B', 'C', 'D']:
        raise ValueError('Correct answer must be A, B, C, or D.')
    row = {
        'Question ID': f'Q{uuid.uuid4().hex[:8]}',
        'question_text': question_text.strip(),
        'difficulty': difficulty,
        'topic': topic.strip(),
        'option_a': str(option_a).strip(),
        'option_b': str(option_b).strip(),
        'option_c': str(option_c).strip(),
        'option_d': str(option_d).strip(),
        'correct_answer': correct_answer,
        'num_words': detect_num_words(question_text),
        'qstn_complexity': detect_complexity(question_text),
    }
    return pd.DataFrame([row])


def show_landing_screen():
    st.subheader('Choose a workspace')
    left, right = st.columns(2)
    with left:
        st.markdown('### Teacher screen')
        st.write('Create question banks, upload CSV/PDF files, review student level, and export sessions.')
        if st.button('Open Teacher Screen', use_container_width=True):
            st.session_state.active_screen = 'teacher'
            st.rerun()
    with right:
        st.markdown('### Student screen')
        st.write('Take the quiz, answer questions, and stop automatically when the model is confident.')
        if st.button('Open Student Screen', use_container_width=True):
            st.session_state.active_screen = 'student'
            st.rerun()


def screen_gate():
    st.sidebar.header('Navigation')
    if st.sidebar.button('Back to home', use_container_width=True):
        st.session_state.active_screen = None
        st.rerun()

    role = st.session_state.get('active_screen')
    if role not in {'teacher', 'student'}:
        return None, False

    st.sidebar.caption(f'Current screen: {role.title()}')
    if role == 'teacher':
        code = st.sidebar.text_input('Teacher access code', type='password', placeholder='try : teacher123')
        authorized = code == ROLE_CODES.get('teacher')
        if not authorized:
            st.info('Enter the teacher access code to open this screen.')
        return role, authorized
    else:
        # Students authenticate with their Student ID (no shared code)
        st.sidebar.markdown('### Student Login')
        first_name = st.sidebar.text_input('First name', placeholder='Enter first name')
        last_name = st.sidebar.text_input('Last name', placeholder='Enter last name')
        sid = st.sidebar.text_input('Enter your Student ID:', placeholder='e.g., 2323+ your unique identifier')
        if st.sidebar.button('Login', use_container_width=True):
            if sid and str(sid).strip():
                student_id = str(sid).strip()
                if backend.student_has_attempted(DB_PATH, student_id):
                    st.sidebar.error('This Student ID has already completed a quiz. Please use a different ID.')
                else:
                    st.session_state['student_id'] = student_id
                    st.session_state['first_name'] = str(first_name).strip()
                    st.session_state['last_name'] = str(last_name).strip()
                    st.rerun()
            else:
                st.sidebar.error('Please enter a valid Student ID')
        
        if not st.session_state.get('student_id'):
            st.info('Enter your Student ID in the sidebar and click Login to proceed.')
            return role, False
        return role, True


def pick_next_question(questions_df, answered_idx, artifacts, session_stats):
    # Uncertainty sampling: pick question with P(correct) closest to 0.5
    # If AlClass engine is available, delegate scoring/selection to it
    al = artifacts.get('al') if artifacts is not None else None
    try:
        if al is not None:
            student_id = st.session_state.get('student_id', '')
            # build pool excluding already answered question IDs
            answered_qids = [a.get('qid') for a in st.session_state.get('answers', []) if a.get('qid')]
            pool = questions_df[~questions_df['Question ID'].isin(answered_qids)].reset_index(drop=True)
            if pool.empty:
                return None
            # build history for AlClass expected format
            history = []
            for a in st.session_state.get('answers', []):
                qid = a.get('qid')
                matched = questions_df[questions_df['Question ID'] == qid]
                if matched.empty:
                    continue
                q = matched.iloc[0]
                diff = str(q.get('difficulty', 'medium')).lower()
                try:
                    diff_enc = int(al.difficulty_map.get(diff, 2))
                except Exception:
                    diff_enc = 2
                history.append({'correct': int(a.get('answer', 0)), 'response_time': 0.0, 'diff_enc': diff_enc, 'topic': q.get('topic', '')})

            probas = al.score_pool(student_id, history, pool)
            # stop if AlClass says so
            if al.should_stop(probas):
                return None
            sel_idx = al.select_next_question('uncertainty', probas, pool)
            # store last question probability for display
            try:
                st.session_state['last_question_probability'] = float(probas[sel_idx])
            except Exception:
                st.session_state['last_question_probability'] = 0.5
            return pool.iloc[int(sel_idx)]
    except Exception:
        # fallback to original lightweight selection
        features = artifacts.get('features') if artifacts is not None else None
        cp = artifacts.get('cp') if artifacts is not None else None
        scaler = artifacts.get('scaler') if artifacts is not None else None

        candidates = questions_df.drop(index=answered_idx)
        if candidates.empty:
            return None

        scores = []
        for idx, row in candidates.iterrows():
            probability = predict_question_probability(row, artifacts, session_stats)
            scores.append((idx, probability, abs(probability - 0.5)))

        idx, best_probability, _ = min(scores, key=lambda item: item[2])
        if best_probability <= CERTAINTY_LOW or best_probability >= CERTAINTY_HIGH:
            return None
        return candidates.loc[idx]


def teacher_tab():
    st.header('Teacher — Quiz / Question Bank')
    st.caption('This area is only visible after teacher access is granted.')
    st.info(
        f'Model-confidence threshold: keep asking while predicted correctness stays between {CERTAINTY_LOW:.2f} and {CERTAINTY_HIGH:.2f}. '
        'When every remaining question is outside that band, the quiz stops.'
    )
    input_mode = st.radio(
        'Choose how to add questions',
        ['Upload CSV', 'Upload PDF', 'Manual question form'],
        horizontal=True,
    )

    staged_df = None

    if input_mode == 'Upload CSV':
        uploaded = st.file_uploader(
            'Upload a CSV with questions and topics',
            type=['csv'],
            help='Accepted columns: question_text or question, topic, difficulty. Word count and complexity are computed automatically.',
        )
        if uploaded is not None:
            try:
                staged_df = parse_csv_or_text_file(uploaded)
                st.caption('Preview - difficulty can be adjusted before saving')
                staged_df = st.data_editor(
                    staged_df,
                    use_container_width=True,
                    num_rows='dynamic',
                    column_config={
                        'difficulty': st.column_config.SelectboxColumn('difficulty', options=DIFFICULTY_OPTIONS),
                        'num_words': st.column_config.NumberColumn('num_words', disabled=True),
                        'qstn_complexity': st.column_config.NumberColumn('qstn_complexity', disabled=True),
                    },
                    disabled=['Question ID', 'num_words', 'qstn_complexity'],
                )
            except Exception as exc:
                st.error(f'Could not read CSV: {exc}')

    elif input_mode == 'Upload PDF':
        uploaded = st.file_uploader(
            'Upload a PDF containing question statements',
            type=['pdf'],
            help='Each non-empty line is treated as a question; topics and difficulty can be edited after extraction.',
        )
        if uploaded is not None:
            try:
                lines = parse_pdf_questions(uploaded)
                staged_df = pd.DataFrame({
                    'Question ID': [f'Q{i+1}' for i in range(len(lines))],
                    'question_text': lines,
                    'difficulty': ['medium'] * len(lines),
                    'topic': [''] * len(lines),
                })
                staged_df['num_words'] = staged_df['question_text'].apply(detect_num_words)
                staged_df['qstn_complexity'] = staged_df['question_text'].apply(detect_complexity)
                staged_df = st.data_editor(
                    staged_df,
                    use_container_width=True,
                    num_rows='dynamic',
                    column_config={
                        'difficulty': st.column_config.SelectboxColumn('difficulty', options=DIFFICULTY_OPTIONS),
                        'num_words': st.column_config.NumberColumn('num_words', disabled=True),
                        'qstn_complexity': st.column_config.NumberColumn('qstn_complexity', disabled=True),
                    },
                    disabled=['Question ID', 'num_words', 'qstn_complexity'],
                )
            except Exception as exc:
                st.error(f'Could not read PDF: {exc}')

    else:
        with st.form('manual_question_form'):
            question_type = st.radio('Question type', ['Short answer', 'Multiple choice'], horizontal=True)
            question_text = st.text_area('Question text', height=120)
            topic = st.text_input('Topic', value='')
            difficulty = st.select_slider('Difficulty', options=DIFFICULTY_OPTIONS, value='medium')

            if question_type == 'Multiple choice':
                col1, col2 = st.columns(2)
                with col1:
                    option_a = st.text_input('Option A', value='', placeholder='Enter first option')
                    option_c = st.text_input('Option C', value='', placeholder='Enter third option')
                with col2:
                    option_b = st.text_input('Option B', value='', placeholder='Enter second option')
                    option_d = st.text_input('Option D', value='', placeholder='Enter fourth option')
                correct_answer = st.radio('Correct answer', ['A', 'B', 'C', 'D'], horizontal=True)
            else:
                option_a = option_b = option_c = option_d = correct_answer = None

            submitted = st.form_submit_button('Add question')
            if submitted:
                try:
                    if question_type == 'Multiple choice':
                        staged_df = build_multiple_choice_question(question_text, topic, difficulty, option_a, option_b, option_c, option_d, correct_answer)
                    else:
                        staged_df = build_question_bank_from_manual(question_text, topic, difficulty)
                    st.session_state.manual_bank = pd.concat(
                        [st.session_state.get('manual_bank', pd.DataFrame()), staged_df],
                        ignore_index=True,
                    )
                except Exception as exc:
                    st.error(str(exc))

        staged_df = st.session_state.get('manual_bank')

    if staged_df is not None and not staged_df.empty:
        st.subheader('Question bank preview')
        st.dataframe(staged_df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Save to Dataset/clean as question_bank_app.csv'):
                try:
                    normalized = normalize_question_frame(staged_df)
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    normalized.to_csv(DATA_DIR / 'question_bank_app.csv', index=False)
                    st.success('Saved question_bank_app.csv')
                except Exception as exc:
                    st.error(f'Could not save file: {exc}')
        with col2:
            if st.button('Clear staged questions'):
                st.session_state.manual_bank = pd.DataFrame()
                st.rerun()

    st.write('Or use existing question bank file in Dataset/clean')
    existing_files = list(DATA_DIR.glob('*.csv')) if DATA_DIR.exists() else []
    for f in existing_files:
        st.write(f.name)
    # Admin: export collected session answers
    st.markdown('---')
    st.subheader('Teacher exports')
    try:
        export_path = backend.export_csv(DB_PATH, BASE_DIR / 'data' / 'session_export.csv')
        with open(export_path, 'rb') as fh:
            export_bytes = fh.read()
        st.download_button(
            'Download all sessions history',
            export_bytes,
            file_name='session_export.csv',
            mime='text/csv',
            use_container_width=True,
        )
    except Exception as e:
        st.error(f'Export failed: {e}')

    st.markdown('---')
    st.subheader('Session overview')
    if DB_PATH.exists():
        try:
            session_summaries = backend.list_session_summaries(DB_PATH)
            global_stats = backend.get_global_session_stats(DB_PATH)
            if session_summaries:
                summary_labels = {
                    f"{(item.get('full_name') or 'Unknown')} ({item['student_id'] or 'Unknown'}) | {item['session_id']}": item
                    for item in session_summaries
                }
                selected_label = st.selectbox('Pick a student session', list(summary_labels.keys()))
                selected = summary_labels[selected_label]
                rows = backend.get_session_answers(DB_PATH, selected['session_id'])
                total = len(rows)
                correct = sum(int(r[2]) for r in rows)
                accuracy = (correct / total) if total else 0.0
                level = 'Strong' if accuracy >= 0.80 else 'Developing' if accuracy >= 0.55 else 'Needs support'

                stat_left, stat_mid, stat_right = st.columns(3)
                stat_left.metric('Answers', total)
                stat_mid.metric('Accuracy', f'{accuracy:.2%}')
                stat_right.metric('Student level', level)

                st.write(f"Student: {selected.get('full_name') or 'Unknown'}")
                st.write(f"Student ID: {selected['student_id'] or 'Unknown'}")

                st.markdown('#### Global session statistics')
                g1, g2, g3 = st.columns(3)
                g1.metric('Sessions', global_stats['sessions'])
                g2.metric('Students', global_stats['students'])
                g3.metric('Mean quiz level', f"{global_stats['mean_quiz_level']:.2f} / 3.00")
                st.write(f"Overall mean accuracy: {global_stats['mean_accuracy']:.2%}")
            else:
                st.write('No student sessions yet.')
        except Exception as exc:
            st.error(f'Could not load sessions: {exc}')


def student_tab(artifacts):
    st.header('Student — Take Quiz (Active Learning)')
    st.caption('This area is only visible after student access is granted.')
    student_id = st.session_state.get('student_id', '')
    first_name = st.session_state.get('first_name', '').strip()
    last_name = st.session_state.get('last_name', '').strip()
    if not student_id:
        st.warning('Student ID not found. Open the Student screen and enter your Student ID in the sidebar.')
        return

    if backend.student_has_attempted(DB_PATH, student_id, st.session_state.get('session_id')):
        st.error('This Student ID has already completed a quiz and cannot take it again.')
        return

    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    qb_path = DATA_DIR / 'question_bank_app.csv'
    if not qb_path.exists():
        # fallback to any question_bank.csv
        qb_files = list(DATA_DIR.glob('*question_bank*.csv')) if DATA_DIR.exists() else []
        qb_path = qb_files[0] if qb_files else None

    if qb_path is None or not qb_path.exists():
        st.warning('No question bank found. Ask the teacher to upload one in the Teacher tab.')
        return

    df_q = pd.read_csv(qb_path)

    if 'session_state_init' not in st.session_state:
        st.session_state.session_state_init = True
        st.session_state.answered_idx = []
        st.session_state.answers = []
        st.session_state.session_stats = {
            'global_correctness': 0.5,
            'topic_correctness': 0.5,
            'easy_correct_avg': 0.5,
            'medium_correct_avg': 0.5,
            'hard_correct_avg': 0.5,
            'avg_response_time': 0.0,
            'questions_answered_so_far': 0,
        }

    st.write(f"Questions available: {len(df_q) - len(st.session_state.answered_idx)}")

    next_q = pick_next_question(df_q, st.session_state.answered_idx, artifacts, st.session_state.session_stats)
    if next_q is None:
        student_level = student_level_summary(st.session_state.session_stats)
        st.success('No more questions.')
        st.info(
            f'Quiz stopped because model confidence is outside the active band {CERTAINTY_LOW:.2f}-{CERTAINTY_HIGH:.2f} for all remaining questions.'
        )
        st.info(f'Estimated level: {student_level}')
        return

    st.subheader('Next Question')
    st.write(next_q['question_text'])

    is_mc = is_multiple_choice(next_q)
    answer_text = ''

    if is_mc:
        options = {
            'A': next_q.get('option_a', ''),
            'B': next_q.get('option_b', ''),
            'C': next_q.get('option_c', ''),
            'D': next_q.get('option_d', ''),
        }
        st.write('**Options:**')
        selected_key = st.radio('Select answer', list(options.keys()), format_func=lambda k: f'{k}: {options[k]}')
        correct_answer = next_q.get('correct_answer', '')
    else:
        choice = st.radio('Answer', ['Correct', 'Incorrect'])

    if st.button('Submit Answer'):
        if is_mc:
            correct = 1 if selected_key == correct_answer else 0
            answer_text = f'{selected_key}: {options[selected_key]}'
        else:
            correct = 1 if choice == 'Correct' else 0
            answer_text = choice
        
        st.session_state.answered_idx.append(int(next_q.name))
        st.session_state.answers.append({
            'qid': next_q.get('Question ID', next_q.name),
            'answer': correct,
            'answer_text': answer_text,
        })
        # update simple stats
        n = st.session_state.session_stats['questions_answered_so_far'] + 1
        prev_gc = st.session_state.session_stats['global_correctness']
        st.session_state.session_stats['global_correctness'] = (prev_gc * (n - 1) + correct) / n
        st.session_state.session_stats['questions_answered_so_far'] = n
        # persist answer to DB (features saved as JSON)
        try:
            features = compute_placeholder_features(next_q.to_dict(), st.session_state.session_stats, artifacts.get('features'))
        except Exception:
            features = None
        try:
            backend.save_answer(
                DB_PATH,
                st.session_state.session_id,
                st.session_state.get('student_id', ''),
                next_q.get('Question ID', next_q.name),
                next_q.get('question_text', ''),
                correct,
                features,
                first_name=st.session_state.get('first_name', ''),
                last_name=st.session_state.get('last_name', ''),
            )
        except Exception as e:
            st.error(f'Failed to save answer: {e}')
        next_question = pick_next_question(df_q, st.session_state.answered_idx, artifacts, st.session_state.session_stats)
        if next_question is None:
            st.session_state.quiz_finished = True
        st.rerun()


def main():
    st.title('Adaptive Exam Platform')
    st.caption('This is a platform for an adaptive exam preparation system based on active learning approaches. Choose your role to get started.')
    artifacts = load_artifacts()
    # ensure DB exists
    backend.init_db(DB_PATH)

    if 'active_screen' not in st.session_state:
        st.session_state.active_screen = None

    if st.session_state.active_screen is None:
        show_landing_screen()
        return

    role, authorized = screen_gate()
    if not authorized:
        return

    if role == 'teacher':
        teacher_tab()
    else:
        student_tab(artifacts)


if __name__ == '__main__':
    main()
