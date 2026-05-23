import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import time
from pathlib import Path
import uuid
from datetime import datetime
import sys

# Resolve project root relative to this file so paths work
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add dashboard directory to sys.path to allow direct imports
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import backend
import AlClass as AlClass_module
import importlib
importlib.reload(AlClass_module)
from AlClass import AlClass, compute_question_features

BASE_DIR = ROOT_DIR
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'Dataset' / 'clean'
DB_PATH = BASE_DIR / 'data' / 'al_sessions.db'

DIFFICULTY_OPTIONS = ['easy', 'medium', 'hard']
# Teacher still uses a simple access code; students log in with their Student ID
ROLE_CODES = {
    'teacher': 'teacher123',
}


def load_artifacts():
    """
    Load model and simulation config.
    No scaler, no label encoders — matching the notebook simulation exactly.
    """
    artifacts = {}

    # Load simulation config
    config_path = MODELS_DIR / 'simulation_config.json'
    try:
        with open(config_path) as f:
            artifacts['config'] = json.load(f)
    except Exception:
        artifacts['config'] = None

    # Load AlClass Engine
    model_path = MODELS_DIR / 'xgboost.pkl'
    try:
        artifacts['engine'] = AlClass(str(model_path), str(config_path))
    except Exception as e:
        artifacts['engine'] = None
        st.warning(f'Could not load AlClass: {e}')

    return artifacts


def normalize_question_frame(df):
    """
    Normalize column names and compute question-level features.
    Features are computed once here so they don't need recomputation per prediction.
    """
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

    # Normalize difficulty text (keep as text for display; encoding happens in compute_question_features)
    df['difficulty'] = df['difficulty'].astype(str).str.lower().where(
        df['difficulty'].astype(str).str.lower().isin(DIFFICULTY_OPTIONS), 'medium'
    )

    base_cols = ['Question ID', 'question_text', 'difficulty', 'topic']
    mc_cols = [c for c in ['option_a', 'option_b', 'option_c', 'option_d', 'correct_answer', 'options'] if c in df.columns]
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
    }
    return pd.DataFrame([row])


def is_multiple_choice(row):
    return pd.notna(row.get('option_a')) and str(row.get('option_a')).strip()


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


def teacher_tab():
    st.header('Teacher — Quiz / Question Bank')
    st.caption('This area is only visible after teacher access is granted.')

    # Load engine config for display and editing
    config_path = MODELS_DIR / 'simulation_config.json'
    try:
        with open(config_path) as f:
            config = json.load(f)
        lower_t = config.get('lower_threshold', 0.25)
        upper_t = config.get('upper_threshold', 0.75)
        strategy = config.get('strategy', 'uncertainty').lower()
        max_quiz_length = int(config.get('max_quiz_length', 20))
    except Exception:
        config = {}
        lower_t, upper_t = 0.25, 0.75
        strategy = 'uncertainty'
        max_quiz_length = 20

    with st.expander('⚙️ Quiz Settings', expanded=False):
        st.info(
            f'Model-confidence threshold: keep asking while predicted correctness stays between {lower_t:.2f} and {upper_t:.2f}. '
            'When every remaining question is outside that band, or the max length is reached, the quiz stops.'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            strategies = ['uncertainty', 'qbc', 'random']
            strategy_idx = strategies.index(strategy) if strategy in strategies else 0
            new_strategy = st.selectbox(
                'Active Learning Selection Strategy', 
                strategies, 
                index=strategy_idx,
                format_func=lambda x: {'uncertainty': 'Uncertainty (Closest to 0.5)', 'qbc': 'Query by Committee (Simulated)', 'random': 'Random Selection'}.get(x, x.title())
            )
            
        with col2:
            new_max_length = st.number_input('Maximum Quiz Length', min_value=1, max_value=500, value=max_quiz_length, step=1)
            
        if st.button('Save Settings'):
            config['strategy'] = new_strategy
            config['max_quiz_length'] = new_max_length
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                st.success('Settings saved successfully!')
                # Update local variables for immediate display if needed
                strategy = new_strategy
                max_quiz_length = new_max_length
            except Exception as e:
                st.error(f'Failed to save settings: {e}')
                
    st.markdown('---')
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
            help='Accepted columns: question_text or question, topic, difficulty, options (pipe-separated) or option_a/b/c/d. '
                 'Question features (num_words, complexity, num_options, avg_option_length) are computed automatically.',
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
                    },
                    disabled=['Question ID'],
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
                staged_df = st.data_editor(
                    staged_df,
                    use_container_width=True,
                    num_rows='dynamic',
                    column_config={
                        'difficulty': st.column_config.SelectboxColumn('difficulty', options=DIFFICULTY_OPTIONS),
                    },
                    disabled=['Question ID'],
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
                    # Compute question features before saving
                    config_path = MODELS_DIR / 'simulation_config.json'
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        diff_map = config.get('difficulty_map', {'easy': 0, 'medium': 1, 'hard': 2})
                    except Exception:
                        diff_map = {'easy': 0, 'medium': 1, 'hard': 2}

                    normalized = compute_question_features(normalized, difficulty_map=diff_map)
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    normalized.to_csv(DATA_DIR / 'question_bank_app.csv', index=False)
                    st.success('Saved question_bank_app.csv (with precomputed features)')
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

                stat_left, stat_right = st.columns(2)
                stat_left.metric('Answers', total)
                stat_right.metric('Correct Answers', f'{correct} / {total}')

                # Show estimated score if available
                est_score = selected.get('estimated_score')
                sim_score = selected.get('simulation_score')
                tot_q = selected.get('total_questions')
                
                col_e1, col_e2 = st.columns(2)
                if est_score is not None:
                    col_e1.metric('Estimated Score', f'{est_score:.1f} / {max_quiz_length}')
                if sim_score is not None and tot_q is not None:
                    col_e2.metric('Simulation Score', f'{sim_score * tot_q:.1f} / {tot_q}')

                st.write(f"Student: {selected.get('full_name') or 'Unknown'}")
                st.write(f"Student ID: {selected['student_id'] or 'Unknown'}")

                st.markdown('#### Global session statistics')
                g1, g2 = st.columns(2)
                g1.metric('Sessions', global_stats['sessions'])
                g2.metric('Students', global_stats['students'])
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

    engine = artifacts.get('engine')
    if engine is None:
        st.error('Model not loaded. Please ensure models/xgboost.pkl and models/simulation_config.json exist.')
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

    # Ensure question features are computed (in case saved CSV lacks them)
    required_feature_cols = ['difficulty', 'num_words', 'qstn_complexity', 'num_options', 'avg_option_length']
    needs_compute = any(col not in df_q.columns for col in required_feature_cols)
    if needs_compute:
        df_q = compute_question_features(df_q, difficulty_map=engine.difficulty_map)
    else:
        # Ensure difficulty is int-encoded
        if df_q['difficulty'].dtype == object:
            df_q['difficulty'] = (
                df_q['difficulty']
                .astype(str)
                .str.lower()
                .map(engine.difficulty_map)
                .fillna(1)
                .astype(int)
            )

    # Initialize session state for quiz
    if 'session_state_init' not in st.session_state:
        st.session_state.session_state_init = True
        st.session_state.history_labels = []
        st.session_state.history_difficulty = []
        st.session_state.history_response_time = []
        st.session_state.answered_qids = []
        st.session_state.quiz_finished = False
        st.session_state.question_start_time = None

    # Check if quiz is finished
    if st.session_state.get('quiz_finished', False):
        _show_quiz_results(engine, df_q)
        return

    # Build remaining questions list
    remaining_indices = [
        i for i in range(len(df_q))
        if df_q.iloc[i].get('Question ID', f'Q{i}') not in st.session_state.answered_qids
    ]

    if not remaining_indices:
        st.session_state.quiz_finished = True
        st.rerun()
        return

    remaining_questions = [df_q.iloc[i].to_dict() for i in remaining_indices]

    # Compute current state
    state = engine.compute_state(
        st.session_state.history_labels,
        st.session_state.history_difficulty,
        st.session_state.history_response_time,
    )

    # Get probabilities for all remaining questions
    probs = engine.predict_pool(remaining_questions, state)

    # Check stopping criterion
    # Stop if confident OR if we reached the maximum quiz length
    answered_count = len(st.session_state.history_labels)
    if engine.should_stop(probs) or answered_count >= getattr(engine, 'max_quiz_length', 20):
        # Save final probs for score estimation
        st.session_state.final_probs_remaining = probs
        st.session_state.quiz_finished = True
        st.rerun()
        return

    # Pick next question
    next_idx_local = engine.pick_next_question_idx(remaining_questions, probs, state)
    next_q = remaining_questions[next_idx_local]
    next_prob = float(probs[next_idx_local])

    # Track question start time for response time measurement
    if st.session_state.question_start_time is None:
        st.session_state.question_start_time = time.time()

    # Display stats
    answered_count = len(st.session_state.history_labels)
    max_quiz_length = getattr(engine, 'max_quiz_length', len(df_q))
    target_length = min(max_quiz_length, len(df_q))
    st.write(f"Questions answered: {answered_count} / {target_length}")
    st.write(f"Questions remaining: {target_length - answered_count}")
    if answered_count > 0:
        st.write(f"Current accuracy: {state['accuracy_so_far']:.2%}")

    st.subheader('Next Question')
    # Show difficulty label
    diff_reverse = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
    diff_label = diff_reverse.get(int(next_q.get('difficulty', 1)), 'Medium')
    st.caption(f'Difficulty: {diff_label}')
    st.write(next_q.get('question_text', ''))

    is_mc = False
    options = {}
    correct_answer = next_q.get('correct_answer', '')

    if next_q.get('option_a') is not None and pd.notna(next_q.get('option_a')) and str(next_q.get('option_a', '')).strip():
        is_mc = True
        options = {
            'A': next_q.get('option_a', ''),
            'B': next_q.get('option_b', ''),
            'C': next_q.get('option_c', ''),
            'D': next_q.get('option_d', ''),
        }
    elif next_q.get('options_cleaned') is not None and pd.notna(next_q.get('options_cleaned')) and str(next_q.get('options_cleaned', '')).strip():
        try:
            import ast
            parts = ast.literal_eval(next_q['options_cleaned'])
            if isinstance(parts, list) and len(parts) > 1:
                is_mc = True
                labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                for i, p in enumerate(parts):
                    label = labels[i] if i < len(labels) else str(i)
                    options[label] = p
        except Exception:
            pass
    
    if not is_mc and next_q.get('options') is not None and pd.notna(next_q.get('options')) and str(next_q.get('options', '')).strip():
        parts = [p.strip() for p in str(next_q['options']).split('|') if p.strip()]
        if len(parts) > 1:
            is_mc = True
            labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i, p in enumerate(parts):
                label = labels[i] if i < len(labels) else str(i)
                options[label] = p

    if is_mc:
        st.write('**Options:**')
        # Clean up prefixes like "A. " or "A) " from the options text to avoid "A: A. text"
        for k in options:
            cleaned = re.sub(rf'^{k}\s*[\.\)]\s*', '', str(options[k]), flags=re.IGNORECASE).strip()
            if cleaned:
                options[k] = cleaned
                
        selected_key = st.radio('Select answer', list(options.keys()), format_func=lambda k: f'{k}: {options[k]}')
    else:
        choice = st.radio('Answer', ['Correct', 'Incorrect'])

    if st.button('Submit Answer'):
        # Measure response time (in milliseconds, matching the dataset)
        if st.session_state.question_start_time is not None:
            response_time_ms = (time.time() - st.session_state.question_start_time) * 1000.0
        else:
            response_time_ms = 0.0
        st.session_state.question_start_time = None

        if is_mc:
            selected_text = options[selected_key]
            # Check correctness: exact match with key 'A'/'B', or matching option text
            correct_ans_str = str(correct_answer).strip()
            if correct_ans_str == selected_key:
                correct = 1
            elif correct_ans_str in selected_text or selected_text in correct_ans_str:
                correct = 1
            else:
                correct = 0
            answer_text = f'{selected_key}: {selected_text}'
        else:
            correct = 1 if choice == 'Correct' else 0
            answer_text = choice

        qid = next_q.get('Question ID', f'Q{remaining_indices[next_idx_local]}')

        # Update history
        st.session_state.history_labels.append(correct)
        st.session_state.history_difficulty.append(int(next_q.get('difficulty', 1)))
        st.session_state.history_response_time.append(response_time_ms)
        st.session_state.answered_qids.append(qid)

        # Persist answer to DB
        features_to_save = {
            'answer_text': answer_text,
            'response_time': response_time_ms,
            'difficulty': int(next_q.get('difficulty', 1)),
            'model_prob': next_prob,
        }
        try:
            backend.save_answer(
                DB_PATH,
                st.session_state.session_id,
                st.session_state.get('student_id', ''),
                qid,
                next_q.get('question_text', ''),
                correct,
                features_to_save,
                first_name=st.session_state.get('first_name', ''),
                last_name=st.session_state.get('last_name', ''),
            )
        except Exception as e:
            st.error(f'Failed to save answer: {e}')

        # Recompute state and check if quiz should stop
        new_state = engine.compute_state(
            st.session_state.history_labels,
            st.session_state.history_difficulty,
            st.session_state.history_response_time,
        )

        # Build new remaining list
        new_remaining = [
            df_q.iloc[i].to_dict() for i in range(len(df_q))
            if df_q.iloc[i].get('Question ID', f'Q{i}') not in st.session_state.answered_qids
        ]

        if not new_remaining:
            st.session_state.final_probs_remaining = np.array([])
            st.session_state.quiz_finished = True
        else:
            new_probs = engine.predict_pool(new_remaining, new_state)
            answered_count = len(st.session_state.history_labels)
            if engine.should_stop(new_probs) or answered_count >= getattr(engine, 'max_quiz_length', 20):
                st.session_state.final_probs_remaining = new_probs
                st.session_state.quiz_finished = True

        st.rerun()


def _show_quiz_results(engine, df_q):
    """Display quiz results with the estimated score."""
    st.success('Quiz completed!')

    history_labels = st.session_state.get('history_labels', [])
    total_questions = len(df_q)
    answered_count = len(history_labels)

    # Get final probabilities for remaining questions
    # Final estimate of the score
    probs = st.session_state.get('final_probs_remaining', np.array([]))
    total_questions = len(df_q)
    estimated_score = engine.estimate_score(history_labels, probs, total_questions)
    sim_score = engine.simulation_score(history_labels, probs, total_questions)

    # Actual accuracy on answered questions
    actual_accuracy = float(np.mean(history_labels)) if history_labels else 0.0

    st.info(
        f'Quiz stopped because model confidence is outside the active band '
        f'{engine.lower_threshold:.2f}–{engine.upper_threshold:.2f} for all remaining questions, '
        f'or all questions have been answered, or the maximum quiz length was reached.'
    )

    actual_correct = int(sum(history_labels))
    target_length = min(engine.max_quiz_length, total_questions)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Questions Answered', f'{answered_count} / {target_length}')
    col2.metric('Correct Answers', f'{actual_correct} / {answered_count}')
    col3.metric('Estimated Score', f'{estimated_score:.1f} / {engine.max_quiz_length}')
    col4.metric('Simulation Score', f'{sim_score * total_questions:.1f} / {total_questions}')

    # Save estimated score to the last answer's features for teacher visibility
    try:
        last_answer = backend.get_session_answers(DB_PATH, st.session_state.session_id)[-1]
        if last_answer:
            answer_id = last_answer[0]
            import json
            existing_features = json.loads(last_answer[4]) if last_answer[4] else {}
            features_summary = {
                'estimated_score': estimated_score,
                'simulation_score': sim_score,
                'actual_accuracy': actual_accuracy,
                'answered_count': answered_count,
                'total_questions': total_questions,
            }
            backend.save_answer(
                DB_PATH,
                session_id,
                st.session_state.get('student_id', ''),
                'SUMMARY',
                'Quiz summary',
                0,
                features_summary,
                first_name=st.session_state.get('first_name', ''),
                last_name=st.session_state.get('last_name', ''),
            )
    except Exception:
        pass


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
