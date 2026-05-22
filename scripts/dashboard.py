"""
Adaptive Exam Preparation System — Streamlit Dashboard
ENSIA Machine Learning | Spring 2025-2026

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, os, time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Exam System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main { background: #0b0f1a; }

.metric-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-val  { font-size: 2rem; font-weight: 800; }
.metric-lbl  { font-size: 0.75rem; color: #64748b; font-family: 'Space Mono', monospace; letter-spacing: 2px; text-transform: uppercase; }

.level-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 1px;
}
.level-Advanced     { background: rgba(0,212,255,.15);  color: #00d4ff;  border: 1px solid rgba(0,212,255,.3); }
.level-Intermediate { background: rgba(245,158,11,.12); color: #f59e0b;  border: 1px solid rgba(245,158,11,.3); }
.level-Beginner     { background: rgba(239,68,68,.12);  color: #ef4444;  border: 1px solid rgba(239,68,68,.3); }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
    cursor: pointer;
}

.question-box {
    background: #111827;
    border: 1px solid #1e2d45;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.question-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.question-text { font-size: 1.05rem; font-weight: 600; }

.conf-bar-bg {
    background: #1e2d45;
    border-radius: 8px;
    height: 14px;
    width: 100%;
    margin: 6px 0;
}

.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DIFFICULTY_MAP = {'easy': 1, 'medium': 2, 'hard': 3}
THRESHOLD      = 0.80
MAX_Q          = 20
RANDOM_SEED    = 42

COLORS = {
    'Advanced':     '#00d4ff',
    'Intermediate': '#f59e0b',
    'Beginner':     '#ef4444',
}
STRATEGY_COLORS = {
    'Uncertainty Sampling': '#00d4ff',
    'Entropy-Based':        '#7c3aed',
    'Query-by-Committee':   '#10b981',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    models_dir = os.path.join(base, '..', 'models')
    data_dir   = os.path.join(base, '..', 'Dataset', 'clean')

    scaler   = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    
    try:
        scaler1 = joblib.load(os.path.join(models_dir, 'knowledge_estimator_scaler.pkl'))
    except FileNotFoundError:
        scaler1 = scaler
        
    try:
        scaler2 = joblib.load(os.path.join(models_dir, 'correctness_predictor_scaler.pkl'))
    except FileNotFoundError:
        scaler2 = scaler
        
    le_topic = joblib.load(os.path.join(models_dir, 'topic_label_encoder.pkl'))
    
    features_path = os.path.join(models_dir, 'features.json')
    if os.path.exists(features_path):
        with open(features_path) as f:
            features = json.load(f)
    else:
        features = []

    try:
        model1 = joblib.load(os.path.join(models_dir, 'knowledge_estimator_logistic_regression.pkl'))
    except FileNotFoundError:
        model1 = joblib.load(os.path.join(models_dir, 'knowledge_estimator.pkl'))
        
    try:
        model2 = joblib.load(os.path.join(models_dir, 'correctness_predictor_logistic_regression.pkl'))
    except FileNotFoundError:
        model2 = joblib.load(os.path.join(models_dir, 'correctness_predictor.pkl'))

    committee = []
    committee_names = [
        'correctness_predictor_logistic_regression.pkl',
        'correctness_predictor_naive_bayes.pkl',
        'correctness_predictor_decision_tree.pkl',
        'correctness_predictor_random_forest.pkl'
    ]
    for name in committee_names:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            committee.append(joblib.load(path))
            
    if not committee:
        committee = [model2]

    df        = pd.read_csv(os.path.join(data_dir, 'interactions_clean.csv'))
    df_labeled = pd.read_csv(os.path.join(base, '..', 'notebooks', 'interactions_labeled.csv'))

    # Load question bank if available, otherwise build from interactions
    qb_path = os.path.join(data_dir, 'question_bank.csv')
    base_qb = df[['Question ID','difficulty','topic']].drop_duplicates().reset_index(drop=True)
    
    if os.path.exists(qb_path):
        qb_file = pd.read_csv(qb_path)
        question_bank = pd.merge(base_qb, qb_file, on='Question ID', how='left')
        question_bank['question_text'] = question_bank['question_text'].fillna('Question ' + question_bank['Question ID'].astype(str))
        if 'correct_answer' not in question_bank.columns:
            question_bank['correct_answer'] = 'N/A'
        if 'options' not in question_bank.columns:
            question_bank['options'] = ''
    else:
        question_bank = base_qb.copy()
        question_bank['question_text']  = 'Question ' + question_bank['Question ID'].astype(str)
        question_bank['correct_answer'] = 'N/A'
        question_bank['options'] = ''

    # Load real question text from topic_mapping.csv
    topic_mapping_path = os.path.join(data_dir, 'topic_mapping.csv')
    if os.path.exists(topic_mapping_path):
        topic_mapping = pd.read_csv(topic_mapping_path)
        id_to_text = dict(zip(topic_mapping['Question ID'], topic_mapping['question_text']))
        question_bank['question_text'] = question_bank['Question ID'].map(id_to_text).fillna(question_bank['question_text'])

    # Load evaluation results if available
    try:
        baseline_summary = json.load(open(os.path.join(models_dir, 'baseline_summary.json')))
        al_summary = pd.read_csv(os.path.join(models_dir, 'al_summary.csv'))
        baseline_acc_curve = np.load(os.path.join(models_dir, 'baseline_acc_curve.npy'))
        baseline_n_range   = np.load(os.path.join(models_dir, 'baseline_n_range.npy'))
    except FileNotFoundError:
        baseline_summary = None
        al_summary = None
        baseline_acc_curve = None
        baseline_n_range = None

    return (scaler, scaler1, scaler2, le_topic, features, model1, model2, committee,
            df, df_labeled, question_bank,
            baseline_summary, al_summary, baseline_acc_curve, baseline_n_range)


(scaler, scaler1, scaler2, le_topic, FEATURES, model1, model2, committee,
 df, df_labeled, question_bank,
 baseline_summary, al_summary, baseline_acc_curve, baseline_n_range) = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(history_df, question_row):
    diff_enc  = DIFFICULTY_MAP.get(question_row['difficulty'], 2)
    topic     = question_row['topic']
    topic_enc = le_topic.transform([topic])[0] if topic in le_topic.classes_ else -1

    if len(history_df) == 0:
        return [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0, diff_enc, topic_enc]

    th = history_df[history_df['topic'] == topic]
    return [
        history_df['correct'].mean(),
        th['correct'].mean() if len(th) > 0 else 0.5,
        history_df[history_df['difficulty']=='easy']['correct'].mean()   if len(history_df[history_df['difficulty']=='easy'])>0   else 0.5,
        history_df[history_df['difficulty']=='medium']['correct'].mean() if len(history_df[history_df['difficulty']=='medium'])>0 else 0.5,
        history_df[history_df['difficulty']=='hard']['correct'].mean()   if len(history_df[history_df['difficulty']=='hard'])>0   else 0.5,
        history_df['Response Time'].mean(),
        len(history_df),
        diff_enc,
        topic_enc,
    ]


def p_correct(history_df, question_row):
    feat = build_feature_vector(history_df, question_row)
    X    = scaler2.transform([feat])
    return model2.predict_proba(X)[0, 1]


def get_confidence(history_df, question_row):
    feat  = build_feature_vector(history_df, question_row)
    feat1 = feat[:8]
    X     = scaler1.transform([feat1])
    proba = model1.predict_proba(X)[0]
    level = model1.classes_[np.argmax(proba)]
    conf  = float(np.max(proba))
    return level, conf, dict(zip(model1.classes_, proba))


def select_next_question(strategy, history_df, pool_df):
    if strategy == 'Random (Baseline)':
        return pool_df.sample(1, random_state=None).iloc[0]

    if strategy == 'Uncertainty Sampling':
        best_q, best_dist = None, float('inf')
        for _, q in pool_df.iterrows():
            p    = p_correct(history_df, q)
            dist = abs(p - 0.5)
            if dist < best_dist:
                best_dist, best_q = dist, q
        return best_q

    if strategy == 'Entropy-Based':
        best_q, best_H = None, -float('inf')
        for _, q in pool_df.iterrows():
            p = np.clip(p_correct(history_df, q), 1e-9, 1 - 1e-9)
            H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            if H > best_H:
                best_H, best_q = H, q
        return best_q

    if strategy == 'Query-by-Committee':
        best_q, best_H = None, -float('inf')
        for _, q in pool_df.iterrows():
            feat  = build_feature_vector(history_df, q)
            X     = scaler2.transform([feat])
            votes = np.array([m.predict(X)[0] for m in committee])
            p_v   = np.clip(votes.mean(), 1e-9, 1 - 1e-9)
            H_v   = -p_v * np.log2(p_v) - (1 - p_v) * np.log2(1 - p_v)
            if H_v > best_H:
                best_H, best_q = H_v, q
        return best_q


def confidence_color(conf):
    if conf >= 0.80: return '#10b981'
    if conf >= 0.60: return '#f59e0b'
    return '#ef4444'


def _process_answer(q, answer, strategy, threshold, max_q):
    if 'question_start_time' in st.session_state and st.session_state.question_start_time is not None:
        rt_ms = (time.time() - st.session_state.question_start_time) * 1000
    else:
        rt_ms = 15000

    st.session_state.history.append({
        'correct':       answer,
        'difficulty':    q['difficulty'],
        'topic':         q['topic'],
        'Response Time': rt_ms,
    })
    st.session_state.step += 1

    # Remove answered question from pool
    st.session_state.pool = st.session_state.pool[
        st.session_state.pool['Question ID'] != q['Question ID']
    ].reset_index(drop=True)

    # Compute confidence after this answer
    history_df = pd.DataFrame(st.session_state.history)
    dummy_q    = q
    level, conf, proba_dict = get_confidence(history_df, dummy_q)
    st.session_state.conf_curve.append(conf)
    st.session_state.level_curve.append(level)

    # Check stopping condition
    if conf >= threshold or st.session_state.step >= max_q or len(st.session_state.pool) == 0:
        st.session_state.finished    = True
        st.session_state.final_level = level
        st.session_state.final_conf  = conf
        st.session_state.awaiting_ans = False
        return

    # Select next question
    next_q = select_next_question(strategy, history_df, st.session_state.pool)
    st.session_state.current_q    = next_q
    st.session_state.awaiting_ans = True
    st.session_state.question_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    st.session_state.history      = []
    st.session_state.pool         = question_bank.copy().reset_index(drop=True)
    st.session_state.current_q    = None
    st.session_state.started      = False
    st.session_state.finished     = False
    st.session_state.conf_curve   = []
    st.session_state.level_curve  = []
    st.session_state.step         = 0
    st.session_state.final_level  = None
    st.session_state.final_conf   = None
    st.session_state.awaiting_ans = False
    st.session_state.question_start_time = None

if 'started' not in st.session_state:
    init_session()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Adaptive Exam System")
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox(
        '', ['Uncertainty Sampling', 'Entropy-Based', 'Query-by-Committee', 'Random (Baseline)'],
        label_visibility='collapsed'
    )

    st.markdown('<div class="sidebar-title">Confidence Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider('', 0.50, 0.95, THRESHOLD, 0.05, label_visibility='collapsed')

    st.markdown('<div class="sidebar-title">Max Questions</div>', unsafe_allow_html=True)
    max_q = st.slider('', 5, 30, MAX_Q, 1, label_visibility='collapsed')

    st.markdown("---")

    if st.button("🔄  Start New Session"):
        init_session()
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Progress</div>', unsafe_allow_html=True)

    if st.session_state.started:
        q_used = st.session_state.step
        conf   = st.session_state.conf_curve[-1] if st.session_state.conf_curve else 0.0
        st.markdown(f"**Questions answered:** {q_used}")
        st.markdown(f"**Current confidence:** {conf:.0%}")
        st.progress(min(conf, 1.0))

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#475569;">'
        'ENSIA · ML Project · Spring 2025-2026</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_exam, tab_eval, tab_results = st.tabs(["🎯  Live Exam", "📊  Evaluation", "📋  Results"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE EXAM
# ═════════════════════════════════════════════════════════════════════════════
with tab_exam:
    st.markdown("## Adaptive Exam")

    col_q, col_m = st.columns([2, 1], gap="large")

    with col_q:
        # ── FINISHED ─────────────────────────────────────────────────────────
        if st.session_state.finished:
            level = st.session_state.final_level
            conf  = st.session_state.final_conf
            q_n   = st.session_state.step

            st.success("✅ Exam complete!")
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {COLORS.get(level,'#64748b')}">
                <div class="metric-lbl">Estimated Knowledge Level</div>
                <div class="metric-val" style="color:{COLORS.get(level,'#e2e8f0')}">{level}</div>
                <br>
                <div class="metric-lbl">Confidence</div>
                <div class="metric-val" style="color:{confidence_color(conf)}">{conf:.0%}</div>
                <br>
                <div class="metric-lbl">Questions used</div>
                <div class="metric-val">{q_n}</div>
            </div>
            """, unsafe_allow_html=True)

            if strategy != 'Random (Baseline)':
                saved = MAX_Q - q_n
                if saved > 0:
                    st.info(f"💡 **{strategy}** saved **{saved} questions** compared to the baseline (N={MAX_Q})")

        # ── NOT STARTED ───────────────────────────────────────────────────────
        elif not st.session_state.started:
            st.markdown("""
            <div class="question-box">
                <div class="question-meta">READY TO BEGIN</div>
                <div class="question-text">
                    Press <strong>Start Exam</strong> to begin the adaptive session.<br><br>
                    The system will ask questions one by one, using the selected strategy to pick 
                    the most informative question at each step. It stops automatically once it is 
                    confident about your knowledge level.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Start Exam"):
                st.session_state.started = True
                history_df = pd.DataFrame(columns=['correct','difficulty','topic','Response Time'])
                pool       = st.session_state.pool
                q          = select_next_question(strategy, history_df, pool)
                st.session_state.current_q    = q
                st.session_state.awaiting_ans = True
                st.session_state.question_start_time = time.time()
                st.rerun()

        # ── QUESTION DISPLAY ──────────────────────────────────────────────────
        elif st.session_state.awaiting_ans and st.session_state.current_q is not None:
            q   = st.session_state.current_q
            qid = q['Question ID']

            # Fetch question text from bank
            qb_row  = question_bank[question_bank['Question ID'] == qid]
            q_text  = qb_row['question_text'].iloc[0]  if len(qb_row) > 0 else f'Question {qid}'
            q_ans   = qb_row['correct_answer'].iloc[0] if len(qb_row) > 0 else 'N/A'
            q_opts  = qb_row['options'].iloc[0] if len(qb_row) > 0 and 'options' in qb_row.columns else ''

            diff_color = {'easy': '#10b981', 'medium': '#f59e0b', 'hard': '#ef4444'}.get(q['difficulty'], '#64748b')

            st.markdown(f"""
            <div class="question-box">
                <div class="question-meta">
                    QUESTION {st.session_state.step + 1} &nbsp;·&nbsp;
                    <span style="color:{diff_color}">{q['difficulty'].upper()}</span> &nbsp;·&nbsp;
                    {q['topic']}
                </div>
                <div class="question-text">{q_text}</div>
            </div>
            """, unsafe_allow_html=True)

            if pd.notna(q_opts) and str(q_opts).strip() != "":
                opts_list = [opt.strip() for opt in str(q_opts).split(' | ')]
                correct_answers = [ans.strip() for ans in str(q_ans).split(' | ')]
                is_multi = len(correct_answers) > 1

                with st.form(key=f"question_form_{st.session_state.step}"):
                    if is_multi:
                        st.markdown("**Select all that apply:**")
                        user_selections = []
                        for i, opt in enumerate(opts_list):
                            if st.checkbox(opt, key=f"q_{st.session_state.step}_opt_{i}"):
                                user_selections.append(opt)
                    else:
                        st.markdown("**Select one answer:**")
                        user_selection = st.radio("", opts_list, index=None, label_visibility="collapsed", key=f"q_{st.session_state.step}_radio")
                        user_selections = [user_selection] if user_selection else []
                        
                    submitted = st.form_submit_button("Submit Answer")
                    if submitted:
                        if not user_selections:
                            st.warning("Please select an answer.")
                        else:
                            # Evaluate correctness
                            user_ans_texts = [opt.split('. ', 1)[1].strip() if '. ' in opt else opt.strip() for opt in user_selections]
                            if set(user_ans_texts) == set(correct_answers):
                                is_correct = 1
                            else:
                                is_correct = 0
                                
                            _process_answer(q, is_correct, strategy, threshold, max_q)
                            st.rerun()
            else:
                # Fallback for questions without options
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅  Correct", use_container_width=True):
                        _process_answer(q, 1, strategy, threshold, max_q)
                        st.rerun()
                with c2:
                    if st.button("❌  Incorrect", use_container_width=True):
                        _process_answer(q, 0, strategy, threshold, max_q)
                        st.rerun()

    with col_m:
        # ── METRICS ───────────────────────────────────────────────────────────
        if st.session_state.conf_curve:
            conf  = st.session_state.conf_curve[-1]
            level = st.session_state.level_curve[-1] if st.session_state.level_curve else '—'

            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {COLORS.get(level,'#64748b')}">
                <div class="metric-lbl">Current Estimate</div>
                <div class="metric-val" style="color:{COLORS.get(level,'#e2e8f0')};font-size:1.4rem">{level}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {confidence_color(conf)}">
                <div class="metric-lbl">Confidence</div>
                <div class="metric-val" style="color:{confidence_color(conf)}">{conf:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Questions Asked</div>
                <div class="metric-val">{st.session_state.step}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence over time mini chart
            if len(st.session_state.conf_curve) > 1:
                fig, ax = plt.subplots(figsize=(4, 2))
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#111827')
                steps = list(range(1, len(st.session_state.conf_curve) + 1))
                ax.plot(steps, st.session_state.conf_curve, color='#00d4ff', linewidth=2)
                ax.axhline(threshold, color='#10b981', linestyle='--', linewidth=1)
                ax.set_ylim(0.3, 1.05)
                ax.set_xlabel('Questions', color='#64748b', fontsize=8)
                ax.set_ylabel('Confidence', color='#64748b', fontsize=8)
                ax.tick_params(colors='#64748b', labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#1e2d45')
                st.pyplot(fig, use_container_width=True)
                plt.close()

        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Strategy</div>
                <div style="font-size:0.95rem;font-weight:700;margin-top:6px">{}</div>
            </div>
            """.format(strategy), unsafe_allow_html=True)

            st.markdown("""
            <div class="metric-card">
                <div class="metric-lbl">Threshold</div>
                <div class="metric-val">{:.0%}</div>
            </div>
            """.format(threshold), unsafe_allow_html=True)




# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## Evaluation — Adaptive vs. Baseline")

    if baseline_summary is None or al_summary is None:
        st.warning("⚠️ Run `Baseline.ipynb` and `Active_Learning_Loop.ipynb` first to generate evaluation results.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("### Summary")
        cols = st.columns(4)
        entries = al_summary.set_index('Strategy').to_dict('index')

        for col, (name, row) in zip(cols, entries.items()):
            with col:
                color = STRATEGY_COLORS.get(name, '#64748b')
                saving = row.get('Saving %', 0)
                st.markdown(f"""
                <div class="metric-card" style="border-top:3px solid {color}">
                    <div class="metric-lbl">{name}</div>
                    <div class="metric-val" style="color:{color}">{row['Accuracy']:.0%}</div>
                    <div style="font-size:0.75rem;color:#64748b;font-family:'Space Mono',monospace">
                        {row['Questions (mean)']:.1f}q &nbsp;·&nbsp; {saving:.0f}% fewer
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Accuracy vs questions chart ───────────────────────────────────────
        st.markdown("### Accuracy vs. Questions Used")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0b0f1a')
        ax.set_facecolor('#0b0f1a')

        ax.plot(baseline_n_range, baseline_acc_curve, 'o-',
                color='#64748b', linewidth=2, label='Baseline (random)')

        for name, row in entries.items():
            color = STRATEGY_COLORS.get(name, '#64748b')
            ax.scatter(row['Questions (mean)'], row['Accuracy'],
                       s=180, color=color, zorder=5,
                       label=f"{name} ({row['Questions (mean)']:.1f}q)")
            ax.annotate(f"{row['Accuracy']:.2f}",
                        (row['Questions (mean)'], row['Accuracy']),
                        textcoords='offset points', xytext=(8, 4),
                        color=color, fontsize=9)

        ax.axhline(0.80, color='#ef4444', linestyle='--', linewidth=1, label='80% target')
        ax.set_xlabel('Number of Questions', color='#94a3b8')
        ax.set_ylabel('Accuracy', color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.legend(facecolor='#111827', edgecolor='#1e2d45', labelcolor='#e2e8f0')
        ax.grid(alpha=0.15, color='#1e2d45')
        ax.set_ylim(0, 1.05)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e2d45')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Full summary table ────────────────────────────────────────────────
        st.markdown("### Full Comparison Table")
        display_df = al_summary.copy()
        st.dataframe(
            display_df.style
            .format({'Accuracy': '{:.1%}', 'Saving %': '{:.1f}%'})
            .background_gradient(subset=['Accuracy'], cmap='Blues'),
            use_container_width=True
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS (session history)
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("## Session History")

    if not st.session_state.history:
        st.info("No answers recorded yet. Start an exam in the Live Exam tab.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df.index = history_df.index + 1
        history_df.index.name = 'Q#'

        # Format response time in seconds for display
        if 'Response Time' in history_df.columns:
            history_df['Response Time (s)'] = (history_df['Response Time'] / 1000).map('{:.1f}s'.format)

        # Add confidence from curve
        if st.session_state.conf_curve:
            history_df['Confidence After'] = [
                f"{c:.0%}" for c in st.session_state.conf_curve[:len(history_df)]
            ]

        columns_to_show = ['correct','difficulty','topic']
        if 'Response Time (s)' in history_df.columns:
            columns_to_show.append('Response Time (s)')
        if 'Confidence After' in history_df.columns:
            columns_to_show.append('Confidence After')

        st.dataframe(
            history_df[columns_to_show],
            use_container_width=True
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            acc = history_df['correct'].mean()
            st.metric("Session Accuracy", f"{acc:.0%}")
        with c2:
            st.metric("Questions Answered", len(history_df))
        with c3:
            if st.session_state.final_level:
                st.metric("Estimated Level", st.session_state.final_level)

        # Per-topic breakdown
        if len(history_df) > 0:
            st.markdown("### Per-Topic Accuracy")
            topic_acc = history_df.groupby('topic')['correct'].agg(['mean','count']).reset_index()
            topic_acc.columns = ['Topic', 'Accuracy', 'Questions']
            topic_acc['Accuracy'] = topic_acc['Accuracy'].map('{:.0%}'.format)
            st.dataframe(topic_acc, use_container_width=True)
