"""
Active Learning Dashboard — Streamlit (CP-Only)
Adaptive Exam Preparation System | ENSIA ML | Spring 2025-2026

Run with:
    streamlit run al_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from al_engine import ALEngine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Exam — Active Learning",
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
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background: #0b0f1a; }

.metric-card {
    background: #111827; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 18px 22px;
    text-align: center; margin-bottom: 8px;
}
.metric-val  { font-size: 2rem; font-weight: 800; }
.metric-lbl  { font-size: 0.75rem; color: #64748b; font-family: 'Space Mono', monospace;
               letter-spacing: 2px; text-transform: uppercase; }

.question-box {
    background: #111827; border: 1px solid #1e2d45;
    border-left: 4px solid #00d4ff; border-radius: 12px;
    padding: 20px 24px; margin: 12px 0;
}
.question-meta { font-family: 'Space Mono', monospace; font-size: 0.7rem;
                 color: #64748b; letter-spacing: 1px; margin-bottom: 8px; }
.question-text { font-size: 1.05rem; font-weight: 600; }

.sidebar-title {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    letter-spacing: 2px; color: #64748b; text-transform: uppercase; margin-bottom: 4px;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    color: white; border: none; border-radius: 10px;
    padding: 10px 28px; font-weight: 700; font-size: 0.95rem;
    width: 100%; cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Cache removed during development to avoid stale artifacts.
# Re-add @st.cache_resource once the pipeline is stable.
def load_engine():
    return ALEngine()

engine = load_engine()
st.cache_resource.clear()  # clear any stale cache from previous runs

STRATEGY_MAP = {
    "Uncertainty Sampling": "uncertainty",
    "Entropy-Based": "entropy",
    "Query-by-Committee": "qbc",
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    st.session_state.started = False
    st.session_state.finished = False
    st.session_state.history = []
    st.session_state.pool = None
    st.session_state.current_q = None
    st.session_state.student_raw_id = None
    st.session_state.student_enc_id = None
    st.session_state.step = 0
    st.session_state.awaiting_ans = False
    st.session_state.question_start_time = None
    st.session_state.score = None

if "started" not in st.session_state:
    init_session()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def process_answer(q, correct, strategy_key):
    """Process a submitted answer and select the next question."""
    # Measure response time
    if st.session_state.question_start_time is not None:
        rt_ms = (time.time() - st.session_state.question_start_time) * 1000
    else:
        rt_ms = 15000.0

    # Add to history
    st.session_state.history.append({
        "correct": int(correct),
        "diff_enc": int(q.get("diff_enc", 2)),
        "topic_enc": int(q.get("topic_enc", -1)),
        "response_time": float(rt_ms),
        "question_id": int(q["Question ID"]),
        "difficulty": str(q.get("difficulty", "medium")),
        "topic": str(q.get("topic", "Unknown")),
    })
    st.session_state.step += 1

    # Remove answered question from pool
    pool = st.session_state.pool
    pool = pool[pool["Question ID"] != q["Question ID"]].reset_index(drop=True)
    st.session_state.pool = pool

    history = st.session_state.history
    n_asked = len(history)

    # ── Check stopping conditions ────────────────────────────────────
    if n_asked >= engine.max_questions or len(pool) == 0:
        st.session_state.finished = True
        st.session_state.score = engine.compute_score(history)
        st.session_state.awaiting_ans = False
        return

    # Score remaining pool
    probas = engine.score_pool(
        st.session_state.student_enc_id, history, pool
    )

    # Stop if model is confident about all remaining questions
    if engine.should_stop(probas):
        st.session_state.finished = True
        st.session_state.score = engine.compute_score(history)
        st.session_state.awaiting_ans = False
        return

    # Select next question
    idx = engine.select_next_question(strategy_key, probas, pool)
    if idx is None:
        st.session_state.finished = True
        st.session_state.score = engine.compute_score(history)
        st.session_state.awaiting_ans = False
        return

    next_q = pool.iloc[idx]
    st.session_state.current_q = next_q
    st.session_state.pool = pool.drop(pool.index[idx]).reset_index(drop=True)
    st.session_state.awaiting_ans = True
    st.session_state.question_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Adaptive Exam (AL)")
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Student ID (optional)</div>', unsafe_allow_html=True)
    student_input = st.text_input("", placeholder="Leave blank for auto-assign", label_visibility="collapsed")

    st.markdown('<div class="sidebar-title">Strategy</div>', unsafe_allow_html=True)
    strategy_label = st.selectbox(
        "", list(STRATEGY_MAP.keys()), label_visibility="collapsed"
    )
    strategy_key = STRATEGY_MAP[strategy_label]

    st.markdown("---")
    if st.button("🔄  New Session"):
        init_session()
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Progress</div>', unsafe_allow_html=True)
    if st.session_state.started:
        st.markdown(f"**Questions asked:** {st.session_state.step}")
        score = engine.compute_score(st.session_state.history) if st.session_state.history else 0
        st.markdown(f"**Current score:** {score:.0%}")
        st.progress(min(score, 1.0))

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#475569;">'
        "ENSIA · ML Project · Spring 2025-2026</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
tab_exam, tab_results = st.tabs(["🎯  Live Exam", "📋  Results"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE EXAM
# ═════════════════════════════════════════════════════════════════════════════
with tab_exam:
    st.markdown("## Adaptive Exam")
    col_q, col_m = st.columns([2, 1], gap="large")

    with col_q:
        # ── FINISHED ─────────────────────────────────────────────────────
        if st.session_state.finished:
            score = st.session_state.score or 0
            n = st.session_state.step
            correct_n = sum(1 for h in st.session_state.history if h["correct"] == 1)

            st.success("✅ Quiz Complete!")
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #10b981">
                <div class="metric-lbl">Final Score</div>
                <div class="metric-val" style="color:#10b981">{score:.0%}</div>
                <br>
                <div class="metric-lbl">Correct Answers</div>
                <div class="metric-val">{correct_n} / {n}</div>
                <br>
                <div class="metric-lbl">Questions Asked</div>
                <div class="metric-val">{n}</div>
            </div>
            """, unsafe_allow_html=True)

            remaining = len(st.session_state.pool) if st.session_state.pool is not None else 0
            if remaining > 0:
                st.info(
                    f"💡 The model stopped early — it was confident about the "
                    f"remaining **{remaining}** questions. Strategy: **{strategy_label}**"
                )

        # ── NOT STARTED ──────────────────────────────────────────────────
        elif not st.session_state.started:
            st.markdown("""
            <div class="question-box">
                <div class="question-meta">READY TO BEGIN</div>
                <div class="question-text">
                    Press <strong>Start Exam</strong> to begin.<br><br>
                    The first question is selected randomly from medium difficulty.
                    After that, the system uses your chosen strategy to pick the
                    most informative question. It stops when the model is confident
                    about all remaining questions.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Start Exam"):
                raw_id, enc_id = engine.resolve_student_id(
                    student_input.strip() if student_input.strip() else None
                )
                pool = engine.get_question_pool()
                first_q = engine.get_initial_question(pool)

                st.session_state.started = True
                st.session_state.student_raw_id = raw_id
                st.session_state.student_enc_id = enc_id
                st.session_state.pool = pool[
                    pool["Question ID"] != first_q["Question ID"]
                ].reset_index(drop=True)
                st.session_state.current_q = first_q
                st.session_state.awaiting_ans = True
                st.session_state.question_start_time = time.time()
                st.rerun()

        # ── QUESTION DISPLAY ─────────────────────────────────────────────
        elif st.session_state.awaiting_ans and st.session_state.current_q is not None:
            q = st.session_state.current_q
            qid = q["Question ID"]

            # Get question text from the full bank
            qb_row = engine.full_question_bank[
                engine.full_question_bank["Question ID"] == qid
            ]
            q_text = qb_row["question_text"].iloc[0] if len(qb_row) > 0 else f"Question {qid}"
            q_opts = qb_row["options"].iloc[0] if len(qb_row) > 0 and "options" in qb_row.columns else ""
            q_ans = qb_row["correct_answer"].iloc[0] if len(qb_row) > 0 and "correct_answer" in qb_row.columns else ""

            diff = q.get("difficulty", "medium")
            diff_color = {"easy": "#10b981", "medium": "#f59e0b", "hard": "#ef4444"}.get(diff, "#64748b")
            topic = q.get("topic", "Unknown")

            st.markdown(f"""
            <div class="question-box">
                <div class="question-meta">
                    QUESTION {st.session_state.step + 1} &nbsp;·&nbsp;
                    <span style="color:{diff_color}">{diff.upper()}</span> &nbsp;·&nbsp;
                    {topic}
                </div>
                <div class="question-text">{q_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Show options if available
            if pd.notna(q_opts) and str(q_opts).strip():
                opts_list = [opt.strip() for opt in str(q_opts).split(" | ")]
                correct_answers = [ans.strip() for ans in str(q_ans).split(" | ")]
                is_multi = len(correct_answers) > 1

                with st.form(key=f"q_form_{st.session_state.step}"):
                    if is_multi:
                        st.markdown("**Select all that apply:**")
                        selections = []
                        for i, opt in enumerate(opts_list):
                            if st.checkbox(opt, key=f"q{st.session_state.step}_o{i}"):
                                selections.append(opt)
                    else:
                        st.markdown("**Select one answer:**")
                        choice = st.radio("", opts_list, index=None,
                                          label_visibility="collapsed",
                                          key=f"q{st.session_state.step}_r")
                        selections = [choice] if choice else []

                    if st.form_submit_button("Submit Answer"):
                        if not selections:
                            st.warning("Please select an answer.")
                        else:
                            user_texts = [
                                o.split(". ", 1)[1].strip() if ". " in o else o.strip()
                                for o in selections
                            ]
                            is_correct = 1 if set(user_texts) == set(correct_answers) else 0
                            process_answer(q, is_correct, strategy_key)
                            st.rerun()
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅  Correct", use_container_width=True):
                        process_answer(q, 1, strategy_key)
                        st.rerun()
                with c2:
                    if st.button("❌  Incorrect", use_container_width=True):
                        process_answer(q, 0, strategy_key)
                        st.rerun()

    with col_m:
        if st.session_state.history:
            score = engine.compute_score(st.session_state.history)
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid #00d4ff">
                <div class="metric-lbl">Score</div>
                <div class="metric-val" style="color:#00d4ff">{score:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Questions Asked</div>
                <div class="metric-val">{st.session_state.step}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Strategy</div>
                <div style="font-size:0.95rem;font-weight:700;margin-top:6px">{strategy_label}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Strategy</div>
                <div style="font-size:0.95rem;font-weight:700;margin-top:6px">{strategy_label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Stopping Criterion</div>
                <div style="font-size:0.8rem;margin-top:6px;color:#94a3b8">
                    No question with<br>0.3 &lt; P(correct) &lt; 0.7
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("## Session History")

    if not st.session_state.history:
        st.info("No answers recorded yet. Start an exam in the Live Exam tab.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df.index = hist_df.index + 1
        hist_df.index.name = "Q#"

        if "response_time" in hist_df.columns:
            hist_df["Response Time (s)"] = (hist_df["response_time"] / 1000).map("{:.1f}s".format)

        cols_show = ["correct", "difficulty", "topic"]
        if "Response Time (s)" in hist_df.columns:
            cols_show.append("Response Time (s)")

        st.dataframe(hist_df[cols_show], use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            acc = hist_df["correct"].mean()
            st.metric("Session Score", f"{acc:.0%}")
        with c2:
            st.metric("Questions Answered", len(hist_df))
        with c3:
            correct_n = hist_df["correct"].sum()
            st.metric("Correct Answers", int(correct_n))

        # Per-topic breakdown
        st.markdown("### Per-Topic Accuracy")
        topic_acc = hist_df.groupby("topic")["correct"].agg(["mean", "count"]).reset_index()
        topic_acc.columns = ["Topic", "Accuracy", "Questions"]
        topic_acc["Accuracy"] = topic_acc["Accuracy"].map("{:.0%}".format)
        st.dataframe(topic_acc, use_container_width=True)
