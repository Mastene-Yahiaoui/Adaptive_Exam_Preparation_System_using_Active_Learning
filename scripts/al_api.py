"""
Active Learning API — FastAPI Service
Adaptive Exam Preparation System | ENSIA ML | Spring 2025-2026

Run with:
    uvicorn al_api:app --reload --port 8000

Or:
    python al_api.py
"""

import uuid
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from al_engine import ALEngine

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Exam API",
    description="Active Learning-powered adaptive exam system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engine once at startup
engine = ALEngine()

# In-memory session store (replace with Redis/DB in production)
sessions: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
class StartSessionRequest(BaseModel):
    student_id: Optional[str] = None
    strategy: str = "uncertainty"  # "uncertainty", "entropy", "qbc"


class StartSessionResponse(BaseModel):
    session_id: str
    student_id: str
    strategy: str
    question: dict
    message: str


class AnswerRequest(BaseModel):
    question_id: int
    correct: int  # 0 or 1
    response_time_ms: float


class AnswerResponse(BaseModel):
    finished: bool
    questions_used: int
    score: Optional[float] = None
    score_percent: Optional[str] = None
    next_question: Optional[dict] = None
    summary: Optional[dict] = None
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    student_id: str
    strategy: str
    questions_used: int
    finished: bool
    score: Optional[float] = None
    remaining_pool_size: int


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def question_to_dict(row):
    """Convert a pool row to a JSON-safe question dict."""
    return {
        "question_id": int(row["Question ID"]),
        "question_text": str(row.get("question_text", f"Question {row['Question ID']}")),
        "options": str(row.get("options", "")),
        "correct_answer": str(row.get("correct_answer", "")),
        "difficulty": str(row.get("difficulty", "medium")),
        "topic": str(row.get("topic", "Unknown")),
    }


def add_to_history(session, question_row, correct, response_time_ms):
    """Add an answered question to the session history."""
    session["history"].append({
        "correct": int(correct),
        "diff_enc": int(question_row.get("diff_enc", 2)),
        "topic_enc": int(question_row.get("topic_enc", -1)),
        "response_time": float(response_time_ms),
        "question_id": int(question_row["Question ID"]),
        "difficulty": str(question_row.get("difficulty", "medium")),
        "topic": str(question_row.get("topic", "Unknown")),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    """
    Start a new adaptive exam session.

    - Assigns a student ID (new or existing).
    - First question is chosen randomly from medium difficulty.
    """
    session_id = uuid.uuid4().hex[:12]
    raw_id, encoded_id = engine.resolve_student_id(req.student_id)
    pool = engine.get_question_pool()

    # First question: random from intermediate difficulty
    first_q = engine.get_initial_question(pool)

    # Remove first question from pool
    pool = pool[pool["Question ID"] != first_q["Question ID"]].reset_index(drop=True)

    sessions[session_id] = {
        "student_raw_id": raw_id,
        "student_enc_id": encoded_id,
        "strategy": req.strategy,
        "history": [],
        "pool": pool,
        "current_question": first_q,
        "finished": False,
        "started_at": time.time(),
    }

    return StartSessionResponse(
        session_id=session_id,
        student_id=raw_id,
        strategy=req.strategy,
        question=question_to_dict(first_q),
        message="Session started. Answer the first question to begin adaptive selection.",
    )


@app.post("/session/{session_id}/answer", response_model=AnswerResponse)
def submit_answer(session_id: str, req: AnswerRequest):
    """
    Submit an answer for the current question.

    After the first answer, the model starts using the chosen strategy
    to select the most informative next question.

    Stops when:
    - No remaining question has 0.3 < P(correct) < 0.7, OR
    - Max questions reached, OR
    - Pool is empty.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if session["finished"]:
        raise HTTPException(status_code=400, detail="Session already finished")

    current_q = session["current_question"]

    # Verify the answer matches the current question
    if req.question_id != int(current_q["Question ID"]):
        raise HTTPException(
            status_code=400,
            detail=f"Expected answer for question {current_q['Question ID']}, got {req.question_id}",
        )

    # Record the answer
    add_to_history(session, current_q, req.correct, req.response_time_ms)
    pool = session["pool"]
    history = session["history"]
    n_asked = len(history)

    # ── Check stopping conditions ────────────────────────────────────
    if n_asked >= engine.max_questions or len(pool) == 0:
        session["finished"] = True
        summary = engine.get_session_summary(history, len(pool))
        return AnswerResponse(
            finished=True,
            questions_used=n_asked,
            score=summary["score"],
            score_percent=summary["score_percent"],
            summary=summary,
            message=f"Quiz complete! Score: {summary['score_percent']}",
        )

    # Score the remaining pool
    probas = engine.score_pool(session["student_enc_id"], history, pool)

    # Check if the model is confident about all remaining questions
    if engine.should_stop(probas):
        session["finished"] = True
        summary = engine.get_session_summary(history, len(pool))
        return AnswerResponse(
            finished=True,
            questions_used=n_asked,
            score=summary["score"],
            score_percent=summary["score_percent"],
            summary=summary,
            message=f"Quiz complete — model is confident. Score: {summary['score_percent']}",
        )

    # ── Select next question via strategy ────────────────────────────
    idx = engine.select_next_question(session["strategy"], probas, pool)
    if idx is None:
        session["finished"] = True
        summary = engine.get_session_summary(history, len(pool))
        return AnswerResponse(
            finished=True,
            questions_used=n_asked,
            score=summary["score"],
            score_percent=summary["score_percent"],
            summary=summary,
            message="No more questions to ask.",
        )

    next_q = pool.iloc[idx]
    session["current_question"] = next_q
    session["pool"] = pool.drop(pool.index[idx]).reset_index(drop=True)

    return AnswerResponse(
        finished=False,
        questions_used=n_asked,
        next_question=question_to_dict(next_q),
        message=f"Question {n_asked + 1} selected via {session['strategy']}.",
    )


@app.get("/session/{session_id}/status", response_model=SessionStatusResponse)
def get_session_status(session_id: str):
    """Get the current status of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    score = engine.compute_score(session["history"]) if session["history"] else None

    return SessionStatusResponse(
        session_id=session_id,
        student_id=session["student_raw_id"],
        strategy=session["strategy"],
        questions_used=len(session["history"]),
        finished=session["finished"],
        score=score,
        remaining_pool_size=len(session["pool"]),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "questions_loaded": len(engine.full_question_bank),
        "features": engine.features,
        "config": engine.config,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
