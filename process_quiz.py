"""
Quiz Data Processor
ENSIA Machine Learning | Spring 2025-2026

Takes:
  - Quiz_N_1_in_Data_Mining_-_Detailed_Answer_Key.json
  - Quiz_N_1_in_Data_Mining__Responses__anonymized.csv

Produces:
  - interactions.csv
  - question_bank.csv
"""

import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────

ANSWER_KEY_PATH  = "data/Quiz N°5 in Data Mining - Detailed Answer Key.json"
RESPONSES_PATH   = "data/Quiz N°5 in Data Mining (Responses)_anonymized.csv"
OUTPUT_DIR       = "."
QUESTION_ID_START = 121   # adjust per member assignment

# ─────────────────────────────────────────────
# 1. LOAD FILES
# ─────────────────────────────────────────────

with open(ANSWER_KEY_PATH, "r", encoding="utf-8") as f:
    answer_key = json.load(f)

responses_df = pd.read_csv(RESPONSES_PATH)

print(f"Loaded {len(answer_key)} questions from answer key")
print(f"Loaded {len(responses_df)} student responses")

# ─────────────────────────────────────────────
# 2. BUILD QUESTION BANK
# ─────────────────────────────────────────────

question_bank_rows = []

for i, q in enumerate(answer_key):
    problem_id    = QUESTION_ID_START + i
    question_text = q["question"].strip()

    # Format options as: A. option1 | B. option2 | ...
    option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_str = " | ".join(
        f"{option_letters[opt['index']]}. {opt['text']}"
        for opt in q["options"]
    )

    # Format correct answer as text of correct options joined by " | "
    correct_texts = [
        opt["text"]
        for opt in q["options"]
        if opt["isCorrect"]
    ]
    correct_answer = " | ".join(correct_texts)

    question_bank_rows.append({
        "Question ID":    problem_id,
        "question_text":  question_text,
        "options":        options_str,
        "correct_answer": correct_answer,
    })

question_bank_df = pd.DataFrame(question_bank_rows)

# ─────────────────────────────────────────────
# 3. BUILD LOOKUP: question_text → question metadata
# ─────────────────────────────────────────────

# Map question text (stripped) → (problem_id, correct_answer_set, question_type)
question_lookup = {}
for i, q in enumerate(answer_key):
    problem_id = QUESTION_ID_START + i
    correct_texts = set(
        opt["text"].strip()
        for opt in q["options"]
        if opt["isCorrect"]
    )
    question_lookup[q["question"].strip()] = {
        "problem_id":    problem_id,
        "correct_texts": correct_texts,
        "q_type":        q["type"],
    }

# ─────────────────────────────────────────────
# 4. GRADING FUNCTION
# ─────────────────────────────────────────────

def grade_answer(student_answer, correct_texts, q_type):
    """
    Grade a student's answer against the correct answer set.
    
    MULTIPLE_CHOICE: single answer text compared directly
    CHECKBOX: comma-separated answer split into set, must match exactly
    
    Returns "correct" or "incorrect"
    """
    if pd.isna(student_answer) or str(student_answer).strip() == "":
        return "incorrect"

    student_answer = str(student_answer).strip()

    if q_type == "MULTIPLE_CHOICE":
        student_set = {student_answer.strip()}
    else:
        # CHECKBOX: split by ", " to get individual selections
        student_set = {s.strip() for s in student_answer.split(",")}

    return "correct" if student_set == correct_texts else "incorrect"

# ─────────────────────────────────────────────
# 5. BUILD INTERACTIONS
# ─────────────────────────────────────────────

# Question columns = all columns except identifier, Timestamp, Score
question_columns = [c for c in responses_df.columns if c not in ["identifier", "Timestamp", "Score"]]

interaction_rows = []

for _, student_row in responses_df.iterrows():
    student_id = student_row["identifier"]

    for col in question_columns:
        col_stripped = col.strip()

        # Match column to question in lookup (strip both sides)
        matched = None
        for q_text, meta in question_lookup.items():
            if q_text == col_stripped:
                matched = meta
                break

        if matched is None:
            print(f"  WARNING: Could not match column: '{col_stripped}'")
            continue

        student_answer = student_row[col]
        correct_label  = grade_answer(
            student_answer,
            matched["correct_texts"],
            matched["q_type"]
        )

        interaction_rows.append({
            "Student ID":    student_id,
            "Question ID":   matched["problem_id"],
            "topic":         "",
            "correct":       correct_label,
            "Response Time": "",
        })

interactions_df = pd.DataFrame(interaction_rows)

# Enforce column order
interactions_df = interactions_df[[
    "Student ID", "Question ID", "topic", "correct", "Response Time"
]]

# ─────────────────────────────────────────────
# 6. SAVE FILES
# ─────────────────────────────────────────────

interactions_path  = os.path.join(OUTPUT_DIR, "interactions.csv")
question_bank_path = os.path.join(OUTPUT_DIR, "question_bank.csv")

interactions_df.to_csv(interactions_path,  index=False)
question_bank_df.to_csv(question_bank_path, index=False)

# ─────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────

print()
print("=" * 55)
print("  Processing Complete")
print("=" * 55)
print(f"  Students processed  : {len(responses_df)}")
print(f"  Questions processed : {len(answer_key)}")
print(f"  Total interactions  : {len(interactions_df)}")
print(f"  Question ID range   : {QUESTION_ID_START} – {QUESTION_ID_START + len(answer_key) - 1}")
print()
print(f"  Saved: {interactions_path}")
print(f"  Saved: {question_bank_path}")
print("=" * 55)

print("\nCorrect/Incorrect distribution:")
print(interactions_df["correct"].value_counts().to_string())

print("\nSample interactions (first 5 rows):")
print(interactions_df.head().to_string(index=False))

print("\nSample question bank (first 3 rows):")
print(question_bank_df[["Question ID", "question_text", "correct_answer"]].head(3).to_string(index=False))
