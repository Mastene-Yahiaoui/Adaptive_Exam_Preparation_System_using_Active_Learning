from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Dataset" / "ready.csv"
OUTPUT_PATH = ROOT / "Dataset" / "processed" / "sequential_student_performance.csv"
QUIZ_LENGTH = 30
DIFFICULTIES = ("easy", "medium", "hard")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        key = column.strip().lower().replace(" ", "_")
        rename_map[column] = key
    return frame.rename(columns=rename_map)


def reconstruct_sequential_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_columns(raw).copy()

    required_columns = {"student_id", "question_id", "difficulty", "correct", "response_time", "topic"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.reset_index(drop=True)
    frame["raw_row_index"] = frame.index

    frame["difficulty"] = frame["difficulty"].astype(str).str.strip().str.lower()
    frame["topic"] = frame["topic"].astype(str).str.strip()
    frame["correct"] = frame["correct"].astype(int)
    frame["question_id"] = frame["question_id"].astype(int)
    frame["response_time"] = pd.to_numeric(frame["response_time"], errors="coerce")

    frame["quiz_id"] = ((frame["question_id"] - 1) // QUIZ_LENGTH) + 1
    frame["question_in_quiz"] = ((frame["question_id"] - 1) % QUIZ_LENGTH) + 1
    frame["session_id"] = frame["student_id"].astype(str) + "_quiz_" + frame["quiz_id"].astype(str)

    group_columns = ["student_id", "quiz_id"]
    rows: list[dict[str, object]] = []

    for (student_id, quiz_id), group in frame.sort_values(group_columns + ["question_in_quiz", "raw_row_index"]).groupby(group_columns, sort=False):
        group = group.sort_values(["question_in_quiz", "raw_row_index"]).reset_index(drop=True)

        total_answered = 0
        total_correct = 0
        total_response_time = 0.0

        difficulty_counts = {difficulty: 0 for difficulty in DIFFICULTIES}
        difficulty_correct = {difficulty: 0 for difficulty in DIFFICULTIES}

        last_correctness: list[int] = []

        for _, row in group.iterrows():
            current_difficulty = row["difficulty"]
            current_response_time = float(row["response_time"]) if pd.notna(row["response_time"]) else 0.0

            difficulty_answered = difficulty_counts.get(current_difficulty, 0)
            difficulty_correct_count = difficulty_correct.get(current_difficulty, 0)

            features = {
                "student_id": student_id,
                "quiz_id": int(quiz_id),
                "session_id": row["session_id"],
                "question_id": int(row["question_id"]),
                "question_in_quiz": int(row["question_in_quiz"]),
                "difficulty": current_difficulty,
                "topic": row["topic"],
                "response_time": current_response_time,
                "accuracy_so_far": (total_correct / total_answered) if total_answered else 0.0,
                "number_of_questions_answered_so_far": int(total_answered),
                "accuracy_easy_so_far": (difficulty_correct["easy"] / difficulty_counts["easy"]) if difficulty_counts["easy"] else 0.0,
                "accuracy_medium_so_far": (difficulty_correct["medium"] / difficulty_counts["medium"]) if difficulty_counts["medium"] else 0.0,
                "accuracy_hard_so_far": (difficulty_correct["hard"] / difficulty_counts["hard"]) if difficulty_counts["hard"] else 0.0,
                "average_response_time_so_far": (total_response_time / total_answered) if total_answered else 0.0,
                "last_1_correctness": last_correctness[-1] if len(last_correctness) >= 1 else 0,
                "last_2_correctness": last_correctness[-2] if len(last_correctness) >= 2 else 0,
                "last_3_correctness": last_correctness[-3] if len(last_correctness) >= 3 else 0,
                "correct": int(row["correct"]),
                "raw_row_index": int(row["raw_row_index"]),
            }
            rows.append(features)

            correct_value = int(row["correct"])
            total_answered += 1
            total_correct += correct_value
            total_response_time += current_response_time
            difficulty_counts[current_difficulty] = difficulty_answered + 1
            difficulty_correct[current_difficulty] = difficulty_correct_count + correct_value
            last_correctness.append(correct_value)

    result = pd.DataFrame(rows)
    result = result.sort_values(["student_id", "quiz_id", "question_in_quiz", "raw_row_index"]).reset_index(drop=True)

    ordered_columns = [
        "student_id",
        "quiz_id",
        "session_id",
        "question_id",
        "question_in_quiz",
        "difficulty",
        "topic",
        "response_time",
        "accuracy_so_far",
        "number_of_questions_answered_so_far",
        "accuracy_easy_so_far",
        "accuracy_medium_so_far",
        "accuracy_hard_so_far",
        "average_response_time_so_far",
        "last_1_correctness",
        "last_2_correctness",
        "last_3_correctness",
        "correct",
        "raw_row_index",
    ]
    return result[ordered_columns]


def main() -> None:
    raw = pd.read_csv(INPUT_PATH)
    cleaned = reconstruct_sequential_dataset(raw)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    # Lightweight validation that session resets are respected.
    first_rows = cleaned.groupby("session_id", sort=False).head(1)
    if not (first_rows[["accuracy_so_far", "number_of_questions_answered_so_far", "average_response_time_so_far"]] == 0).all().all():
        raise AssertionError("Session-level history did not reset cleanly at the first interaction of a session.")

    print(f"Wrote {len(cleaned):,} rows to {OUTPUT_PATH}")
    print(cleaned.head(5).to_string(index=False))


if __name__ == "__main__":
    main()