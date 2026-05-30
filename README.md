# Adaptive Exam Preparation System using Active Learning

> ENSIA — Machine Learning Project, Spring 2025/2026

Traditional exams ask every student the same fixed set of questions. This system does the opposite: it selects each next question based on what it has already learned about you, stopping as soon as it has enough information to accurately estimate your knowledge level. The result is fewer questions, same accuracy.

**Target:** reduce question count by while maintaining ≥ 90% level estimation accuracy.

---

## How it works

```
Student answers a question
        │
        ▼
Model updates its belief about student's knowledge level
        │
        ▼
Active learning strategy selects the next question:
  - Which question am I most uncertain about?
        │
        ▼
Stop when confidence threshold is reached
        │
        ▼
Output: estimated knowledge level
```

---

## Active Learning Strategy

The core selection strategy relies on **Uncertainty Sampling (Entropy-Based Selection)**. 
- The model prefers questions where its prediction probability is closest to 0.5. 
- Those are considered the most informative samples because they carry the highest uncertainty (entropy).


---

## Dataset

Built from **real Data Mining quizzes conducted at our school (ENSIA)**:

| Property | Value |
|---|---|
| Quizzes | 5 distinct quizzes |
| Features | Student history (response time, past accuracy), Question metadata (difficulty, complexity, word count) |
| Format | `student.csv` (responses) and `questions.csv` (metadata) |
| Difficulty levels | Easy / Medium / Hard |

The system uses features such as `accuracy_so_far`, `average_response_time_so_far`, and `last_n_correctness` together with NLP metrics from questions to estimate knowledge dynamically.

---

## Results

| Condition | Questions asked | Accuracy |
|---|---|---|
| Random baseline (full quiz) | 100% | ~85% |
| Uncertainty sampling | **50–70%** | **≥ 80%** |
| Entropy-based selection | **50–65%** | **≥ 80%** |

---

## Project Structure

```bash
Dataset/        # Contains questions.csv, student.csv, and sequential data
models/         # Saved joblib models and scalers (entropy_*)
notebooks/      # Jupyter notebooks for model experiments and feature engineering
  ├── model_with_emb.ipynb
  ├── model_without_emb.ipynb
  ├── question_feature_engineering.ipynb
  └── time_response_prediction.ipynb
scripts/        # python scripts for data preprocessing
  └── prepare_sequential_dataset.py
```

---

## Evaluation metrics

- Accuracy of knowledge level estimation
- Number of questions asked vs. baseline
- Accuracy / questions trade-off curve
- Confidence scores
- Learning curves
- Confusion matrix, Precision / Recall
- K-fold cross-validation
- Robustness to noisy answers

