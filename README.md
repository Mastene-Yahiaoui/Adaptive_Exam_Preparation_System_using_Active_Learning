# Adaptive Exam Preparation System using Active Learning

> ENSIA — Machine Learning Project, Spring 2025/2026

Traditional exams ask every student the same fixed set of questions. This system does the opposite: it selects each next question based on what it has already learned about you, stopping as soon as it has enough information to accurately estimate your knowledge level. The result is fewer questions, same accuracy.

**Target:** reduce question count by 30–50% while maintaining ≥ 80% level estimation accuracy.

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
  - Which answer would reduce my uncertainty the most?
        │
        ▼
Stop when confidence threshold is reached
        │
        ▼
Output: estimated knowledge level + confidence score
```

---

## Active learning strategies implemented

| Strategy | Description |
|---|---|
| **Uncertainty Sampling** | Ask the question the model is least sure about |
| **Entropy-Based Selection** | Pick the question with highest prediction entropy |
| **Query-by-Committee** | Multiple models vote; ask where they disagree most |
| **Random Baseline** | Fixed-length quiz, random order (for comparison) |

---

## Models for knowledge estimation

- Logistic Regression
- Naive Bayes
- Decision Tree
- *(extensible — any sklearn-compatible classifier)*

---

## Dataset

Built on the **ASSISTments** and **EdNet** educational datasets:

| Property | Value |
|---|---|
| Students | ≥ 100 |
| Questions | 100–200 |
| Topics | 5–10 |
| Features | student ID, question ID, difficulty, topic, correct/incorrect, response time |
| Difficulty levels | Easy / Medium / Hard (balanced) |

---

## Results

| Condition | Questions asked | Accuracy |
|---|---|---|
| Random baseline (full quiz) | 100% | ~85% |
| Uncertainty sampling | **50–70%** | **≥ 80%** |
| Entropy-based selection | **50–65%** | **≥ 80%** |

---

## Setup

```bash
git clone https://github.com/nacermissouni23/adaptive-exam-system
cd adaptive-exam-system
pip install -r requirements.txt
```

```bash
# Run adaptive exam simulation
python run_exam.py --strategy entropy --dataset data/assistments.csv

# Compare all strategies
python evaluate.py --all-strategies --plot

# Interactive demo
python demo.py
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

---

## Stack

```
scikit-learn    classifiers, cross-validation
pandas / numpy  data handling, entropy computation
matplotlib      learning curves, confusion matrices
```

---

## Deliverables

- [x] Labeled dataset (ASSISTments + EdNet)
- [x] Documented Jupyter notebook
- [x] Trained adaptive models (all three strategies)
- [x] Evaluation section with metrics and visualizations
- [x] Live demo: real-time adaptive question selection
- [ ] Final presentation
