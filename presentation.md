# Adaptive Exam Preparation System using Active Learning

## Slide 1. Title
**Adaptive Exam Preparation System using Active Learning**

A smart quiz platform for university professors that adapts questions to each student’s level, estimates performance in real time, and improves over time through active learning.


---

## Slide 2. The problem we wanted to solve
University professors often evaluate students with the same quiz, even though students do not share the same level.

That creates several issues:
- strong students may get bored or under-challenged
- weaker students may get discouraged too early
- fixed quizzes do not adapt to individual knowledge levels
- grading and evaluation can take unnecessary time

Our idea was to build a platform that adapts the quiz to the student instead of forcing every student through the exact same path.

**Core objective:** estimate how a student will answer the next question and use that prediction to choose the most informative next question.

---

## Slide 3. Why this project matters
This project is useful because it supports both teaching and assessment.

For professors:
- faster evaluation
- more informative assessment
- better understanding of student level
- personalized quiz paths

For students:
- a quiz that matches their level
- less frustration from questions that are too hard too early
- a more realistic evaluation experience

For the platform:
- every interaction becomes useful data
- the system gets better as more quizzes are used
- the model can be retrained with new real platform data

---

## Slide 4. High-level idea of the system
The system has two major parts:

1. **Quiz creation by the professor**
   - enters quiz questions
   - adds options
   - selects the correct answer
   - assigns a difficulty level

2. **Adaptive quiz execution**
   - student joins using a quiz code
   - system starts with an initial student state
   - model predicts the probability of correctness for the remaining questions
   - the next question is selected from the most uncertain predictions
   - the student answers
   - the student state is updated
   - the process repeats until the quiz becomes sufficiently certain

The key principle is **uncertainty sampling**.  
The model prefers questions where its prediction is closest to 0.5, because those are the most informative samples.

---

## Slide 5. Data source
The data initially came from **data mining quizzes done in our school**.

We started from:
- **5 quizzes**
- each quiz had **2 files**
  - one file for student responses
  - one file for question metadata in JSON format

### Responses file
This file contained columns like:
- `student_id`
- `score`
- `question1`, `question2`, ..., until the end of the quiz

Each question column contained the student answer.

### Question metadata file
This file stored, for each question:
- question text
- question type
- answer options
- correct answer index or indices

Example structure:
```json
{
  "question": "In the Apriori algorithm, what is the significance of the "pruning" step?",
  "type": "MULTIPLE_CHOICE",
  "options": [
    { "index": 0, "text": "It removes infrequent itemsets from consideration.", "isCorrect": true },
    { "index": 1, "text": "It adds more itemsets to the frequent itemsets list.", "isCorrect": false },
    { "index": 2, "text": "It increases the minimum support threshold.", "isCorrect": false },
    { "index": 3, "text": "It reorganizes the transaction database.", "isCorrect": false }
  ],
  "correctAnswers": [0]
}
```

---

## Slide 6. Raw dataset after integration
After collecting and aligning the quiz data, we built a unified interaction dataset.

### Global statistics
- **121 students**
- **150 questions**
- **17,340 student-question interactions**

Each row represents one student answering one question in one quiz session.

### Main raw columns included
- student id
- question id
- difficulty
- topic
- response time
- correctness label

### Important derived fields
- `difficulty` was computed using the number of correct answers from students
- `response time` was predicted using a separate model trained on a similar dataset containing response-time information

This gave us a richer table for modeling student behavior and question behavior together.

---

## Slide 7. Why this data design was useful
This data format allowed us to study both sides of adaptive testing:

### Student side
How strong is the student so far?
- accuracy so far
- accuracy on easy questions
- accuracy on medium questions
- accuracy on hard questions
- average response time so far
- recent correctness patterns

### Question side
How difficult and complex is the question?
- difficulty
- number of words
- complexity
- number of options
- average option length
- question embedding in the first version

This is important because adaptive learning is not only about the student or only about the question. It is about the interaction between both.

---

## Slide 8. Transformation 1. From raw quizzes to interaction rows
The first transformation was the conversion from quiz files into one row per student-question interaction.

Instead of keeping data separated by quiz files, we unified everything into a single structured dataset.

This step gave us:
- a single modeling table
- a consistent row format
- easier feature engineering
- easier train/test splitting
- easier active learning simulation

This was the foundation for everything that came after.

---

## Slide 9. Transformation 2. Building the student dataset
From the raw interaction data, we created a **student.csv** file with **17,340 rows**.

### Student-level fields
- `student_id`
- `quiz_id`
- `question_id`
- `question_in_quiz`
- `response_time`
- `accuracy_so_far`
- `number_of_questions_answered_so_far`
- `accuracy_easy_so_far`
- `accuracy_medium_so_far`
- `accuracy_hard_so_far`
- `average_response_time_so_far`
- `correct`

### Why this mattered
This dataset captured the evolving state of the student during the quiz.

A major design choice was that these features were computed using only **previous answers**, not the current answer.

That avoided **data leakage** and made the system realistic.

---

## Slide 10. Transformation 3. Building the question dataset
We also created a **question.csv** file with **150 rows**, one row per question.

### Question-level fields
- `question_id`
- `question_text`
- `correct_answer`
- `difficulty`
- `num_words`
- `qstn_complexity`
- `options_cleaned`
- `num_options`
- `avg_option_length`
- `embedding`

### Why this mattered
This dataset represented the intrinsic properties of each question.

It allowed us to describe questions in a machine-learning-friendly way, not just as raw text.

---

## Slide 11. Question feature engineering
We engineered several question-level features:

- **difficulty**  
  calculated from student answers

- **num_words**  
  number of words in the question text

- **qstn_complexity**  
  a computed complexity score

- **num_options**  
  number of possible answers

- **avg_option_length**  
  average length of answer options

- **embedding**  
  transformer-based text embedding of the question

These features helped the model understand the question beyond simple identifiers.

---

## Slide 12. Student feature engineering
We engineered a rich student state representation:

- **accuracy_so_far**
- **number_of_questions_answered_so_far**
- **accuracy_easy_so_far**
- **accuracy_medium_so_far**
- **accuracy_hard_so_far**
- **average_response_time_so_far**
- **last_1_correctness**
- **last_2_correctness**
- **last_3_correctness**

These features allowed the model to estimate how the student is performing in the current session.

The idea was to model not only correctness, but also momentum, recent performance, and response behavior.

---

## Slide 13. Final modeling table before training
After merging the student and question datasets, we created the final table used for model training.

### Final shape used by the model
- `session_id`
- `accuracy_so_far`
- `number_of_questions_answered_so_far`
- `accuracy_easy_so_far`
- `accuracy_medium_so_far`
- `accuracy_hard_so_far`
- `average_response_time_so_far`
- `correct`
- `difficulty`
- `num_words`
- `qstn_complexity`
- `num_options`
- `avg_option_length`

### Notes
- `session_id` was kept for group-based splitting
- the target label was `correct`
- the `difficulty` column was mapped into integers:
  - easy = 0
  - medium = 1
  - hard = 2

---

## Slide 14. What we removed and why
We removed several columns before training:

- `student_id`
- `question_id`
- `quiz_id`
- `question_text`
- `difficulty_x`
- `options_cleaned`
- `correct_answer`
- `topic`
- `embedding`
- `response_time`
- `last_1_correctness`
- `last_2_correctness`
- `last_3_correctness`
- `question_in_quiz`

### Why we removed them
Some were identifiers, some were redundant, and some were too dominant or risky.

The most important case was the **embedding feature**:
- in the first try, the model relied too heavily on embeddings
- embeddings reached about **97 percent feature importance**
- this reduced the influence of student state features
- the model started learning question text too strongly instead of adapting based on the student

So we removed embeddings to force the model to learn a more balanced representation.

---

## Slide 15. Train-test strategy
We used **GroupShuffleSplit** with `session_id`.

### Why group-based splitting?
This was important because interactions from the same student session should not appear in both train and test sets.

If we had split randomly at row level, the model could have seen nearly the same session in both training and testing, which would have inflated the results.

### Our split
- **70 percent train**
- **30 percent test**

This gave us a more realistic evaluation of generalization.

---

## Slide 16. Initial labeled and unlabeled pools
To simulate active learning, we did not train on the full training set at once.

Instead, we created:
- a **small labeled set**
- a **large unlabeled pool**

### Initial labeled ratio
We started with approximately:
- **10 percent labeled**
- **90 percent unlabeled**

This simulates a realistic situation where the model starts with limited annotated data and improves gradually by querying the most informative instances.

---

## Slide 17. Active learning simulation
We simulated active learning using **uncertainty sampling**.

### Idea
At each iteration:
1. train the model on the labeled set
2. predict probabilities on unlabeled samples
3. compute uncertainty using entropy
4. select the most uncertain samples
5. move them from unlabeled to labeled
6. retrain
7. repeat

### Why uncertainty sampling?
Because samples with probabilities near 0.5 are the most informative.  
The model is least confident there, so these are the best candidates to improve learning.

This is the same principle we later wanted to use in the real platform.

---

## Slide 18. The active learning loop in our project
The loop works like this:

- start from a small labeled seed set
- train a classifier
- measure uncertainty on the remaining data
- select the most uncertain samples
- add them to the labeled set
- retrain the model
- repeat until improvement stabilizes

We used a stopping condition based on metric plateau.

### Stopping logic
If LogLoss did not improve enough for several iterations, we stopped the loop.

This prevented unnecessary retraining once the model became stable.

---

## Slide 19. Models we compared
We compared several classifiers:

- **XGBoost**
- **Random Forest**
- **Logistic Regression**
- **Naive Bayes**
- **Gradient Boosting**

### Why compare several models?
We wanted to see which model handled:
- tabular features
- nonlinear relationships
- mixed feature types
- unscaled data

best in the context of our adaptive quiz problem.

---

## Slide 20. Why XGBoost was the best choice
We selected **XGBoost** as the final model because it performed best overall.

### Why it fit the problem
- handles nonlinear relationships well
- works well on structured tabular data
- does not require feature scaling
- robust with mixed feature importance patterns
- strong performance on probability-based binary classification

### In our results
XGBoost achieved the best LogLoss during active learning and the best final evaluation metrics.

---

## Slide 21. Model comparison results
### Best LogLoss during active learning
| Model | Best LogLoss | Final LogLoss | Final Labeled Ratio |
|---|---:|---:|---:|
| XGBoost | 0.3725 | 0.3726 | 48.77% |
| Random Forest | 0.3791 | 0.3791 | 43.25% |
| Gradient Boosting | 0.3808 | 0.3808 | 40.27% |
| Logistic Regression | 0.4016 | 0.4025 | 30.35% |
| Naive Bayes | 0.4219 | 0.4509 | 46.08% |

### Interpretation
XGBoost gave the best combination of:
- probability quality
- ranking quality
- stable convergence
- practical performance with limited labeled data

---

## Slide 22. Final evaluation metrics
We evaluated the final model on the test set.

### Final test results
- **Log Loss:** 0.3726321252395159
- **AUC:** 0.8657233802161338
- **Brier Score:** 0.11465199760822176

### What these mean
- **Log Loss** shows strong probabilistic prediction quality
- **AUC** shows good separation between correct and incorrect answers
- **Brier Score** shows decent probability calibration

Overall, the model performed well as a correctness predictor for adaptive quiz selection.

---

## Slide 23. Learning efficiency result
One of the most important findings was that we did not need the full labeled training set to reach a stable performance level.

We reached strong performance using only about:
- **48 percent of the available training data**

### Why this matters
This is a strong active learning result because it means:
- the model can learn efficiently from a smaller labeled subset
- the system saves labeling effort
- uncertainty sampling focuses the annotation process on the most useful examples

This is exactly what makes the approach scalable.

---

## Slide 24. Feature importance insight
The feature importance analysis gave us an important lesson.

### What happened
When embeddings were included, the model relied heavily on them, with around:
- **97 percent feature importance**

### Why this was a problem
That meant the model was not paying enough attention to:
- student accuracy history
- current session behavior
- response-time related signals

In other words, the model was becoming too text-driven and not adaptive enough.

### Our solution
We removed embeddings from the final training set to make the model focus on student state and structured quiz features.

---

## Slide 25. How the adaptive quiz works at runtime
During real use, the platform works like this:

1. The professor creates the quiz and enters the correct answers and difficulty labels.
2. The professor gives the quiz code to students.
3. A student enters the quiz with an initial student state.
4. The model predicts the probability of correctness for all remaining questions.
5. The question with the highest uncertainty is selected.
6. The student answers it.
7. The student state is updated.
8. The model predicts again using the new state.
9. The next most uncertain question is selected.
10. The process continues until the model becomes confident enough.

### Confidence threshold
We stop when predicted probabilities are:
- lower than **0.3**
- or higher than **0.7**

This means the model is no longer uncertain enough to justify continuing adaptation.

---

## Slide 26. Scoring mechanism
The final score combines:
- questions actually solved by the student
- questions predicted by the model when the student did not answer them

### Score formula
\[
	ext{Score} = rac{w_1 \cdot (	ext{number of correct answers}) + w_2 \cdot (	ext{number of predicted correct answers})}{	ext{total number of questions}}
\]

### Why weights were used
We gave:
- more importance to questions actually solved
- less importance to questions that were not solved and had to be estimated

This makes the score more realistic and fair.

---

## Slide 27. Data collection strategy in the platform
The platform is not only for testing. It is also a data collection engine.

### What we collect
From many professors and many students, we collect:
- student states
- answers
- predicted probabilities
- uncertainty cases
- quiz metadata
- real outcomes

### Why uncertainty data is valuable
The system intentionally collects the cases where the model was uncertain, because those are the most informative examples for improving the model.

That is the core logic of active learning.

---

## Slide 28. Future retraining plan
Our future plan is to retrain the model periodically.

### Retraining strategy
- collect new data from the platform
- add it to the training dataset when a threshold `k` is reached
- retrain the model on the expanded dataset
- redeploy the updated model

### Why periodic retraining matters
Because the platform will evolve:
- new students
- new quizzes
- new topics
- new difficulty distributions

Periodic retraining keeps the model aligned with real usage and improves performance over time.

---

## Slide 29. Limitations
We also identified some important limitations.

### 1. Small initial dataset
Our first dataset was not very large, so the model had limited starting information.

### 2. Difficulty estimation is imperfect
Question difficulty was derived from student answers, which is not the same as a teacher-defined pedagogical difficulty.

### 3. Response time is predicted
Since response time was not directly available in the same shape, it was predicted from another model rather than fully observed.

### 4. Embeddings were too dominant in the first version
The embedding feature caused the model to focus too much on text semantics and not enough on student behavior.

These limitations are useful because they show what we would improve in the next version.

---

## Slide 30. What we learned
This project taught us several important lessons:

- adaptive testing can be built with real ML pipelines
- student state features are very important
- question features are also important, but they must not dominate everything
- active learning helps reduce labeling needs
- uncertainty sampling is a practical way to improve a quiz model
- evaluation must avoid leakage and session overlap
- model choice matters, especially for tabular adaptive systems

---

## Slide 31. Final conclusion
We built an **Adaptive Exam Preparation System using Active Learning** that:
- helps professors evaluate students more intelligently
- adapts quiz flow to student level
- uses uncertainty to choose the most informative next question
- learns efficiently from limited labeled data
- improves as the platform collects more real data

### Final result
The best model was **XGBoost**, and the system reached strong performance with only about **48 percent** of the data needed for stable metrics.

This makes the project a solid foundation for a real adaptive learning platform.

---

## Slide 32. Closing
Thank you.

**Adaptive Exam Preparation System using Active Learning**  
A platform that makes quizzes smarter, faster, and more adaptive over time.

---
