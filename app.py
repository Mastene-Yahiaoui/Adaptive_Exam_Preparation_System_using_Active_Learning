import streamlit as st
import pandas as pd
import os

##### NOTE ######
# 1. The features used in the datasets are temporary.
# They will be modified to match the features used
# to train the model.

# 2. Only the most recent answer of the student gets updated
# unlike his previous answers.

QUESTIONS_FILE = 'questions.csv'
ANSWERS_FILE = 'answers.csv'

def init_csv_files():
    """Initialize CSV files if they don't exist."""
    if not os.path.exists(QUESTIONS_FILE):
        df = pd.DataFrame(columns=[
            'questionID', 'topic', 'difficulty', 'answer', 'flesch_reading_ease'
        ])
        df.to_csv(QUESTIONS_FILE, index=False)
    
    if not os.path.exists(ANSWERS_FILE):
        df = pd.DataFrame(columns=[
            'studentID', 'questionID', 'topic', 'difficulty', 'flesch_reading_ease', 'is_correct',
            'student_avg_score', 'student_questions_answered', 'student_topic_performance',
            'student_difficulty_tolerance'
        ])
        df.to_csv(ANSWERS_FILE, index=False)

# --- API Endpoints as Functions ---

def get_all_questions():
    """Retrieve all questions."""
    return pd.read_csv(QUESTIONS_FILE)

def add_question(question_data):
    """Add a new question."""
    df = pd.read_csv(QUESTIONS_FILE)
    df = pd.concat([df, pd.DataFrame([question_data])], ignore_index=True)
    df.to_csv(QUESTIONS_FILE, index=False)
    return True

def get_all_answers():
    """Retrieve all answers."""
    return pd.read_csv(ANSWERS_FILE)

def submit_answer(answer_data):
    """Submit an answer for correct/incorrect evaluation.\n

    answer_data should be a dictionary with keys:
    - studentID
    - questionID
    - answer
    """
    df = pd.read_csv(QUESTIONS_FILE)
    answer = df[df['questionID'] == answer_data['questionID']]['answer'].values[0]
    is_correct = int(answer_data['answer'] == answer)
    add_answer(answer_data, is_correct)
    
    return True

def add_answer(answer_data, is_correct):
    """Add a new answer to the dataset `answers.csv` with updated student performance metrics. \n

    answer_data should be a dictionary with keys:
    - studentID
    - questionID
    - answer
    """
    df = pd.read_csv(ANSWERS_FILE)
    student_id = answer_data['studentID']
    student_answers = df[df['studentID'] == student_id]
    
    # Corrected metrics calculation
    student_questions_answered = len(student_answers) + 1
    student_avg_score = (student_answers['is_correct'].sum() + is_correct) / student_questions_answered
    
    topic, difficulty, flesch_reading_ease = get_question_details(answer_data['questionID'])
    
    topic_answers = student_answers[student_answers['topic'] == topic]
    student_topic_performance = (topic_answers['is_correct'].sum() + is_correct) / (len(topic_answers) + 1)
    
    diff_answers = student_answers[student_answers['difficulty'] == difficulty]
    student_difficulty_tolerance = (diff_answers['is_correct'].sum() + is_correct) / (len(diff_answers) + 1)

    new_answer = {
        'studentID': student_id,
        'questionID': answer_data['questionID'],
        'topic': topic,
        'difficulty': difficulty,
        'flesch_reading_ease': flesch_reading_ease,
        'is_correct': is_correct,
        'student_avg_score': student_avg_score,
        'student_questions_answered': student_questions_answered,
        'student_topic_performance': student_topic_performance,
        'student_difficulty_tolerance': student_difficulty_tolerance
    }
    df = pd.concat([df, pd.DataFrame([new_answer])], ignore_index=True)
    df.to_csv(ANSWERS_FILE, index=False)
    return True


def get_question_details(questionID):
    """Helper function to get all the features of a question based on questionID."""
    df = pd.read_csv(QUESTIONS_FILE)
    row = df[df['questionID'] == questionID]
    if row.empty:
        raise ValueError(f"Question ID {questionID} not found.")
    return row['topic'].values[0], row['difficulty'].values[0], row['flesch_reading_ease'].values[0]


####### REMAINIG FUNCTIONS ########
# - select_most_uncertain_question(studentID, method='entropy')
# - is_threshold_met(studentID, threshold=0.8)
# - calculate_student_score(studentID)
# maybe there are more functions to implement...


# --- Streamlit UI ---

def main():
    st.title("Adaptive Exam Preparation System")
    init_csv_files()
    
    st.write("Streamlit interactive UI and remaining logic goes here...")
    
    # Placeholder for specific streamlit components
    st.header("Questions")
    st.dataframe(get_all_questions())
    
    st.header("Answers")
    st.dataframe(get_all_answers())

if __name__ == "__main__":
    main()
