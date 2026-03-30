"""
Topic Extraction & Population Script
Analyzes quiz questions and fills the topic column in interactions_clean.csv

Process:
1. Extract topics from question_bank.csv for all 5 quizzes
2. Create topic_mapping.csv for review
3. Join mapping into interactions_clean.csv to fill empty topic column
4. Validate results
"""

import pandas as pd
import os
from pathlib import Path

# ─────────────────────────────────────────────
# TOPIC KEYWORDS MAPPING
# ─────────────────────────────────────────────
TOPIC_KEYWORDS = {
    # Quiz 1: Data Fundamentals
    "Data Types": ["attribute", "discrete", "continuous", "nominal", "ordinal", "interval", "ratio", "binary", "social security", "movie ratings"],
    "Similarity Measures": ["cosine", "euclidean", "jaccard", "distance", "similarity", "measure", "correlation", "linear correlation"],
    "Scaling & Normalization": ["scaling", "normalize", "normalization", "z-score", "min-max", "range", "standard deviation", "mean of 0"],
    "Time Series Data": ["time series", "temporal", "daily"],
    "Programming Tools (NumPy/Pandas)": ["numpy", "pandas", "dataframe", "array", "fillna", "drop", "axis", "python list", "shape of a"],
    "Data Preprocessing": ["preprocessing", "missing values", "impute", "sampling", "discretization", "aggregate", "binning"],
    "Data Mining Tasks": ["data mining task", "mining activities", "dimensionality reduction"],
    
    # Quiz 2: Feature Engineering
    "Feature Selection": ["feature selection", "subset selection", "recursive feature elimination", "rfe", "forward selection", "relevance", "redundant"],
    "Feature Extraction": ["feature extraction"],
    "Principal Component Analysis": ["pca", "eigenvalue", "eigenvector", "orthogonal", "svd", "decomposition", "centered data", "principal component"],
    
    # Quiz 3: Clustering
    "K-means Clustering": ["k-means", "kmeans", "centroid", "elbow", "sum of squared errors", "sse", "squared error"],
    "Hierarchical Clustering": ["hierarchical", "agglomerative", "linkage", "dendrogram", "divisive"],
    "DBSCAN": ["dbscan", "density-based", "eps", "minpts", "density-reachable", "border point", "core point", "noise", "non-elliptical"],
    "Unsupervised Learning": ["unsupervised", "pattern", "relationship", "unlabeled"],
    "Clustering Evaluation": ["silhouette", "purity", "clustering evaluation", "clustering metrics"],
    
    # Quiz 4: Classification & Validation
    "Decision Trees": ["decision tree", "tree", "pruning", "splitting", "leaf", "gini", "entropy", "information gain"],
    "Classification": ["classification", "classify", "classifier", "binary outcome", "non-parametric", "parametric", "baseline", "unbalanced"],
    "Overfitting & Regularization": ["overfitting", "overfit", "underfitting", "regularization", "generalization", "unseen data", "learning curve"],
    "Cross-Validation": ["cross-validation", "cv", "train-test", "fold", "stratified", "loocv", "leave-one-out"],
    "Model Evaluation": ["evaluation", "confusion matrix", "false positive", "false negative", "precision", "recall", "accuracy", "f1-score", "roc", "auc"],
    
    # Quiz 5: Association Rules
    "Association Rules": ["association rules", "rule", "itemset", "transaction", "personalized", "recommendation", "historical"],
    "Apriori Algorithm": ["apriori", "join operation", "candidate generation"],
    "FP-growth": ["fp-growth", "fp-tree", "frequent pattern", "header table"],
    "Rule Metrics": ["lift", "confidence", "support", "interestingness", "minconf", "minsup"],
}
# ─────────────────────────────────────────────
# TOPIC EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def extract_topic(question_text: str) -> str:
    """
    Extract topic from question text using keyword matching.
    Returns the topic with highest keyword match count.
    """
    if not question_text:
        return "General"
    
    question_lower = question_text.lower()
    topic_scores = {}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword.lower() in question_lower)
        if score > 0:
            topic_scores[topic] = score
    
    if topic_scores:
        return max(topic_scores, key=topic_scores.get)
    else:
        return "General"

# ─────────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────────

def main():
    # Set base path
    base_path = Path(__file__).parent.parent / "Dataset"
    all_mappings = []
    
    print("=" * 70)
    print("STEP 1: EXTRACTING TOPICS FROM ALL QUIZZES")
    print("=" * 70)
    
    # Process each quiz
    for quiz_num in range(1, 6):
        quiz_path = base_path / f"quiz{quiz_num}"
        question_bank_file = quiz_path / "question_bank.csv"
        
        if not question_bank_file.exists():
            print(f"  Quiz {quiz_num} not found, skipping...")
            continue
        
        print(f"\n Processing Quiz {quiz_num}...")
        qb_df = pd.read_csv(question_bank_file)
        
        for idx, row in qb_df.iterrows():
            q_id = row["Question ID"]
            q_text = row["question_text"]
            topic = extract_topic(q_text)
            
            all_mappings.append({
                "Question ID": q_id,
                "question_text": q_text,
                "suggested_topic": topic,
                "quiz_num": quiz_num
            })
            
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1} questions")
        
        print(f"   Completed {len(qb_df)} questions for Quiz {quiz_num}")
    
    # Create mapping dataframe
    mapping_df = pd.DataFrame(all_mappings)
    mapping_file = base_path.parent / "topic_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False, encoding="utf-8")
    print(f"\n   Topic mapping created: {mapping_file}")
    print(f"   Total questions: {len(mapping_df)}")
    
    # Show topic distribution
    print("\n" + "=" * 70)
    print("TOPIC DISTRIBUTION")
    print("=" * 70)
    print(mapping_df["suggested_topic"].value_counts().to_string())
    
    # Show sample
    print(f"\n   SAMPLE MAPPINGS (first 10 rows):")
    print(mapping_df.head(10).to_string(index=False))
    
    return mapping_df

def join_topics_to_interactions(mapping_df):
    """
    Join topic mapping into interactions_clean.csv
    """
    base_path = Path(__file__).parent.parent / "Dataset"
    clean_path = base_path / "clean"
    interactions_file = clean_path / "interactions_clean.csv"
    
    print("\n" + "=" * 70)
    print("STEP 2: JOINING TOPICS INTO INTERACTIONS_CLEAN.CSV")
    print("=" * 70)
    
    if not interactions_file.exists():
        print(f" interactions_clean.csv not found at {interactions_file}")
        return
    
    print(f"\n Processing {interactions_file}...")
    
    # Read interactions_clean.csv
    interactions_df = pd.read_csv(interactions_file)
    initial_rows = len(interactions_df)
    
    # Drop existing topic column if it exists to ensure fresh update on reruns
    if "topic" in interactions_df.columns:
        interactions_df = interactions_df.drop("topic", axis=1)
    
    # Prepare mapping with only needed columns
    topic_mapping = mapping_df[["Question ID", "suggested_topic"]]
    
    # Join on Question ID
    interactions_df = interactions_df.merge(
        topic_mapping,
        on="Question ID",
        how="left"
    )
    
    # Rename suggested_topic to topic
    interactions_df = interactions_df.rename(columns={"suggested_topic": "topic"})
    
    # Save updated interactions_clean.csv
    interactions_df.to_csv(interactions_file, index=False, encoding="utf-8")
    
    null_count = interactions_df["topic"].isna().sum()
    print(f"   Rows: {initial_rows} → {len(interactions_df)}")
    print(f"   Topic column filled: {len(interactions_df) - null_count}/{len(interactions_df)}")
    
    if null_count > 0:
        print(f"    {null_count} null values remaining")
    
    # Show sample
    print(f"  Sample (first 5 rows):")
    print(interactions_df[["Question ID", "topic"]].head(5).to_string(index=False))
    
    print("\n interactions_clean.csv updated!")

def validate_results():
    """
    Validate that all topics are properly filled in interactions_clean.csv
    """
    base_path = Path(__file__).parent.parent / "Dataset"
    clean_path = base_path / "clean"
    interactions_file = clean_path / "interactions_clean.csv"
    
    print("\n" + "=" * 70)
    print("STEP 3: VALIDATION")
    print("=" * 70)
    
    if not interactions_file.exists():
        print(f" interactions_clean.csv not found at {interactions_file}")
        return
    
    df = pd.read_csv(interactions_file)
    total_rows = len(df)
    null_count = df["topic"].isna().sum()
    
    print(f"\ninteractions_clean.csv:")
    print(f"  Total rows: {total_rows}")
    print(f"  Topic nulls: {null_count}")
    print(f"  Completion: {((total_rows - null_count) / total_rows * 100):.1f}%")
    print(f"  Unique topics: {df['topic'].nunique()}")
    
    print(f"\n{'=' * 70}")
    print(f"OVERALL VALIDATION:")
    print(f"  Total rows: {total_rows}")
    print(f"  Filled topics: {total_rows - null_count}")
    print(f"  Null topics: {null_count}")
    print(f"  Overall completion: {((total_rows - null_count) / total_rows * 100):.1f}%")
    
    if null_count == 0:
        print(f"   ALL TOPICS SUCCESSFULLY FILLED!")
    else:
        print(f"    {null_count} rows still have null topics")
def print_qstn_topic_mapping(mapping_df):
    interactions_path = Path(__file__).parent.parent / "Dataset" / "clean" / "interactions_clean.csv"
    interactions_df = pd.read_csv(interactions_path)
    merged_df = interactions_df.merge(mapping_df[["Question ID", "suggested_topic"]], on="Question ID", how="left")
    # read all the questions from question_bank.csv files
    question_banks = []
    for quiz_num in range(1, 6):
        quiz_path = Path(__file__).parent.parent / "Dataset" / f"quiz{quiz_num}" / "question_bank.csv"
        if quiz_path.exists():
            qb_df = pd.read_csv(quiz_path)
            question_banks.append(qb_df)
    all_questions_df = pd.concat(question_banks, ignore_index=True)
    all_questions_df = all_questions_df.rename(columns={"Question ID": "Question ID", "question_text": "question_text"})
    final_df = merged_df.merge(all_questions_df[["Question ID", "question_text"]], on="Question ID", how="left") 
    #save into a csv file
    final_df.to_csv(Path(__file__).parent.parent / "Dataset" / "question_topic_mapping.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    # Step 1: Extract topics
    mapping_df = main()
    
    # Step 2: Join into interactions.csv
    join_topics_to_interactions(mapping_df)
    
    # Step 3: Validate
    validate_results()
    
    print("\n" + "=" * 70)
    print(" PROCESS COMPLETE!")
    print("=" * 70)
