import os
import csv
import glob

def merge_question_banks():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'Dataset')
    
    # Use glob to find all question_bank.csv files in the quiz directories
    file_pattern = os.path.join(dataset_dir, 'quiz*', 'question_bank.csv')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print("No question_bank.csv files found.")
        return
    
    # Sort files to ensure quiz1, quiz2, etc. are in order
    file_list.sort()
    
    header = None
    all_rows = []
    
    # Read all files
    for file in file_list:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if not rows:
                    continue
                    
                # Capture the header from the first file
                if header is None:
                    header = rows[0]
                
                # Append data rows (skipping the header)
                all_rows.extend(rows[1:])
                print(f"Loaded: {os.path.relpath(file, project_root)} ({len(rows)-1} questions)")
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    # Write to the merged file
    if all_rows:
        output_dir = os.path.join(dataset_dir, 'clean')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'question_bank.csv')
        
        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if header:
                    writer.writerow(header)
                writer.writerows(all_rows)
                
            print(f"\nSuccessfully merged {len(file_list)} files.")
            print(f"Total questions merged: {len(all_rows)}")
            print(f"Saved to: {os.path.relpath(output_file, project_root)}")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    merge_question_banks()
