#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse

def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data

def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def map_question_text(questions_csv, choices_csv, question_map_path, output_path):
    print(f"Loading question map from {question_map_path}...")
    question_map = open_json(question_map_path)

    print(f"Reading Questions CSV: {questions_csv}")
    question_df = pd.read_csv(questions_csv, encoding='ISO-8859-1', low_memory=False,
                     usecols=['id', 'question_text']).dropna()
    
    print(f"Reading Choices CSV: {choices_csv}")
    choices_df = pd.read_csv(choices_csv, encoding='ISO-8859-1', low_memory=False, usecols =['question_id','choice_text','is_correct']).dropna()
    
    question_text_map = {}
    print("Mapping questions...")
    
    # Pre-group choices by question_id for faster lookup
    choices_grouped = choices_df.groupby('question_id')

    for _, row in question_df.iterrows():
        original_question_id = str(row['id'])
        question_text = row['question_text']
        
        if original_question_id in question_map:
            mapped_question_id = question_map[original_question_id]

            # Get choices for this question
            try:
                choices = choices_grouped.get_group(int(original_question_id))
            except KeyError:
                choices = pd.DataFrame() # No choices found

            # Format choices and find the correct answer
            formatted_choices = []
            correct_answer = None
            for _, choice_row in choices.iterrows():
                choice_text = choice_row['choice_text']
                # is_correct might be boolean or 0/1
                is_correct = choice_row['is_correct']
                formatted_choices.append(choice_text)
                if is_correct:  # Check if the choice is the correct answer
                    correct_answer = choice_text

            # Add question with choices and correct answer to the map
            question_text_map[mapped_question_id] = {
                'question_text': question_text,
                'choices': formatted_choices,
                'correct_answer': correct_answer
            }

    # Simpan ke file JSON
    print(f"Saving to {output_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump_json(output_path, question_text_map)
    print('Number of Problems DBEKT22:', len(question_text_map))

if __name__ == "__main__":
    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=os.path.join(project_root, 'raw_data', 'DBEKT22', 'datasets', 'Questions.csv'), help="Path to Questions.csv")
    parser.add_argument("--choices", default=os.path.join(project_root, 'raw_data', 'DBEKT22', 'datasets', 'Question_Choices.csv'), help="Path to Question_Choices.csv")
    parser.add_argument("--map", default=os.path.join(project_root, 'data', 'question_map_dbekt22.json'), help="Path to question_map_dbekt22.json")
    parser.add_argument("--out", default=os.path.join(project_root, 'data', 'question_text_map_dbekt22.json'), help="Output JSON path")
    
    args = parser.parse_args()
    
    if os.path.exists(args.questions) and os.path.exists(args.choices) and os.path.exists(args.map):
        map_question_text(args.questions, args.choices, args.map, args.out)
    else:
        print("Error: One or more input files not found.")
        print(f"Questions: {args.questions} ({os.path.exists(args.questions)})")
        print(f"Choices: {args.choices} ({os.path.exists(args.choices)})")
        print(f"Map: {args.map} ({os.path.exists(args.map)})")