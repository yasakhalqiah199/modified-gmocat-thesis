#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from multiprocessing import Pool
import argparse
from collections import defaultdict

# Global variables for multiprocessing
question_map = {}
df = None

def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data

def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def f_dbekt22(uuid):
    global df, question_map
    # Filter dataframe for the specific student
    user_df = df[df.student_id == uuid]
    
    # Note: sorting by 'student_id' on a filtered dataframe (where student_id is constant) 
    # doesn't change the order. We rely on the original CSV order (usually chronological).
    # If there is a timestamp column, it would be better to sort by that.
    
    q_ids, labels = [], []
    q_ids_set = set()
    
    for _, row in user_df.iterrows():
        q_id = str(row['question_id'])
        if q_id in q_ids_set:
            continue
        q_ids_set.add(q_id)
        
        if q_id in question_map:
            q_ids.append(question_map[q_id])
            
            # Robust answer parsing (handles int, float, string)
            val = row['answer_state']
            try:
                ans = 1 if int(float(val)) == 1 else 0
            except (ValueError, TypeError):
                ans = 0
            labels.append(ans)
            
    out = {'student_id': int(uuid), 'q_ids': q_ids, 'labels': labels, 'log_num':len(labels)}
    return out

def featurize_dbekt22(csv_path, kc_path, output_dir):
    global question_map, df
    
    print(f"Reading Transaction CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', low_memory=False,
                     usecols=['student_id', 'question_id', 'answer_state']).dropna()

    print(f"Reading KC CSV: {kc_path}")
    q2k_df = pd.read_csv(kc_path, encoding='ISO-8859-1', low_memory=False).dropna()

    print("Data Types:")
    print(df.dtypes)
    print(q2k_df.dtypes)

    user_ids = df['student_id'].unique()
    problems = df['question_id'].unique()

    # Build Question Map
    question_map = {}
    for p in problems:
        question_map[str(p)] = len(question_map)
        
    print(f"Processing {len(user_ids)} users with multiprocessing...")
    with Pool(30) as p:
        results = p.map(f_dbekt22, user_ids)

    # Filter short sequences
    bad_interactions = [len(d['q_ids']) for d in results if len(d['q_ids']) < 40]
    results = [d for d in results if len(d['q_ids']) >= 40]
    interactions = [len(d['q_ids']) for d in results]

    # Build Concept Map
    table = q2k_df.loc[:, ['question_id', 'knowledgecomponent_id']].drop_duplicates()
    q2k = {}
    k2n = {}
    for _, row in table.iterrows():
        qid_str = str(row['question_id'])
        if qid_str in question_map:
            qid = str(question_map[qid_str])
            # Format KC ID as string of list to match existing convention if needed
            kid = str([row['knowledgecomponent_id']])
            
            if kid not in k2n:
                k2n[kid] = len(k2n)

            if qid not in q2k:
                q2k[qid] = []

            if k2n[kid] not in q2k[qid]:
                q2k[qid].append(k2n[kid])

    print('Number of DBEKT22 User: ', len(results))
    print('Number of DBEKT22 Interactions: ', sum(interactions))
    print('Ignored DBEKT22 Interactions: ', sum(bad_interactions))
    print('Number of Problems DBEKT22:', len(question_map))
    print('Number of Knowledge DBEKT22:', len(k2n))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    dump_json(os.path.join(output_dir, 'train_task_dbekt22.json'), results)
    dump_json(os.path.join(output_dir, 'question_map_dbekt22.json'), question_map)
    dump_json(os.path.join(output_dir, 'concept_map_dbekt22.json'), q2k)

if __name__ == "__main__":
    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join(project_root, 'raw_data', 'DBEKT22', 'datasets', 'Transaction.csv'), help="Path to Transaction.csv")
    parser.add_argument("--kc", default=os.path.join(project_root, 'raw_data', 'DBEKT22', 'datasets', 'Question_KC_Relationships.csv'), help="Path to Question_KC_Relationships.csv")
    parser.add_argument("--out", default=os.path.join(project_root, 'data'), help="Output directory")
    args = parser.parse_args()
    
    if os.path.exists(args.csv) and os.path.exists(args.kc):
        featurize_dbekt22(args.csv, args.kc, args.out)
    else:
        print(f"Error: Input files not found.\nCSV: {args.csv}\nKC: {args.kc}")
        print("Please check paths or provide arguments via --csv and --kc")
