#!/usr/bin/env python3
"""Debug: Check if greedy is being called"""

import sys
sys.path.insert(0, '.')
from demo_interactive_v2 import load_data, greedy_coverage_selection

data = load_data('./data', 'dbekt22')

# Get student 20
student = None
for s in data['train_task']:
    if s['student_id'] == 20:
        student = s
        break

print("\n" + "=" * 60)
print("DEBUG: Checking first 10 questions")
print("=" * 60)

available_q_ids = student['q_ids']
greedy_q_ids = greedy_coverage_selection(
    available_questions=available_q_ids,
    concept_map=data['concept_map'],
    tested_concepts=set()
)

print(f"\nOriginal first 10 q_ids: {available_q_ids[:10]}")
print(f"Greedy first 10 q_ids:   {greedy_q_ids[:10]}")
print(f"\nAre they different? {available_q_ids[:10] != greedy_q_ids[:10]}")

# Check concepts for first 3 questions
print("\n" + "=" * 60)
print("Concepts for first 3 questions:")
print("=" * 60)

tested = set()
for i in range(3):
    q_id = greedy_q_ids[i]
    concepts = set(data['concept_map'].get(str(q_id), []))
    new_concepts = concepts - tested
    tested.update(concepts)
    print(f"Q{i+1} (ID {q_id}): {len(new_concepts)} new concepts (total: {len(tested)})")
    
print(f"\nExpected coverage after 3 questions: {len(tested)}/93 = {len(tested)/93*100:.1f}%")
