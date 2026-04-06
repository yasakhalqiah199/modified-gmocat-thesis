#!/usr/bin/env python3
"""Find student with highest coverage potential"""

import sys
sys.path.insert(0, '.')
from demo_interactive_v2 import load_data

data = load_data('./data', 'dbekt22')

best_students = []

for student in data['train_task']:
    student_id = student['student_id']
    q_ids = student['q_ids']
    
    # Calculate max possible coverage
    all_concepts = set()
    for q_id in q_ids:
        concepts = data['concept_map'].get(str(q_id), [])
        all_concepts.update(concepts)
    
    coverage = len(all_concepts) / data['n_concepts']
    
    best_students.append({
        'student_id': student_id,
        'n_questions': len(q_ids),
        'concepts_covered': len(all_concepts),
        'coverage': coverage
    })

# Sort by coverage
best_students.sort(key=lambda x: x['coverage'], reverse=True)

print("=" * 80)
print("TOP 10 STUDENTS BY MAXIMUM COVERAGE POTENTIAL")
print("=" * 80)
print(f"{'Rank':<6} {'Student ID':<12} {'Questions':<12} {'Concepts':<12} {'Coverage':<12}")
print("-" * 80)

for i, s in enumerate(best_students[:10], 1):
    print(f"{i:<6} {s['student_id']:<12} {s['n_questions']:<12} {s['concepts_covered']:<12} {s['coverage']*100:.1f}%")

print("=" * 80)
