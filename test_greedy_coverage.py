#!/usr/bin/env python3
"""Quick test to see coverage improvement"""

import sys
sys.path.insert(0, '.')

from demo_interactive_v2 import load_data, greedy_coverage_selection

# Load data
data = load_data('./data', 'dbekt22')

# Get student 54
student = None
for s in data['train_task']:
    if s['student_id'] == 54:
        student = s
        break

if student:
    original_q_ids = student['q_ids']
    
    # Original sequence coverage
    original_concepts = set()
    for q_id in original_q_ids:
        concepts = data['concept_map'].get(str(q_id), [])
        original_concepts.update(concepts)
    
    # Greedy sequence coverage
    greedy_q_ids = greedy_coverage_selection(
        available_questions=original_q_ids,
        concept_map=data['concept_map'],
        tested_concepts=set(),
        max_questions=None
    )
    
    greedy_concepts = set()
    for q_id in greedy_q_ids:
        concepts = data['concept_map'].get(str(q_id), [])
        greedy_concepts.update(concepts)
    
    print("=" * 60)
    print("COVERAGE COMPARISON - Student 54")
    print("=" * 60)
    print(f"Total questions available: {len(original_q_ids)}")
    print(f"Total concepts in dataset: {data['n_concepts']}")
    print("")
    print(f"ORIGINAL sequence:")
    print(f"  Concepts covered: {len(original_concepts)}/{data['n_concepts']}")
    print(f"  Coverage: {len(original_concepts)/data['n_concepts']*100:.1f}%")
    print("")
    print(f"GREEDY sequence:")
    print(f"  Concepts covered: {len(greedy_concepts)}/{data['n_concepts']}")
    print(f"  Coverage: {len(greedy_concepts)/data['n_concepts']*100:.1f}%")
    print("")
    print(f"IMPROVEMENT: +{len(greedy_concepts) - len(original_concepts)} concepts")
    print("=" * 60)
