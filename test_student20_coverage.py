#!/usr/bin/env python3
"""Test greedy coverage with student 20"""

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

if student:
    q_ids = student['q_ids']
    
    # Greedy selection
    greedy_q_ids = greedy_coverage_selection(
        available_questions=q_ids,
        concept_map=data['concept_map'],
        tested_concepts=set(),
        max_questions=None
    )
    
    # Track coverage progression
    tested_concepts = set()
    milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("=" * 70)
    print("GREEDY COVERAGE PROGRESSION - Student 20")
    print("=" * 70)
    print(f"{'Questions':<12} {'Concepts':<12} {'Coverage':<12} {'Milestone'}")
    print("-" * 70)
    
    for i, q_id in enumerate(greedy_q_ids, 1):
        concepts = set(data['concept_map'].get(str(q_id), []))
        tested_concepts.update(concepts)
        
        coverage = len(tested_concepts) / data['n_concepts']
        
        if i in milestones or coverage == 1.0:
            milestone = ""
            if coverage >= 0.75:
                milestone = "← 75% KKM reached!" if coverage >= 0.75 and len(tested_concepts) - len(set(data['concept_map'].get(str(greedy_q_ids[i-2] if i > 1 else q_id), []))) < 70 else ""
            if coverage == 1.0:
                milestone = "🎯 100% COVERAGE!"
                
            print(f"{i:<12} {len(tested_concepts):<12} {coverage*100:>6.1f}%      {milestone}")
        
        if coverage == 1.0:
            print("=" * 70)
            print(f"✅ Reached 100% coverage in {i} questions!")
            print(f"   (vs {len(q_ids)} total available)")
            print(f"   Efficiency: {(1 - i/len(q_ids))*100:.1f}% fewer questions needed")
            print("=" * 70)
            break
