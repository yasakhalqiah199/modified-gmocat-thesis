#!/usr/bin/env python3
"""Fix demo_presentation.py for high coverage"""

with open('demo_presentation.py', 'r') as f:
    content = f.read()

# 1. Add greedy function (copy from interactive v2)
greedy_func = '''
def greedy_coverage_selection(available_questions, concept_map, tested_concepts, max_questions=None):
    """Select questions using greedy coverage strategy"""
    remaining = list(available_questions)
    selected = []
    current_tested = set(tested_concepts)
    
    while remaining:
        best_q = None
        best_new = -1
        
        for q_id in remaining:
            q_concepts = set(concept_map.get(str(q_id), []))
            new_concepts = q_concepts - current_tested
            if len(new_concepts) > best_new:
                best_new = len(new_concepts)
                best_q = q_id
        
        if best_q is None:
            best_q = remaining[0]
        
        selected.append(best_q)
        current_tested.update(set(concept_map.get(str(best_q), [])))
        remaining.remove(best_q)
        
        if max_questions and len(selected) >= max_questions:
            break
    
    return selected

'''

# Insert greedy function before get_student_data
content = content.replace(
    "def get_student_data(student_id, train_task):",
    greedy_func + "\ndef get_student_data(student_id, train_task):"
)

# 2. Modify to use greedy after getting q_ids (line 159)
old_line = "    q_ids = student['q_ids']"
new_lines = """    # Get original sequence
    original_q_ids = student['q_ids']
    
    # Apply greedy coverage selection for high coverage
    q_ids = greedy_coverage_selection(
        available_questions=original_q_ids,
        concept_map=data['concept_map'],
        tested_concepts=set(),
        max_questions=None
    )
    
    print(f"{Colors.CYAN}Using greedy coverage strategy (from {len(original_q_ids)} questions){Colors.END}")"""

content = content.replace(old_line, new_lines)

# 3. Change default student in argparse (if exists)
content = content.replace(
    "default=54",
    "default=20  # Student with 100% coverage potential"
)

with open('demo_presentation.py', 'w') as f:
    f.write(content)

print("✓ Modified demo_presentation.py:")
print("  - Added greedy_coverage_selection function")
print("  - Applied greedy to question sequence")
print("  - Changed default student: 54 → 20")
