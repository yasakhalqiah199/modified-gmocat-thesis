#!/usr/bin/env python3
"""Add greedy coverage-based question selection to demo_interactive_v2.py"""

with open('demo_interactive_v2.py', 'r') as f:
    content = f.read()

# Add greedy selector function after get_student_questions function
greedy_function = '''
def greedy_coverage_selection(available_questions, concept_map, tested_concepts, max_questions=None):
    """
    Select questions using greedy coverage strategy.
    Prioritize questions that cover the most NEW (untested) concepts.
    
    Args:
        available_questions: List of question IDs available
        concept_map: Dict mapping question_id -> list of concepts
        tested_concepts: Set of concepts already tested
        max_questions: Maximum number of questions to return (None = all)
    
    Returns:
        List of question IDs in optimal order for coverage
    """
    remaining_questions = list(available_questions)
    selected_sequence = []
    current_tested = set(tested_concepts)
    
    while remaining_questions:
        best_q = None
        best_new_concepts = -1
        
        # Find question that adds most new concepts
        for q_id in remaining_questions:
            q_concepts = set(concept_map.get(str(q_id), []))
            new_concepts = q_concepts - current_tested
            n_new = len(new_concepts)
            
            if n_new > best_new_concepts:
                best_new_concepts = n_new
                best_q = q_id
        
        # If no question adds new concepts, just pick first remaining
        if best_q is None:
            best_q = remaining_questions[0]
        
        # Add to sequence
        selected_sequence.append(best_q)
        
        # Update tested concepts
        q_concepts = set(concept_map.get(str(best_q), []))
        current_tested.update(q_concepts)
        
        # Remove from remaining
        remaining_questions.remove(best_q)
        
        # Check if we've reached max_questions
        if max_questions and len(selected_sequence) >= max_questions:
            break
    
    return selected_sequence

'''

# Insert after get_student_questions function (around line 97)
insert_marker = "def display_question(q_id, q_text, concepts, step, total_steps):"
content = content.replace(insert_marker, greedy_function + "\n" + insert_marker)

# Save modified file
with open('demo_interactive_v2.py', 'w') as f:
    f.write(content)

print("=" * 60)
print("✓ Added greedy_coverage_selection function")
print("=" * 60)
