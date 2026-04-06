#!/usr/bin/env python3
"""Apply greedy to BOTH paths (with student_id and without)"""

with open('demo_interactive_v2.py', 'r') as f:
    lines = f.readlines()

# Find line: q_ids, student = get_student_questions(student_id, data['train_task'])
# Replace the next block

modified = False
for i, line in enumerate(lines):
    if "q_ids, student = get_student_questions(student_id, data['train_task'])" in line:
        # Find the "if not q_ids:" block and modify
        indent = ' ' * (len(line) - len(line.lstrip()))
        
        # Replace the block
        new_block = [
            line,  # Keep original line
            lines[i+1],  # if not q_ids:
            lines[i+2],  # print error
            lines[i+3],  # return
            indent + "\n",
            indent + "# Apply greedy coverage selection\n",
            indent + "available_q_ids = q_ids\n",
            indent + "q_ids = greedy_coverage_selection(\n",
            indent + "    available_questions=available_q_ids,\n",
            indent + "    concept_map=data['concept_map'],\n",
            indent + "    tested_concepts=set(),\n",
            indent + "    max_questions=None\n",
            indent + ")\n",
            indent + f"print(f\"{{Colors.CYAN}}Using greedy coverage strategy for student {{student_id}}{{Colors.END}}\")\n",
        ]
        
        # Replace 4 lines with new block
        lines[i:i+4] = new_block
        modified = True
        print(f"✓ Applied greedy to student_id path at line {i+1}")
        break

if not modified:
    print("❌ Could not find student_id path!")
    exit(1)

with open('demo_interactive_v2.py', 'w') as f:
    f.writelines(lines)

print("=" * 60)
print("✓ Greedy now applied to BOTH paths")
print("=" * 60)
