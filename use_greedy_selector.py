#!/usr/bin/env python3
"""Modify run_interactive_test to use greedy coverage selection"""

with open('demo_interactive_v2.py', 'r') as f:
    lines = f.readlines()

# Find the line where q_ids is assigned (around line 272-273)
# Replace with greedy selection call

modified = False
for i, line in enumerate(lines):
    # Look for: q_ids = student['q_ids']
    if "q_ids = student['q_ids']" in line and not modified:
        indent = ' ' * (len(line) - len(line.lstrip()))
        
        # Replace with greedy selection
        new_lines = [
            indent + "# Get available questions from student\n",
            indent + "available_q_ids = student['q_ids']\n",
            indent + "\n",
            indent + "# Use greedy coverage-based selection for high coverage\n",
            indent + "q_ids = greedy_coverage_selection(\n",
            indent + "    available_questions=available_q_ids,\n",
            indent + "    concept_map=data['concept_map'],\n",
            indent + "    tested_concepts=set(),  # Start with no tested concepts\n",
            indent + "    max_questions=None  # Use all available questions\n",
            indent + ")\n",
            indent + "\n",
            indent + f"print(f\"{{Colors.CYAN}}Using greedy coverage strategy (optimized from {{len(available_q_ids)}} questions){{Colors.END}}\")\n",
        ]
        
        # Replace the line
        lines[i:i+1] = new_lines
        modified = True
        print(f"✓ Modified question selection at line {i+1}")
        break

if not modified:
    print("❌ Could not find q_ids assignment line!")
    exit(1)

# Write back
with open('demo_interactive_v2.py', 'w') as f:
    f.writelines(lines)

print("=" * 60)
print("✓ Demo now uses greedy coverage-based selection")
print("=" * 60)
