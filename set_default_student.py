#!/usr/bin/env python3
"""Set default student to one with 100% coverage potential"""

with open('demo_interactive_v2.py', 'r') as f:
    content = f.read()

# Find and replace default student selection
# Look for: student = data['train_task'][0]
old_pattern = "student = data['train_task'][0]"
new_pattern = """# Use student with 100% coverage potential (has all 212 questions)
        # Options: 20, 25, 42, 55, 69, 79, 118, 132, 26, 135
        target_student_id = 20
        student = None
        for s in data['train_task']:
            if s['student_id'] == target_student_id:
                student = s
                break
        if not student:
            student = data['train_task'][0]  # Fallback"""

content = content.replace(old_pattern, new_pattern)

with open('demo_interactive_v2.py', 'w') as f:
    f.write(content)

print("=" * 60)
print("✓ Set default student to ID 20 (100% coverage)")
print("=" * 60)
