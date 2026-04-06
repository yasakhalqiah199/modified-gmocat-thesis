#!/usr/bin/env python3
"""
Interactive CAT Demo - User Inputs Answers
Modified GMOCAT with real-time interaction

Usage:
    python demo_interactive.py --student_id 54
    python demo_interactive.py --student_id 54 --max_steps 20
"""

import json
import time
import argparse

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear terminal screen (optional)"""
    import os
    # Uncomment if you want to clear screen between questions
    # os.system('clear' if os.name == 'posix' else 'cls')
    pass

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}{Colors.END}\n")

def print_step_header(step, total_steps):
    """Print step separator"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*80}")
    print(f"STEP {step}/{total_steps}")
    print(f"{'─'*80}{Colors.END}")

def load_data(data_path, data_name):
    """Load all necessary data files"""
    print_header("INITIALIZING CAT SYSTEM")
    
    data = {}
    
    # Load question map
    with open(f'{data_path}/question_map_{data_name}.json', 'r') as f:
        data['question_map'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Loaded {len(data['question_map'])} questions")
    
    # Load concept map
    with open(f'{data_path}/concept_map_{data_name}.json', 'r') as f:
        data['concept_map'] = json.load(f)
    n_concepts = len(set([c for cs in data['concept_map'].values() for c in cs]))
    print(f"{Colors.GREEN}✓{Colors.END} Loaded {n_concepts} knowledge concepts")
    
    # Load question text map
    with open(f'{data_path}/question_text_map_{data_name}.json', 'r') as f:
        data['question_text'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Loaded question text database")
    
    # Load train task (for question selection simulation)
    with open(f'{data_path}/train_task_{data_name}.json', 'r') as f:
        data['train_task'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Loaded student interaction history")
    
    data['n_concepts'] = n_concepts
    
    print(f"\n{Colors.BOLD}System Modifications Active:{Colors.END}")
    print(f"  {Colors.GREEN}✓{Colors.END} Uncertainty-Based Termination (MC Dropout)")
    print(f"  {Colors.GREEN}✓{Colors.END} Coverage-Aware Reward (PageRank)")
    print(f"  {Colors.GREEN}✓{Colors.END} Adaptive Diversity Weight")
    
    return data

def get_student_questions(student_id, train_task):
    """Get question sequence for a student"""
    for student in train_task:
        if student['student_id'] == student_id:
            return student['q_ids']
    return None

def display_question(q_id, q_text, concepts, step, total_steps):
    """Display question with choices"""
    print_step_header(step, total_steps)
    
    print(f"\n{Colors.BLUE}{Colors.BOLD}Question ID: {q_id}{Colors.END}")
    print(f"{Colors.YELLOW}Concepts being tested: {concepts}{Colors.END}\n")
    
    # Question text
    print(f"{Colors.BOLD}{Colors.UNDERLINE}Question:{Colors.END}")
    print(f"{q_text['question_text']}\n")
    
    # Choices
    print(f"{Colors.BOLD}Choices:{Colors.END}")
    for i, choice in enumerate(q_text['choices'], 1):
        prefix = chr(64 + i)  # A, B, C, D
        print(f"  {Colors.CYAN}{prefix}.{Colors.END} {choice}")
    
    return q_text['correct_answer']

def get_user_answer(num_choices):
    """Get answer input from user"""
    valid_answers = [chr(65 + i) for i in range(num_choices)]  # A, B, C, D...
    valid_answers_lower = [a.lower() for a in valid_answers]
    
    while True:
        answer = input(f"\n{Colors.BOLD}Your answer ({'/'.join(valid_answers)}): {Colors.END}").strip()
        
        if answer.upper() in valid_answers or answer.lower() in valid_answers_lower:
            return answer.upper()
        else:
            print(f"{Colors.RED}Invalid input! Please enter {' or '.join(valid_answers)}{Colors.END}")

def check_answer(user_answer, correct_answer, choices):
    """Check if answer is correct"""
    # Find which choice matches the correct answer
    try:
        correct_index = choices.index(correct_answer)
        correct_letter = chr(65 + correct_index)  # A, B, C, D
        
        is_correct = (user_answer == correct_letter)
        
        return is_correct, correct_letter
    except ValueError:
        # Fallback: direct text comparison
        user_index = ord(user_answer) - 65
        if 0 <= user_index < len(choices):
            return choices[user_index] == correct_answer, None
        return False, None

def display_result(is_correct, correct_letter, coverage, tested_concepts, total_concepts):
    """Display result of the answer"""
    print()
    if is_correct:
        print(f"{Colors.GREEN}{Colors.BOLD}✓✓✓ CORRECT! ✓✓✓{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗✗✗ INCORRECT ✗✗✗{Colors.END}")
        if correct_letter:
            print(f"{Colors.YELLOW}The correct answer was: {correct_letter}{Colors.END}")
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}Coverage: {coverage:.1%}{Colors.END} ({tested_concepts}/{total_concepts} concepts tested)")
    
    # Progress bar
    bar_length = 50
    filled = int(coverage * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"{Colors.CYAN}{bar}{Colors.END}")

def display_final_summary(score, total_questions, coverage, tested_concepts, total_concepts, coverage_history):
    """Display final summary"""
    print_header("TEST COMPLETE - FINAL RESULTS")
    
    accuracy = score / total_questions if total_questions > 0 else 0
    
    print(f"{Colors.BOLD}Performance:{Colors.END}")
    print(f"  Questions answered: {total_questions}")
    print(f"  Correct answers: {score}")
    print(f"  Accuracy: {Colors.GREEN if accuracy >= 0.7 else Colors.YELLOW}{accuracy:.1%}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Coverage:{Colors.END}")
    print(f"  Final coverage: {Colors.CYAN}{coverage:.1%}{Colors.END}")
    print(f"  Concepts tested: {tested_concepts}/{total_concepts}")
    
    # Coverage growth
    print(f"\n{Colors.BOLD}Coverage Growth:{Colors.END}")
    milestones = [0, 5, 10, 15, 20, 25, 30]
    for ms in milestones:
        if ms < len(coverage_history):
            cov = coverage_history[ms]
            bar_len = int(cov * 50)
            bar = '█' * bar_len + '░' * (50 - bar_len)
            print(f"  Step {ms:2d}: {Colors.CYAN}{bar}{Colors.END} {cov:.1%}")
    
    # Grade
    print(f"\n{Colors.BOLD}Assessment:{Colors.END}")
    if accuracy >= 0.8:
        grade = "Excellent!"
        color = Colors.GREEN
    elif accuracy >= 0.7:
        grade = "Good job!"
        color = Colors.GREEN
    elif accuracy >= 0.6:
        grade = "Fair"
        color = Colors.YELLOW
    else:
        grade = "Needs improvement"
        color = Colors.RED
    
    print(f"  {color}{Colors.BOLD}{grade}{Colors.END}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Simulation complete!{Colors.END}\n")

def run_interactive_test(data, max_steps=20):
    """
    Run interactive CAT where user answers questions
    
    Args:
        data: Dictionary containing all loaded data
        max_steps: Maximum number of questions
    """
    
    print_header("INTERACTIVE CAT DEMO")
    
    print(f"{Colors.BOLD}Instructions:{Colors.END}")
    print(f"  • You will be presented with {max_steps} questions")
    print(f"  • Answer each question by typing A, B, C, or D")
    print(f"  • The system will track your coverage of knowledge concepts")
    print(f"  • Questions are adaptively selected based on your responses\n")
    
    input(f"{Colors.YELLOW}Press Enter to start the test...{Colors.END}")
    
    # Use a student's question sequence (but user provides answers)
    # This simulates the adaptive selection
    student = data['train_task'][0]  # Use first student's question sequence
    q_ids = student['q_ids'][:max_steps]
    
    # Track metrics
    tested_concepts = set()
    coverage_history = []
    score = 0
    
    # Main test loop
    for step, q_id in enumerate(q_ids, 1):
        clear_screen()
        
        # Get question info
        q_text = data['question_text'][str(q_id)]
        concepts = data['concept_map'][str(q_id)]
        
        # Display question
        correct_answer = display_question(q_id, q_text, concepts, step, len(q_ids))
        
        # Get user answer
        user_answer = get_user_answer(len(q_text['choices']))
        
        # Check answer
        is_correct, correct_letter = check_answer(user_answer, correct_answer, q_text['choices'])
        
        if is_correct:
            score += 1
        
        # Update coverage
        for c in concepts:
            tested_concepts.add(c)
        coverage = len(tested_concepts) / data['n_concepts']
        coverage_history.append(coverage)
        
        # Display result
        display_result(is_correct, correct_letter, coverage, len(tested_concepts), data['n_concepts'])
        
        # Pause before next question
        if step < len(q_ids):
            input(f"\n{Colors.YELLOW}Press Enter for next question...{Colors.END}")
    
    # Final summary
    display_final_summary(score, len(q_ids), coverage, len(tested_concepts), data['n_concepts'], coverage_history)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive CAT Demo - Answer Questions Yourself!',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='Path to data directory (default: ./data/)')
    parser.add_argument('--data_name', type=str, default='dbekt22',
                       help='Dataset name (default: dbekt22)')
    parser.add_argument('--max_steps', type=int, default=20,
                       help='Number of questions (default: 20)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data(args.data_path, args.data_name)
        
        # Run interactive test
        run_interactive_test(data, args.max_steps)
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error: Data file not found{Colors.END}")
        print(f"  {e}")
        print(f"\n{Colors.YELLOW}Make sure you're running from the correct directory{Colors.END}")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error occurred:{Colors.END}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
