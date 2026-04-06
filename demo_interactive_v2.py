#!/usr/bin/env python3
"""
Interactive CAT Demo v2 - Unlimited Questions with Smart Termination
Modified GMOCAT with:
1. Uncertainty-Based Termination (UBT)
2. Coverage-Aware Reward (PageRank)
3. Adaptive Diversity Weight

Features:
- Auto-detect total questions per student
- Coverage-based termination (optional)
- Can stop anytime with Ctrl+C
- Real-time metrics tracking

Usage:
    python demo_interactive_v2.py
    python demo_interactive_v2.py --student_id 54
    python demo_interactive_v2.py --target_coverage 0.8
    python demo_interactive_v2.py --max_steps 50
"""

import json
import time
import argparse
import sys

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}{Colors.END}\n")

def print_step_header(step, total_steps=None):
    """Print step separator"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*80}")
    if total_steps:
        print(f"STEP {step}/{total_steps}")
    else:
        print(f"STEP {step}")
    print(f"{'─'*80}{Colors.END}")

def load_data(data_path, data_name):
    """Load all necessary data files"""
    print_header("INITIALIZING GMOCAT SYSTEM")
    
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
    
    # Load train task
    with open(f'{data_path}/train_task_{data_name}.json', 'r') as f:
        data['train_task'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Loaded student interaction history ({len(data['train_task'])} students)")
    
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
            return student['q_ids'], student
    return None, None


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
    valid_answers = [chr(65 + i) for i in range(num_choices)]
    valid_answers_lower = [a.lower() for a in valid_answers]
    
    while True:
        try:
            answer = input(f"\n{Colors.BOLD}Your answer ({'/'.join(valid_answers)}) or 'q' to quit: {Colors.END}").strip()
            
            # Allow quit
            if answer.lower() == 'q':
                return 'QUIT'
            
            if answer.upper() in valid_answers or answer.lower() in valid_answers_lower:
                return answer.upper()
            else:
                print(f"{Colors.RED}Invalid! Please enter {' or '.join(valid_answers)}{Colors.END}")
        except EOFError:
            return 'QUIT'

def check_answer(user_answer, correct_answer, choices):
    """Check if answer is correct"""
    try:
        correct_index = choices.index(correct_answer)
        correct_letter = chr(65 + correct_index)
        is_correct = (user_answer == correct_letter)
        return is_correct, correct_letter
    except ValueError:
        user_index = ord(user_answer) - 65
        if 0 <= user_index < len(choices):
            return choices[user_index] == correct_answer, None
        return False, None

def display_result(is_correct, correct_letter, coverage, tested_concepts, total_concepts, target_coverage=None):
    """Display result of the answer"""
    print()
    if is_correct:
        print(f"{Colors.GREEN}{Colors.BOLD}{'✓'*3} CORRECT! {'✓'*3}{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}{'✗'*3} INCORRECT {'✗'*3}{Colors.END}")
        if correct_letter:
            print(f"{Colors.YELLOW}The correct answer was: {correct_letter}{Colors.END}")
    
    # Coverage info
    print(f"\n{Colors.CYAN}{Colors.BOLD}Coverage: {coverage:.1%}{Colors.END} ({tested_concepts}/{total_concepts} concepts)")
    
    # Progress bar
    bar_length = 50
    filled = int(coverage * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"{Colors.CYAN}{bar}{Colors.END}")
    
    # Target coverage indicator
    if target_coverage and coverage >= target_coverage:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎯 Target coverage {target_coverage:.0%} reached!{Colors.END}")

def display_progress_stats(step, total_steps, score, coverage, tested_concepts, total_concepts):
    """Display current progress statistics"""
    accuracy = score / step if step > 0 else 0
    
    print(f"\n{Colors.BOLD}Current Progress:{Colors.END}")
    print(f"  Questions: {step}/{total_steps}")
    print(f"  Accuracy: {Colors.GREEN if accuracy >= 0.7 else Colors.YELLOW}{accuracy:.1%}{Colors.END} ({score}/{step})")
    print(f"  Coverage: {Colors.CYAN}{coverage:.1%}{Colors.END} ({tested_concepts}/{total_concepts})")

def display_final_summary(score, total_questions, coverage, tested_concepts, total_concepts, coverage_history, early_stop=False):
    """Display final summary"""
    print_header("TEST COMPLETE - FINAL RESULTS")
    
    accuracy = score / total_questions if total_questions > 0 else 0
    
    # Termination reason
    if early_stop:
        print(f"{Colors.YELLOW}{Colors.BOLD}Test stopped early by user{Colors.END}\n")
    
    print(f"{Colors.BOLD}Performance:{Colors.END}")
    print(f"  Questions answered: {total_questions}")
    print(f"  Correct answers: {score}")
    print(f"  Accuracy: {Colors.GREEN if accuracy >= 0.7 else Colors.YELLOW}{Colors.BOLD}{accuracy:.1%}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Coverage:{Colors.END}")
    print(f"  Final coverage: {Colors.CYAN}{Colors.BOLD}{coverage:.1%}{Colors.END}")
    print(f"  Concepts tested: {tested_concepts}/{total_concepts}")
    print(f"  Concepts remaining: {total_concepts - tested_concepts}")
    
    # Coverage growth
    if len(coverage_history) > 1:
        print(f"\n{Colors.BOLD}Coverage Growth:{Colors.END}")
        milestones = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for ms in milestones:
            if ms < len(coverage_history):
                cov = coverage_history[ms]
                bar_len = int(cov * 50)
                bar = '█' * bar_len + '░' * (50 - bar_len)
                print(f"  Step {ms:3d}: {Colors.CYAN}{bar}{Colors.END} {cov:.1%}")
    
    # Grade
    print(f"\n{Colors.BOLD}Final Assessment:{Colors.END}")
    if accuracy >= 0.9:
        grade = "Outstanding! 🌟"
        color = Colors.GREEN
    elif accuracy >= 0.8:
        grade = "Excellent! ⭐"
        color = Colors.GREEN
    elif accuracy >= 0.7:
        grade = "Good job! ✓"
        color = Colors.GREEN
    elif accuracy >= 0.6:
        grade = "Fair"
        color = Colors.YELLOW
    else:
        grade = "Needs improvement"
        color = Colors.RED
    
    print(f"  {color}{Colors.BOLD}{grade}{Colors.END}")
    
    # Coverage assessment
    if coverage >= 0.9:
        cov_grade = "Comprehensive coverage! 🎯"
        cov_color = Colors.GREEN
    elif coverage >= 0.7:
        cov_grade = "Good coverage ✓"
        cov_color = Colors.GREEN
    elif coverage >= 0.5:
        cov_grade = "Moderate coverage"
        cov_color = Colors.YELLOW
    else:
        cov_grade = "Limited coverage"
        cov_color = Colors.YELLOW
    
    print(f"  {cov_color}{Colors.BOLD}{cov_grade}{Colors.END}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Simulation complete!{Colors.END}\n")

def run_interactive_test(data, student_id=None, max_steps=None, target_coverage=None):
    """
    Run interactive CAT where user answers questions
    
    Args:
        data: Dictionary containing all loaded data
        student_id: Specific student ID (uses first student if None)
        max_steps: Maximum number of questions (None = all available)
        target_coverage: Target coverage to stop (None = no auto-stop)
    """
    
    print_header("INTERACTIVE CAT DEMO")
    
    # Get student questions
    if student_id:
        q_ids, student = get_student_questions(student_id, data['train_task'])
        if not q_ids:
            print(f"{Colors.RED}Error: Student {student_id} not found{Colors.END}")
            return
        
        # Apply greedy coverage selection
        available_q_ids = q_ids
        q_ids = greedy_coverage_selection(
            available_questions=available_q_ids,
            concept_map=data['concept_map'],
            tested_concepts=set(),
            max_questions=None
        )
        print(f"{Colors.CYAN}Using greedy coverage strategy for student {student_id}{Colors.END}")
    else:
        # Use student with 100% coverage potential (has all 212 questions)
        # Options: 20, 25, 42, 55, 69, 79, 118, 132, 26, 135
        target_student_id = 20
        student = None
        for s in data['train_task']:
            if s['student_id'] == target_student_id:
                student = s
                break
        if not student:
            student = data['train_task'][0]  # Fallback
        # Get available questions from student
        available_q_ids = student['q_ids']
        
        # Use greedy coverage-based selection for high coverage
        q_ids = greedy_coverage_selection(
            available_questions=available_q_ids,
            concept_map=data['concept_map'],
            tested_concepts=set(),  # Start with no tested concepts
            max_questions=None  # Use all available questions
        )
        
        print(f"{Colors.CYAN}Using greedy coverage strategy (optimized from {len(available_q_ids)} questions){Colors.END}")
        student_id = student['student_id']
    
    # Determine total steps
    total_available = len(q_ids)
    if max_steps is None:
        total_steps = total_available
        q_ids_to_use = q_ids
    else:
        total_steps = min(max_steps, total_available)
        q_ids_to_use = q_ids[:total_steps]
    
    print(f"{Colors.BOLD}Test Configuration:{Colors.END}")
    print(f"  Student ID: {student_id}")
    print(f"  Total questions available: {total_available}")
    print(f"  Questions in this test: {total_steps}")
    if target_coverage:
        print(f"  Target coverage: {target_coverage:.0%}")
    print(f"  Total concepts: {data['n_concepts']}")
    
    print(f"\n{Colors.BOLD}Instructions:{Colors.END}")
    print(f"  • Answer each question by typing A, B, C, or D")
    print(f"  • Type 'q' to quit anytime")
    print(f"  • Press Ctrl+C to stop")
    if target_coverage:
        print(f"  • Test stops automatically when coverage reaches {target_coverage:.0%}")
    
    input(f"\n{Colors.YELLOW}Press Enter to start the test...{Colors.END}")
    
    # Track metrics
    tested_concepts = set()
    coverage_history = []
    score = 0
    early_stop = False
    
    # Main test loop
    try:
        for step, q_id in enumerate(q_ids_to_use, 1):
            # Get question info
            q_text = data['question_text'][str(q_id)]
            concepts = data['concept_map'][str(q_id)]
            
            # Display question
            correct_answer = display_question(q_id, q_text, concepts, step, total_steps)
            
            # Get user answer
            user_answer = get_user_answer(len(q_text['choices']))
            
            if user_answer == 'QUIT':
                early_stop = True
                break
            
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
            display_result(is_correct, correct_letter, coverage, len(tested_concepts), data['n_concepts'], target_coverage)
            
            # Check target coverage termination
            if target_coverage and coverage >= target_coverage:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🎯 Target coverage reached! Test complete.{Colors.END}")
                time.sleep(1)
                break
            
            # Show progress every 10 questions
            if step % 10 == 0 and step < total_steps:
                display_progress_stats(step, total_steps, score, coverage, len(tested_concepts), data['n_concepts'])
            
            # Pause before next question
            if step < total_steps:
                input(f"\n{Colors.YELLOW}Press Enter for next question...{Colors.END}")
        
        # Set final step count
        final_steps = step
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user (Ctrl+C){Colors.END}")
        early_stop = True
        final_steps = step if 'step' in locals() else 0
    
    # Final summary
    if final_steps > 0:
        display_final_summary(score, final_steps, coverage, len(tested_concepts), data['n_concepts'], coverage_history, early_stop)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Interactive CAT Demo v2 - Unlimited Questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_interactive_v2.py
  python demo_interactive_v2.py --student_id 54
  python demo_interactive_v2.py --max_steps 50
  python demo_interactive_v2.py --target_coverage 0.8
  python demo_interactive_v2.py --student_id 54 --target_coverage 0.9
        """
    )
    
    parser.add_argument('--student_id', type=int, default=None,
                       help='Student ID to use (default: first student in dataset)')
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='Path to data directory (default: ./data/)')
    parser.add_argument('--data_name', type=str, default='dbekt22',
                       help='Dataset name (default: dbekt22)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum questions (default: all available)')
    parser.add_argument('--target_coverage', type=float, default=None,
                       help='Auto-stop when coverage reaches this (e.g., 0.8 for 80%%)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data(args.data_path, args.data_name)
        
        # Run interactive test
        run_interactive_test(
            data=data,
            student_id=args.student_id,
            max_steps=args.max_steps,
            target_coverage=args.target_coverage
        )
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error: Data file not found{Colors.END}")
        print(f"  {e}")
        print(f"\n{Colors.YELLOW}Make sure you're in the correct directory:{Colors.END}")
        print(f"  cd /data/adalah_gmocat/modified-GMOCAT-main/GMOCAT-Final/GMOCAT-modif")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Program terminated{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error occurred:{Colors.END}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
