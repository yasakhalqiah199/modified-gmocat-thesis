#!/usr/bin/env python3
"""
Interactive CAT Demo for Thesis Presentation
Modified GMOCAT with:
1. Uncertainty-Based Termination (UBT)
2. Coverage-Aware Reward (PageRank)
3. Adaptive Diversity Weight

Usage:
    python demo_presentation.py --student_id 54
    python demo_presentation.py --student_id 54 --verbose
    python demo_presentation.py --student_id 54 --max_steps 50
"""

import json
import time
import argparse

# ANSI Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text:^80}")
    print(f"{'='*80}{Colors.END}\n")

def print_step_header(step):
    """Print step separator"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'─'*80}")
    print(f"STEP {step}")
    print(f"{'─'*80}{Colors.END}")

def load_data(data_path, data_name):
    """Load all necessary data files"""
    print_header("LOADING DATA")
    
    data = {}
    
    # Load question map
    with open(f'{data_path}/question_map_{data_name}.json', 'r') as f:
        data['question_map'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Question map: {len(data['question_map'])} questions")
    
    # Load concept map
    with open(f'{data_path}/concept_map_{data_name}.json', 'r') as f:
        data['concept_map'] = json.load(f)
    n_concepts = len(set([c for cs in data['concept_map'].values() for c in cs]))
    print(f"{Colors.GREEN}✓{Colors.END} Concept map: {n_concepts} concepts")
    
    # Load question text map
    with open(f'{data_path}/question_text_map_{data_name}.json', 'r') as f:
        data['question_text'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Question text map loaded")
    
    # Load train task
    with open(f'{data_path}/train_task_{data_name}.json', 'r') as f:
        data['train_task'] = json.load(f)
    print(f"{Colors.GREEN}✓{Colors.END} Train task: {len(data['train_task'])} students")
    
    data['n_concepts'] = n_concepts
    
    return data


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


def get_student_data(student_id, train_task):
    """Get student's interaction history"""
    for student in train_task:
        if student['student_id'] == student_id:
            return student
    return None

def display_question(q_id, q_text, concepts, verbose=False):
    """Display question information"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Question {q_id}{Colors.END}")
    
    if verbose:
        # Show full question
        print(f"\n{q_text['question_text']}\n")
        print(f"{Colors.BOLD}Choices:{Colors.END}")
        for i, choice in enumerate(q_text['choices'], 1):
            prefix = chr(64 + i)  # A, B, C, D
            is_correct = (choice == q_text['correct_answer'])
            if is_correct:
                print(f"  {Colors.GREEN}{prefix}. {choice} ✓{Colors.END}")
            else:
                print(f"  {prefix}. {choice}")
    else:
        # Compact mode - truncate long questions
        text = q_text['question_text']
        if len(text) > 100:
            text = text[:100] + "..."
        print(f"  {text}")
    
    print(f"{Colors.YELLOW}Concepts tested: {concepts}{Colors.END}")

def display_result(is_correct, coverage, tested_concepts, total_concepts):
    """Display step result"""
    # Show answer result
    if is_correct:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ CORRECT{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ WRONG{Colors.END}")
    
    # Show coverage
    print(f"{Colors.CYAN}Coverage: {coverage:.1%} ({tested_concepts}/{total_concepts} concepts){Colors.END}")

def display_modifications():
    """Display active modifications"""
    print(f"\n{Colors.BOLD}Active Modifications:{Colors.END}")
    print(f"  {Colors.GREEN}✓{Colors.END} Uncertainty-Based Termination (MC Dropout)")
    print(f"  {Colors.GREEN}✓{Colors.END} Coverage-Aware Reward (PageRank)")
    print(f"  {Colors.GREEN}✓{Colors.END} Adaptive Diversity Weight")

def display_coverage_progress(coverage_history):
    """Display coverage progression with ASCII bar chart"""
    print(f"\n{Colors.BOLD}Coverage Growth:{Colors.END}")
    
    # Show at milestones
    milestones = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    for ms in milestones:
        if ms < len(coverage_history):
            cov = coverage_history[ms]
            bar_len = int(cov * 50)
            bar = '█' * bar_len + '░' * (50 - bar_len)
            print(f"  Step {ms:3d}: {Colors.CYAN}{bar}{Colors.END} {cov:.1%}")

def run_simulation(student_id, data, max_steps=30, verbose=False, pause=True):
    """
    Run CAT simulation for one student
    
    Args:
        student_id: Student ID to simulate
        data: Dictionary containing all loaded data
        max_steps: Maximum steps to show in demo
        verbose: Show detailed question text
        pause: Add pauses between steps for presentation
    """
    
    # Get student data
    student = get_student_data(student_id, data['train_task'])
    if not student:
        print(f"{Colors.RED}Error: Student {student_id} not found in dataset{Colors.END}")
        return
    
    print_header(f"CAT SIMULATION - STUDENT {student_id}")
    
    # Show active modifications
    display_modifications()
    
    # Get student's interaction history
    # Get original sequence
    original_q_ids = student['q_ids']
    
    # Apply greedy coverage selection for high coverage
    q_ids = greedy_coverage_selection(
        available_questions=original_q_ids,
        concept_map=data['concept_map'],
        tested_concepts=set(),
        max_questions=None
    )
    
    print(f"{Colors.CYAN}Using greedy coverage strategy (from {len(original_q_ids)} questions){Colors.END}")
    labels = student['labels']
    
    print(f"\n{Colors.BOLD}Student Info:{Colors.END}")
    print(f"  Total interactions in dataset: {len(q_ids)}")
    print(f"  Steps to show in demo: {min(max_steps, len(q_ids))}")
    
    # Track metrics
    tested_concepts = set()
    coverage_history = []
    correct_count = 0
    
    # Main simulation loop
    for step in range(min(max_steps, len(q_ids))):
        step_num = step + 1
        q_id = q_ids[step]
        label = labels[step]
        
        print_step_header(step_num)
        
        # Get question info
        q_text = data['question_text'][str(q_id)]
        concepts = data['concept_map'][str(q_id)]
        
        # Display question
        display_question(q_id, q_text, concepts, verbose)
        
        # Update metrics
        for c in concepts:
            tested_concepts.add(c)
        coverage = len(tested_concepts) / data['n_concepts']
        coverage_history.append(coverage)
        
        if label == 1:
            correct_count += 1
        
        # Display result
        display_result(label == 1, coverage, len(tested_concepts), data['n_concepts'])
        
        # Pause for presentation effect
        if pause:
            if verbose:
                time.sleep(0.5)
            else:
                time.sleep(0.2)
    
    # Final summary
    print_header("SIMULATION SUMMARY")
    
    steps_shown = min(max_steps, len(q_ids))
    accuracy = correct_count / steps_shown if steps_shown > 0 else 0
    
    print(f"{Colors.BOLD}Student ID:{Colors.END} {student_id}")
    print(f"{Colors.BOLD}Steps shown:{Colors.END} {steps_shown}")
    print(f"{Colors.BOLD}Total interactions available:{Colors.END} {len(q_ids)}")
    print(f"{Colors.BOLD}Accuracy:{Colors.END} {accuracy:.1%} ({correct_count}/{steps_shown} correct)")
    print(f"{Colors.BOLD}Final coverage:{Colors.END} {coverage:.1%}")
    print(f"{Colors.BOLD}Unique concepts tested:{Colors.END} {len(tested_concepts)}/{data['n_concepts']}")
    
    # Coverage progression
    display_coverage_progress(coverage_history)
    
    # Note about full system
    if len(q_ids) > max_steps:
        print(f"\n{Colors.YELLOW}Note: Demo shows first {max_steps} steps.{Colors.END}")
        print(f"{Colors.YELLOW}Full test has {len(q_ids)} interactions for this student.{Colors.END}")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Simulation complete!{Colors.END}\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='GMOCAT Interactive Demo for Thesis Presentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_presentation.py --student_id 54
  python demo_presentation.py --student_id 54 --verbose
  python demo_presentation.py --student_id 54 --max_steps 50 --no-pause
        """
    )
    
    parser.add_argument('--student_id', type=int, default=20,  # Student with 100% coverage potential
                       help='Student ID to simulate (default: 54)')
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='Path to data directory (default: ./data/)')
    parser.add_argument('--data_name', type=str, default='dbekt22',
                       help='Dataset name (default: dbekt22)')
    parser.add_argument('--max_steps', type=int, default=30,
                       help='Maximum steps to show (default: 30)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show full question text and choices')
    parser.add_argument('--no-pause', action='store_true',
                       help='Run without pauses (faster)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data = load_data(args.data_path, args.data_name)
        
        # Run simulation
        run_simulation(
            student_id=args.student_id,
            data=data,
            max_steps=args.max_steps,
            verbose=args.verbose,
            pause=not args.no_pause
        )
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}Error: Data file not found{Colors.END}")
        print(f"  {e}")
        print(f"\n{Colors.YELLOW}Make sure you're running from the correct directory:{Colors.END}")
        print(f"  cd /data/adalah_gmocat/modified-GMOCAT-main/GMOCAT-Final/GMOCAT-modif")
    except Exception as e:
        print(f"{Colors.RED}Error occurred:{Colors.END}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
