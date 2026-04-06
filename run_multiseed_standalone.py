#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner - STANDALONE (50 EPOCHS VERSION)
Modified for Priority 2A: Skip seed 42, run seeds [123, 456, 789, 2024]
50 epochs, NO early stopping (must match Run 3 configuration)
"""

import sys
import os
import json
import glob
import shutil
import numpy as np
from datetime import datetime

# Path setup
GMOCAT_PATH = '/data/adalah_gmocat/modified-GMOCAT-main/GMOCAT-Final/GMOCAT-modif'
os.chdir(GMOCAT_PATH)
sys.path.insert(0, GMOCAT_PATH)

# Import setelah path setup
import importlib.util

# ============================================================================
# KONFIGURASI - MODIFIED FOR 50 EPOCHS
# ============================================================================
SEEDS = [456, 789, 2024]  # Remaining 3 seeds  # Run one at a time  # All 4 remaining seeds
EXPERIMENT_NAME = "Run3_MultiSeed_50Epochs"
MAX_EPOCHS = 50  # MUST BE 50 to match Run 3


def load_run_experiment():
    """Load run_experiment.py module secara dinamis"""
    spec = importlib.util.spec_from_file_location("run_experiment",
                                                   f"{GMOCAT_PATH}/run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_final_metrics(log_file):
    """Parse metrics dari log file"""
    metrics = {}

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Cari "Best Mean Metrics" terakhir
        best_idx = -1
        for i in range(len(lines)-1, -1, -1):
            if "Best Mean Metrics:" in lines[i]:
                best_idx = i
                break

        if best_idx == -1:
            # Fallback: cari "Mean Metrics" terakhir
            for i in range(len(lines)-1, -1, -1):
                if "Mean Metrics:" in lines[i] and "Best" not in lines[i]:
                    best_idx = i
                    break

        if best_idx == -1:
            return metrics

        # Parse metrics
        for line in lines[best_idx:best_idx+25]:
            # Format: "1AUC: 0.6168, ACC: 0.7249"
            if "AUC:" in line and "ACC:" in line and "," in line:
                try:
                    parts = line.strip().split(',')
                    # Parse AUC
                    auc_part = parts[0].strip()
                    step = auc_part.split('AUC:')[0].strip()
                    auc_val = float(auc_part.split(':')[1].strip())

                    # Parse ACC
                    acc_part = parts[1].strip()
                    acc_val = float(acc_part.split(':')[1].strip())

                    metrics[f'auc@{step}'] = auc_val
                    metrics[f'acc@{step}'] = acc_val
                except:
                    continue

            # Format: "20COV: 0.616"
            elif "COV:" in line:
                try:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        step = parts[0].replace('COV', '').strip()
                        cov_val = float(parts[1].strip())
                        metrics[f'cov@{step}'] = cov_val
                except:
                    continue

        # Check actual epoch completed
        max_epoch_seen = 0
        for line in lines:
            if "epoch:" in line.lower():
                try:
                    # Extract epoch number from patterns like "epoch: 49"
                    parts = line.split("epoch:")
                    if len(parts) > 1:
                        epoch_num = int(parts[1].split(',')[0].strip())
                        max_epoch_seen = max(max_epoch_seen, epoch_num)
                except:
                    pass
        
        if max_epoch_seen > 0:
            metrics['final_epoch'] = max_epoch_seen

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")

    return metrics


def run_single_seed(seed, output_dir, run_experiment_module):
    """Jalankan eksperimen untuk satu seed"""

    print(f"\n{'='*70}")
    print(f"RUNNING SEED: {seed}")
    print(f"{'='*70}\n")

    # Import Args class dan main function
    Args = run_experiment_module.Args
    main = run_experiment_module.main

    # Create args dengan seed baru
    args = Args()
    args.seed = seed
    
    # FORCE 50 epochs
    args.training_epoch = MAX_EPOCHS

    # Simpan config
    config_path = os.path.join(output_dir, f'config_seed_{seed}.json')
    config_dict = {k: v for k, v in args.__dict__.items()
                   if not k.startswith('_') and v is not None}
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Config: seed={seed}, epoch={args.training_epoch}, emb_dim={args.emb_dim}")
    print(f"Early stopping: DISABLED (will run full {MAX_EPOCHS} epochs)\n")

    # Run experiment
    try:
        main(args)
        print(f"\n✓ Seed {seed} completed")

        # Find latest log file
        log_dir = f'{GMOCAT_PATH}/baseline_log/dbekt22'
        log_files = glob.glob(f'{log_dir}/*.txt')
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)

            # Copy log file ke output dir dengan nama yang jelas
            log_copy = os.path.join(output_dir, f'log_seed_{seed}.txt')
            shutil.copy2(latest_log, log_copy)
            print(f"✓ Log copied to: {log_copy}")

            # Parse metrics
            metrics = parse_final_metrics(latest_log)
            metrics['seed'] = seed
            metrics['log_file'] = latest_log

            # Save individual result
            result_path = os.path.join(output_dir, f'result_seed_{seed}.json')
            with open(result_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"✓ Metrics parsed:")
            for k, v in sorted(metrics.items()):
                if k not in ['seed', 'log_file']:
                    print(f"    {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

            return metrics
        else:
            print(f"⚠ No log file found")
            return {'seed': seed, 'error': 'No log file'}

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'seed': seed, 'error': str(e)}


def aggregate_results(results):
    """Aggregate multi-seed results"""
    valid = [r for r in results if 'error' not in r]

    if not valid:
        return {}

    # Collect all metrics
    all_metrics = set()
    for r in valid:
        all_metrics.update([k for k in r.keys()
                           if k not in ['seed', 'log_file', 'error', 'final_epoch']])

    summary = {}
    for metric in sorted(all_metrics):
        values = [r[metric] for r in valid if metric in r]
        if values:
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.mean(values) * 100),  # Coefficient of variation
                'values': [float(v) for v in values],
                'n_seeds': len(values)
            }

    # Info tentang final epochs
    final_epochs = [r.get('final_epoch') for r in valid if 'final_epoch' in r]
    if final_epochs:
        summary['epoch_info'] = {
            'mean_epoch': float(np.mean(final_epochs)),
            'epochs': final_epochs,
            'all_completed_50': all(e >= 49 for e in final_epochs)  # epoch 49 = 50th epoch (0-indexed)
        }

    return summary


def print_summary(summary, results):
    """Print summary table"""
    print(f"\n\n{'='*70}")
    print("MULTI-SEED EXPERIMENT SUMMARY (50 EPOCHS)")
    print(f"{'='*70}\n")

    valid_count = len([r for r in results if 'error' not in r])
    print(f"Valid results: {valid_count}/{len(results)} seeds\n")

    # Epoch info
    if 'epoch_info' in summary:
        ep_info = summary['epoch_info']
        print(f"Epochs completed: {ep_info['epochs']}")
        if ep_info['all_completed_50']:
            print(f"✓ All seeds completed full 50 epochs\n")
        else:
            print(f"⚠ Some seeds did not complete 50 epochs\n")

    # Metrics per step
    for step in ['1', '5', '10', '20']:
        print(f"@{step} SOAL:")
        print("-" * 70)

        for metric_type in ['cov', 'auc', 'acc']:
            key = f'{metric_type}@{step}'
            if key in summary:
                s = summary[key]
                print(f"{key.upper():10s}: {s['mean']:7.4f} ± {s['std']:6.4f}  " +
                      f"[{s['min']:.4f} - {s['max']:.4f}]  CV: {s['cv']:5.2f}%")
        print()

    # Comparison dengan Run 3 original (seed 42)
    print("="*70)
    print("COMPARISON WITH RUN 3 (SEED 42)")
    print("="*70 + "\n")

    run3_original = {'cov@20': 0.6308, 'auc@20': 0.6312, 'acc@20': 0.7525}

    print(f"{'Metric':<12s} {'Run3(42)':>10s} {'MultiSeed':>18s} {'Diff':>12s}")
    print("-" * 70)

    for metric, orig in run3_original.items():
        if metric in summary:
            ms = summary[metric]['mean']
            std = summary[metric]['std']
            diff = ms - orig
            diff_pct = (diff / orig) * 100
            print(f"{metric.upper():<12s} {orig:10.4f}  {ms:10.4f} ± {std:5.4f}  " +
                  f"{diff:+7.4f} ({diff_pct:+6.2f}%)")


def main():
    """Main function"""
    print("="*70)
    print("GMOCAT RUN 3 - MULTI-SEED VALIDATION (50 EPOCHS)")
    print("="*70)
    print(f"\nSeeds to run: {SEEDS}")
    print(f"Seed 42: SKIPPED (already done as Run 3 main result)")
    print(f"Total: {len(SEEDS)} experiments")
    print(f"\nConfiguration:")
    print(f"  - Max epochs: {MAX_EPOCHS} (MUST match Run 3)")
    print(f"  - Early stopping: DISABLED")
    print(f"  - Hyperparameters: Same as Run 3")
    print(f"\nEstimated time: ~{len(SEEDS) * 6} hours ({len(SEEDS) * 6 / 24:.1f} days)")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'{GMOCAT_PATH}/multiseed_50epoch_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory:\n  {output_dir}")

    # Confirm
    print("\n" + "="*70)
    response = input("Start multi-seed experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Load run_experiment module
    print("\nLoading run_experiment.py...")
    run_exp = load_run_experiment()
    print("✓ Module loaded\n")

    # Run experiments
    print("="*70)
    print("STARTING EXPERIMENTS")
    print("="*70)

    results = []
    start_time = datetime.now()

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT {i}/{len(SEEDS)} - SEED {seed}")
        print(f"{'#'*70}")

        exp_start = datetime.now()
        result = run_single_seed(seed, output_dir, run_exp)
        exp_time = (datetime.now() - exp_start).total_seconds() / 60

        result['duration_minutes'] = exp_time
        results.append(result)

        print(f"\n✓ Completed in {exp_time:.1f} minutes ({exp_time/60:.2f} hours)")

        # Save intermediate
        inter_path = os.path.join(output_dir, 'intermediate_results.json')
        with open(inter_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Estimate remaining
        if i < len(SEEDS):
            avg = sum([r.get('duration_minutes', 0) for r in results]) / len(results)
            remaining = avg * (len(SEEDS) - i)
            print(f"Estimated remaining: {remaining:.0f} min ({remaining/60:.1f} hours)")

    total_time = (datetime.now() - start_time).total_seconds() / 60

    # Aggregate
    print("\n\nAggregating results...")
    summary = aggregate_results(results)

    # Save summary
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print_summary(summary, results)

    # Save report
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()) as output:
            print_summary(summary, results)

        f.write("GMOCAT RUN 3 - MULTI-SEED VALIDATION REPORT (50 EPOCHS)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Duration: {total_time:.1f} min ({total_time/60:.1f} hours)\n")
        f.write(f"Seeds: {SEEDS} (Seed 42 skipped - already done as Run 3)\n")
        f.write(f"Epochs: {MAX_EPOCHS}\n\n")
        f.write(output.getvalue())

    print(f"\n\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"\nResults saved in:\n  {output_dir}/")
    print(f"\nFiles:")
    print(f"  - summary_{timestamp}.json")
    print(f"  - report_{timestamp}.txt")
    print(f"  - config_seed_*.json (per seed)")
    print(f"  - result_seed_*.json (per seed)")
    print(f"  - log_seed_*.txt (per seed)")
    print(f"\nNext: Run ablation studies (Run A, B, C)")


if __name__ == "__main__":
    main()
