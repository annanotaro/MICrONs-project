"""
MICrONs Project: Master Pipeline Runner
Runs Q0, Q1, Q2, and Q3 for a chosen session.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# --- ANSI COLORS ---
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
CYAN = "\033[96m"
RESET = "\033[0m"

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent

# Define the full pipeline structure
PIPELINE = {
    "Q0: Exploration": [
        "src/step0_explore_session.py",
    ],
    "Q1: Categorization": [
        "src/step1_features.py",
        "src/step1b_behavioral_clean.py",
        "src/step4_learning_curves.py",
        "src/step4_learning_curves_CLEAN.py",
        "src/step5_confusion.py",
        "src/step5_confusion_CLEAN.py",
    ],
    "Q2: Temporal Decoding": [
        "q2/step0_trial_labels.py",
        "q2/step1_features.py",
        "q2/step1b_features_clean.py",
        "q2/step2_decode.py",
        "q2/step3_plot.py",
    ],
    "Q3: Naturalness Deep-Dive": [
        "q3/step0_labels.py",
        "q3/step1_features.py",
        "q3/step1b_features_clean.py",
        "q3/step2_natural_vs_parametric.py",
        "q3/step3_natural_subgroups.py",
        "q3/step4_plots.py",
        "q3/step5_time_features.py",
        "q3/step5b_time_features_clean.py",
        "q3/step6_time_decode.py",
        "q3/step7_time_plots.py",
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full MICrONs analysis pipeline.")
    parser.add_argument("session", nargs="?", default="7_4", help="Session ID (default: 7_4)")
    return parser.parse_args()

def run_task(script_path: str, session: str, log_file, current_step, total_steps):
    full_path = PROJECT_ROOT / script_path
    cmd = [sys.executable, str(full_path), session]
    
    # Progress indicator
    prefix = f"[{current_step}/{total_steps}]"
    print(f"  {CYAN}{prefix}{RESET} Running {YELLOW}{script_path}{RESET}...", end="", flush=True)
    
    log_file.write(f"\n{'='*80}\nTASK: {script_path}\nCOMMAND: {' '.join(cmd)}\n{'='*80}\n")
    log_file.flush()
    
    start_time = time.time()
    
    # Run process and capture output to log file only
    with subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True) as proc:
        proc.wait()
        success = (proc.returncode == 0)
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"\r  {CYAN}{prefix}{RESET} {GREEN}✓{RESET} {script_path} ({elapsed:.1f}s)      ")
    else:
        print(f"\r  {CYAN}{prefix}{RESET} {RED}✗{RESET} {script_path} ({elapsed:.1f}s) - {BOLD}FAILED{RESET}")
        return False
    return True

def main():
    args = parse_args()
    session = args.session
    
    # Setup results directory and log file
    results_dir = PROJECT_ROOT / "results" / session
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "master_pipeline.log"
    
    # Total tasks calculation
    total_tasks = sum(len(tasks) for tasks in PIPELINE.values())
    
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"{BOLD}{BLUE}MICrONs Master Pipeline{RESET}")
    print(f"{BOLD}Session:{RESET} {YELLOW}{session}{RESET}")
    print(f"{BOLD}Log File:{RESET} {log_path}")
    print(f"{'='*50}\n")
    
    current_task_count = 0
    overall_start = time.time()
    
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            for chapter, tasks in PIPELINE.items():
                print(f"{BOLD}{chapter}{RESET}")
                
                for task in tasks:
                    current_task_count += 1
                    ok = run_task(task, session, log_file, current_task_count, total_tasks)
                    if not ok:
                        print(f"\n{RED}{BOLD}Pipeline aborted due to failure in {task}.{RESET}")
                        print(f"Check the log for details: {log_path}")
                        return 1
                print() # Newline between chapters
                
        total_time = time.time() - overall_start
        print(f"{'='*50}")
        print(f"{BOLD}{GREEN}Pipeline Completed Successfully!{RESET}")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print(f"\n\n{RED}{BOLD}Pipeline Interrupted by User.{RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
