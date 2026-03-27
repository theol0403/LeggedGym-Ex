"""Monitor active experiment runs by reading TensorBoard event files.

Usage:
    python legged_gym/scripts/monitor_experiments.py
    python legged_gym/scripts/monitor_experiments.py --watch 60   # refresh every 60s
"""

import argparse
import glob
import os
import sys
import time

LOGS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _read_latest_scalars(log_dir, tags):
    """Read the latest value for each tag from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("tensorboard not installed — pip install tensorboard")
        sys.exit(1)

    ea = EventAccumulator(log_dir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    results = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            if events:
                results[tag] = (events[-1].step, events[-1].value)
    return results


METRICS = [
    "Episode/success_stairs",
    "Episode/success_hurdle_block",
    "Episode/success_gap",
    "Episode/progress_gap",
    "Loss/action",
    "Loss/yaw",
    "Train/mean_reward",
]


def scan_runs():
    """Find all recent run directories and report metrics."""
    logs_root = os.path.normpath(LOGS_ROOT)
    if not os.path.isdir(logs_root):
        print(f"Logs directory not found: {logs_root}")
        return

    runs = []
    for experiment_dir in sorted(os.listdir(logs_root)):
        exp_path = os.path.join(logs_root, experiment_dir)
        if not os.path.isdir(exp_path):
            continue
        for run_dir in sorted(os.listdir(exp_path)):
            run_path = os.path.join(exp_path, run_dir)
            if not os.path.isdir(run_path):
                continue
            event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
            if not event_files:
                continue
            runs.append((experiment_dir, run_dir, run_path))

    if not runs:
        print("No runs found.")
        return

    print(f"\n{'='*120}")
    print(f"  {'Experiment':<40} {'Run':<45} {'Step':>6}  {'Stairs':>7} {'Hurdle':>7} {'Gap':>7} {'GapPrg':>7} {'ActLoss':>8} {'Reward':>7}")
    print(f"{'='*120}")

    for experiment_dir, run_dir, run_path in runs:
        metrics = _read_latest_scalars(run_path, METRICS)
        if not metrics:
            continue

        step = max(v[0] for v in metrics.values()) if metrics else 0
        stairs = metrics.get("Episode/success_stairs", (0, float("nan")))[1]
        hurdle = metrics.get("Episode/success_hurdle_block", (0, float("nan")))[1]
        gap = metrics.get("Episode/success_gap", (0, float("nan")))[1]
        gap_prg = metrics.get("Episode/progress_gap", (0, float("nan")))[1]
        act_loss = metrics.get("Loss/action", (0, float("nan")))[1]
        reward = metrics.get("Train/mean_reward", (0, float("nan")))[1]

        # Flag status
        flags = ""
        if gap >= 0.5:
            flags += " ** PROMISING **"
        if gap >= 0.8:
            flags = " *** TARGET ***"
        if stairs < 0.5 and step > 200:
            flags += " !! DIVERGED !!"

        short_run = run_dir[-35:] if len(run_dir) > 35 else run_dir
        print(
            f"  {experiment_dir:<40} {short_run:<45} {step:>6}  "
            f"{stairs:>6.1%} {hurdle:>6.1%} {gap:>6.1%} {gap_prg:>6.2f}  "
            f"{act_loss:>7.4f} {reward:>7.1f}{flags}"
        )

    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor experiment runs")
    parser.add_argument("--watch", type=int, default=0,
                        help="Refresh interval in seconds (0 = single shot)")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                os.system("clear")
                print(f"Monitoring experiments (refresh every {args.watch}s, Ctrl+C to stop)")
                scan_runs()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        scan_runs()


if __name__ == "__main__":
    main()
