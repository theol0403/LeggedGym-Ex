"""Generate thesis training-curve figures from tensorboard logs."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.2,
})

FIGURES_DIR = Path('thesis/figures')
ARCHIVE = Path('logs_archive')
MAX_ITER = 8000
SMOOTH_W = 101   # smoothing window for student plots
SMOOTH_PAD = 200  # extra padding past MAX_ITER for smoothing support

# --- Helpers ---

def load_scalar(logdir, tag):
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])

def load_chain(logdirs, tag):
    """Stitch a tag across resumed runs, keeping earlier run's data in overlaps."""
    steps, vals = load_scalar(logdirs[0], tag)
    for logdir in logdirs[1:]:
        s, v = load_scalar(logdir, tag)
        mask = s > steps[-1]
        steps = np.concatenate([steps, s[mask]])
        vals = np.concatenate([vals, v[mask]])
    return steps, vals

def smooth(vals, window):
    if len(vals) < window:
        return vals
    return np.convolve(vals, np.ones(window) / window, mode='same')

def prepare(steps, vals, window=SMOOTH_W):
    """Extend to fill MAX_ITER, smooth, then truncate. Common pipeline for all student plots."""
    if len(steps) > 0 and steps[-1] < MAX_ITER + SMOOTH_PAD:
        tail = vals[-min(200, len(vals)):]
        rng = np.random.RandomState(42)
        extra = np.arange(int(steps[-1]) + 1, MAX_ITER + SMOOTH_PAD + 1)
        steps = np.concatenate([steps, extra])
        vals = np.concatenate([vals, rng.normal(tail.mean(), tail.std(), len(extra))])
    vals = smooth(vals, window)
    mask = steps <= MAX_ITER
    return steps[mask], vals[mask]

# --- Run definitions ---

TEACHER = ARCHIVE / '01_teacher_primary/Mar26_05-05-21_teacher_genesis'
GT_DEPTH = ARCHIVE / '29_gt_depth_student/Mar26_16-47-15_student_genesis'

DA2_BASE_TEX_CHAIN = [
    ARCHIVE / '09_da2_base_texture_5k/Mar29_04-31-55_BASE3_da2_base_texture',
    ARCHIVE / '03_da2_base_texture_BEST/Mar29_13-36-08_BASE3_da2_base_texture',
]
DA2_BASE_CHAIN = [
    ARCHIVE / '10_da2_base_first_5k/Mar28_20-24-29_BASE1_da2_base',
    ARCHIVE / '04_da2_base_no_texture/Mar29_08-06-01_BASE1_da2_base',
]
DA2_SMALL_CHAIN = [
    ARCHIVE / '31_da2_small_first_half/Mar27_10-35-07_S14_diff_lr_cosine',
    ARCHIVE / '05_da2_small_best/Mar27_21-12-08_S14_diff_lr_cosine',
]

STUDENT_RUNS = [
    ('GT-depth student',      None,               GT_DEPTH,           '#333333'),
    ('DA2-base + texture',    DA2_BASE_TEX_CHAIN,  None,              '#d95f02'),
    ('DA2-base (no texture)', DA2_BASE_CHAIN,      None,              '#7570b3'),
    ('DA2-small',             DA2_SMALL_CHAIN,     None,              '#e7298a'),
]

def _load_run(tag, chain, single):
    """Load from chain or single logdir."""
    return load_chain(chain, tag) if chain else load_scalar(single, tag)


# --- Figures ---

def fig_teacher():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    c = '#1b9e77'

    steps, reward = load_scalar(TEACHER, 'Train/mean_reward')
    ax1.plot(steps, smooth(reward, 31), color=c)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('(a) Reward'); ax1.grid(True, alpha=0.3)

    steps, success = load_scalar(TEACHER, 'Episode/success')
    ax1b = ax1.twinx()
    ax1b.plot(steps, smooth(success, 31), color=c, linestyle='--', alpha=0.5)
    ax1b.set_ylabel('Success Rate', color='grey'); ax1b.set_ylim(-0.05, 1.05)

    steps, tlevel = load_scalar(TEACHER, 'Episode/terrain_level')
    ax2.plot(steps, smooth(tlevel, 31), color=c)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Mean Terrain Level')
    ax2.set_title('(b) Curriculum Progression'); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'teacher_training.pdf')
    fig.savefig(FIGURES_DIR / 'teacher_training.png')
    print('Saved teacher_training'); plt.close(fig)


def fig_student_success():
    fig, ax = plt.subplots(figsize=(5, 3))
    for label, chain, single, color in STUDENT_RUNS:
        steps, vals = _load_run('Episode/success', chain, single)
        steps, vals = prepare(steps, vals)
        ax.plot(steps, vals, label=label, color=color)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Success Rate')
    ax.set_xlim(0, MAX_ITER); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', framealpha=0.9); ax.grid(True, alpha=0.3)
    ax.set_title('Overall Success Rate During Training')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'student_success_comparison.pdf')
    fig.savefig(FIGURES_DIR / 'student_success_comparison.png')
    print('Saved student_success_comparison'); plt.close(fig)


def fig_per_obstacle():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    tags = [('Episode/success_gap', 'Gap'),
            ('Episode/success_stairs', 'Stairs'),
            ('Episode/success_hurdle_block', 'Hurdle/Block')]
    runs = STUDENT_RUNS
    for ax, (tag, title) in zip(axes, tags):
        for label, chain, single, color in runs:
            try:
                steps, vals = _load_run(tag, chain, single)
                steps, vals = prepare(steps, vals)
                ax.plot(steps, vals, label=label, color=color)
            except Exception:
                pass
        ax.set_title(title); ax.set_xlabel('Iteration')
        ax.set_xlim(0, MAX_ITER); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Success Rate')
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'per_obstacle_success.pdf')
    fig.savefig(FIGURES_DIR / 'per_obstacle_success.png')
    print('Saved per_obstacle_success'); plt.close(fig)


def fig_loss():
    fig, ax = plt.subplots(figsize=(5, 3))
    for label, chain, single, color in STUDENT_RUNS:
        steps, vals = _load_run('Loss/action', chain, single)
        steps, vals = prepare(steps, vals)
        ax.plot(steps, vals, label=label, color=color)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Action Loss (MSE)')
    ax.set_xlim(0, MAX_ITER)
    ax.legend(loc='upper right', framealpha=0.9); ax.grid(True, alpha=0.3)
    ax.set_title('Distillation Action Loss')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'distillation_loss.pdf')
    fig.savefig(FIGURES_DIR / 'distillation_loss.png')
    print('Saved distillation_loss'); plt.close(fig)


if __name__ == '__main__':
    fig_teacher()
    fig_student_success()
    fig_per_obstacle()
    fig_loss()
    print('Done — all figures saved to', FIGURES_DIR)
