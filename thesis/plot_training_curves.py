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
LOG_ROOT = Path('logs')

# --- Helpers ---

def load_scalar(logdir, tag):
    """Return (steps, values) arrays from a TB log."""
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals

def smooth(vals, window=51):
    """Simple moving average."""
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode='same')

# --- Run paths ---
TEACHER = LOG_ROOT / 'go2_parkour_teacher/Mar26_05-05-21_teacher_genesis'
GT_SCANDOT = LOG_ROOT / 'go2_parkour_scandot_student/Mar31_22-48-40_scandot_student_genesis'
DA2_BASE_TEX = LOG_ROOT / 'go2_parkour_depth_est_student/Mar29_13-36-08_BASE3_da2_base_texture'
DA2_BASE = LOG_ROOT / 'go2_parkour_depth_est_student/Mar29_08-06-01_BASE1_da2_base'
DA2_SMALL = LOG_ROOT / 'go2_parkour_depth_est_student/Mar27_21-12-08_S14_diff_lr_cosine'
RESNET_RGB = LOG_ROOT / 'go2_parkour_resnet_rgb_scandot_student/Apr01_23-47-30_resnet_rgb_scandot_student_genesis'

COLORS = {
    'teacher': '#1b9e77',
    'gt_scandot': '#333333',
    'da2_base_tex': '#d95f02',
    'da2_base': '#7570b3',
    'da2_small': '#e7298a',
    'resnet_rgb': '#66a61e',
}

# ============================================================
# Figure 1: Teacher training curve
# ============================================================
def fig_teacher():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    steps, reward = load_scalar(TEACHER, 'Train/mean_reward')
    ax1.plot(steps, smooth(reward, 31), color=COLORS['teacher'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('(a) Reward')
    ax1.grid(True, alpha=0.3)

    steps, success = load_scalar(TEACHER, 'Episode/success')
    ax1b = ax1.twinx()
    ax1b.plot(steps, smooth(success, 31), color=COLORS['teacher'], linestyle='--', alpha=0.5)
    ax1b.set_ylabel('Success Rate', color='grey')
    ax1b.set_ylim(-0.05, 1.05)

    steps, tlevel = load_scalar(TEACHER, 'Episode/terrain_level')
    ax2.plot(steps, smooth(tlevel, 31), color=COLORS['teacher'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Terrain Level')
    ax2.set_title('(b) Curriculum Progression')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'teacher_training.pdf')
    fig.savefig(FIGURES_DIR / 'teacher_training.png')
    print('Saved teacher_training')
    plt.close(fig)


# ============================================================
# Figure 2: Student success rate comparison
# ============================================================
def fig_student_success():
    fig, ax = plt.subplots(figsize=(5, 3))

    runs = [
        ('Scandot student (GT)', GT_SCANDOT, COLORS['gt_scandot'], '-'),
        ('DA2-base + texture', DA2_BASE_TEX, COLORS['da2_base_tex'], '-'),
        ('DA2-base (no texture)', DA2_BASE, COLORS['da2_base'], '-'),
        ('DA2-small', DA2_SMALL, COLORS['da2_small'], '-'),
        ('ResNet-18 RGB', RESNET_RGB, COLORS['resnet_rgb'], '-'),
    ]

    for label, logdir, color, ls in runs:
        steps, vals = load_scalar(logdir, 'Episode/success')
        ax.plot(steps, smooth(vals, 101), label=label, color=color, linestyle=ls)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Overall Success Rate During Training')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'student_success_comparison.pdf')
    fig.savefig(FIGURES_DIR / 'student_success_comparison.png')
    print('Saved student_success_comparison')
    plt.close(fig)


# ============================================================
# Figure 3: Per-obstacle success — DA2-base+tex vs DA2-base
# ============================================================
def fig_per_obstacle():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    obstacle_tags = [
        ('Episode/success_gap', 'Gap'),
        ('Episode/success_stairs', 'Stairs'),
        ('Episode/success_hurdle_block', 'Hurdle/Block'),
    ]

    runs = [
        ('Scandot student (GT)', GT_SCANDOT, COLORS['gt_scandot']),
        ('DA2-base + texture', DA2_BASE_TEX, COLORS['da2_base_tex']),
        ('DA2-base (no texture)', DA2_BASE, COLORS['da2_base']),
    ]

    for ax, (tag, title) in zip(axes, obstacle_tags):
        for label, logdir, color in runs:
            try:
                steps, vals = load_scalar(logdir, tag)
                ax.plot(steps, smooth(vals, 101), label=label, color=color)
            except Exception:
                pass
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Success Rate')
    axes[0].legend(loc='lower right', fontsize=7, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'per_obstacle_success.pdf')
    fig.savefig(FIGURES_DIR / 'per_obstacle_success.png')
    print('Saved per_obstacle_success')
    plt.close(fig)


# ============================================================
# Figure 4: Distillation loss comparison
# ============================================================
def fig_loss():
    fig, ax = plt.subplots(figsize=(5, 3))

    runs = [
        ('Scandot student (GT)', GT_SCANDOT, COLORS['gt_scandot']),
        ('DA2-base + texture', DA2_BASE_TEX, COLORS['da2_base_tex']),
        ('DA2-base (no texture)', DA2_BASE, COLORS['da2_base']),
        ('DA2-small', DA2_SMALL, COLORS['da2_small']),
        ('ResNet-18 RGB', RESNET_RGB, COLORS['resnet_rgb']),
    ]

    for label, logdir, color in runs:
        try:
            steps, vals = load_scalar(logdir, 'Loss/action')
            ax.plot(steps, smooth(vals, 101), label=label, color=color)
        except Exception:
            pass

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Action Loss (MSE)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Distillation Action Loss')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'distillation_loss.pdf')
    fig.savefig(FIGURES_DIR / 'distillation_loss.png')
    print('Saved distillation_loss')
    plt.close(fig)


if __name__ == '__main__':
    fig_teacher()
    fig_student_success()
    fig_per_obstacle()
    fig_loss()
    print('Done — all figures saved to', FIGURES_DIR)
