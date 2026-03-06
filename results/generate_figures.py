#!/usr/bin/env python3
"""
Generate publication-ready figures with English labels.
Reproduces key paper figures from analysis outputs.

Usage: python generate_figures.py
Output: results/figures/fig_*.png
"""

import os
import sys
import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def fig_balance_model():
    """Figure 1: The cognitive obesity balance model (conceptual)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    # Balance equation visualization
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.5, r'$L = \alpha_1 \cdot I - \alpha_2 \cdot C$',
            fontsize=24, ha='center', va='center', fontfamily='serif', style='italic')

    # Term 1
    ax.add_patch(plt.Rectangle((0.5, 2), 3.5, 2.5, facecolor='#fee2e2',
                                edgecolor='#dc2626', linewidth=1.5, zorder=1))
    ax.text(2.25, 3.9, r'Term 1: $\alpha_1 \cdot I$', fontsize=12,
            ha='center', va='center', fontweight='bold', color='#dc2626')
    ax.text(2.25, 3.2, 'Cognitive Input', fontsize=10, ha='center', va='center')
    ax.text(2.25, 2.6, 'Low loop closure:\nScrolling, notifications,\nalgorithmic feeds, ads',
            fontsize=8, ha='center', va='center', color='#666')

    # Minus sign
    ax.text(4.75, 3.25, '−', fontsize=28, ha='center', va='center', fontweight='bold')

    # Term 2
    ax.add_patch(plt.Rectangle((5.5, 2), 3.5, 2.5, facecolor='#dbeafe',
                                edgecolor='#2563eb', linewidth=1.5, zorder=1))
    ax.text(7.25, 3.9, r'Term 2: $\alpha_2 \cdot C$', fontsize=12,
            ha='center', va='center', fontweight='bold', color='#2563eb')
    ax.text(7.25, 3.2, 'Experiential Processing', fontsize=10, ha='center', va='center')
    ax.text(7.25, 2.6, 'High loop closure:\nExercise, creative work,\nface-to-face, embodied play',
            fontsize=8, ha='center', va='center', color='#666')

    # Bottom note
    ax.text(5, 1.2, 'Imbalance (L > 0) → cognitive obesity → depression risk',
            fontsize=9, ha='center', va='center', style='italic', color='#444')
    ax.text(5, 0.7, 'Both terms contribute independently (additive model; interaction p = 0.95)',
            fontsize=8, ha='center', va='center', color='#888')

    fig.savefig(os.path.join(OUTDIR, 'fig01_balance_model.png'))
    plt.close()
    print("  ✓ fig01_balance_model.png")


def fig_twfe_reversal():
    """Figure 2: TWFE sign reversal (schematic)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Country FE
    ax = axes[0]
    np.random.seed(42)
    x = np.linspace(20, 90, 50)
    y = 3.5 + 0.01 * x + np.random.normal(0, 0.3, 50)
    ax.scatter(x, y, c='#3b82f6', alpha=0.5, s=20)
    z = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(z, x), 'r-', linewidth=2)
    ax.set_xlabel('Internet Penetration (%)')
    ax.set_ylabel('Depression Prevalence (rate)')
    ax.set_title('A: Country FE\nβ = +0.074 (t = 7.30)', fontweight='bold')
    ax.text(0.05, 0.95, 'Positive\nassociation', transform=ax.transAxes,
            fontsize=9, va='top', color='red', fontweight='bold')

    # Panel B: TWFE
    ax = axes[1]
    y2 = 5.0 - 0.005 * x + np.random.normal(0, 0.3, 50)
    ax.scatter(x, y2, c='#3b82f6', alpha=0.5, s=20)
    z2 = np.polyfit(x, y2, 1)
    ax.plot(x, np.polyval(z2, x), 'b-', linewidth=2)
    ax.set_xlabel('Internet Penetration (%)')
    ax.set_ylabel('Depression Prevalence (rate)')
    ax.set_title('B: Two-Way FE (+ Year FE)\nβ = −0.031 (t = −4.86)', fontweight='bold')
    ax.text(0.05, 0.95, 'Sign reversal\n(common trends\nremoved)', transform=ax.transAxes,
            fontsize=9, va='top', color='blue', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig02_twfe_reversal.png'))
    plt.close()
    print("  ✓ fig02_twfe_reversal.png")


def fig_first_difference():
    """Figure 3: First-difference horse race (Δproxy vs ΔGDP)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    variables = ['Δproxy\n(ad ecosystem)', 'ΔGDP\n(pure growth)']
    t_values = [3.19, 0.16]
    colors = ['#dc2626', '#94a3b8']
    significance = ['***', 'n.s.']

    bars = ax.barh(variables, t_values, color=colors, height=0.5, edgecolor='white')
    ax.axvline(x=1.96, color='gray', linestyle='--', alpha=0.5, label='t = 1.96 (p = 0.05)')
    ax.axvline(x=-1.96, color='gray', linestyle='--', alpha=0.5)

    for i, (bar, sig) in enumerate(zip(bars, significance)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f't = {t_values[i]:.2f} {sig}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('t-statistic')
    ax.set_title('First-Difference Identification:\nAd Proxy Survives GDP Control', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.5, 4.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig03_first_difference.png'))
    plt.close()
    print("  ✓ fig03_first_difference.png")


def fig_atus_quadrant():
    """Figure 4: ATUS 2×2 quadrant (exercise × cognitive leisure)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    labels = ['Both\npresent', 'Exercise\nonly', 'Cognitive\nleisure only', 'Neither\npresent']
    cantril = [7.46, 7.35, 7.13, 7.02]
    fair_poor = [6.7, 9.9, 17.1, 20.9]
    colors_c = ['#22c55e', '#86efac', '#fbbf24', '#ef4444']
    colors_h = ['#22c55e', '#86efac', '#fbbf24', '#ef4444']

    # Panel A: Cantril
    ax = axes[0]
    bars = ax.bar(labels, cantril, color=colors_c, edgecolor='white', width=0.6)
    for bar, val in zip(bars, cantril):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Cantril Ladder (0-10)')
    ax.set_title('A: Life Satisfaction', fontweight='bold')
    ax.set_ylim(6.8, 7.6)

    # Panel B: Fair/Poor
    ax = axes[1]
    bars = ax.bar(labels, fair_poor, color=colors_h, edgecolor='white', width=0.6)
    for bar, val in zip(bars, fair_poor):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Fair/Poor Health Rate (%)')
    ax.set_title('B: Self-Rated Health', fontweight='bold')
    ax.set_ylim(0, 25)

    fig.suptitle('ATUS 2×2 Contrapositive Test (N = 21,736)', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig04_atus_quadrant.png'))
    plt.close()
    print("  ✓ fig04_atus_quadrant.png")


def fig_loop_closure_spectrum():
    """Figure 5: Sensorimotor loop closure spectrum."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn',
              extent=[0.5, 9.5, 1.2, 1.8], zorder=1)

    # Labels
    ax.text(0.5, 2.2, 'LOW CLOSURE\n(Term 1)', fontsize=9, ha='left',
            fontweight='bold', color='#dc2626')
    ax.text(9.5, 2.2, 'HIGH CLOSURE\n(Term 2)', fontsize=9, ha='right',
            fontweight='bold', color='#16a34a')

    # Activity examples
    activities = [
        (1.0, 'Rumination'),
        (2.0, 'SNS scrolling'),
        (3.0, 'Passive video'),
        (4.5, 'Podcast\n(passive)'),
        (5.5, 'Narrative\nRPG'),
        (6.5, 'Podcast\n(active)'),
        (7.5, 'Creative\nwriting'),
        (8.5, 'Competitive\ngaming'),
        (9.0, 'Exercise'),
    ]

    for x, label in activities:
        ax.plot(x, 1.5, 'ko', markersize=6, zorder=3)
        ax.text(x, 0.7, label, fontsize=7, ha='center', va='top')

    ax.text(5, 2.8, 'Sensorimotor Loop Closure Spectrum', fontsize=12,
            ha='center', fontweight='bold')
    ax.text(5, 0.1, 'Note: Same activity can shift along spectrum depending on engagement mode',
            fontsize=7, ha='center', color='#888', style='italic')

    fig.savefig(os.path.join(OUTDIR, 'fig05_loop_closure_spectrum.png'))
    plt.close()
    print("  ✓ fig05_loop_closure_spectrum.png")


def fig_game_typology():
    """Figure 6: Game typology by loop closure."""
    fig, ax = plt.subplots(figsize=(8, 4))

    categories = ['Gacha/Idle\nGrinding', 'Level-up\nGrinding', 'Narrative\nRPG', 'Strategy/\nPuzzle', 'Competitive\nFPS/Fighting', 'Creative\n(Minecraft)']
    closure = [1.5, 2.5, 4.5, 6.5, 8.0, 9.0]
    colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#0ea5e9', '#8b5cf6']

    bars = ax.barh(categories, closure, color=colors, height=0.6, edgecolor='white')
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.3)
    ax.text(2.5, -0.8, '← Term 1 (Input)', fontsize=9, ha='center', color='#dc2626')
    ax.text(7.5, -0.8, 'Term 2 (Processing) →', fontsize=9, ha='center', color='#16a34a')

    ax.set_xlabel('Loop Closure Degree')
    ax.set_title('Video Game Typology by Sensorimotor Loop Closure', fontweight='bold')
    ax.set_xlim(0, 10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'fig06_game_typology.png'))
    plt.close()
    print("  ✓ fig06_game_typology.png")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig_balance_model()
    fig_twfe_reversal()
    fig_first_difference()
    fig_atus_quadrant()
    fig_loop_closure_spectrum()
    fig_game_typology()
    print(f"\nAll figures saved to {OUTDIR}/")
