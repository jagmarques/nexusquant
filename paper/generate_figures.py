"""Generate all paper figures — NeurIPS style, matching document font."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Match NeurIPS document: Computer Modern serif, small sizes
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'axes.linewidth': 0.4,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'grid.linewidth': 0.3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

COL_WIDTH = 5.5  # NeurIPS text width in inches


def fig2_pareto():
    """Pareto frontier — NexusQuant vs competitors."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 3.0))

    nq_short_ratio = [6.6, 9.8, 10.1, 13.0, 21.3, 32.0]
    nq_short_ppl =   [1.10, 0.95, 0.90, 3.63, 4.81, 6.58]
    nq_long_ratio =  [6.8, 10.4, 13.4, 16.6, 22.1, 32.7]
    nq_long_ppl =    [0.28, 0.14, 0.56, 0.82, 1.52, 2.13]

    ax.plot(nq_short_ratio, nq_short_ppl, 'o-', color='#3498db',
            label='NexusQuant (500-tok prefix)', zorder=5)
    ax.plot(nq_long_ratio, nq_long_ppl, 's-', color='#2c3e50',
            label='NexusQuant (1664-tok prefix)', zorder=6)

    # Competitors
    ax.scatter([5.5], [0.1], marker='s', color='#e74c3c', s=25, zorder=4, label='TurboQuant')
    ax.scatter([8.0], [0.1], marker='D', color='#f39c12', s=25, zorder=4, label='CommVQ')
    ax.scatter([20.0], [0.8], marker='p', color='#2ecc71', s=30, zorder=4, label='KVTC')
    # Palu off-chart
    ax.scatter([11.4], [9.3], marker='^', color='#9b59b6', s=25, zorder=4)
    ax.annotate('Palu (~25%)', xy=(11.4, 9.3), fontsize=5.5, ha='center', va='bottom', color='#9b59b6')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.text(34, 1.15, '<1% threshold', fontsize=5.5, color='gray', ha='right')

    ax.set_xlabel('Compression ratio')
    ax.set_ylabel('PPL degradation (%)')
    ax.set_xlim(3, 35)
    ax.set_ylim(-0.5, 10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.15)

    plt.tight_layout(pad=0.3)
    fig.savefig('fig2_pareto_final.pdf', bbox_inches='tight')
    fig.savefig('fig2_pareto_final.png', bbox_inches='tight')
    print("fig2_pareto_final.pdf")
    plt.close()


def fig4_context_scaling():
    """Context-length scaling — the key finding."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 3.0))

    evict_pcts = [0, 35, 50, 60, 70, 80]
    ppl_500 =  [1.10, 0.90, 3.63, 4.53, 4.81, 6.58]
    ppl_1664 = [0.28, 0.14, 0.56, 0.82, 1.52, 2.13]

    ax.plot(evict_pcts, ppl_500, 'o-', color='#e74c3c', label='500-token prefix')
    ax.plot(evict_pcts, ppl_1664, 's-', color='#2c3e50', label='1664-token prefix')

    ax.fill_between(evict_pcts, ppl_500, ppl_1664, alpha=0.08, color='#3498db',
                    label='Quality gain from longer context')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.text(82, 1.12, '<1%', fontsize=6, color='gray')

    ax.annotate('10x, <1%', xy=(35, 0.90), fontsize=6, color='#e74c3c',
               ha='center', xytext=(28, 2.2),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=0.6))
    ax.annotate('16.6x, <1%', xy=(60, 0.82), fontsize=6, color='#2c3e50',
               ha='center', xytext=(67, 2.2),
               arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=0.6))

    ax.set_xlabel('Eviction rate (%)')
    ax.set_ylabel('PPL degradation (%)')
    ax.set_xlim(-5, 85)
    ax.set_ylim(-0.3, 7.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.15)

    plt.tight_layout(pad=0.3)
    fig.savefig('fig4_context_scaling.pdf', bbox_inches='tight')
    fig.savefig('fig4_context_scaling.png', bbox_inches='tight')
    print("fig4_context_scaling.pdf")
    plt.close()


def fig3_quality_cliff():
    """Fine-grained quality cliff: PPL delta vs eviction rate, easy vs hard text."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 3.0))

    evict_rates = [32, 33, 34, 35, 36, 37, 38]
    easy_ppl = [0.90, 0.60, 0.60, 0.53, 0.51, 1.62, 1.94]
    hard_ppl = [0.90, 0.95, 1.00, 0.99, 1.22, 1.39, 1.57]

    # Shade the <1% quality zone
    ax.axhspan(0, 1.0, color='#2ecc71', alpha=0.07, zorder=0)
    ax.text(32.2, 0.06, '<1% quality threshold', fontsize=5.5, color='#27ae60', va='bottom')

    ax.plot(evict_rates, easy_ppl, 'o-', color='#3498db', label='Easy text (general science)', zorder=5)
    ax.plot(evict_rates, hard_ppl, 's-', color='#e74c3c', label='Hard text (advanced technical)', zorder=5)

    # Mark the cliff at 36%
    ax.axvline(x=36, color='#e67e22', linestyle='--', alpha=0.7, linewidth=0.7, zorder=3)
    ax.text(36.15, 1.85, 'Cliff at 36%', fontsize=5.5, color='#e67e22', va='bottom')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)

    ax.set_xlabel('Eviction rate (%)')
    ax.set_ylabel('PPL degradation (%)')
    ax.set_xlim(31, 39)
    ax.set_ylim(0, 2.3)
    ax.set_xticks(evict_rates)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.15)

    plt.tight_layout(pad=0.3)
    fig.savefig('fig3_quality_cliff.pdf', bbox_inches='tight')
    fig.savefig('fig3_quality_cliff.png', bbox_inches='tight')
    print("fig3_quality_cliff.pdf")
    plt.close()


def fig6_cross_arch():
    """Cross-architecture comparison: Mistral-7B vs Llama-3-8B PPL delta."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 2.6))

    # Eviction rates and PPL deltas
    evict_labels = ['0%\n(quant only)', '35%', '60%', '80%']
    mistral_ppl = [0.28, 0.14, 0.82, 2.13]
    llama3_ppl  = [0.85, 1.05, 1.50, 3.15]

    x = np.arange(len(evict_labels))
    width = 0.35

    bars_m = ax.bar(x - width/2, mistral_ppl, width, label='Mistral-7B',
                    color='#2c3e50', alpha=0.85, edgecolor='white', linewidth=0.3)
    bars_l = ax.bar(x + width/2, llama3_ppl, width, label='Llama-3-8B',
                    color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.3)

    # Value labels on bars
    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'+{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=5.5, color='#2c3e50')
    for bar in bars_l:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'+{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=5.5, color='#2980b9')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.text(3.5, 1.06, '<1%', fontsize=5.5, color='gray', ha='right')

    ax.set_ylabel('PPL degradation (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(evict_labels, fontsize=6.5)
    ax.set_xlabel('Eviction rate (~1500-tok prefix)')
    ax.set_ylim(0, 3.8)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.12, axis='y')

    plt.tight_layout(pad=0.3)
    fig.savefig('fig6_cross_arch.pdf', bbox_inches='tight')
    fig.savefig('fig6_cross_arch.png', bbox_inches='tight')
    print("fig6_cross_arch.pdf")
    plt.close()


def fig5_ablation():
    """Ablation: eviction vs quantization vs combined."""
    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH, 2.6))

    configs = ['Quant only\n(2-bit E8)', 'Evict only\n(35%)', 'Combined\n(2b+35%)',
               'Evict only\n(60%)', 'Combined\n(2b+60%)', 'Combined\n(2b+80%)']
    ratios = [6.6, 1.54, 10.1, 2.50, 16.6, 32.7]
    ppls =   [1.10, 0.0, 0.90, 0.0, 0.82, 2.13]
    colors = ['#3498db', '#e74c3c', '#2c3e50', '#e74c3c', '#2c3e50', '#95a5a6']

    x = np.arange(len(configs))
    bars = ax.bar(x, ratios, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3, width=0.7)

    for bar, ppl in zip(bars, ppls):
        label = f'+{ppl:.1f}%' if ppl > 0 else 'mask\nonly'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4,
                label, ha='center', va='bottom', fontsize=5.5,
                color='#2c3e50' if ppl <= 1 else '#e74c3c')

    ax.set_ylabel('Compression ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=6)
    ax.set_ylim(0, 38)
    ax.grid(True, alpha=0.12, axis='y')

    plt.tight_layout(pad=0.3)
    fig.savefig('fig5_ablation.pdf', bbox_inches='tight')
    fig.savefig('fig5_ablation.png', bbox_inches='tight')
    print("fig5_ablation.pdf")
    plt.close()


if __name__ == '__main__':
    fig2_pareto()
    fig3_quality_cliff()
    fig4_context_scaling()
    fig5_ablation()
    fig6_cross_arch()
    print("All figures generated.")
