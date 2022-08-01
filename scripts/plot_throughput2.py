import argparse
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sns.set_theme(style="white")
sns.set(font_scale=2)


def plot(depth, n_micro, ax, color, ms, alpha, throughput=True, ylabel=False):
    y_base = []
    y_recomp = []
    y_kfac_save_err = []
    y_kfac_no_save = []
    y_recomp_kfac_save_err = []
    y_recomp_kfac_no_save = []

    pipes_kfac_save_err = []
    pipes_kfac_no_save = []
    pipes_recomp_kfac_save_err = []
    pipes_recomp_kfac_no_save = []

    for micro_bs in micro_bs_list:
        if args.chimera:
            bubble_f = 0 if depth == n_micro else depth/2 - 1
            bubble_b = depth - 2 if depth == n_micro else depth/2 - 1
        else:
            bubble_f = depth - 1
            bubble_b = depth - 1
        count_f = bubble_f + n_micro
        count_b = bubble_b + n_micro
        time_f = data['time_f'][micro_bs]
        time_b = data['time_b'][micro_bs]
        time_kron = data['time_kron'][micro_bs]
        time_inv = data['time_inv'][micro_bs]
        time_prec = data['time_prec'][micro_bs]
        time_pipe = count_f * time_f + count_b * time_b
        time_bubble = time_pipe - n_micro * (time_f + time_b)

        y_base.append(time_pipe)
        y_kfac_save_err.append(time_pipe + time_prec)
        y_kfac_no_save.append(time_pipe + time_prec + time_kron * n_micro)
        pipes_kfac_save_err.append((time_inv + time_kron * n_micro) / time_bubble)
        pipes_kfac_no_save.append(time_inv / time_bubble)

        # recompute
        time_pipe = count_f * time_f + count_b * (time_f + time_b)
        time_bubble = time_pipe - n_micro * (time_f + time_f + time_b)
        y_recomp.append(time_pipe)
        y_recomp_kfac_save_err.append(time_pipe + time_prec)
        y_recomp_kfac_no_save.append(time_pipe + time_prec + time_kron * n_micro)
        pipes_recomp_kfac_save_err.append((time_inv + time_kron * n_micro) / time_bubble)
        pipes_recomp_kfac_no_save.append(time_inv / time_bubble)

    base_name = 'Chimera' if args.chimera else 'GPipe/1F1B'

    if throughput:
        for y in [y_base, y_recomp, y_kfac_save_err, y_kfac_no_save, y_recomp_kfac_save_err, y_recomp_kfac_no_save]:
            for i in range(len(y)):
                # throughput (sequences/s)
                y[i] = n_micro * micro_bs_list[i] / y[i] * 1000  # ms -> s

#        ax.plot(x, y_base, marker='.', color=color, ms=ms, alpha=alpha, ls='--')
#        ax.plot(x, y_recomp, label=f'{base_name} (R)', marker='.', ms=20, color='gray', ls='--')
        ax.plot(x, y_kfac_save_err, label=f'D={depth}, N_micro={n_micro}', marker='.', color=color, ms=ms, alpha=alpha)
#        ax.plot(x, y_recomp_kfac_save_err, label='PipeFisher (R)', marker='.', ms=20, color='red', ls='--')
        if ylabel:
            ax.set_ylabel('Throughput (seqs/s)')
    else:
        ax.plot(x, pipes_kfac_save_err, label=f'D={depth}, N_micro={n_micro}', marker='.', color=color, ms=ms, alpha=alpha)
#        ax.plot(x, pipes_recomp_kfac_save_err, label='PipeFisher (R)', marker='.', ms=20, color='red', ls='--')
        if ylabel:
            ax.set_ylabel('(curv+inv) / bubble')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--fig_path', type=str)
    parser.add_argument('--chimera', action='store_true')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8')
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, header=0, index_col=0).to_dict()

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(2, 1)

    micro_bs_list = [int(s) for s in args.batch_sizes.split(',')]
    depth_list = [4, 8, 16, 32]
    n_micro_scale_list = [3, 2, 1]

    x = np.arange(len(micro_bs_list))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    alphas = [1, 0.5, 0.25]
    marker_sizes = [20, 15, 10]

    for i, depth in enumerate(depth_list):
        color = colors[i]
        for j, n_micro_scale in enumerate(n_micro_scale_list):
            alpha = alphas[j]
            ms = marker_sizes[j]
            n_micro = depth * n_micro_scale
            plot(depth, n_micro, ax1, color, ms, alpha, ylabel=True)
            ax1.legend(loc='lower right', fontsize=18, ncol=len(depth_list))
            plot(depth, n_micro, ax2, color, ms, alpha, throughput=False, ylabel=True)
#            ax2.legend(loc='upper right', fontsize=18, ncol=len(depth_list))

    for ax in [ax1, ax2]:
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
#        ax.set_xticklabels([f'B_micro={micro_bs}' for micro_bs in micro_bs_list])
        ax.set_xticklabels([micro_bs for micro_bs in micro_bs_list])
    ax2.set_xlabel('Micro-batch size (B_micro)')

    plt.tight_layout()
    plt.savefig(args.fig_path)
