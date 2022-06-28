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


def plot(micro_bs, ax, depth_list, throughput=True, ylabel=False):
    x = np.arange(len(depth_list))
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

    for i, depth in enumerate(depth_list):
        count_f = depth if args.chimera else 2 * depth - 1
        count_b = 2 * depth - 2 if args.chimera else 2 * depth - 1
        n_micro = depth
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
                y[i] = depth_list[i] * micro_bs / y[i] * 1000  # ms -> s

        ax.plot(x, y_base, label=base_name, marker='.', ms=20, color='gray')
        ax.plot(x, y_recomp, label=f'{base_name} (R)', marker='.', ms=20, color='gray', ls='--')
        ax.plot(x, y_kfac_save_err, label='PipeFisher', marker='.', ms=20, color='red')
        ax.plot(x, y_recomp_kfac_save_err, label='PipeFisher (R)', marker='.', ms=20, color='red', ls='--')
#        ax.plot(x, y_base, label='base', marker='.', ms=20, color=colors[0])
#        ax.plot(x, y_recomp, label='base (R)', marker='.', ms=20, color=colors[1])
#        ax.plot(x, y_kfac_save_err, label='save err', marker='.', ms=20, color=colors[2])
#        ax.plot(x, y_kfac_no_save, label='no save', marker='.', ms=20, color=colors[3])
#        ax.plot(x, y_recomp_kfac_save_err, label='save err, R', marker='.', ms=20, color=colors[4])
#        ax.plot(x, y_recomp_kfac_no_save, label='no save, R', marker='.', ms=20, color=colors[5])
        if ylabel:
            ax.set_ylabel('Throughput (seqs/s)')
    else:
        ax.plot(x, pipes_kfac_save_err, label='PipeFisher', marker='.', ms=20, color='red')
        ax.plot(x, pipes_recomp_kfac_save_err, label='PipeFisher (R)', marker='.', ms=20, color='red', ls='--')
#        ax.plot(x, pipes_kfac_save_err, label='save err', marker='.', ms=20, color=colors[2])
#        ax.plot(x, pipes_kfac_no_save, label='no save', marker='.', ms=20, color=colors[3])
#        ax.plot(x, pipes_recomp_kfac_save_err, label='save err, R', marker='.', ms=20, color=colors[4])
#        ax.plot(x, pipes_recomp_kfac_no_save, label='no save, R', marker='.', ms=20, color=colors[5])
        if ylabel:
            ax.set_ylabel('(curv+inv) / bubble')

    ax.set_title(f'B_micro={micro_bs}', )
    ax.set_xticks(x)
    ax.set_xticklabels([f'D={depth}' for depth in depth_list])
    ax.set_ylim(bottom=0)


def main():
    micro_bs_list = [8, 16, 32]
    depth_list = [4, 8, 16]
    n_cols = len(micro_bs_list)
    fig = plt.figure(figsize=(8 * n_cols, 8))
    gs = fig.add_gridspec(2, n_cols)

    for i, micro_bs in enumerate(micro_bs_list):
        ax1 = fig.add_subplot(gs[0, i])
        plot(micro_bs, ax1, depth_list, ylabel=True)
        ax1.legend(loc='lower right', fontsize=18, ncol=2)
        ax2 = fig.add_subplot(gs[1, i])
        plot(micro_bs, ax2, depth_list, throughput=False, ylabel=True)
        ax2.legend(loc='upper right', fontsize=18)

    plt.tight_layout()
    plt.savefig(args.fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--fig_path', type=str)
    parser.add_argument('--chimera', action='store_true')
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, header=0, index_col=0).to_dict()
    main()
