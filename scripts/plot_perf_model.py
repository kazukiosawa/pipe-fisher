import argparse
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

sns.set_theme(style="white")
sns.set(font_scale=2)


def plot(bs, ax1, ax2, depth_list, ylabel=False):
    x = np.arange(len(depth_list))

    # time
    width = 0.15
    ax = ax1
    bc_list = []
    for i, depth in enumerate(depth_list):
        count_f = depth if args.chimera else 2 * depth - 1
        count_b = 2 * depth - 2 if args.chimera else 2 * depth - 1
        n_micro = depth
        time_f = data['time_f'][bs] / 1000
        time_b = data['time_b'][bs] / 1000
        time_kron = data['time_kron'][bs] / 1000
        time_inv = data['time_inv'][bs] / 1000
        time_prec = data['time_prec'][bs] / 1000
        time_pipe = count_f * time_f + count_b * time_b
        time_bubble = time_pipe - n_micro * (time_f + time_b)
        ax.bar(x[i] - 2 * width, count_f * time_f, width=width, label='fwd', color='C0')
        ax.bar(x[i] - 2 * width, count_b * time_b, bottom=count_f * time_f, width=width, label='bwd', color='C1')
        ax.bar(x[i] - 2 * width, time_prec, bottom=time_pipe, width=width, label='prec', color='C8')
        ax.bar(x[i] - width, time_bubble, width=width, label='bubble', color='gray', hatch='/')

        # recompute
        time_pipe = count_f * time_f + count_b * (time_f + time_b)
        time_f_all = (count_f + count_b) * time_f
        time_b_all = count_b * time_b
        ax.bar(x[i], time_f_all, width=width, color='C0')
        ax.bar(x[i], time_b_all, bottom=time_f_all, width=width, color='C1')
        bc1 = ax.bar(x[i], time_prec, bottom=time_pipe, width=width, color='C8')
        time_bubble = time_pipe - n_micro * (time_f + time_f + time_b)
        bc2 = ax.bar(x[i] + width, time_bubble, width=width, color='gray', hatch='/')
        bc_list.extend([bc1, bc2])

        # kfac
        ax.bar(x[i] + 2 * width, n_micro * time_kron, width=width, label='curv', color='C2')
        ax.bar(x[i] + 2 * width, time_inv, bottom=n_micro * time_kron, width=width, label='inv', color='C4')
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=18, loc='upper left', ncol=2)
    ymax = ax.get_ylim()[1]
    for bc in bc_list:
        for rect in bc.patches:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() + 0.02 * ymax,
                    'R', ha='center')
    ax.set_title(f'B_micro={bs}',)
    if ylabel:
        ax.set_ylabel('Time (s)')
    ax.set_ylim(bottom=0, top=ymax * 1.05)
    ax.set_xticks(x, [f'D={depth}' for depth in depth_list])

    # memory
    width = 0.3
    ax = ax2
    bc_list = []
    for i, depth in enumerate(depth_list):
        n_micro = depth
        mem = {
            'act': data['mem_act'][bs] * n_micro,
            'peak_err': data['mem_peak_err'][bs],
            'save_err': data['mem_save_err'][bs] * n_micro,
            'curv+inv': data['mem_kron'][bs] * 2,
            'param+grad': data['mem_param'][bs] * 2,
        }
        colors = ['C0', 'C3', 'C1', 'C2', 'C6']
        if args.chimera:
            mem['param+grad'] *= 2
        # MB -> GB
        for key in mem:
            mem[key] /= 1000

        bottom = 0
        for j, key in enumerate(mem):
            ax.bar(x[i] - width / 2, mem[key], bottom=bottom, width=width, label=key, color=colors[j])
            bottom += mem[key]

        # recompute
        mem['act'] = data['mem_act'][bs] / 1000
        bottom = 0
        bc = None
        for j, key in enumerate(mem):
            bc = ax.bar(x[i] + width / 2, mem[key], bottom=bottom, width=width, color=colors[j])
            bottom += mem[key]
        bc_list.append(bc)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=18, loc='upper left', ncol=2)
    ymax = ax.get_ylim()[1]
    for bc in bc_list:
        for rect in bc.patches:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() + 0.02 * ymax,
                    'R', ha='center')
    ax.set_title(f'B_micro={bs}',)
    if ylabel:
        ax.set_ylabel('Memory (GB)')
    ax.set_ylim(bottom=0, top=ymax * 1.05)
    ax.set_xticks(x, [f'D={depth}' for depth in depth_list])


def main():
    bs_list = [8, 16, 32]
    depth_list = [4, 8, 16]
    n_cols = len(bs_list)
    n_rows = 2
    fig = plt.figure(figsize=(8 * n_cols, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    for i, bs in enumerate(bs_list):
        ax1 = fig.add_subplot(gs[0, i])
        ax2 = fig.add_subplot(gs[1, i])
        plot(bs, ax1, ax2, depth_list, ylabel=True)

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
