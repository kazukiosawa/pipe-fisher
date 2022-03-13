import argparse
import pickle
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


key_to_color_label = OrderedDict(
    {
        'call_forward': ('C0', 'forward'),
        'call_backward': ('C1', 'backward'),
        'cov_kron_A': ('C2', 'curvature'),
        'cov_kron_B': ('C2', None),
        'cov_unit_wise': ('C2', None),
        'inv_kron_A': ('C4', 'inv'),
        'inv_kron_B': ('C4', None),
        'inv_unit_wise': ('C4', None),
        'sync_grad': ('C7', 'sync-grad'),
        'nb_sync_grad': ('C7', 'sync-grad'),
        'reduce_scatter_curvature': ('C9', 'sync-curvature'),
        'all_reduce_undivided_curvature': ('C9', None),
        'reduce_scatter_grad': ('C7', 'sync-grad'),
        'all_reduce_undivided_grad': ('C7', None),
        'precondition': ('C8', 'precondition'),
        'all_gather_grad': ('C7', None),
        'all_reduce_no_curvature_grad': ('C7', None),
    }
)


def main():
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    min_time = sys.maxsize
    max_time = 0
    for start_end_list in timelines[0].values():
        for s, e in start_end_list:
            min_time = min(min_time, s)
            max_time = max(max_time, e)

    def time_shift(t):
        return (t - min_time) / 10 ** 6  # ns -> ms

    verts = []
    verts_alpha = []
    colors = []
    colors_alpha = []
    used_keys = set()
    width = .95
    for idx, timeline in enumerate(timelines):
        y = len(timelines) - idx - 1
        for i, event_txt in enumerate(timeline):
            if not any(key in event_txt for key in key_to_color_label):
                continue
            key = next(key for key in key_to_color_label if key in event_txt)
            used_keys.add(key)
            start_end_list = timeline[event_txt]
            for s, e in start_end_list:
                s = time_shift(s)
                e = time_shift(e)
                v = [(s, y-width/2), (s, y+width/2), (e, y+width/2), (e, y-width/2), (s, y-width/2)]
                color, label = key_to_color_label[key]
                if 'sync' in key:
                    verts_alpha.append(v)
                    colors_alpha.append(color)
                else:
                    verts.append(v)
                    colors.append(color)

    bars = PolyCollection(verts, facecolors=colors)
    ax.add_collection(bars)
    bars = PolyCollection(verts_alpha, facecolors=colors_alpha, alpha=0.5, hatch='//')
    ax.add_collection(bars)
    ax.autoscale()

    ax.set_xlabel('Time (ms)')
    ax.set_yticks(range(len(timelines)))
    ax.set_yticklabels([f'GPU {i}' for i in range(len(timelines))][::-1])
    ax.set_title(args.title)
    ax.set_xlim(time_shift(min_time), time_shift(max_time))

    for i, (start, _) in enumerate(timelines[0]['start_end'][1:]):
        ax.axvline(time_shift(start), color='r', lw=7, label='flush @ GPU0' if i == 0 else None)
    for key, (color, label) in key_to_color_label.items():
        if key in used_keys:
            if 'sync' in key:
                ax.bar(0, 0, label=label, color=color, alpha=0.5, hatch='//')
            else:
                ax.bar(0, 0, label=label, color=color)
    if len(used_keys) + 1 > 6:
        ax.legend(bbox_to_anchor=(0, 1.2), loc='upper left', ncol=6)
    else:
        ax.legend(bbox_to_anchor=(0, 1.15), loc='upper left', ncol=len(used_keys)+1)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_paths', type=str)
    parser.add_argument('--fig_path', type=str, default='prof.png')
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()

    timelines = []
    for pickle_path in args.pickle_paths.split(','):
        if pickle_path == '':
            continue
        timelines.append(pickle.load(open(pickle_path, 'rb')))
    main()
