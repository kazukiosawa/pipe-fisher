import argparse
import pickle
from collections import OrderedDict
import numpy as np

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
        'inv_kron_A': ('C4', 'inverse'),
        'inv_kron_B': ('C4', None),
        'inv_unit_wise': ('C4', None),
        'sync_grad': ('C7', 'sync-grad'),
        'nb_sync_grad': ('C7', None),
        'reduce_scatter_grad': ('C7', None),
        'all_reduce_undivided_grad': ('C7', None),
        'all_gather_grad': ('C7', None),
        'all_reduce_no_curvature_grad': ('C7', None),
        'reduce_scatter_curvature': ('C9', 'sync-curvature'),
        'all_reduce_undivided_curvature': ('C9', None),
        'precondition': ('C8', 'precondition'),
    }
)


def sort(array, num_split):
    if num_split == 1:
        return array
    array_sorted = []
    for i in range(num_split):
        array_sorted += array[i:len(array):num_split]
    return array_sorted


def plot_timeline(ax, timelines, xlabel=True, num_replicas=1, title=None):
    min_time = timelines[0]['call_forward'][0][0]
    max_time = 0
    for start_end_list in timelines[0].values():
        for s, e in start_end_list:
            if e is not None:
                max_time = max(max_time, e)

    def time_shift(t):
        return (t - min_time) / 10 ** 6  # ns -> ms

    num_iterations = len(timelines[0]['start_end'])
    num_forward_per_iteration = len(timelines[0]['call_forward']) // num_iterations
    first_pipeline_time = time_shift(timelines[0]['call_forward'][num_forward_per_iteration][0])

    verts = []
    verts_alpha = []
    colors = []
    colors_alpha = []
    used_keys = set()
    width = .95
    usages = []
    for idx, timeline in enumerate(sort(timelines, num_replicas)):
        total_time_in_first_pipeline = 0
        y = len(timelines) - idx - 1
        for i, event_txt in enumerate(timeline):
            if not any(key in event_txt for key in key_to_color_label):
                continue
            key = next(key for key in key_to_color_label if key in event_txt)
            used_keys.add(key)
            start_end_list = timeline[event_txt]
            for s, e in start_end_list:
                if s is None or e is None:
                    continue
                s = time_shift(s)
                e = time_shift(e)
                if e < first_pipeline_time:
                    total_time_in_first_pipeline += e - s
                v = [(s, y-width/2), (s, y+width/2), (e, y+width/2), (e, y-width/2), (s, y-width/2)]
                color, label = key_to_color_label[key]
                if any(keyword in key for keyword in ['sync', 'reduce', 'gather']):
                    verts_alpha.append(v)
                    colors_alpha.append(color)
                else:
                    verts.append(v)
                    colors.append(color)
        usages.append(total_time_in_first_pipeline / first_pipeline_time)
    usage = np.mean(usages)

    bars = PolyCollection(verts, facecolors=colors)
    ax.add_collection(bars)
    bars = PolyCollection(verts_alpha, facecolors=colors_alpha, alpha=.5, hatch='//')
    ax.add_collection(bars)
    ax.autoscale()

    if xlabel:
        ax.set_xlabel('Time (ms)')
    ax.set_yticks(range(len(timelines)))
    ax.set_yticklabels([f'GPU {i+1}' for i in range(len(timelines))][::-1])
    ax.set_title(f'{title} [GPU util. {usage * 100:.1f}%]')
    ax.set_xlim(time_shift(min_time), time_shift(max_time))

    for i in range(1, num_iterations):
        ax.axvline(time_shift(timelines[0]['call_forward'][num_forward_per_iteration * i][0]),
                   color='r', lw=7, label='flush @ GPU1' if i == 1 else None)
    ax.axvline(time_shift(max_time), color='r', lw=7)
    for key, (color, label) in key_to_color_label.items():
        if key in used_keys:
            if any(keyword in key for keyword in ['sync', 'reduce', 'gather']):
                ax.bar(0, 0, label=label, color=color, alpha=0.5, hatch='//')
            else:
                ax.bar(0, 0, label=label, color=color)


def main():
    fig = plt.figure(figsize=(24, 2 + 2.6 * len(timelines_list)))
    n_cols = 2 if len(timelines_list2) > 0 else 1
    gs = fig.add_gridspec(len(timelines_list), n_cols)

    for col_id in range(n_cols):
        axes = []
        tlist = timelines_list if col_id == 0 else timelines_list2
        titles = titles1 if col_id == 0 else titles2
        for i in range(len(tlist)):
            ax = fig.add_subplot(gs[i, col_id])
            plot_timeline(ax, tlist[i],
                          xlabel=i == len(tlist)-1,
                          num_replicas=1 if i < len(tlist) - 1 else args.num_replicas,
                          title=titles[i])
            axes.append(ax)

        _, right = axes[-1].get_xlim()
        for i in range(len(tlist) - 1):
            axes[i].set_xlim(right=right)

        if args.x_max:
            for i in range(len(tlist)):
                axes[i].set_xlim(right=args.x_max)

        if col_id == n_cols - 1:
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.04), loc='lower center', ncol=8, fontsize=20)

    plt.tight_layout()
    plt.savefig(args.fig_path, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_paths1', type=str)
    parser.add_argument('--pickle_paths2', type=str)
    parser.add_argument('--pickle_paths3', type=str, default=None)
    parser.add_argument('--pickle_paths4', type=str, default=None)
    parser.add_argument('--pickle_paths5', type=str, default=None)
    parser.add_argument('--pickle_paths6', type=str, default=None)
    parser.add_argument('--titles1', type=str, default=None)
    parser.add_argument('--titles2', type=str, default=None)
    parser.add_argument('--fig_path', type=str, default='prof.png')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--num_replicas', type=int, default=1)
    parser.add_argument('--x_max', type=float, default=None)
    args = parser.parse_args()

    def get_timelines(pickle_paths):
        timelines = []
        for pickle_path in pickle_paths.split(','):
            if pickle_path == '':
                continue
            timelines.append(pickle.load(open(pickle_path, 'rb')))
        return timelines

    timelines1 = get_timelines(args.pickle_paths1)
    timelines2 = get_timelines(args.pickle_paths2)
    timelines_list = [timelines1, timelines2]
    if args.pickle_paths3 is not None:
        timelines3 = get_timelines(args.pickle_paths3)
        timelines_list.append(timelines3)
    if args.titles1 is not None:
        titles1 = args.titles1.split(',')
    else:
        titles1 = [None] * len(timelines_list)

    timelines_list2 = []
    for paths in [args.pickle_paths4, args.pickle_paths5, args.pickle_paths6]:
        if paths is not None:
            timelines_list2.append(get_timelines(paths))
    if args.titles2 is not None:
        titles2 = args.titles2.split(',')
    else:
        titles2 = [None] * len(timelines_list2)
    main()
