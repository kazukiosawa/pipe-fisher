import argparse
import pickle
import math
import copy


PIPELINE_START = '-- start --'
PIPELINE_END = '-- end --'
FORWARD = 'call_forward'
BACKWARD = 'call_backward'
COV_KRON_A = 'cov_kron_A'
COV_KRON_B = 'cov_kron_B'
INV_KRON_A = 'inv_kron_A'
INV_KRON_B = 'inv_kron_B'
SYNC_KRON_A = 'sync_kron_A'
SYNC_KRON_B = 'sync_kron_B'
SYNC_GRAD = 'sync_grad'
NB_SYNC_GRAD = 'nb_sync_grad'
SYNC_CURVATURE = 'sync_curvature'
SYNC_GRAD_PRE_PRECOND = 'sync_grad_pre_precondition'
SYNC_GRAD_POST_PRECOND = 'sync_grad_post_precondition'
WAIT_ALL = 'wait_all'
BUBBLE = 'bubble'
TURN_ON_SAVE = 'turn_on_save_inputs_outgrads'
TURN_OFF_SAVE = 'turn_off_save_inputs_outgrads'

TAG_UP_PIPE = ':up_pipe'


pipeline_events = [FORWARD, BACKWARD]
cov_events = [COV_KRON_A, COV_KRON_B]
ngd_events = [COV_KRON_A, COV_KRON_B, SYNC_KRON_A, SYNC_KRON_B, INV_KRON_A, INV_KRON_B]


class Workload:
    def __init__(self, label, start, end, priority=-1):
        self.label = label
        self.start = start
        self.end = end
        self.priority = priority

    def __repr__(self):
        return repr((self.label, self.start, self.end, self.duration))

    @property
    def duration(self):
        return self.end - self.start


class WorkloadQueue(Workload):
    def __init__(self, label, start, end, queue_size, priority=-1):
        super(WorkloadQueue, self).__init__(label, start, end, priority)
        avg_duration = (end - start) // queue_size
        workloads = []
        s = start
        for i in range(queue_size):
            e = s + avg_duration
            workloads.append(Workload(label, s, e, priority))
            s = e
        self.queue = workloads
        self.avg_duration = avg_duration
        self.next_workload_id = 0

    def __repr__(self):
        return repr((self.label, self.start, self.end, self.duration, f'queue_size={len(self)}'))

    def pop(self):
        assert len(self.queue) > 0
        res = self.queue.pop(0)
        if len(self.queue) == 0:
            self.next_workload_id = None
        else:
            self.next_workload_id += 1
        return res

    def __len__(self):
        return len(self.queue)


def assign_workloads_to_bubbles(workloads, schedule, fwd_count=0, bwd_count=0, margin_ratio=.0):
    new_schedule = []
    last_workload = schedule.pop(0)
    new_schedule.append(last_workload)
    while len(schedule) > 0:
        if last_workload.label == FORWARD:
            fwd_count += 1
        elif last_workload.label == BACKWARD:
            bwd_count += 1
        next_workload = schedule.pop(0)
        bubble_start = last_workload.end
        bubble_end = next_workload.start
        while True:
            num_workloads_before = len(new_schedule)
            for workload in workloads:
                if isinstance(workload, WorkloadQueue):
                    while bubble_start + workload.avg_duration * (1 + margin_ratio) < bubble_end:
                        if COV_KRON_A in workload.label and (fwd_count < workload.next_workload_id + 1):
                            break
                        elif COV_KRON_B in workload.label and (bwd_count < workload.next_workload_id + 1):
                            break
                        sub_workload = workload.pop()
                        sub_workload.end = bubble_start + sub_workload.duration
                        sub_workload.start = bubble_start
                        new_schedule.append(sub_workload)
                        bubble_start += sub_workload.duration
                        if len(workload) == 0:
                            workloads.remove(workload)
                            break
                elif bubble_start + workload.duration * (1 + margin_ratio) < bubble_end:
                    if SYNC_KRON_A in workload.label and any(COV_KRON_A in w.label for w in workloads):
                        continue
                    elif SYNC_KRON_B in workload.label and any(COV_KRON_B in w.label for w in workloads):
                        continue
                    elif INV_KRON_A in workload.label and any(COV_KRON_A in w.label or SYNC_KRON_A in w.label for w in workloads):
                        continue
                    elif INV_KRON_B in workload.label and any(COV_KRON_B in w.label or SYNC_KRON_B in w.label for w in workloads):
                        continue
                    workload.end = bubble_start + workload.duration
                    workload.start = bubble_start
                    new_schedule.append(workload)
                    bubble_start += workload.duration
                    workloads.remove(workload)
            if len(new_schedule) == num_workloads_before:
                # loop until no workload is added
                break
        new_schedule.append(next_workload)
        last_workload = next_workload
    return new_schedule


def main():
    # get start and end time by the timeline of the node 0
    base_time = timelines[0]['call_forward'][0][0]
    if 'call_forward' + TAG_UP_PIPE in timelines[0]:
        base_time = min(base_time, timelines[0]['call_forward' + TAG_UP_PIPE][0][0])

    def time_shift(t):
        if t is None:
            return 0
        return t - base_time

    start_time = 0
    end_time = timelines[0]['call_backward'][-1][-1]
    if 'call_backward' + TAG_UP_PIPE in timelines[0]:
        end_time = max(end_time, timelines[0]['call_backward' + TAG_UP_PIPE][-1][-1])
    end_time = time_shift(end_time)
    pipeline_time = end_time - start_time

    schedules = []
    num_pipeline_iterations_list = []
    for node_id, timeline in enumerate(timelines):
        pipeline_workloads = [Workload(PIPELINE_START, start_time, start_time)]
        for event in pipeline_events:
            for s, e in timeline[event]:
                pipeline_workloads.append(Workload(event, time_shift(s), time_shift(e)))
        pipeline_workloads.append(Workload(PIPELINE_END, end_time, end_time))
        pipeline_workloads.sort(key=lambda x: x.start)

        num_micro_batches = sum(map(lambda x: x.label == FORWARD, pipeline_workloads))
        assert num_micro_batches == sum(map(lambda x: x.label == BACKWARD, pipeline_workloads))

        ngd_workloads = []
        for i, event in enumerate(ngd_events):
            for key in timeline:
                if event not in key:
                    continue
                if event in cov_events:
                    for s, e in timeline[key]:
                        ngd_workloads.append(WorkloadQueue(key, time_shift(s), time_shift(e), num_micro_batches, priority=i))
                else:
                    for s, e in timeline[key]:
                        ngd_workloads.append(Workload(key, time_shift(s), time_shift(e), priority=i))
        ngd_workloads.sort(key=lambda x: x.start)
        ngd_workloads.sort(key=lambda x: x.priority)

        # assign as many workloads as possible to the 1st pipeline
        schedule = assign_workloads_to_bubbles(ngd_workloads, pipeline_workloads.copy())
        schedule.append(Workload(TURN_OFF_SAVE, schedule[-1].end, schedule[-1].end))

        # assign all remaining workloads to the 1st and extra pipelines
        while len(ngd_workloads) > 0:
            remaining_workloads = copy.deepcopy(ngd_workloads)
            # try to assign all the remaining workloads to the bubbles in the current schedule
            new_schedule = assign_workloads_to_bubbles(remaining_workloads,
                                                       schedule.copy(),
                                                       fwd_count=num_micro_batches,
                                                       bwd_count=num_micro_batches)
            if len(remaining_workloads) == 0:
                schedule = new_schedule
                break

            # add one pipeline
            additional_schedule = assign_workloads_to_bubbles(ngd_workloads,
                                                              pipeline_workloads.copy(),
                                                              fwd_count=num_micro_batches,
                                                              bwd_count=num_micro_batches)
            schedule += additional_schedule

        schedule.append(Workload(TURN_ON_SAVE, schedule[-1].end, schedule[-1].end))

        total_time = 0
        for workload in schedule:
            total_time += workload.duration
        num_pipeline_iterations = sum(map(lambda x: x.label == PIPELINE_START, schedule))
        usage = total_time / (pipeline_time * num_pipeline_iterations) * 100
        print('*****************')
        print(f'node{node_id}:{num_pipeline_iterations} pipeline iterations (usage: {usage:.2f} %)')

        if args.print_workloads:
            last_workload = schedule[0]
            schedule_with_bubbles = []
            for i in range(1, len(schedule)):
                schedule_with_bubbles.append(last_workload)
                next_workload = schedule[i]
                bubble_time = next_workload.start - last_workload.end
                if bubble_time > 0:
                    schedule_with_bubbles.append(Workload(BUBBLE, last_workload.end, next_workload.start))
                last_workload = next_workload
            schedule_with_bubbles.append(last_workload)
            for workload in schedule_with_bubbles:
                print(workload)

        num_pipeline_iterations_list.append(num_pipeline_iterations)
        schedules.append([workload.label for workload in schedule])

    # set the number of pipeline iterations to the least common multiple of iterations between all nodes
    lcm_num_pipeline_iterations = math.lcm(*num_pipeline_iterations_list)
    for schedule, num_pipeline_iterations in zip(schedules, num_pipeline_iterations_list):
        schedule *= lcm_num_pipeline_iterations // num_pipeline_iterations

    if args.save_path is not None:
        with open(args.save_path, 'wb') as f:
            pickle.dump(schedules, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_paths', type=str)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--print_workloads', action='store_true')
    args = parser.parse_args()

    timelines = []
    for pickle_path in args.pickle_paths.split(','):
        if pickle_path == '':
            continue
        timelines.append(pickle.load(open(pickle_path, 'rb')))
    main()
