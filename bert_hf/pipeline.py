import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator, Union, Dict
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist

import threading
import threadsafe_queue


PIPELINE_1F1B = '1f1b'
PIPELINE_GPIPE = 'gpipe'
PIPELINE_CHIMERA = 'chimera'


class StageModule(nn.Module):
    @property
    def keys_from_source(self) -> List[str]:
        raise NotImplementedError

    @property
    def sizes_from_prev_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError

    @property
    def sizes_for_next_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError


def recv_comm_thread(num_iterations, queue, src_rank, tag, tensor_shape, device):
    for _ in range(num_iterations):
        recv_tensor = torch.zeros(tensor_shape, requires_grad=True)
        if dist.get_backend() == dist.Backend.NCCL:
            recv_tensor = recv_tensor.to(device)
        dist.recv(tensor=recv_tensor, src=src_rank, tag=tag)
        queue.add(recv_tensor.to(device))


def send_comm_thread(num_iterations, queue, dst_rank, tag):
    for _ in range(num_iterations):
        send_tensor = queue.remove()
        if dist.get_backend() != dist.Backend.NCCL:
            send_tensor = send_tensor.cpu()
        dist.send(tensor=send_tensor, dst=dst_rank, tag=tag)


def start_comm_thread(func, kwargs):
    comm_thread = threading.Thread(target=func, kwargs=kwargs)
    comm_thread.daemon = True
    comm_thread.start()


class PipelineStage:
    def __init__(self,
                 stage_id: int,
                 num_stages: int,
                 stage_module: Union[StageModule, DistributedDataParallel],
                 prev_rank: int = None,
                 next_rank: int = None,
                 grad_sync_group: dist.ProcessGroup = None,
                 pipeline_method: str = None,
                 recompute: bool = False,
                 is_up_pipe: bool = False,
                 up_pipe_stage = None):
        assert dist.is_initialized(), 'torch.distributed needs to be initialized.'
        assert num_stages > 1, 'num_stages has to be > 1.'
        assert stage_id in range(num_stages), 'stage_id has be in range(num_stage).'
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.stage_module = stage_module
        self.input_output_queue: Deque[Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]] = deque()
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        self.grad_sync_group = grad_sync_group
        self.device = next(stage_module.parameters()).device
        self.total_loss = 0.
        self.pipeline_method = pipeline_method
        self.recompute = recompute
        self.is_up_pipe = is_up_pipe
        if not self.is_up_pipe and self.pipeline_method == PIPELINE_CHIMERA:
            assert up_pipe_stage is not None, 'Up pipeline should be created.'
        self.up_pipe_stage = up_pipe_stage
        self.tag = 2 if is_up_pipe else 1

        self.forward_recv_queues = {}
        self.backward_recv_queues = {}
        self.forward_send_queues = {}
        self.backward_send_queues = {}

        self.handles = []
        self.grads = []
        self.packed_grads = []

        self.init_comm_queues()

    @property
    def is_first_stage(self):
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        return self.stage_id == self.num_stages - 1

    @property
    def keys_from_source(self):
        if isinstance(self.stage_module, DistributedDataParallel):
            return self.stage_module.module.keys_from_source
        return self.stage_module.keys_from_source

    @property
    def sizes_from_prev_stage(self) -> Dict[str, Tuple]:
        stage_module = self.stage_module
        if isinstance(stage_module, DistributedDataParallel):
            stage_module = stage_module.module
        return stage_module.sizes_from_prev_stage

    @property
    def keys_from_prev_stage(self) -> List[str]:
        return list(self.sizes_from_prev_stage.keys())

    @property
    def sizes_for_next_stage(self) -> Dict[str, Tuple]:
        stage_module = self.stage_module
        if isinstance(stage_module, DistributedDataParallel):
            stage_module = stage_module.module
        return stage_module.sizes_for_next_stage

    @property
    def keys_for_next_stage(self):
        return list(self.sizes_for_next_stage.keys())

    def init_comm_queues(self):
        if not self.is_last_stage:
            for key in self.keys_for_next_stage:
                self.backward_recv_queues[key] = threadsafe_queue.Queue()
                self.forward_send_queues[key] = threadsafe_queue.Queue()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = threadsafe_queue.Queue()
                self.backward_send_queues[key] = threadsafe_queue.Queue()

    def start_comm_threads(self, num_iterations, batch_sizes):
        def start_recv_threads(recv_queues, src_rank, tensor_shapes):
            for key, queue in recv_queues.items():
                start_comm_thread(recv_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       src_rank=src_rank,
                                       tag=self.tag,
                                       tensor_shape=batch_sizes + tensor_shapes[key],
                                       device=self.device))

        def start_send_threads(queues, dst_rank):
            for queue in queues.values():
                start_comm_thread(send_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       dst_rank=dst_rank,
                                       tag=self.tag))

        start_recv_threads(self.forward_recv_queues, self.prev_rank, self.sizes_from_prev_stage)
        start_send_threads(self.forward_send_queues, self.next_rank)
        start_recv_threads(self.backward_recv_queues, self.next_rank, self.sizes_for_next_stage)
        start_send_threads(self.backward_send_queues, self.prev_rank)

    def send_outputs_to_queue(self, key, tensor):
        self.forward_send_queues[key].add(tensor)

    def send_input_grads_to_queue(self, key, tensor):
        self.backward_send_queues[key].add(tensor)

    def recv_inputs_from_queue(self, key):
        return self.forward_recv_queues[key].remove()

    def recv_output_grads_from_queue(self, key):
        return self.backward_recv_queues[key].remove()

    def call_forward(self, input_source: OrderedDict[str, Tensor]):
        inputs = collections.OrderedDict()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                inputs[key] = self.recv_inputs_from_queue(key)
        for key in self.keys_from_source:
            inputs[key] = input_source[key].to(self.device)
        assert len(inputs) > 0, 'No input is set.'

        no_grad_if_recompute = torch.no_grad if self.recompute else nullcontext
        with no_grad_if_recompute():
            outputs = self.stage_module(**inputs)

        if not self.is_last_stage:
            for key in outputs:
                self.send_outputs_to_queue(key, outputs[key])
        else:
            self.total_loss += float(outputs['loss'])

        # push inputs/outputs to the queue
        self.input_output_queue.append((inputs, outputs))

    def call_backward(self, no_sync=True):
        assert len(self.input_output_queue) > 0, 'No input/output is set.'
        # pop inputs/outputs from the queue
        inputs, outputs = self.input_output_queue.popleft()
        if self.recompute:
            outputs = self.stage_module(**inputs)

        out_tensors = tuple(outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            grad_dict = {}
            for key in outputs:
                grad_dict[key] = self.recv_output_grads_from_queue(key)
            grad_tensors = tuple(grad_dict[key] for key in grad_dict)
            assert len(out_tensors) == len(grad_tensors), 'output_grads are not set yet.'

        input_grads = {}
        def hook_wrapper(key):
            def hook(input_gradient):
                input_grads[key] = input_gradient
            return hook

        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                inputs[key].register_hook(hook_wrapper(key))

        with self.no_sync_if_need(no_sync):
            torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.send_input_grads_to_queue(key, input_grads[key])

        del inputs, outputs

    def no_sync_if_need(self, no_sync: bool):
        if isinstance(self.stage_module, DistributedDataParallel) and no_sync:
            return self.stage_module.no_sync()
        return nullcontext()

    def sync_grad(self):
        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        dist.barrier(group=self.grad_sync_group)
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        packed_tensor = parameters_to_vector(grads)
        dist.all_reduce(packed_tensor, group=self.grad_sync_group)
        packed_tensor /= self.grad_sync_group.size()
        vector_to_parameters(packed_tensor, grads)

    def nb_sync_grad(self):
        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        dist.barrier(group=self.grad_sync_group)
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        self.grads.append(grads)
        packed_tensor = parameters_to_vector(self.grads[-1])
        self.packed_grads.append(packed_tensor)
        self.handles.append(dist.all_reduce(self.packed_grads[-1], group=self.grad_sync_group, async_op=True))

    def waitall(self):
        l = len(self.handles)
        for _ in range(l):
            self.handles.pop(0).wait()
            packed_tensor = self.packed_grads.pop(0) / self.grad_sync_group.size()
            vector_to_parameters(packed_tensor, self.grads.pop(0))

    def call_pipeline(self, data_iterator: Iterator, num_micro_batches, pipeline_method=None):
        if pipeline_method is None:
            pipeline_method = self.pipeline_method
        if pipeline_method == PIPELINE_1F1B:
            _call_pipeline = self._call_1f1b_pipeline
        elif pipeline_method == PIPELINE_GPIPE:
            _call_pipeline = self._call_gpipe_pipeline
        elif pipeline_method == PIPELINE_CHIMERA:
            _call_pipeline = self._call_chimera_pipeline
        else:
            raise ValueError(f'Invalid pipeline_method: {pipeline_method}')

        self.total_loss = 0.
        assert len(self.input_output_queue) == 0
        _call_pipeline(data_iterator, num_micro_batches)
        assert len(self.input_output_queue) == 0
        return self.total_loss

    def call_1f1b_pipeline(self, data_iterator: Iterator, num_micro_batches):
        return self.call_pipeline(data_iterator, num_micro_batches, PIPELINE_1F1B)

    def _call_1f1b_pipeline(self, data_iterator: Iterator, num_micro_batches):
        num_warmup_steps = self.num_stages - self.stage_id - 1

        for _ in range(num_warmup_steps):
            self.call_forward(next(data_iterator))
        for _ in range(num_micro_batches - num_warmup_steps - 1):
            self.call_forward(next(data_iterator))
            self.call_backward()
        self.call_forward(next(data_iterator))
        for _ in range(num_warmup_steps):
            self.call_backward()
        self.call_backward()
        
        if self.grad_sync_group is not None and self.grad_sync_group.size() > 1:
            self.sync_grad()

    def _call_gpipe_pipeline(self, data_iterator: Iterator, num_micro_batches):
        for _ in range(num_micro_batches):
            self.call_forward(next(data_iterator))

        for _ in range(num_micro_batches):
            self.call_backward()

        if self.grad_sync_group is not None and self.grad_sync_group.size() > 1:
            self.sync_grad()

    def _call_chimera_pipeline(self, data_iterator: Iterator, num_micro_batches):
        assert self.num_stages % 2 == 0, 'The number of stages should be an even value.'
        assert num_micro_batches % self.num_stages == 0, 'Num_micro_batches should be a multiple of num_stages.'
        acc_steps = num_micro_batches // self.num_stages        
        half_stages = self.num_stages // 2
        first_half = self.stage_id // half_stages == 0

        schedule_number_a = half_stages - self.stage_id
        if schedule_number_a <= 0:
            schedule_number_a = -schedule_number_a
            schedule_number_a += 1

        schedule_number_b = half_stages - schedule_number_a

        for acc_step in range(acc_steps):
            if acc_step == 0:
                for _ in range(schedule_number_a):
                    if first_half:
                        self.call_forward(next(data_iterator))
                    else:
                        self.up_pipe_stage.call_forward(next(data_iterator))
            else:
                for _ in range(schedule_number_a):
                    if first_half:
                        self.call_backward()
                        self.call_forward(next(data_iterator))
                    else:
                        self.up_pipe_stage.call_backward()
                        self.up_pipe_stage.call_forward(next(data_iterator))

            for _ in range(schedule_number_b):
                if first_half:
                    self.up_pipe_stage.call_forward(next(data_iterator))
                    self.call_forward(next(data_iterator))
                else:
                    self.call_forward(next(data_iterator))
                    self.up_pipe_stage.call_forward(next(data_iterator))

            for _ in range(schedule_number_a):
                if first_half:
                    self.up_pipe_stage.call_forward(next(data_iterator))
                    self.up_pipe_stage.call_backward()
                else:
                    self.call_forward(next(data_iterator))
                    self.call_backward()

            for _ in range(schedule_number_b):
                if first_half:
                    self.call_backward()
                    self.up_pipe_stage.call_backward()
                else:
                    self.up_pipe_stage.call_backward()
                    self.call_backward()

            # early invoke grad_sync
            if acc_step == acc_steps - 1:
                if self.stage_id > half_stages:
                    self.nb_sync_grad()
                if self.up_pipe_stage.stage_id > half_stages:
                    self.up_pipe_stage.nb_sync_grad()

        for _ in range(schedule_number_a):
            if first_half:
                self.call_backward()
            else:
                self.up_pipe_stage.call_backward()

        if self.stage_id == half_stages:
            self.nb_sync_grad()
            self.up_pipe_stage.nb_sync_grad()
        if self.up_pipe_stage.stage_id == half_stages:
            self.up_pipe_stage.nb_sync_grad()
            self.nb_sync_grad()

        if self.stage_id > half_stages:
            self.up_pipe_stage.nb_sync_grad()
        if self.up_pipe_stage.stage_id > half_stages:
            self.nb_sync_grad()

        if first_half:
            self.up_pipe_stage.waitall()
            self.waitall()
        else:
            self.waitall()
            self.up_pipe_stage.waitall()
