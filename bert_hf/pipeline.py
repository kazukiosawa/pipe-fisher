import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator, Union
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
    def keys_and_sizes_from_prev_stage(self) -> List[Tuple[str, tuple]]:
        raise NotImplementedError


def recv_comm_thread(num_iterations, queue, src_rank, tag, tensor_shape, device):
    for i in range(num_iterations):
        recv_tensor = torch.zeros(tensor_shape, requires_grad=True)
        dist.recv(tensor=recv_tensor, src=src_rank, tag=tag)
        queue.add(recv_tensor.to(device))


def send_comm_thread(num_iterations, queue, dst_rank, tag):
    for i in range(num_iterations):
        send_tensor = queue.remove()

        send_tensor = send_tensor.cpu()
        dist.send(tensor=send_tensor, dst=dst_rank, tag=tag)


class PipelineStage:
    def __init__(self,
                 stage_id: int,
                 num_stages: int,
                 input_tensor_shape: tuple,
                 output_tensor_shape: tuple,
                 stage_module: Union[StageModule, DistributedDataParallel],
                 num_batch_dims: int = 1,
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
        self.input_tensor_shape = input_tensor_shape
        self.output_tensor_shape = output_tensor_shape
        self.stage_module = stage_module
        self.num_batch_dims = num_batch_dims
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
    def keys_and_sizes_from_prev_stage(self):
        if isinstance(self.stage_module, DistributedDataParallel):
            return self.stage_module.module.keys_and_sizes_from_prev_stage
        return self.stage_module.keys_and_sizes_from_prev_stage

    @property
    def keys_from_prev_stage(self):
        return [v[0] for v in self.keys_and_sizes_from_prev_stage]

    @property
    def keys_and_sizes_of_next_stage(self):
        if isinstance(self.stage_module, DistributedDataParallel):
            return self.stage_module.module.keys_and_sizes_of_next_stage
        return self.stage_module.keys_and_sizes_of_next_stage

    @property
    def keys_of_next_stage(self):
        return [v[0] for v in self.keys_and_sizes_of_next_stage]

    def init_comm_queues(self):
        if self.is_first_stage:
            for key in self.keys_of_next_stage:
                self.backward_recv_queues[key] = threadsafe_queue.Queue()
                self.forward_send_queues[key] = threadsafe_queue.Queue()
        elif self.is_last_stage:
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = threadsafe_queue.Queue()
                self.backward_send_queues[key] = threadsafe_queue.Queue()
        else:
            for key in self.keys_of_next_stage:
                self.backward_recv_queues[key] = threadsafe_queue.Queue()
                self.forward_send_queues[key] = threadsafe_queue.Queue()
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = threadsafe_queue.Queue()
                self.backward_send_queues[key] = threadsafe_queue.Queue()

    def start_comm_thread(self, func, func_args):
        comm_thread = threading.Thread(target=func, args=func_args)
        comm_thread.daemon = True
        comm_thread.start()

    def start_comm_threads(self, num_iterations):
        for key in self.forward_recv_queues:
            queue = self.forward_recv_queues[key]
            self.start_comm_thread(recv_comm_thread,
                                   (num_iterations, queue, self.prev_rank, self.tag, self.input_tensor_shape, self.device))

        for key in self.forward_send_queues:
            queue = self.forward_send_queues[key]
            self.start_comm_thread(send_comm_thread, (num_iterations, queue, self.next_rank, self.tag))

        for key in self.backward_recv_queues:
            queue = self.backward_recv_queues[key]
            self.start_comm_thread(recv_comm_thread,
                                   (num_iterations, queue, self.next_rank, self.tag, self.output_tensor_shape, self.device))

        for key in self.backward_send_queues:
            queue = self.backward_send_queues[key]
            self.start_comm_thread(send_comm_thread, (num_iterations, queue, self.prev_rank, self.tag))

    def send_outputs_to_queue(self, key, tensor):
        self.forward_send_queues[key].add(tensor)

    def send_input_grads_to_queue(self, key, tensor):
        self.backward_send_queues[key].add(tensor)

    def recv_inputs_from_queue(self, key):
        return self.forward_recv_queues[key].remove()

    def recv_output_grads_from_queue(self, key):
        return self.backward_recv_queues[key].remove()

    def call_forward(self, input_source: OrderedDict[str, Tensor]):
        no_grad_if_need = nullcontext() if not self.recompute else torch.no_grad

        with no_grad_if_need():
            if not self.is_first_stage:
                inputs = collections.OrderedDict()
                for key in self.keys_from_prev_stage:
                    inputs[key] = self.recv_inputs_from_queue(key)
            else:
                inputs = {}
            for key in self.keys_from_source:
                inputs[key] = input_source[key].to(self.device)
            assert len(inputs) > 0, 'No input is set.'

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

    def _get_zero_inputs(self, input_source: OrderedDict[str, Tensor]):
        batch_size = tuple(next(iter(input_source.values())).shape[:self.num_batch_dims])
        inputs = collections.OrderedDict()
        for key, size in self.keys_and_sizes_from_prev_stage:
            inputs[key] = torch.zeros(batch_size + size, device=self.device, requires_grad=True)
        return inputs

    def _receive_inputs(self, inputs: OrderedDict[str, Tensor]):
        assert self.prev_rank is not None, 'prev_rank is not specified.'
        for key in self.keys_from_prev_stage:
            dist.recv(inputs[key], src=self.prev_rank)

    def _send_outputs(self, outputs: OrderedDict[str, Tensor]):
        assert self.next_rank is not None, 'next_rank is not specified.'
        for key in outputs:
            dist.send(outputs[key], dst=self.next_rank)

    def _receive_output_grads(self, outputs: OrderedDict[str, Tensor]):
        assert self.next_rank is not None, 'next_rank is not specified.'
        for key in outputs:
            dist.recv(outputs[key].grad, src=self.next_rank)

    def _send_input_grads(self, inputs: OrderedDict[str, Tensor]):
        assert self.prev_rank is not None, 'prev_rank is not specified.'
        for key in self.keys_from_prev_stage:
            dist.send(inputs[key].grad, dst=self.prev_rank)

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
