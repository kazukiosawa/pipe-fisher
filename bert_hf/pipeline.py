import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator, Union
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
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
        recv_tensor = torch.zeros(tensor_shape, device=device, requires_grad=True)
        dist.recv(tensor=recv_tensor, src=src_rank, tag=tag)
        queue.add(recv_tensor)

def send_comm_thread(num_iterations, queue, dst_rank, tag):
    for i in range(num_iterations):
        tensor = queue.remove()
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)


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
                 pipeline_method: str = None,
                 is_up_pipe: bool = False):
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
        self.device = next(stage_module.parameters()).device
        self.total_loss = 0.
        self.pipeline_method = pipeline_method
        self.is_up_pipe = is_up_pipe
        self.tag = 2 if is_up_pipe else 1

        self.forward_recv_queues = {}
        self.backward_recv_queues = {}
        self.forward_send_queues = {}
        self.backward_send_queues = {}

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
        if not self.is_first_stage:
            inputs = collections.OrderedDict()
            for key in self.keys_from_prev_stage:
                inputs[key] = self.recv_inputs_from_queue(key)
            #inputs = self._get_zero_inputs(input_source)
            #self._receive_inputs(inputs)
        else:
            inputs = {}
        for key in self.keys_from_source:
            inputs[key] = input_source[key].to(self.device)
        assert len(inputs) > 0, 'No input is set.'

        outputs = self.stage_module(**inputs)
        if not self.is_last_stage:
            #self._send_outputs(outputs)
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

        out_tensors = tuple(outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            #for tensor in outputs.values():
            #    tensor.grad = torch.zeros_like(tensor)
            #self._receive_output_grads(outputs)
            #grad_tensors = tuple(tensor.grad for tensor in outputs.values())
            grad_dict = {}
            for key in outputs:
                grad_dict[key] = self.recv_output_grads_from_queue(key)
            grad_tensors = tuple(grad_dict[key] for key in grad_dict) # Shigang: Is this safe for autograd.backward?

            assert len(out_tensors) == len(grad_tensors), 'output_grads are not set yet.'

        with self.no_sync_if_need(no_sync):
            torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)
        if not self.is_first_stage:
            #self._send_input_grads(inputs)
            for key in self.keys_from_prev_stage:
                self.send_input_grads_to_queue(key, inputs[key].grad)

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

    def call_pipeline(self, data_iterator: Iterator, num_micro_batches, pipeline_method=None):
        if pipeline_method is None:
            pipeline_method = self.pipeline_method
        if pipeline_method == PIPELINE_1F1B:
            _call_pipeline = self._call_1f1b_pipeline
        elif pipeline_method == PIPELINE_GPIPE:
            _call_pipeline = self._call_gpipe_pipeline
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
        num_warmup_steps = self.num_stages - self.stage_id

        for _ in range(num_warmup_steps):
            self.call_forward(next(data_iterator))
        for _ in range(num_micro_batches - num_warmup_steps):
            self.call_backward()
            self.call_forward(next(data_iterator))
        for _ in range(num_warmup_steps):
            self.call_backward()

    def _call_gpipe_pipeline(self, data_iterator: Iterator, num_micro_batches):
        for _ in range(num_micro_batches):
            self.call_forward(next(data_iterator))

        for _ in range(num_micro_batches):
            self.call_backward()
