import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

PIPELINE_1F1B = '1f1b'


class StageModule(nn.Module):
    @property
    def keys_from_source(self) -> List[str]:
        raise NotImplementedError

    @property
    def keys_and_sizes_from_prev_stage(self) -> List[Tuple[str, tuple]]:
        raise NotImplementedError


class PipelineStage:
    def __init__(self,
                 stage_id: int,
                 num_stages: int,
                 stage_module: StageModule,
                 num_batch_dims: int = 1,
                 prev_rank: int = None,
                 next_rank: int = None,
                 pipeline_method: str = None):
        assert dist.is_initialized(), 'torch.distributed needs to be initialized.'
        assert num_stages > 1, 'num_stages has to be > 1.'
        assert stage_id in range(num_stages), 'stage_id has be in range(num_stage).'
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.stage_module = stage_module
        self.num_batch_dims = num_batch_dims
        self.input_output_queue: Deque[Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]] = deque()
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        self.device = next(stage_module.parameters()).device
        self.total_loss = 0.
        self.pipeline_method = pipeline_method

    @property
    def is_first_stage(self):
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        return self.stage_id == self.num_stages - 1

    @property
    def keys_from_source(self):
        return self.stage_module.keys_from_source

    @property
    def keys_and_sizes_from_prev_stage(self):
        return self.stage_module.keys_and_sizes_from_prev_stage

    @property
    def keys_from_prev_stage(self):
        return [v[0] for v in self.keys_and_sizes_from_prev_stage]

    def no_sync_if_need(self, no_sync):
        if isinstance(self.stage_module, DistributedDataParallel) and no_sync:
            return self.stage_module.no_sync()
        return nullcontext()

    def call_forward(self, input_source: OrderedDict[str, Tensor]):
        if not self.is_first_stage:
            inputs = self._get_zero_inputs(input_source)
            self._receive_inputs(inputs)
        else:
            inputs = {}
        for key in self.keys_from_source:
            inputs[key] = input_source[key].to(self.device)
        assert len(inputs) > 0, 'No input is set.'

        outputs = self.stage_module(**inputs)
        if not self.is_last_stage:
            self._send_outputs(outputs)
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
            for tensor in outputs.values():
                tensor.grad = torch.zeros_like(tensor)
            self._receive_output_grads(outputs)
            grad_tensors = tuple(tensor.grad for tensor in outputs.values())
            assert len(out_tensors) == len(grad_tensors), 'output_grads are not set yet.'

        with self.no_sync_if_need(no_sync):
            torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)
        if not self.is_first_stage:
            self._send_input_grads(inputs)

        del inputs, outputs

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
        self.call_backward(no_sync=False)
