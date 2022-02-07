import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist


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
                 grad_sync_group: dist.ProcessGroup = None):
        assert dist.is_initialized(), 'torch.distributed needs to be initialized.'
        assert num_stages > 1, 'num_stages has to be > 1.'
        assert stage_id in range(num_stages), 'stage_id has be in range(num_stage).'
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.stage_module = stage_module
        self.num_batch_dims = num_batch_dims
        self.input_output_queue: Deque[Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]] = deque()
        self.grad_sync_group = grad_sync_group
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        self.device = next(stage_module.parameters()).device
        self.total_loss = 0.

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

    def call_forward(self, input_source):
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

    def call_backward(self):
        assert len(self.input_output_queue) > 0, 'No input/output is set.'
        # pop inputs/outputs from the queue
        inputs, outputs = self.input_output_queue.popleft()

        out_tensors = tuple(outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            for tensor in outputs.values():
                tensor.grad = torch.zeros_like(tensor)
            self._receive_output_grads(outputs)
            grad_tensors = tuple(tensor.grad for tensor in outputs)
            assert len(out_tensors) == len(grad_tensors), 'output_grads are not set yet.'

        torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)

        if not self.is_first_stage:
            self._send_input_grads(inputs)

        del inputs, outputs

    def _get_zero_inputs(self, input_source):
        batch_size = tuple(input_source[0].shape[:self.num_batch_dims])
        inputs = collections.OrderedDict()
        for key, size in self.keys_and_sizes_from_prev_stage:
            size = batch_size + size
            inputs[key] = torch.zeros(size, device=self.device, requires_grad=True)
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
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        packed_tensor = parameters_to_vector(grads)
        dist.all_reduce(packed_tensor, group=self.grad_sync_group)
        packed_tensor /= self.grad_sync_group.size()
        vector_to_parameters(packed_tensor, grads)
