from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
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
                 batch_size: int,
                 max_seq_length: int = None,
                 prev_rank: int = None,
                 next_rank: int = None,
                 grad_sync_group: dist.ProcessGroup = None):
        assert dist.is_initialized(), 'torch.distributed needs to be initialized.'
        assert num_stages > 1, 'num_stages has to be > 1.'
        assert stage_id in range(num_stages), 'stage_id has be in range(num_stage).'
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.stage_module = stage_module
        self.inputs: OrderedDict[str, Tensor] = OrderedDict()
        self.outputs: OrderedDict[str, Tensor] = OrderedDict()
        self.input_grads: OrderedDict[str, Tensor] = OrderedDict()
        self.output_grads: OrderedDict[str, Tensor] = OrderedDict()
        self.grad_sync_group = grad_sync_group
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = next(stage_module.parameters()).device

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
            self._init_inputs()
            self._receive_inputs()
        for key in self.stage_module.keys_from_source:
            self.inputs[key] = input_source[key].to(self.device)
        assert len(self.inputs) > 0, 'No input is set.'

        self.outputs = self.stage_module(**self.inputs)
        if not self.is_last_stage:
            self._send_outputs()

    def call_backward(self):
        assert len(self.outputs) > 0, 'No output is set.'
        handles = []
        for key in self.keys_from_prev_stage:
            if key in self.inputs and self.inputs[key].requires_grad:
                def backward_hook(grad):
                    self.input_grads[key] = grad
                handle = self.inputs[key].register_hook(backward_hook)
                handles.append(handle)

        tensors = tuple(self.outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            self._init_output_grads()
            self._receive_output_grads()
            grad_tensors = tuple(self.output_grads.values())
            assert len(tensors) == len(grad_tensors), 'output_grads are not set yet.'

        torch.autograd.backward(tensors, grad_tensors=grad_tensors)
        for handle in handles:
            handle.remove()

        if not self.is_first_stage:
            self._send_input_grads()

    def _init_inputs(self):
        batch_size = (self.batch_size,)
        if self.max_seq_length:
            batch_size += (self.max_seq_length,)
        for key, size in self.keys_and_sizes_from_prev_stage:
            size = batch_size + size
            self.inputs[key] = torch.zeros(size, device=self.device)

    def _init_output_grads(self):
        for key, tensor in self.outputs:
            self.output_grads[key] = torch.zeros_like(tensor)

    def _receive_inputs(self):
        assert self.prev_rank, 'prev_rank is not specified.'
        for key in self.keys_from_prev_stage:
            dist.recv(self.inputs[key], src=self.prev_rank)

    def _send_outputs(self):
        assert self.next_rank, 'next_rank is not specified.'
        for tensor in self.outputs.values():
            dist.send(tensor, dst=self.next_rank)

    def _receive_output_grads(self):
        assert self.next_rank, 'next_rank is not specified.'
        for key in self.output_grads:
            dist.recv(self.output_grads[key], src=self.next_rank)

    def _send_input_grads(self):
        assert self.prev_rank, 'prev_rank is not specified.'
        for key in self.keys_from_prev_stage:
            dist.send(self.input_grads[key], dst=self.prev_rank)

    def sync_grad(self):
        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        packed_tensor = parameters_to_vector(grads)
        dist.all_reduce(packed_tensor, group=self.grad_sync_group)
        packed_tensor /= self.grad_sync_group.size()
        vector_to_parameters(packed_tensor, grads)

    def train_one_epoch_with_1f1b(self,
                                  train_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  num_optimization_steps: int,
                                  num_micro_batches_per_step: int):
        self.stage_module.train()
        num_warmup_steps = self.num_stages - self.stage_id - 1
        assert train_loader.drop_last, 'All micro-batches has to have the same batch size.'
        input_source = iter(train_loader)

        for _ in range(num_optimization_steps):
            optimizer.zero_grad()

            for _ in range(num_warmup_steps):
                self.call_forward(next(input_source))

            for _ in range(num_micro_batches_per_step - num_warmup_steps - 1):
                self.call_forward(next(input_source))
                self.call_backward()

            self.call_forward(next(input_source))

            for _ in range(num_warmup_steps):
                self.call_backward()

            self.call_backward()

            if self.grad_sync_group is not None:
                self.sync_grad()

            optimizer.step()
