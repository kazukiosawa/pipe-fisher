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
from torch.cuda import nvtx

import asdfghjkl as asdl

import threading
import threadsafe_queue
from auto_schedule import PIPELINE_END, FORWARD, BACKWARD, \
    COV_KRON_A, COV_KRON_B, INV_KRON_A, INV_KRON_B, SYNC_KRON_A, SYNC_KRON_B, \
    TURN_OFF_SAVE, TURN_ON_SAVE, TAG_UP_PIPE


PIPELINE_1F1B = '1f1b'
PIPELINE_GPIPE = 'gpipe'
PIPELINE_GPIPE_NGD = 'gpipe_ngd'
PIPELINE_CHIMERA = 'chimera'
PIPELINE_INTER = 'interleave'


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


def start_comm_thread(func, kwargs):
    comm_thread = threading.Thread(target=func, kwargs=kwargs)
    comm_thread.daemon = True
    comm_thread.start()


class PipelineStage:
    def __init__(self,
                 stage_id: int,
                 num_stages: int,
                 stage_module: Union[StageModule, DistributedDataParallel],
                 batch_sizes: Tuple[int, ...],
                 prev_rank: int = None,
                 next_rank: int = None,
                 rank: int = None,
                 grad_sync_group: dist.ProcessGroup = None,
                 pipeline_method: str = None,
                 recompute: bool = False,
                 is_up_pipe: bool = False,
                 chunks: int = None,
                 up_pipe_stage=None,
                 interleaved_stages: List = [],
                 nvtx_tag=''):
        assert dist.is_initialized(), 'torch.distributed needs to be initialized.'
        assert num_stages > 1, 'num_stages has to be > 1.'
        assert stage_id in range(num_stages), 'stage_id has be in range(num_stage).'
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.stage_module = stage_module
        self.batch_sizes = batch_sizes
        self.input_output_queue: Deque[Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]] = deque()
        self.prev_rank = prev_rank
        self.next_rank = next_rank
        self.rank = rank
        self.grad_sync_group = grad_sync_group
        self.device = next(stage_module.parameters()).device
        self.total_loss = 0.
        self.pipeline_method = pipeline_method
        self.recompute = recompute
        self.is_up_pipe = is_up_pipe
        if not self.is_up_pipe and self.pipeline_method == PIPELINE_CHIMERA:
            assert up_pipe_stage is not None, 'Up pipeline should be created.'
        self.up_pipe_stage = up_pipe_stage
        self.interleaved_stages = interleaved_stages
        self.chunks = chunks
        self.tag = 2 if is_up_pipe else 1
        self.nvtx_tag = nvtx_tag

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

    @property
    def is_distributed(self):
        return self.grad_sync_group is not None and self.grad_sync_group.size() > 1

    def init_comm_queues(self):
        if not self.is_last_stage:
            for key in self.keys_for_next_stage:
                self.backward_recv_queues[key] = threadsafe_queue.Queue()
                self.forward_send_queues[key] = threadsafe_queue.Queue()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = threadsafe_queue.Queue()
                self.backward_send_queues[key] = threadsafe_queue.Queue()

    @staticmethod
    def recv_comm_thread(num_iterations, queue, src_rank, tag, tensor_shape, device):
        for _ in range(num_iterations):
            recv_tensor = torch.zeros(tensor_shape, requires_grad=True)
            if dist.get_backend() == dist.Backend.NCCL:
                recv_tensor = recv_tensor.to(device)
            dist.recv(tensor=recv_tensor, src=src_rank, tag=tag)
            queue.add(recv_tensor.to(device))

    @staticmethod
    def send_comm_thread(num_iterations, queue, dst_rank, tag):
        for _ in range(num_iterations):
            send_tensor = queue.remove()
            if dist.get_backend() != dist.Backend.NCCL:
                send_tensor = send_tensor.cpu()
            dist.send(tensor=send_tensor, dst=dst_rank, tag=tag)

    def start_comm_threads(self, num_iterations):
        def start_recv_threads(recv_queues, src_rank, tensor_shapes):
            for key, queue in recv_queues.items():
                start_comm_thread(self.recv_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       src_rank=src_rank,
                                       tag=self.tag,
                                       tensor_shape=self.batch_sizes + tensor_shapes[key],
                                       device=self.device))

        def start_send_threads(queues, dst_rank):
            for queue in queues.values():
                start_comm_thread(self.send_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       dst_rank=dst_rank,
                                       tag=self.tag))

        start_recv_threads(self.forward_recv_queues, self.prev_rank, self.sizes_from_prev_stage)
        start_send_threads(self.forward_send_queues, self.next_rank)
        start_recv_threads(self.backward_recv_queues, self.next_rank, self.sizes_for_next_stage)
        start_send_threads(self.backward_send_queues, self.prev_rank)

    def start_interleaved_pipeline_comm_threads(self, num_iterations):
        def start_recv_threads(recv_queues, src_rank, tensor_shapes, tag):
            for key, queue in recv_queues.items():
                start_comm_thread(self.recv_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       src_rank=src_rank,
                                       tag=tag,
                                       tensor_shape=self.batch_sizes + tensor_shapes[key],
                                       device=self.device))

        def start_send_threads(queues, dst_rank, tag):
            for queue in queues.values():
                start_comm_thread(self.send_comm_thread,
                                  dict(num_iterations=num_iterations,
                                       queue=queue,
                                       dst_rank=dst_rank,
                                       tag=tag))

        start_recv_threads(self.forward_recv_queues, self.prev_rank, self.sizes_from_prev_stage, self.stage_id)
        start_send_threads(self.forward_send_queues, self.next_rank, self.stage_id+1)
        start_recv_threads(self.backward_recv_queues, self.next_rank, self.sizes_for_next_stage, self.stage_id+1)
        start_send_threads(self.backward_send_queues, self.prev_rank, self.stage_id)
        #start_recv_threads(self.forward_recv_queues, self.prev_rank, self.sizes_from_prev_stage, 1)
        #start_send_threads(self.forward_send_queues, self.next_rank, 1)
        #start_recv_threads(self.backward_recv_queues, self.next_rank, self.sizes_for_next_stage, 1)
        #start_send_threads(self.backward_send_queues, self.prev_rank, 1)

    def send_outputs_to_queue(self, key, tensor):
        self.forward_send_queues[key].add(tensor)

    def send_input_grads_to_queue(self, key, tensor):
        self.backward_send_queues[key].add(tensor)

    def recv_inputs_from_queue(self, key):
        return self.forward_recv_queues[key].remove()

    def recv_output_grads_from_queue(self, key):
        return self.backward_recv_queues[key].remove()

    def call_forward(self, input_source: OrderedDict[str, Tensor]):
        nvtx.range_push('call_forward' + self.nvtx_tag)

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

        nvtx.range_pop()

    def call_backward(self, no_sync=True):
        nvtx.range_push('call_backward' + self.nvtx_tag)
        assert len(self.input_output_queue) > 0, 'No input/output is set.'
        # pop inputs/outputs from the queue
        inputs, outputs = self.input_output_queue.popleft()
        if self.recompute:
            with nvtx.range('recompute'):
                outputs = self.stage_module(**inputs)

        out_tensors = tuple(outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            grad_tensors = tuple(self.recv_output_grads_from_queue(key) for key in outputs)

        input_grads = collections.OrderedDict()

        def get_hook(key):
            def hook(grad):
                input_grads[key] = grad
            return hook

        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                inputs[key].register_hook(get_hook(key))

        with self.no_sync_if_need(no_sync):
            torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.send_input_grads_to_queue(key, input_grads[key])

        del inputs, outputs

        nvtx.range_pop()

    def no_sync_if_need(self, no_sync: bool):
        if isinstance(self.stage_module, DistributedDataParallel) and no_sync:
            return self.stage_module.no_sync()
        return nullcontext()

    @nvtx.range('sync_grad')
    def sync_grad(self):
        nvtx.range_push('sync_grad' + self.nvtx_tag)

        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        dist.barrier(group=self.grad_sync_group)
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        packed_tensor = parameters_to_vector(grads)
        dist.all_reduce(packed_tensor, group=self.grad_sync_group)
        packed_tensor /= self.grad_sync_group.size()
        vector_to_parameters(packed_tensor, grads)

        nvtx.range_pop()

    def nb_sync_grad(self):
        nvtx.range_push('nb_sync_grad' + self.nvtx_tag)

        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        dist.barrier(group=self.grad_sync_group)
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        self.grads.append(grads)
        packed_tensor = parameters_to_vector(self.grads[-1])
        self.packed_grads.append(packed_tensor)
        self.handles.append(dist.all_reduce(self.packed_grads[-1], group=self.grad_sync_group, async_op=True))

        nvtx.range_pop()

    def wait_all(self):
        nvtx.range_push('wait_all' + self.nvtx_tag)

        for _ in range(len(self.handles)):
            self.handles.pop(0).wait()
            packed_tensor = self.packed_grads.pop(0) / self.grad_sync_group.size()
            vector_to_parameters(packed_tensor, self.grads.pop(0))

        nvtx.range_pop()

    def assert_intermediate_queues_are_empty(self):
        assert len(self.input_output_queue) == 0, f'input_output_queue of stage{self.stage_id} is not empty.'
        for name, queues in [('forward_send', self.forward_send_queues),
                             ('backward_recv', self.backward_recv_queues)]:
            for key, queue in queues.items():
                assert len(queue) == 0, f'{name}_queue for {key} of stage{self.stage_id} is not empty.'

    @nvtx.range('call_pipeline')
    def call_pipeline(self,
                      data_iterator: Iterator,
                      num_micro_batches,
                      pipeline_method=None,
                      data_iterator_for_up_pipe: Iterator = None,
                      ngd: asdl.NaturalGradient = None,
                      iteration: int = None,
                      no_sync_grad=False):
        if pipeline_method is None:
            pipeline_method = self.pipeline_method

        kwargs = dict(data_iterator=data_iterator, num_micro_batches=num_micro_batches, no_sync_grad=no_sync_grad)
        if pipeline_method == PIPELINE_1F1B:
            _call_pipeline = self._call_1f1b_pipeline
        elif pipeline_method == PIPELINE_INTER:
            _call_pipeline = self._call_interleaved_1f1b_pipeline
        elif pipeline_method == PIPELINE_GPIPE:
            _call_pipeline = self._call_gpipe_pipeline
        elif pipeline_method == PIPELINE_GPIPE_NGD:
            _call_pipeline = self._call_gpipe_pipeline_ngd
            kwargs['ngd'] = ngd
            kwargs['iteration'] = iteration
            kwargs.pop('no_sync_grad')
        elif pipeline_method == PIPELINE_CHIMERA:
            _call_pipeline = self._call_chimera_pipeline
            kwargs['data_iterator_for_up_pipe'] = data_iterator_for_up_pipe
        else:
            raise ValueError(f'Invalid pipeline_method: {pipeline_method}')

        self.total_loss = 0.
        self.assert_intermediate_queues_are_empty()
        _call_pipeline(**kwargs)
        self.assert_intermediate_queues_are_empty()
        return self.total_loss

    @nvtx.range('call_pipeline')
    def call_scheduled_pipeline(self,
                                schedule: List[str],
                                data_iterator,
                                ngd: asdl.NaturalGradient,
                                cxt: asdl.OperationContext,
                                data_iterator_up_pipe=None,
                                ngd_up_pipe: asdl.NaturalGradient = None,
                                cxt_up_pipe: asdl.OperationContext = None,
                                up_side_down=False):
        self.total_loss = 0.
        self.assert_intermediate_queues_are_empty()

        dual_pipelines = data_iterator_up_pipe is not None or ngd_up_pipe is not None
        if dual_pipelines:
            assert data_iterator_up_pipe is not None
            assert ngd_up_pipe is not None

        if up_side_down:
            stage_down_pipe = self.up_pipe_stage
            stage_up_pipe = self
            data_iterator_down_pipe = data_iterator_up_pipe
            data_iterator_up_pipe = data_iterator
            ngd_down_pipe = ngd_up_pipe
            ngd_up_pipe = ngd
            cxt_down_pipe = cxt_up_pipe
            cxt_up_pipe = cxt
        else:
            stage_down_pipe = self
            stage_up_pipe = self.up_pipe_stage
            data_iterator_down_pipe = data_iterator
            ngd_down_pipe = ngd
            cxt_down_pipe = cxt

        def get_ngd_and_module_name(_label, _event):
            _module_name = _label.replace(f'{_event}:', '')
            if TAG_UP_PIPE in _label:
                assert dual_pipelines
                return ngd_up_pipe, _module_name.replace(TAG_UP_PIPE, '')
            return ngd_down_pipe, _module_name

        for label in schedule:
            if label == FORWARD:
                stage_down_pipe.call_forward(next(data_iterator_down_pipe))
            elif label == FORWARD + TAG_UP_PIPE:
                stage_up_pipe.call_forward(next(data_iterator_up_pipe))
            elif label == BACKWARD:
                stage_down_pipe.call_backward()
            elif label == BACKWARD + TAG_UP_PIPE:
                stage_up_pipe.call_backward()
            elif COV_KRON_A in label or COV_KRON_B in label:
                if COV_KRON_A in label:
                    A_or_B = 'A'
                    _ngd, module_name = get_ngd_and_module_name(label, COV_KRON_A)
                else:
                    A_or_B = 'B'
                    _ngd, module_name = get_ngd_and_module_name(label, COV_KRON_B)
                _cxt = cxt_up_pipe if TAG_UP_PIPE in label else cxt_down_pipe
                _ngd.accumulate_curvature(cxt=_cxt, module_name=module_name, kron=[A_or_B], num_batches=1)
            elif SYNC_KRON_A in label or SYNC_KRON_B in label:
                if SYNC_KRON_A in label:
                    A_or_B = 'A'
                    _module_name = label.replace(f'{SYNC_KRON_A}:', '')
                else:
                    A_or_B = 'B'
                    _module_name = label.replace(f'{SYNC_KRON_B}:', '')
                if dual_pipelines:
                    ngd_down_pipe.sync_curvature(module_name=_module_name, kron=[A_or_B], async_op=True)
                    ngd_up_pipe.sync_curvature(module_name=_module_name, kron=[A_or_B], async_op=True)
                    ngd_down_pipe.wait_all_curvature_sync()
                    ngd_up_pipe.wait_all_curvature_sync()
                else:
                    ngd_down_pipe.sync_curvature(module_name=_module_name, kron=[A_or_B], enabled=self.is_distributed)
#            elif label in [SYNC_KRON_A, SYNC_KRON_B]:
#                A_or_B = 'A' if label == SYNC_KRON_A else 'B'
#                if dual_pipelines:
#                    ngd_down_pipe.sync_curvature(kron=[A_or_B], async_op=True)
#                    ngd_up_pipe.sync_curvature(kron=[A_or_B], async_op=True)
#                    ngd_down_pipe.wait_all_curvature_sync()
#                    ngd_up_pipe.wait_all_curvature_sync()
#                else:
#                    ngd_down_pipe.sync_curvature(kron=[A_or_B], enabled=self.is_distributed)
            elif INV_KRON_A in label or INV_KRON_B in label:
                if INV_KRON_A in label:
                    A_or_B = 'A'
                    _ngd, module_name = get_ngd_and_module_name(label, INV_KRON_A)
                else:
                    A_or_B = 'B'
                    _ngd, module_name = get_ngd_and_module_name(label, INV_KRON_B)
                _ngd.update_inv(module_name=module_name, kron=[A_or_B], zero_curvature=True, partition_aware=True)
            elif label == PIPELINE_END:
                if dual_pipelines:
                    ngd_down_pipe.sync_grad_pre_precondition(self.is_distributed, async_op=True)
                    ngd_up_pipe.sync_grad_pre_precondition(self.is_distributed, async_op=True)
                    ngd_down_pipe.wait_all_grad_sync()
                    ngd_up_pipe.wait_all_grad_sync()

                    ngd_down_pipe.precondition()
                    ngd_up_pipe.precondition()

                    ngd_down_pipe.sync_grad_post_precondition(self.is_distributed, async_op=True)
                    ngd_up_pipe.sync_grad_post_precondition(self.is_distributed, async_op=True)
                    ngd_down_pipe.wait_all_grad_sync()
                    ngd_up_pipe.wait_all_grad_sync()
                else:
                    ngd_down_pipe.sync_grad_pre_precondition(self.is_distributed)
                    ngd_down_pipe.precondition()
                    ngd_down_pipe.sync_grad_post_precondition(self.is_distributed)
            elif label == TURN_OFF_SAVE:
                cxt_down_pipe.turn_off_save_inputs_outgrads()
                if dual_pipelines:
                    cxt_up_pipe.turn_off_save_inputs_outgrads()
            elif label == TURN_ON_SAVE:
                cxt_down_pipe.turn_on_save_inputs_outgrads()
                if dual_pipelines:
                    cxt_up_pipe.turn_on_save_inputs_outgrads()

        self.assert_intermediate_queues_are_empty()
        return self.total_loss

    def _call_1f1b_pipeline(self, data_iterator: Iterator, num_micro_batches, no_sync_grad=False):
        """
        1F1B
        """
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
        
        if self.is_distributed and not no_sync_grad:
            self.sync_grad()

    def _call_interleaved_1f1b_pipeline(self, data_iterator: Iterator, num_micro_batches, no_sync_grad=False):
        """
        Interleaved 1F1B
        """
        num_micro_batches = num_micro_batches*self.chunks
        pipeline_parallel_size = self.num_stages // self.chunks
        pipeline_parallel_rank = self.stage_id % pipeline_parallel_size

        num_warmup_steps = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_steps += (self.chunks - 1) * pipeline_parallel_size

        forward_counter = 0
        backward_counter = 0

        for _ in range(num_warmup_steps):
            forward_chunk_id = (forward_counter // pipeline_parallel_size) % self.chunks
            if forward_chunk_id == 0:
                self.call_forward(next(data_iterator))
            else:
                self.interleaved_stages[forward_chunk_id-1].call_forward(next(data_iterator))
            forward_counter += 1
        for _ in range(num_micro_batches - num_warmup_steps):
            forward_chunk_id = (forward_counter // pipeline_parallel_size) % self.chunks
            if forward_chunk_id == 0:
                self.call_forward(next(data_iterator))
            else:
                self.interleaved_stages[forward_chunk_id-1].call_forward(next(data_iterator))
            forward_counter += 1

            backward_chunk_id = self.chunks - (backward_counter // pipeline_parallel_size) % self.chunks - 1
            if backward_chunk_id == 0:
                self.call_backward()
            else:
                self.interleaved_stages[backward_chunk_id-1].call_backward()
            backward_counter += 1

        for _ in range(num_warmup_steps):
            backward_chunk_id = self.chunks - (backward_counter // pipeline_parallel_size) % self.chunks - 1
            if backward_chunk_id == 0:
                self.call_backward()
            else:
                self.interleaved_stages[backward_chunk_id-1].call_backward()
            backward_counter += 1
        
        if self.is_distributed and not no_sync_grad:
            self.sync_grad()
            for stage in self.interleaved_stages:
                stage.sync_grad()

    def _call_gpipe_pipeline(self, data_iterator: Iterator, num_micro_batches, no_sync_grad=False):
        """
        GPipe
        """
        for _ in range(num_micro_batches):
            self.call_forward(next(data_iterator))

        for _ in range(num_micro_batches):
            self.call_backward()

        if self.is_distributed and not no_sync_grad:
            self.sync_grad()

    def _call_gpipe_pipeline_ngd(self,
                                 data_iterator: Iterator,
                                 num_micro_batches,
                                 ngd: asdl.NaturalGradient,
                                 iteration):
        """
        GPipe with K-FAC
        """
        update_prev_B_inv = self.stage_id in [0, 1] and iteration > 0

        with asdl.save_inputs_outgrads(self.stage_module, ignore_modules=ngd.ignore_modules) as cxt:
            for _ in range(num_micro_batches):
                self.call_forward(next(data_iterator))

            if not self.is_last_stage:
                ngd.accumulate_curvature(cxt=cxt)
                ngd.sync_curvature(kron=['A'], enabled=self.is_distributed)
                ngd.update_inv(kron=['A'], zero_curvature=True)
                if update_prev_B_inv:
                    ngd.sync_curvature(kron=['A'], enabled=self.is_distributed)
                    ngd.update_inv(kron=['B'], zero_curvature=True)

            for i in range(num_micro_batches):
                self.call_backward()
                if not self.is_last_stage:
                    if (self.is_first_stage and i < num_micro_batches - 1) or (not self.is_first_stage):
                        ngd.accumulate_curvature(cxt=cxt)
            if self.is_last_stage:
                if iteration % 2 == 0:
                    ngd.accumulate_curvature(cxt=cxt)

        if self.is_last_stage:
            if iteration == 0 or iteration % 2 == 1:
                ngd.sync_curvature(kron=['A', 'B'], enabled=self.is_distributed)
                ngd.update_inv(kron=['A', 'B'], zero_curvature=True)
        elif not update_prev_B_inv:
            ngd.sync_curvature(kron=['B'], enabled=self.is_distributed)
            ngd.update_inv(kron=['B'], zero_curvature=True)

        ngd.sync_grad_pre_precondition(self.is_distributed)
        ngd.precondition()
        ngd.sync_grad_post_precondition(self.is_distributed)

    def _call_chimera_pipeline(self,
                               data_iterator: Iterator,
                               data_iterator_for_up_pipe: Iterator,
                               num_micro_batches,
                               no_sync_grad=False):
        """
        Chimera with dual pipelines
        """
        assert self.num_stages % 2 == 0, 'The number of stages should be an even value.'
        assert num_micro_batches * 2 % self.num_stages == 0, 'num_micro_batches*2 should be a multiple of num_stages.'
        acc_steps = num_micro_batches * 2 // self.num_stages
        half_stages = self.num_stages // 2
        first_half = self.stage_id // half_stages == 0

        schedule_number_a = half_stages - self.stage_id
        if schedule_number_a <= 0:
            schedule_number_a = -schedule_number_a + 1
        schedule_number_b = half_stages - schedule_number_a

        def call(func_name, down_or_up, up_side_down=False, with_data=False):
            if up_side_down:
                down_or_up = 'up' if down_or_up == 'down' else 'down'
            args = []
            if with_data:
                data = next(data_iterator) if down_or_up == 'down' else next(data_iterator_for_up_pipe)
                args.append(data)
            if down_or_up == 'down':
                getattr(self, func_name)(*args)
            else:
                getattr(self.up_pipe_stage, func_name)(*args)

        def forward(down_or_up):
            call('call_forward', down_or_up, up_side_down=not first_half, with_data=True)

        def backward(down_or_up):
            call('call_backward', down_or_up, up_side_down=not first_half)

        def wait_all(down_or_up):
            if not no_sync_grad:
                call('wait_all', down_or_up, up_side_down=not first_half)

        def sync_grad(down_or_up):
            if not no_sync_grad:
                call('nb_sync_grad', down_or_up)

        for acc_step in range(acc_steps):
            for _ in range(schedule_number_a):
                if acc_step > 0:
                    backward('down')
                forward('down')

            for _ in range(schedule_number_b):
                forward('up')
                forward('down')

            for _ in range(schedule_number_a):
                forward('up')
                backward('up')

            for _ in range(schedule_number_b):
                backward('down')
                backward('up')

            # early invoke sync_grad
            if acc_step == acc_steps - 1:
                if self.stage_id > half_stages:
                    sync_grad('down')
                elif self.up_pipe_stage.stage_id > half_stages:
                    sync_grad('up')

        for _ in range(schedule_number_a):
            backward('down')

        if self.stage_id == half_stages:
            sync_grad('down')
            sync_grad('up')
        elif self.up_pipe_stage.stage_id == half_stages:
            sync_grad('up')
            sync_grad('down')
        elif self.stage_id > half_stages:
            sync_grad('up')
        elif self.up_pipe_stage.stage_id > half_stages:
            sync_grad('down')

        wait_all('up')
        wait_all('down')
