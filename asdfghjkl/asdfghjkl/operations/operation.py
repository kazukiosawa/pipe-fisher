from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda import nvtx
from ..utils import original_requires_grad
from ..matrices import *
from ..symmatrix import *
from ..vector import ParamVector

# compute no-centered covariance
OP_FULL_COV = 'full_cov'  # full covariance
OP_FULL_CVP = 'full_cvp'  # full covariance-vector product
OP_COV = 'cov'  # layer-wise covariance
OP_CVP = 'cvp'  # layer-wise covariance-vector product
OP_COV_KRON = 'cov_kron'  # Kronecker-factored
OP_COV_DIAG = 'cov_diag'  # diagonal
OP_COV_UNIT_WISE = 'cov_unit_wise'  # unit-wise
OP_RFIM_RELU = 'rfim_relu'  # relative FIM for ReLU
OP_RFIM_SOFTMAX = 'rfim_softmax'  # relative FIM for sotmax

# compute Gram matrix
OP_GRAM_DIRECT = 'gram_direct'  # direct
OP_GRAM_HADAMARD = 'gram_hada'  # Hadamard-factored

OP_BATCH_GRADS = 'batch_grads'  # compute batched gradients (per-example gradients)
OP_SAVE_INPUTS = 'save_inputs'  # save inputs during a forward-pass
OP_SAVE_OUTGRADS = 'save_outgrads'  # save outgrads during a backward-pass

ALL_OPS = [OP_FULL_COV, OP_FULL_CVP, OP_COV, OP_CVP,
           OP_COV_KRON, OP_COV_DIAG, OP_COV_UNIT_WISE,
           OP_RFIM_RELU, OP_RFIM_SOFTMAX,
           OP_GRAM_DIRECT, OP_GRAM_HADAMARD, OP_BATCH_GRADS,
           OP_SAVE_INPUTS, OP_SAVE_OUTGRADS]

FWD_OPS = [OP_SAVE_INPUTS, OP_COV_KRON, OP_GRAM_HADAMARD, OP_RFIM_RELU, OP_RFIM_SOFTMAX]
BWD_OPS_WITH_INPUTS = [OP_COV, OP_CVP, OP_COV_DIAG, OP_COV_UNIT_WISE, OP_BATCH_GRADS, OP_GRAM_DIRECT]
BWD_OPS = [OP_SAVE_OUTGRADS, OP_COV_KRON, OP_GRAM_HADAMARD, OP_RFIM_RELU, OP_RFIM_SOFTMAX] + BWD_OPS_WITH_INPUTS


class Operation:
    def __init__(self, module, op_names, model_for_kernel=None):
        if hasattr(module, 'weight'):
            requires_grad = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                requires_grad = requires_grad or module.bias.requires_grad
            assert requires_grad, f'One of weight or bias has to require grad (module: {module}).'
        self._module = module
        self._model_for_kernel = model_for_kernel
        if isinstance(op_names, str):
            op_names = [op_names]
        assert isinstance(op_names, list)
        # remove duplicates
        op_names = set(op_names)
        for name in op_names:
            assert name in ALL_OPS, f'Invalid operation name: {name}.'
        self._op_names = op_names
        self._op_results = {}

    def accumulate_result(self, value, *keys, extend=False):
        """
        Examples:
             accumulate_result(data, OP_COV_UNIT_WISE)
             accumulate_result(data, OP_BATCH_GRADS, 'weight')
             accumulate_result(A, OP_COV_KRON, 'A')
        """
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                if results.get(key, None) is None:
                    results[key] = {}
                results = results[key]
        key = keys[-1]
        if results.get(key, None) is None:
            results[key] = value
        elif extend:
            results[key].extend(value)
        else:
            results[key] += value

    def get_result(self, *keys, pop=False, default=None):
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                try:
                    results = results[key]
                except KeyError:
                    return None
        key = keys[-1]
        if pop:
            return results.pop(key, default)
        return results.get(key, default)

    def clear_result(self, *keys):
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                results = results[key]
        key = keys[-1]
        results.pop(key, None)

    def clear_results(self):
        self._op_results = {}

    def add_op_name(self, op_name):
        if op_name not in self._op_names:
            self._op_names.add(op_name)

    def remove_op_name(self, op_name):
        while op_name in self._op_names:
            self._op_names.remove(op_name)

    def forward_post_process(self, in_data: torch.Tensor, out_data: torch.Tensor):
        module = self._module
        op_names = self._op_names

        if any(op_name in FWD_OPS for op_name in op_names):
            in_data = self.preprocess_in_data(module, in_data, out_data)
            if OP_SAVE_INPUTS in op_names:
                self.accumulate_result([in_data], OP_SAVE_INPUTS, extend=True)

        for op_name in op_names:
            if op_name not in FWD_OPS:
                continue
            if op_name == OP_COV_KRON:
                A = self.cov_kron_A(module, in_data)
                self.accumulate_result(A, OP_COV_KRON, 'A')
            elif op_name == OP_GRAM_HADAMARD:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    A = self.gram_A(module, in_data, in_data)
                else:
                    A = self.gram_A(module, in_data[:n1], in_data[n1:])
                if original_requires_grad(module, 'bias'):
                    A += 1.
            elif op_name == OP_RFIM_RELU:
                self.accumulate_result(self.rfim_relu(module, in_data, out_data), OP_RFIM_RELU)
            elif op_name == OP_RFIM_SOFTMAX:
                self.accumulate_result(self.rfim_softmax(module, in_data, out_data), OP_RFIM_SOFTMAX)

    def backward_pre_process(self, in_data, out_data, out_grads, vector: torch.Tensor = None):
        module = self._module
        op_names = self._op_names

        if any(op_name in BWD_OPS for op_name in op_names):
            out_grads = self.preprocess_out_grads(module, out_grads)
            if OP_SAVE_OUTGRADS in op_names:
                self.accumulate_result([out_grads], OP_SAVE_OUTGRADS, extend=True)
        if any(op_name in BWD_OPS_WITH_INPUTS for op_name in op_names):
            in_data = self.preprocess_in_data(module, in_data, out_data)

        for op_name in op_names:
            if op_name not in BWD_OPS:
                continue
            if op_name == OP_COV:
                _, _, batch_g = self.collect_batch_grads(in_data, out_grads)
                self.accumulate_result(torch.matmul(batch_g.T, batch_g), OP_COV)
            elif op_name == OP_CVP:
                _, _, batch_g = self.collect_batch_grads(in_data, out_grads)
                assert vector is not None
                assert vector.ndim == 1
                batch_gtv = batch_g.mul(vector.unsqueeze(0)).sum(dim=1)
                cvp = torch.einsum('ni,n->i', batch_g, batch_gtv)
                if original_requires_grad(module, 'weight'):
                    if original_requires_grad(module, 'bias'):
                        w_numel = module.weight.numel()
                        self.accumulate_result(cvp[:w_numel].view_as(module.weight), OP_CVP, 'weight')
                        self.accumulate_result(cvp[w_numel:].view_as(module.bias), OP_CVP, 'bias')
                    else:
                        self.accumulate_result(cvp.view_as(module.weight), OP_CVP, 'weight')
                else:
                    self.accumulate_result(cvp.view_as(module.bias), OP_CVP, 'bias')
            elif op_name == OP_COV_KRON:
                B = self.cov_kron_B(module, out_grads)
                self.accumulate_result(B, OP_COV_KRON, 'B')
            elif op_name == OP_COV_UNIT_WISE:
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    assert original_requires_grad(module, 'weight') and original_requires_grad(module, 'bias'), \
                        f'Both weight and bias have to require grad for {OP_COV_UNIT_WISE} (module: {module}).'
                elif original_requires_grad(module, 'bias'):
                    in_data = self.extend_in_data(in_data)
                rst = self.cov_unit_wise(module, in_data, out_grads)
                self.accumulate_result(rst, OP_COV_UNIT_WISE)
            elif op_name == OP_GRAM_HADAMARD:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    B = self.gram_B(module, out_grads, out_grads)
                else:
                    B = self.gram_B(module, out_grads[:n1], out_grads[n1:])
                A = self._op_results[OP_GRAM_HADAMARD]['A']
                self._model_for_kernel.kernel += B.mul(A)
            elif op_name == OP_GRAM_DIRECT:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_DIRECT}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]

                batch_grads_w, batch_grads_b, batch_g = self.collect_batch_grads(in_data, out_grads)

                precond = getattr(module, 'gram_precond', None)
                if precond is not None:
                    precond.precondition_module(module, vec_weight=batch_grads_w, vec_bias=batch_grads_b)
                    grads = [g.flatten(start_dim=1) for g in [batch_grads_w, batch_grads_b] if g is not None]
                    batch_g2 = torch.cat(grads, dim=1)
                else:
                    batch_g2 = batch_g

                if n_data == n1:
                    self._model_for_kernel.kernel += torch.matmul(batch_g, batch_g2.T)
                else:
                    self._model_for_kernel.kernel += torch.matmul(batch_g[:n1], batch_g2[n1:].T)
            elif op_name in [OP_COV_DIAG, OP_BATCH_GRADS]:
                if original_requires_grad(module, 'weight'):
                    rst = getattr(self,
                                  f'{op_name}_weight')(module, in_data, out_grads)
                    self.accumulate_result(rst, op_name, 'weight')
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    self.accumulate_result(rst, op_name, 'bias')

    @staticmethod
    def preprocess_in_data(module, in_data, out_data):
        # no preprocess by default
        return in_data

    @staticmethod
    def preprocess_out_grads(module, out_grads):
        # no preprocess by default
        return out_grads

    @torch.no_grad()
    def collect_batch_grads(self, in_data, out_grads):
        module = self._module
        batch_grads_w = None
        batch_grads_b = None
        if original_requires_grad(module, 'weight'):
            batch_grads_w = self.batch_grads_weight(module, in_data, out_grads)
        if original_requires_grad(module, 'bias'):
            batch_grads_b = self.batch_grads_bias(module, out_grads)
        assert batch_grads_w is not None or batch_grads_b is not None, \
            f'At least one of weight or bias has to require grad (module: {module}).'

        grads = [batch_grads_w, batch_grads_b]
        batch_g = torch.cat([g.flatten(start_dim=1) for g in grads if g is not None], dim=1)
        return batch_grads_w, batch_grads_b, batch_g

    @staticmethod
    def extend_in_data(in_data):
        # no extension by default
        return in_data

    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def grad_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def grad_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_kron_A(module, in_data):
        raise NotImplementedError

    @staticmethod
    def cov_kron_B(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise NotImplementedError

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise NotImplementedError

    @staticmethod
    def rfim_relu(module, in_data, out_data):
        raise NotImplementedError

    @staticmethod
    def rfim_softmax(module, in_data, out_data):
        raise NotImplementedError


class OperationContext:
    def __init__(self, vectors: ParamVector = None):
        self._operations: Dict[nn.Module, Operation] = {}
        self._vectors: ParamVector = vectors
        self._input_scale = 1.
        self._output_scale = 1.

    def is_operation_registered(self, module: nn.Module):
        return module in self._operations

    def register_operation(self, module: nn.Module, operation: Operation):
        self._operations[module] = operation

    def get_vectors_by_module(self, module: nn.Module, flatten=False):
        if self._vectors is None:
            return None
        vectors = self._vectors.get_vectors_by_module(module)
        if vectors is None:
            return None
        if flatten:
            return vectors.get_flatten_vector()
        return vectors

    def get_operation(self, module: nn.Module) -> Operation:
        try:
            return self._operations[module]
        except KeyError:
            raise KeyError(f'No operation is registered to {module}.')

    def clear_operation(self, module: nn.Module):
        try:
            self._operations.pop(module)
        except KeyError:
            raise KeyError(f'No operation is registered to {module}.')

    def clear_operations(self):
        keys = list(self._operations.keys())
        for key in keys:
            del self._operations[key]

    def call_operations_in_forward(self, module, in_data, out_data):
        self.get_operation(module).forward_post_process(in_data, out_data)

    def call_operations_in_backward(self, module, in_data, out_data, out_grads):
        vector = self.get_vectors_by_module(module, flatten=True)
        self.get_operation(module).backward_pre_process(in_data, out_data, out_grads, vector)

    def get_result(self, module, *keys, pop=False, default=None):
        try:
            return self.get_operation(module).get_result(*keys, pop=pop, default=default)
        except KeyError:
            return default

    def accumulate_result(self, module, value, *keys):
        return self.get_operation(module).accumulate_result(value, *keys)

    def clear_result(self, module, *keys):
        return self.get_operation(module).clear_result(*keys)

    def turn_on_op(self, op_name):
        for operation in self._operations.values():
            operation.add_op_name(op_name)

    def turn_off_op(self, op_name):
        for operation in self._operations.values():
            operation.remove_op_name(op_name)

    def turn_on_save_inputs_outgrads(self):
        self.turn_on_op(OP_SAVE_INPUTS)
        self.turn_on_op(OP_SAVE_OUTGRADS)

    def turn_off_save_inputs_outgrads(self):
        self.turn_off_op(OP_SAVE_INPUTS)
        self.turn_off_op(OP_SAVE_OUTGRADS)

    def calc_grads(self, scale=None):
        for module in self._operations.keys():
            if original_requires_grad(module, 'weight') \
                    or original_requires_grad(module, 'bias'):
                self.calc_grad(module, scale)

    def calc_grad(self, module, scale=None):
        operation, in_data, out_grads = self.load_op_in_out(module)
        if scale is not None:
            assert scale.shape[0] == out_grads.shape[0]
            out_grads.mul_(scale.unsqueeze(-1))
        if original_requires_grad(module, 'weight'):
            grad = operation.grad_weight(module, in_data, out_grads)
            if module.weight.grad is None:
                module.weight.grad = grad
            else:
                module.weight.grad.data.copy_(grad)
        if original_requires_grad(module, 'bias'):
            grad = operation.grad_bias(module, out_grads)
            if module.bias.grad is None:
                module.bias.grad = grad
            else:
                module.bias.grad.data.copy_(grad)

    def calc_kernel(self):
        kernel = None
        for module in self._operations.keys():
            if original_requires_grad(module, 'weight'):
                operation, in_data, out_grads = self.load_op_in_out(module)
                A = operation.gram_A(module, in_data, in_data)
                B = operation.gram_B(module, out_grads, out_grads)
                if kernel is None:
                    kernel = torch.zeros_like(A)
                kernel += A.mul(B)
                if original_requires_grad(module, 'bias'):
                    kernel += B
        return kernel

    def batch_grads(self, module, flatten=False):
        grads = self.get_result(module, OP_BATCH_GRADS)
        if grads is not None and flatten:
            return torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)
        else:
            return grads

    def clear_batch_grads(self):
        for operation in self._operations.values():
            if operation.get_result(OP_BATCH_GRADS, default=None) is not None:
                operation.clear_result(OP_BATCH_GRADS)

    def full_batch_grads(self, module):
        bg = [self.batch_grads(m, flatten=True) for m in module.modules()]
        bg = [_bg for _bg in bg if _bg is not None]
        if len(bg) == 0:
            return None
        return torch.cat(bg, dim=1)  # n x p

    def full_cov(self, module):
        return self.get_result(module, OP_FULL_COV)

    def calc_full_cov(self, module):
        """
        g: (p,)
        cov = sum[gg^t]: (p, p)
        """
        bg = self.full_batch_grads(module)
        if bg is None:
            return
        cov = torch.matmul(bg.T, bg)  # p x p
        self.accumulate_result(module, cov, OP_FULL_COV)

    def full_cvp(self, module):
        return self.get_result(module, OP_FULL_CVP)

    def calc_full_cvp(self, module):
        """
        g: (p,)
        c = sum[gg^t]: (p, p)
        v: (p,)
        cvp = sum[gg^t]v = sum[g(g^t)v]: (p,)
        """
        vector = self.get_vectors_by_module(module, flatten=True)
        bg = self.full_batch_grads(module)
        if bg is None:
            return
        assert bg.shape[1] == vector.shape[0]
        bgtv = bg.mul(vector.unsqueeze(0)).sum(dim=1)
        cvp = torch.einsum('ni,n->i', bg, bgtv)
        self.accumulate_result(module, cvp, OP_FULL_CVP)

    def load_op_in_out(self, module) -> (Operation, List[torch.Tensor], List[torch.Tensor]):
        operation = self.get_operation(module)
        in_data = self.in_data(module)
        out_grads = self.out_grads(module)
        return operation, in_data, out_grads

    def cov(self, module):
        return self.get_result(module, OP_COV)

    def cvp(self, module):
        return self.get_result(module, OP_CVP)

    def cov_kron(self, module):
        return self.get_result(module, OP_COV_KRON)

    def cov_unit_wise(self, module):
        return self.get_result(module, OP_COV_UNIT_WISE)

    def cov_diag(self, module):
        return self.get_result(module, OP_COV_DIAG)

    def calc_cov(self, module, shape=SHAPE_LAYER_WISE, kron=None, clear_in_out=False, num_batches=None, tag=''):
        operation, in_data, out_grads = self.load_op_in_out(module)
        if shape == SHAPE_KRON:
            if kron is None:
                kron = ['A', 'B']
            if 'A' in kron and in_data is not None:
                max_len = len(in_data) if num_batches is None else min(len(in_data), num_batches)
                with nvtx.range('cov_kron_A' + tag):
                    for i in range(max_len):
                        if clear_in_out:
                            tensor = in_data.pop(0)
                        else:
                            tensor = in_data[i]
                        if self._input_scale != 1:
                            tensor = tensor * self._input_scale
                        A = operation.cov_kron_A(module, tensor)
                        self.accumulate_result(module, A, OP_COV_KRON, 'A')
            if 'B' in kron and out_grads is not None:
                max_len = len(out_grads) if num_batches is None else min(len(out_grads), num_batches)
                with nvtx.range('cov_kron_B' + tag):
                    for i in range(max_len):
                        if clear_in_out:
                            tensor = out_grads.pop(0)
                        else:
                            tensor = out_grads[i]
                        if self._output_scale != 1:
                            tensor = tensor * self._output_scale
                        B = operation.cov_kron_B(module, tensor)
                        self.accumulate_result(module, B, OP_COV_KRON, 'B')
        else:
            if in_data is None or out_grads is None:
                return
            max_len = min(len(in_data), len(out_grads))
            if num_batches is not None:
                max_len = min(max_len, num_batches)
            with nvtx.range(f'cov_{shape}' + tag):
                for i in range(max_len):
                    if clear_in_out:
                        tensor1 = in_data.pop(0)
                        tensor2 = out_grads.pop(0)
                    else:
                        tensor1 = in_data[i]
                        tensor2 = out_grads[i]
                    if shape == SHAPE_LAYER_WISE:
                        _, _, batch_g = operation.collect_batch_grads(tensor1, tensor2)
                        cov = torch.matmul(batch_g.T, batch_g)
                        self.accumulate_result(module, cov, OP_COV)
                    elif shape == SHAPE_UNIT_WISE:
                        unit = operation.cov_unit_wise(module, tensor1, tensor2)
                        self.accumulate_result(module, unit, OP_COV_UNIT_WISE)
                    elif shape == SHAPE_DIAG:
                        diag_w = operation.cov_diag_weight(module, tensor1, tensor2)
                        self.accumulate_result(module, diag_w, OP_COV_DIAG, 'weight')
                        if original_requires_grad(module, 'bias'):
                            diag_b = operation.cov_diag_bias(module, tensor2)
                            self.accumulate_result(module, diag_b, OP_COV_DIAG, 'bias')

    def calc_cov_kron(self, module, clear_in_out=False):
        self.calc_cov(module, shape=SHAPE_KRON, clear_in_out=clear_in_out)

    def calc_cov_unit_wise(self, module, clear_in_out=False):
        self.calc_cov(module, shape=SHAPE_UNIT_WISE, clear_in_out=clear_in_out)

    def calc_cov_diag(self, module, clear_in_out=False):
        self.calc_cov(module, shape=SHAPE_DIAG, clear_in_out=clear_in_out)

    def cov_symmatrix(self, module, pop=False):
        kwargs = dict(data=self.get_result(module, OP_COV, pop=pop),
                      unit_data=self.get_result(module, OP_COV_UNIT_WISE, pop=pop))
        cov_kron = self.get_result(module, OP_COV_KRON, pop=pop)
        if cov_kron is not None:
            kwargs['kron_A'] = cov_kron.pop('A', None)
            kwargs['kron_B'] = cov_kron.pop('B', None)
        cov_diag = self.get_result(module, OP_COV_DIAG, pop=pop)
        if cov_diag is not None:
            if original_requires_grad(module, 'weight'):
                kwargs['diag_weight'] = cov_diag['weight']
            if original_requires_grad(module, 'bias'):
                kwargs['diag_bias'] = cov_diag['bias']
        if all(v is None for v in kwargs.values()):
            return None
        return SymMatrix(**kwargs)

    def full_cov_symmatrix(self, module, pop=False):
        cov = self.get_result(module, OP_FULL_COV, pop=pop)
        if cov is None:
            return
        return SymMatrix(data=cov)

    def cvp_paramvector(self, module, pop=False):
        cvp = self.get_result(module, OP_CVP, pop=pop)
        if cvp is None:
            return None
        params = []
        vectors = []
        if original_requires_grad(module, 'weight'):
            params.append(module.weight)
            vectors.append(cvp['weight'])
        if original_requires_grad(module, 'bias'):
            params.append(module.bias)
            vectors.append(cvp['bias'])
        return ParamVector(params, vectors)

    def full_cvp_paramvector(self, module, pop=False):
        cvp = self.get_result(module, OP_FULL_CVP, pop=pop)
        if cvp is None:
            return None
        params = [p for p in module.parameters() if original_requires_grad(param=p)]
        return ParamVector(params, cvp)

    def rfim_relu(self, module):
        return self.get_result(module, OP_RFIM_RELU)

    def rfim_softmax(self, module):
        return self.get_result(module, OP_RFIM_SOFTMAX)

    def set_input_scale(self, scale):
        self._input_scale = scale

    def set_output_scale(self, scale):
        self._output_scale = scale

    def in_data(self, module):
        return self.get_result(module, OP_SAVE_INPUTS)

    def out_grads(self, module):
        return self.get_result(module, OP_SAVE_OUTGRADS)
